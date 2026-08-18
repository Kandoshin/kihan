import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -------------------
# 参数配置
# -------------------
data_dir = r"G:\god\pycharm\PythonProject\test\cnn\dataset\train"
val_dir = r"G:\god\pycharm\PythonProject\test\cnn\dataset\val"
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)
class_map_path = os.path.join(save_dir, "class_to_idx.json")

img_size = 70
batch_size = 64
epochs = 64  # 建议多跑几轮
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义两个模型路径
last_path = os.path.join(save_dir, "last.pth")  # 专门用于记录最新进度
best_path = os.path.join(save_dir, "best.pth")  # 专门用于记录历史最佳

# -------------------
# 数据集准备 & 智能 ID 管理
# -------------------
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- [新增] 自定义 Dataset 类，强制使用固定的 ID 映射 ---
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, custom_class_to_idx=None):
        self.custom_class_to_idx = custom_class_to_idx
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        """
        重写父类方法：直接返回我们指定的 ID 映射，
        而不是让 PyTorch 自动按字母排序重新生成。
        """
        if self.custom_class_to_idx:
            classes = list(self.custom_class_to_idx.keys())
            return classes, self.custom_class_to_idx
        return super().find_classes(directory)

# --- [新增] 智能 ID 更新逻辑 ---
# 1. 加载旧映射
if os.path.exists(class_map_path):
    try:
        with open(class_map_path, 'r', encoding='utf-8') as f:
            final_class_to_idx = json.load(f)
        print(f"📖 加载现有类别映射: {len(final_class_to_idx)} 类")
    except Exception as e:
        print(f"⚠️ 读取旧映射失败: {e}，将重新创建")
        final_class_to_idx = {}
else:
    final_class_to_idx = {}
    print("🆕 未找到旧映射，将创建新映射")

# 2. 扫描磁盘上的实际文件夹 (新类别)
current_classes_on_disk = []
if os.path.exists(data_dir):
    current_classes_on_disk = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])

# 3. 增量追加新类别 (Append-Only)
if final_class_to_idx:
    next_id = max(final_class_to_idx.values()) + 1
else:
    next_id = 0

has_new_class = False
for cls_name in current_classes_on_disk:
    if cls_name not in final_class_to_idx:
        final_class_to_idx[cls_name] = next_id
        print(f"➕ 发现新类别并追加: '{cls_name}' -> ID {next_id}")
        next_id += 1
        has_new_class = True

# 4. 立即保存更新后的映射
if has_new_class or not os.path.exists(class_map_path):
    with open(class_map_path, "w") as f:
        json.dump(final_class_to_idx, f, indent=2)
    print(f"💾 ID 映射已更新并保存至: {class_map_path}")

# 5. 使用自定义 Dataset 加载数据
train_ds = CustomImageFolder(data_dir, transform=train_transform, custom_class_to_idx=final_class_to_idx)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# 验证集也强制使用相同的映射
val_ds = CustomImageFolder(val_dir, transform=val_transform, custom_class_to_idx=final_class_to_idx)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

num_classes = len(final_class_to_idx)
print(f"Num samples: {len(train_ds)}  Num classes: {num_classes}")

# -------------------
# 模型初始化 (ResNet18)
# -------------------
print("正在初始化 ResNet18 (Pretrained)...")
model = models.resnet18(weights='DEFAULT')

# 修改全连接层
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# -------------------
# 智能加载逻辑
# -------------------
start_epoch = 1
resume_path = None

if os.path.exists(last_path):
    print(f"🔄 检测到上次训练进度: {last_path}")
    resume_path = last_path
elif os.path.exists(best_path):
    print(f"⚠️ 未找到上次进度，但发现最佳模型: {best_path}，将基于此继续微调")
    resume_path = best_path
else:
    print("🚀 未发现已有模型，开始全新训练")

if resume_path:
    try:
        state_dict = torch.load(resume_path, map_location=device)
        model_dict = model.state_dict()

        # 1. 过滤掉形状不匹配的层 (主要是 fc 层)
        # 这样即使类别数变了，卷积层的权重也能保留，不用从头练
        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        # 2. 统计匹配情况
        match_count = len(pretrained_dict)
        total_count = len(model_dict)

        if match_count == total_count:
            print("✅ 权重完美匹配，完整加载")
        else:
            print(f"⚠️ 权重部分匹配 ({match_count}/{total_count})。")
            print("⚠️ 已加载卷积层权重，重置了全连接层 (fc) 以适应新类别。")

        # 3. 更新权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    except Exception as e:
        print(f"❌ 加载模型失败: {e}，将重新开始训练")

model = model.to(device)

# -------------------
# 损失 & 优化器
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -------------------
# 训练循环
# -------------------
best_acc = 0.0
best_loss = float("inf")

for epoch in range(start_epoch, epochs + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_avg_loss = running_loss / len(train_loader)

    # 验证
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for val_imgs, val_labels in val_loader:
            val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
            val_outs = model(val_imgs)
            val_loss = criterion(val_outs, val_labels)
            val_running_loss += val_loss.item()
            _, val_preds = torch.max(val_outs, 1)
            val_correct += (val_preds == val_labels).sum().item()
            val_total += val_labels.size(0)

    val_acc = val_correct / val_total if val_total > 0 else 0.0
    val_avg_loss = val_running_loss / len(val_loader) if len(val_loader) > 0 else float("inf")

    print(f"Epoch {epoch}/{epochs}")
    print(f"  Train Loss: {train_avg_loss:.4f}  Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_avg_loss:.4f}  Acc: {val_acc:.4f}")

    # 保存 Last
    torch.save(model.state_dict(), last_path)

    # 保存 Best
    is_best = False
    if val_acc > best_acc:
        is_best = True
    elif val_acc == best_acc and val_avg_loss < best_loss:
        is_best = True

    if is_best:
        best_acc = val_acc
        best_loss = val_avg_loss
        torch.save(model.state_dict(), best_path)
        print(f"  🏆 新纪录！Acc={val_acc:.4f} Loss={val_avg_loss:.4f} -> 已保存 best.pth")

    print("-" * 30)

print("训练结束。")