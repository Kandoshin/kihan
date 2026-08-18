import time
import json
import cv2
import numpy as np
import torch
import mss
from PIL import Image
from torchvision import transforms, models
import os
import random
import torch.nn as nn

# ---------------------------
# 1. 基础配置
# ---------------------------
img_size = 70
device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = "models"
model_path = f"{save_dir}/best.pth"
class_map_path = f"{save_dir}/class_to_idx.json"
save_low_conf_dir = r"G:\god\pycharm\PythonProject\test\data\images"
os.makedirs(save_low_conf_dir, exist_ok=True)

# 读取类别映射
with open(class_map_path, "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

# ---------------------------
# 2. 加载模型 (ResNet18)
# ---------------------------
print("正在加载 ResNet18 模型...")
model = models.resnet18(weights=None)
# 修改全连接层 (ResNet 是 .fc)
model.fc = nn.Linear(model.fc.in_features, num_classes)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ 成功加载权重: {model_path}")
except Exception as e:
    print(f"❌ 权重加载失败: {e}")
    print("请确保你已经运行了新的 train.py 训练了 ResNet18 模型！")
    exit()

model.to(device).eval()

# ---------------------------
# 3. 图像预处理
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ---------------------------
# 4. 定义检测区域
# ---------------------------
# ... (保持原来的 regions 不变)
regions = [
    {"left": 30, "top": 58, "width": 70, "height": 70, "name": "1p"},
    {"left": 845, "top": 58, "width": 70, "height": 70, "name": "2p"},
    {"left": 824, "top": 151, "width": 56, "height": 56, "name": "summon"},
    {"left": 824, "top": 238, "width": 56, "height": 56, "name": "scroll"},
    {"left": 624, "top": 506, "width": 30, "height": 30, "name": "stand"},
    {"left": 726, "top": 505, "width": 30, "height": 30, "name": "skill1"},
    {"left": 738, "top": 400, "width": 30, "height": 30, "name": "skill2"},
    # {"left": 96, "top": 102, "width": 60, "height": 16, "name": "1p_energy"},
    # {"left": 790, "top": 102, "width": 60, "height": 16, "name": "2p_energy"},
    {"left": 428, "top": 83, "width": 88, "height": 66, "name": "time"},
]


# ---------------------------
# 5. 单张图推理函数
# ---------------------------
def predict_region(img_np):
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGB")
    x = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(prob.argmax())
    return idx_to_class[pred], float(prob[pred])


# ---------------------------
# 6. 主循环
# ---------------------------
sct = mss.mss()
print("开始实时检测，按 Ctrl+C 停止...\n")
try:
    while True:
        tic = time.time()
        full_img = np.array(sct.grab(sct.monitors[0]))[..., :3]

        result_line = []
        for r in regions:
            crop = full_img[r["top"]:r["top"] + r["height"],
                   r["left"]:r["left"] + r["width"]]
            label, conf = predict_region(crop)
            result_line.append(f"{r['name']}:{label}({conf:.2f})")

            # 保存低置信度图片
            if conf < 0.8:
                name_base = f"{int(time.time() * 1000) % 1000000:06d}{random.randint(0, 999):03d}"
                save_name = os.path.join(save_low_conf_dir, f"{name_base}.png")
                cv2.imwrite(save_name, crop)

        print(" | ".join(result_line))

        elapsed = time.time() - tic
        time.sleep(max(0, 0.1 - elapsed))

except KeyboardInterrupt:
    print("\n已停止。")