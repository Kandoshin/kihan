import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from concurrent.futures import ThreadPoolExecutor

# 导入配置
from game_config import (
    YOLO_PATH, MODEL_PATH, CONF_THRES, IOU_THRES, CNN_WEIGHT_PATH,
    IMG_SIZE, HSV_LOW, HSV_HIGH, num_classes, idx_to_class, class_to_idx,
    cnn_regions
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def normalize_tuple(x, y, w, h, W, H):
    # [修改] 统一为 6 位小数
    return (round(x / W, 6), round(y / H, 6), round(w / W, 6), round(h / H, 6))


def get_health(crop):
    """获取血量百分比"""
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
        if mask.shape[0] == 0: return 1.00
        row1 = int(mask[0].sum() // 255)
        row_last = int(mask[-1].sum() // 255)
        if row1 > 82: return round(row1 / 164, 6)
        return round(row_last / 164, 6)
    except Exception:
        return 1.00


def init_yolo_model():
    """初始化 YOLO 模型"""
    try:
        if not os.path.exists(YOLO_PATH) or not os.path.exists(MODEL_PATH):
            return None
        yolo = torch.hub.load(YOLO_PATH, 'custom', path=MODEL_PATH, source='local', device=DEVICE)
        yolo.conf, yolo.iou = CONF_THRES, IOU_THRES
        yolo.eval()
        return yolo
    except Exception as e:
        print(f"YOLO 模型加载失败: {e}")
        return None


def init_cnn_model():
    """初始化 CNN 模型 (ResNet18)"""
    try:
        print("🔄 正在初始化 CNN 模型 (ResNet18)...")

        # 1. 定义模型架构
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # 2. 加载本地权重
        if os.path.exists(CNN_WEIGHT_PATH):
            try:
                state_dict = torch.load(CNN_WEIGHT_PATH, map_location=DEVICE)
                model.load_state_dict(state_dict, strict=True)
                print(f"✅ 已加载CNN权重: {CNN_WEIGHT_PATH}")
            except Exception as e:
                print(f"⚠️ 权重加载失败: {e}")
                return None, None, None
        else:
            print(f"⚠️ 未找到CNN权重: {CNN_WEIGHT_PATH}，无法继续")
            return None, None, None

        model.to(DEVICE).eval()

        # 3. Embedding 层
        embedding_layer = nn.Embedding(num_classes, 16).to(DEVICE)

        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        return model, embedding_layer, transform
    except Exception as e:
        print(f"❌ CNN 初始化失败: {e}")
        return None, None, None


class ThreadCNNProcessor:
    """使用线程池的 CNN 处理器"""

    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self.executor = None
        self.cnn_model = None
        self.embedding_layer = None
        self.transform = None
        self.last_energy_features = {"blue_energy": 0.0, "red_energy": 0.0}

    def start(self):
        try:
            self.cnn_model, self.embedding_layer, self.transform = init_cnn_model()
            if self.cnn_model is None: return False
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            return True
        except Exception:
            return False

    def stop(self):
        if self.executor: self.executor.shutdown(wait=True)

    def get_expected_dim(self):
        dim = 0
        for reg in cnn_regions:
            name = reg["name"]
            if name in {"1p", "2p", "summon", "scroll"}:
                dim += 16
            else:
                dim += 1
        return dim

    def convert_predictions_to_features(self, pred_ids, assign_flag=None):
        try:
            cnn_vecs = []
            name_to_idx = {reg["name"]: i for i, reg in enumerate(cnn_regions)}

            if assign_flag == 'left':
                player_order, energy_order = ['1p', '2p'], ['1p_energy', '2p_energy']
            else:
                player_order, energy_order = ['2p', '1p'], ['2p_energy', '1p_energy']

            # 1. 玩家 (Embedding 16维)
            for pname in player_order:
                pred_id = pred_ids[name_to_idx[pname]]
                label = idx_to_class.get(int(pred_id), "none")
                if label == "none":
                    cnn_vecs.extend([0.0] * 16)
                else:
                    vec = self.embedding_layer(torch.tensor([pred_id], device=DEVICE)).squeeze(0).detach().cpu().numpy()
                    cnn_vecs.extend(vec.tolist())

            # 2. 召唤物/卷轴
            for name in ["summon", "scroll"]:
                pred_id = pred_ids[name_to_idx[name]]
                label = idx_to_class.get(int(pred_id), "none")
                if label.isdigit():
                    val = min(1.0, int(label) / 60.0)
                    cnn_vecs.extend([val] * 16)
                elif label == "none":
                    cnn_vecs.extend([0.0] * 16)
                else:
                    vec = self.embedding_layer(torch.tensor([pred_id], device=DEVICE)).squeeze(0).detach().cpu().numpy()
                    cnn_vecs.extend(vec.tolist())

            # 3. CD
            for name in ["stand", "skill1", "skill2"]:
                pred_id = pred_ids[name_to_idx[name]]
                label = idx_to_class.get(int(pred_id), "none")
                val = int(label) / 60.0 if label.isdigit() else 0.0
                cnn_vecs.append(min(1.0, val))

            # 4. 能量
            for i, ename in enumerate(energy_order):
                key = "blue_energy" if i == 0 else "red_energy"
                pred_id = pred_ids[name_to_idx[ename]]
                label = idx_to_class.get(int(pred_id), "none")
                if label.isdigit():
                    val = min(1.0, int(label) / 4.0)
                    self.last_energy_features[key] = val
                    cnn_vecs.append(val)
                else:
                    cnn_vecs.append(self.last_energy_features[key])

            # 5. 时间
            pred_id = pred_ids[name_to_idx["time"]]
            label = idx_to_class.get(int(pred_id), "none")
            val = int(label) / 60.0 if label.isdigit() else -1.0
            cnn_vecs.append(min(1.0, val) if val != -1.0 else -1.0)

            return cnn_vecs
        except Exception as e:
            print(f"Feature convert error: {e}")
            return [0.0] * self.get_expected_dim()

    def predict(self, crop_images, assign_flag=None, timeout=2):
        """
        [修改] 返回 (features, raw_ids) 元组
        features: 给 PPO 的输入
        raw_ids: 原始类别 ID，给 main_training 判断死亡用
        """
        if self.cnn_model is None or self.executor is None:
            return [0.0] * self.get_expected_dim(), []
        try:
            future = self.executor.submit(self._predict_sync, crop_images, assign_flag)
            return future.result(timeout=timeout)
        except Exception:
            return [0.0] * self.get_expected_dim(), []

    def _predict_sync(self, crop_images, assign_flag):
        try:
            tensors = [self.transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crop_images]
            if not tensors: return [0.0] * self.get_expected_dim(), []

            with torch.no_grad():
                probs = torch.softmax(self.cnn_model(torch.stack(tensors).to(DEVICE)), dim=1)
                confs, preds = torch.max(probs, dim=1)

                adjusted = []
                for p, c in zip(preds.cpu().numpy(), confs.cpu().numpy()):
                    adjusted.append(class_to_idx.get("none", 0) if c < 0.6 else int(p))

                # [关键修改] 返回特征的同时，也返回 adjusted 原始 ID 列表
                features = self.convert_predictions_to_features(adjusted, assign_flag)
                return features, adjusted
        except Exception as e:
            print(f"Prediction error: {e}")
            return [0.0] * self.get_expected_dim(), []


# 全局单例
_global_cnn_processor = None


def get_global_cnn_processor():
    global _global_cnn_processor
    if _global_cnn_processor is None:
        _global_cnn_processor = ThreadCNNProcessor()
        _global_cnn_processor.start()
    return _global_cnn_processor