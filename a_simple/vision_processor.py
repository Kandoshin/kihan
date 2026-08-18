import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from concurrent.futures import ThreadPoolExecutor

from game_config import (
    YOLO_PATH, MODEL_PATH, CONF_THRES, IOU_THRES, CNN_WEIGHT_PATH,
    IMG_SIZE, num_classes, idx_to_class, class_to_idx, cnn_regions,
    NONE_ID  # [新增] 导入 NONE_ID
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def normalize_tuple(x, y, w, h, W, H):
    return (round(x / W, 6), round(y / H, 6), round(w / W, 6), round(h / H, 6))


def init_yolo_model():
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
    try:
        print("正在初始化cv模型")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        if os.path.exists(CNN_WEIGHT_PATH):
            try:
                state_dict = torch.load(CNN_WEIGHT_PATH, map_location=DEVICE)
                model.load_state_dict(state_dict, strict=True)
                print("已加载CNN权重")
            except Exception as e:
                print(f"⚠️ 权重加载失败: {e}")
                return None, None
        else:
            print(f"⚠️ 未找到CNN权重: {CNN_WEIGHT_PATH}")
            return None, None

        model.to(DEVICE).eval()
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        return model, transform
    except Exception as e:
        print(f"❌ CNN 初始化失败: {e}")
        return None, None


class ThreadCNNProcessor:
    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self.executor = None
        self.cnn_model = None
        self.transform = None
        self.last_time_val = 0.0  # [新增] 存储上一帧的时间值，默认为1.0

    def start(self):
        try:
            self.cnn_model, self.transform = init_cnn_model()
            if self.cnn_model is None: return False
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            return True
        except Exception:
            return False

    def stop(self):
        if self.executor: self.executor.shutdown(wait=True)

    def parse_cnn_output(self, pred_ids):
        try:
            name_to_idx = {reg["name"]: i for i, reg in enumerate(cnn_regions)}

            # 1. 提取 ID (1p, 2p)
            id_1p = int(pred_ids[name_to_idx["1p"]])
            id_2p = int(pred_ids[name_to_idx["2p"]])

            # 2. 提取标量值 (Scalars)
            scalars = []

            # Time
            t_id = pred_ids[name_to_idx["time"]]
            t_label = idx_to_class.get(int(t_id), "none")

            # [修改] 时间平滑逻辑：如果识别为数字则更新，否则使用上一帧
            if t_label.isdigit():
                t_val = int(t_label) / 60.0
                self.last_time_val = t_val  # 更新缓存
            else:
                # 识别为 none 或置信度低时，使用上一帧的值，防止归零导致异常结束
                t_val = self.last_time_val

            scalars.append(max(0.0, t_val))

            # CD类
            for name in ["stand", "skill1", "skill2"]:
                pid = pred_ids[name_to_idx[name]]
                label = idx_to_class.get(int(pid), "none")
                val = int(label) / 60.0 if label.isdigit() else 0.0
                scalars.append(min(1.0, val))

            # 物品/CD类
            for name in ["summon", "scroll"]:
                pid = pred_ids[name_to_idx[name]]
                label = idx_to_class.get(int(pid), "none")
                if label.isdigit():
                    val = min(1.0, int(label) / 60.0)
                    scalars.append(val)
                else:
                    scalars.append(0.0)

            return id_1p, id_2p, scalars, t_val

        except Exception as e:
            # [修正] 异常时返回 NONE_ID 而不是 0
            return NONE_ID, NONE_ID, [0.0] * 6, -1.0

    def predict(self, crop_images, timeout=2):
        if self.cnn_model is None or self.executor is None:
            return NONE_ID, NONE_ID, [0.0] * 6, []
        try:
            future = self.executor.submit(self._predict_sync, crop_images)
            return future.result(timeout=timeout)
        except Exception:
            return NONE_ID, NONE_ID, [0.0] * 6, []

    def _predict_sync(self, crop_images):
        try:
            tensors = [self.transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crop_images]
            if not tensors: return NONE_ID, NONE_ID, [0.0] * 6, []

            with torch.no_grad():
                probs = torch.softmax(self.cnn_model(torch.stack(tensors).to(DEVICE)), dim=1)
                confs, preds = torch.max(probs, dim=1)

                raw_ids = []
                for p, c in zip(preds.cpu().numpy(), confs.cpu().numpy()):
                    # [修正] 置信度低时返回 NONE_ID
                    raw_ids.append(NONE_ID if c < 0.6 else int(p))

                id_1p, id_2p, scalars, t_val = self.parse_cnn_output(raw_ids)
                return id_1p, id_2p, scalars, raw_ids
        except Exception as e:
            return NONE_ID, NONE_ID, [0.0] * 6, []


_global_cnn_processor = None


def get_global_cnn_processor():
    global _global_cnn_processor
    if _global_cnn_processor is None:
        _global_cnn_processor = ThreadCNNProcessor()
        _global_cnn_processor.start()
    return _global_cnn_processor