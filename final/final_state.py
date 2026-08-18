# realtime_state_embedding.py  (最终整合版)
import cv2
import numpy as np
import time
import json
import torch
import mss
from PIL import Image
from torchvision import transforms, models
import os
import torch.nn as nn
from typing import Tuple

# -------------------------------------------------
# YOLO 配置
# -------------------------------------------------
YOLO_PATH   = r"G:\god\yolov5\yolov5-7.0\yolov5-7.0"
MODEL_PATH  = r"G:\god\yolov5\yolov5-7.0\yolov5-7.0\best.pt"
GAME_REGION = {"left": 0, "top": 46, "width": 945, "height": 535}
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

CONF_THRES, IOU_THRES = 0.6, 0.45
yolo_model = torch.hub.load(YOLO_PATH, 'custom', path=MODEL_PATH, source='local', device=DEVICE)
yolo_model.conf, yolo_model.iou = CONF_THRES, IOU_THRES
yolo_model.eval()
TARGET_CLASSES = {'p', 'b', 'r', 's', 'wm', 'rm'}

# -------------------------------------------------
# CNN 配置
# -------------------------------------------------
img_size = 70
save_dir = r"G:\god\pycharm\PythonProject\test\cnn\models"
class_map_path = os.path.join(save_dir, "class_to_idx.json")
with open(class_map_path, "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

cnn_model = models.resnet18(weights=None)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, num_classes)
cnn_model.load_state_dict(torch.load(os.path.join(save_dir, "best.pth"), map_location=DEVICE))
cnn_model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

embedding_dim = 16
embedding_layer = nn.Embedding(num_classes, embedding_dim).to(DEVICE)

# 区域列表：已追加 1p_energy / 2p_energy
cnn_regions = [
    {"left": 30,  "top": 48,  "width": 70, "height": 70, "name": "1p"},
    {"left": 845, "top": 48,  "width": 70, "height": 70, "name": "2p"},
    {"left": 824, "top": 151, "width": 56, "height": 56, "name": "summon"},
    {"left": 824, "top": 238, "width": 56, "height": 56, "name": "scroll"},
    {"left": 624, "top": 506, "width": 30, "height": 30, "name": "stand"},
    {"left": 726, "top": 505, "width": 30, "height": 30, "name": "skill1"},
    {"left": 738, "top": 400, "width": 30, "height": 30, "name": "skill2"},
    {"left": 96,  "top": 92,  "width": 60, "height": 16, "name": "1p_energy"},
    {"left": 790, "top": 92,  "width": 60, "height": 16, "name": "2p_energy"},
]

# -------------------------------------------------
# ROI
# -------------------------------------------------
MONITOR_1P = {"left": 99,  "top": 78, "width": 175, "height": 9}
MONITOR_2P = {"left": 672, "top": 78, "width": 175, "height": 9}
time_region = {"left": 428, "top": 83, "width": 88, "height": 66}

HSV_LOW  = np.array([0, 200, 200])
HSV_HIGH = np.array([10, 255, 255])
KERNEL   = np.ones((3, 3), np.uint8)

hu_file = r'G:\god\pycharm\PythonProject\test\time\hutxt\allhu.txt'
base_hu = {}
with open(hu_file, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx_str, vec_str = line.split(':', 1)
        idx = int(idx_str)
        vec = np.fromstring(vec_str.strip('()'), sep=',', dtype=np.float64)
        base_hu.setdefault(idx, []).append(vec)

# -------------------------------------------------
# 工具函数
# -------------------------------------------------
def cnn_predict(img_np: np.ndarray) -> Tuple[str, int]:
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        id_int = int(torch.argmax(cnn_model(x), dim=1))
    label_str = idx_to_class[id_int]
    return label_str, id_int

def normalize_tuple(x, y, w, h, W, H):
    return (round(x / W, 4), round(y / H, 4), round(w / W, 4), round(h / H, 4))

def get_health(crop, last):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    row_counts = np.sum(mask[2:7] > 0, axis=1)
    if len(row_counts) == 0:
        return last
    max_val = int(np.max(row_counts))
    use_val = int(np.min(row_counts)) if max_val > 81 else max_val
    return round(max(0, min(use_val / 166.0, 1.0)) * 100, 4)

def get_time(crop):
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 0), (0, 70, 255))
        white = cv2.bitwise_and(crop, crop, mask=mask)
        gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        mask_merge = np.zeros_like(gray)
        cv2.drawContours(mask_merge, contours, -1, 255, cv2.FILLED)
        M = cv2.moments(mask_merge)
        live_hu = cv2.HuMoments(M).flatten()
        best = min(base_hu, key=lambda num: min(np.linalg.norm(hu - live_hu) for hu in base_hu[num]))
        return int(best)
    except Exception:
        return 0

# -------------------------------------------------
# 主循环
# -------------------------------------------------
print("开始检测，Ctrl+C退出")
with mss.mss() as sct:
    while True:
        tic = time.time()
        full_img = np.array(sct.grab(sct.monitors[0]))[..., :3]

        # ---- YOLO ----
        g = GAME_REGION
        game_crop = full_img[g['top']:g['top'] + g['height'], g['left']:g['left'] + g['width']]
        results = yolo_model(cv2.cvtColor(game_crop, cv2.COLOR_BGR2RGB), size=640)
        boxes = {k: [] for k in TARGET_CLASSES}
        for *xyxy, conf, cls in results.pred[0]:
            cls_name = yolo_model.names[int(cls)]
            if cls_name in boxes:
                x1, y1, x2, y2 = map(float, xyxy)
                x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2
                boxes[cls_name].append(
                    normalize_tuple(x_c, y_c, x2 - x1, y2 - y1, g['width'], g['height'])
                )
        for k in boxes:
            boxes[k] += [(0, 0, 0, 0)] * (4 - len(boxes[k]))

        # ---- CNN 特征提取 ----
        cnn_features = {}
        embed_ids = []

        for reg in cnn_regions:
            crop = full_img[reg["top"]:reg["top"] + reg["height"],
                            reg["left"]:reg["left"] + reg["width"]]
            label, id_int = cnn_predict(crop)

            if label == "none":
                cnn_features[reg["name"]] = 0.0
            elif label.isdigit():
                # 能量区除以 4，其余除以 60
                div = 4.0 if reg["name"] in {"1p_energy", "2p_energy"} else 60.0
                value = int(label) / div
                cnn_features[reg["name"]] = value
            else:  # 英文标签
                embed_ids.append(id_int)
                cnn_features[reg["name"]] = id_int  # 暂存 id，后面替换为向量

        # 一次性 Embedding
        if embed_ids:
            embeds = embedding_layer(torch.tensor(embed_ids, device=DEVICE)) \
                      .detach() \
                      .cpu() \
                      .numpy()
            idx = 0
            for reg in cnn_regions:
                label, _ = cnn_predict(full_img[reg["top"]:reg["top"] + reg["height"],
                                                reg["left"]:reg["left"] + reg["width"]])
                if (not label.isdigit()) and label != "none":
                    cnn_features[reg["name"]] = embeds[idx].tolist()
                    idx += 1

        # ---- 血量 & 时间 ----
        crop1 = full_img[MONITOR_1P['top']:MONITOR_1P['top'] + MONITOR_1P['height'],
                         MONITOR_1P['left']:MONITOR_1P['left'] + MONITOR_1P['width']]
        crop2 = full_img[MONITOR_2P['top']:MONITOR_2P['top'] + MONITOR_2P['height'],
                         MONITOR_2P['left']:MONITOR_2P['left'] + MONITOR_2P['width']]
        time_crop = full_img[time_region['top']:time_region['top'] + time_region['height'],
                             time_region['left']:time_region['left'] + time_region['width']]

        p1_blood = get_health(crop1, 0.0) / 100
        p2_blood = get_health(crop2, 0.0) / 100
        current_time = get_time(time_crop) / 60

        # ---- 最终状态 ----
        state = {
            "time": round(current_time, 4),
            "p1_blood": round(p1_blood, 4),
            "p2_blood": round(p2_blood, 4),
            "yolo": boxes,
            "cnn_features": cnn_features
        }

        print(state)
        time.sleep(max(0, 0.1 - (time.time() - tic)))