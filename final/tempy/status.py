import cv2
import numpy as np
import time
import json
import torch
import mss
from PIL import Image
from torchvision import transforms, models
from mss.exception import ScreenShotError
import os

# --------------------------- 1. 血量 ---------------------------
MONITOR_1P = {"left": 99,  "top": 78, "width": 175, "height": 9}
MONITOR_2P = {"left": 672, "top": 78, "width": 175, "height": 9}

HSV_LOW  = np.array([0, 200, 200])
HSV_HIGH = np.array([10, 255, 255])
KERNEL   = np.ones((3, 3), np.uint8)

_last_1p = 0.0
_last_2p = 0.0

def get_health(monitor: dict, last_val: float) -> float:
    try:
        with mss.mss() as sct:
            frame = np.array(sct.grab(monitor))[:, :, :3]
    except ScreenShotError:
        return last_val

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    row_counts = np.sum(mask[2:7] > 0, axis=1)
    max_val = int(np.max(row_counts))
    use_val = int(np.min(row_counts)) if max_val > 81 else max_val
    return round(max(0.0, min(use_val / 166.0, 1.0)) * 100, 4)

# --------------------------- 2. 时间 ---------------------------
hu_file = r'G:\god\pycharm\PythonProject\test\time\hutxt\allhu.txt'
base_hu = {}
with open(hu_file, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        idx_str, vec_str = line.split(':', 1)
        idx = int(idx_str)
        vec = np.fromstring(vec_str.strip('()'), sep=',', dtype=np.float64)
        base_hu.setdefault(idx, []).append(vec)

time_region = {"left": 428, "top": 83, "width": 88, "height": 66}
time_hsv_low  = np.array([0,   0,   0])
time_hsv_high = np.array([0,  70, 255])

def get_time() -> int:
    try:
        with mss.mss() as sct:
            img = np.array(sct.grab(time_region))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, time_hsv_low, time_hsv_high)
        white = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return 0
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        mask_merge = np.zeros_like(gray)
        cv2.drawContours(mask_merge, contours, -1, 255, cv2.FILLED)
        M = cv2.moments(mask_merge)
        live_hu = cv2.HuMoments(M).flatten()
        best_num = min(base_hu, key=lambda num: min(np.linalg.norm(hu - live_hu) for hu in base_hu[num]))
        return int(best_num)
    except Exception:
        return 0

# --------------------------- 3. CNN 区域 ---------------------------
img_size = 70
device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = r"G:\god\pycharm\PythonProject\test\cnn\models"
model_path      = os.path.join(save_dir, "best.pth")
class_map_path  = os.path.join(save_dir, "class_to_idx.json")

with open(class_map_path, "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = models.resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

regions = [
    {"left": 30, "top": 48,"width": 70, "height": 70,"name": "1p"},
    {"left": 845, "top": 48, "width": 70, "height": 70, "name": "2p"},
    {"left": 824, "top": 151, "width": 56, "height": 56, "name": "summon"},
    {"left": 824, "top": 238, "width": 56, "height": 56, "name": "scroll"},
    {"left": 624, "top": 506, "width": 30, "height": 30,"name": "stand"},
    {"left": 726, "top": 505, "width": 30, "height": 30,"name": "skill1"},
    {"left": 738, "top": 400, "width": 30, "height": 30,"name": "skill2"},
    {"left": 96,  "top": 92, "width": 60, "height": 16,"name": "1p_energy"},
    {"left": 790, "top": 92, "width": 60, "height": 16,"name": "2p_energy"},
]

def predict_region(img_np):
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGB")
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(prob.argmax())
    return idx_to_class[pred]

# --------------------------- 4. 主循环 ---------------------------
current_time = 0
p1_blood = 0.0
p2_blood = 0.0
region_vars = {r["name"]: "" for r in regions}

try:
    while True:
        tic = time.time()

        current_time = get_time()
        p1_blood = get_health(MONITOR_1P, p1_blood)
        p2_blood = get_health(MONITOR_2P, p2_blood)

        with mss.mss() as sct:
            full_img = np.array(sct.grab(sct.monitors[0]))[..., :3]
        for r in regions:
            crop = full_img[r["top"]:r["top"]+r["height"], r["left"]:r["left"]+r["width"]]
            region_vars[r["name"]] = predict_region(crop)

        print(f"current_time:{current_time} p1_blood:{p1_blood:.4f} p2_blood:{p2_blood:.4f} "
              + " ".join([f"{k}:{v}" for k, v in region_vars.items()]))

        elapsed = time.time() - tic
        time.sleep(max(0, 0.1 - elapsed))

except KeyboardInterrupt:
    pass