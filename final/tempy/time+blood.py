import cv2
import numpy as np
import time
from mss import mss
from mss.exception import ScreenShotError

# 1. 读取基准 Hu 矩
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

# 2. 血量监控区域
MONITOR_1P = {"left": 99,  "top": 78, "width": 175, "height": 9}
MONITOR_2P = {"left": 672, "top": 78, "width": 175, "height": 9}
HSV_LOW  = np.array([0, 200, 200])
HSV_HIGH = np.array([10, 255, 255])
KERNEL   = np.ones((3, 3), np.uint8)

_last_1p = 0.0
_last_2p = 0.0

def get_health(monitor: dict, last_val: float) -> float:
    global _last_1p, _last_2p
    try:
        frame = np.array(mss().grab(monitor))[:, :, :3]
    except ScreenShotError:
        return last_val

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    row_counts = np.sum(mask[2:7] > 0, axis=1)
    max_val = int(np.max(row_counts))
    use_val = int(np.min(row_counts)) if max_val > 81 else max_val
    ratio = max(0.0, min(use_val / 166.0, 1.0))
    return round(ratio * 100, 4)

# 3. 时间数字识别区域
time_region = {"left": 428, "top": 83, "width": 88, "height": 66}
time_hsv_low  = np.array([0,   0,   0])
time_hsv_high = np.array([0,  70, 255])

def get_time() -> int:
    try:
        img = np.array(mss().grab(time_region))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, time_hsv_low, time_hsv_high)
        white = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        mask_merge = np.zeros_like(gray)
        cv2.drawContours(mask_merge, contours, -1, 255, cv2.FILLED)
        M = cv2.moments(mask_merge)
        live_hu = cv2.HuMoments(M).flatten()
        best_num = min(base_hu, key=lambda num: min(np.linalg.norm(hu - live_hu) for hu in base_hu[num]))
        return int(best_num)
    except Exception:
        return 0

# 4. 主循环
if __name__ == "__main__":
    try:
        while True:
            # 血量
            _last_1p = get_health(MONITOR_1P, _last_1p)
            _last_2p = get_health(MONITOR_2P, _last_2p)
            # 时间
            time_val = get_time()
            # 仅输出结果
            print(f"{time_val} {_last_1p} {_last_2p}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass