import cv2
import numpy as np
import time
from mss import mss
from mss.exception import ScreenShotError

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
        print("截图异常")
        return last_val

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    row_counts = np.sum(mask[2:7] > 0, axis=1)
    max_val = int(np.max(row_counts))
    use_val = int(np.min(row_counts))

    ratio = max(0.0, min(use_val / 166.0, 1.0))
    return round(ratio * 100, 4)          # 乘100

def main():
    global _last_1p, _last_2p
    while True:
        _last_1p = get_health(MONITOR_1P, _last_1p)
        _last_2p = get_health(MONITOR_2P, _last_2p)
        print(f"1p_blood:{_last_1p:.4f}  2p_blood:{_last_2p:.4f}")  # 打印百分比
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass