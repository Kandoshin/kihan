import cv2
import numpy as np
import time
from mss import mss

# ROI：1 行
ROI = {"left": 90, "top": 95, "width": 50, "height": 1}

# HSV 区间
BLUE_LOW  = np.array([1,   46, 251])
BLUE_HIGH = np.array([93,  51, 255])
GRAY_LOW  = np.array([109, 187, 98])
GRAY_HIGH = np.array([109, 187, 98])

sct = mss()

def white_indices(mask):
    """返回该行里白色像素的列号元组"""
    # mask 形状 (1, width) → 取第 0 行即可
    return tuple(np.where(mask[0] > 0)[0])

def main():
    while True:
        # 1. 截 1 行
        img = np.array(sct.grab(ROI))[:, :, :3]   # BGR
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 2. 两个掩膜
        blue_mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
        gray_mask = cv2.inRange(hsv, GRAY_LOW, GRAY_HIGH)

        # 3. 取白色列号
        blue_cols = white_indices(blue_mask)
        gray_cols = white_indices(gray_mask)

        # 4. 实时输出
        print(f"blue：{blue_cols}  gray：{gray_cols}")
        time.sleep(0.05)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass