# p12_blood_meter.py
import cv2
import numpy as np
from mss import mss

# 两个 ROI
MONITOR_1P = {"left": 98, "top": 91, "width": 176, "height": 5}
MONITOR_2P = {"left": 672, "top": 91, "width": 176, "height": 5}

HSV_LOW  = np.array([0,   0, 251])
HSV_HIGH = np.array([179, 255, 255])

sct = mct = mss()

def blood_value(mask):
    """根据 mask 返回血量比例（6 位小数）"""
    row1 = int(mask[0].sum() // 255)   # 第 1 行白像素数
    row6 = int(mask[4].sum() // 255)   # 第 5 行白像素数
    if row1 > 82:
        return round(row1 / 164, 5)
    return round(row6 / 164, 5)

while True:
    # 1P
    roi  = np.array(sct.grab(MONITOR_1P))[:, :, :3]
    mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), HSV_LOW, HSV_HIGH)
    p1_blood = blood_value(mask)

    # 2P
    roi  = np.array(sct.grab(MONITOR_2P))[:, :, :3]
    mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), HSV_LOW, HSV_HIGH)
    p2_blood = blood_value(mask)

    print(f"p1_blood:{p1_blood:.6f}  p2_blood:{p2_blood:.6f}")