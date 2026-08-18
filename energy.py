import cv2
import numpy as np
import time
from mss import mss

# --------------------------------
# 1. 两个 ROI 区域
ROI_1P = {"left":  90, "top": 95, "width": 50, "height": 1}
ROI_2P = {"left": 774, "top": 97, "width": 50, "height": 1}

# 2. HSV 阈值（直接沿用）
BLUE_LOW  = np.array([92,  46, 251])
BLUE_HIGH = np.array([93,  61, 255])
GRAY_LOW  = np.array([109, 187, 98])
GRAY_HIGH = np.array([109, 188, 99])

sct = mss()

# --------------------------------
def count_blocks(mask):
    """
    返回连续白色区域的个数（单个像素也算，连续只算 1 个）。
    mask 形状 (1, width)
    """
    row = mask[0]
    padded = np.concatenate(([0], row, [0]))
    return int(np.sum((padded[1:] > 0) & (padded[:-1] == 0)))

# --------------------------------
def get_energy_from_roi(roi):
    """给定 ROI，返回该角色的能量值"""
    img = np.array(sct.grab(roi))[:, :, :3]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blue_blocks = count_blocks(cv2.inRange(hsv, BLUE_LOW,  BLUE_HIGH))
    gray_blocks = count_blocks(cv2.inRange(hsv, GRAY_LOW,  GRAY_HIGH))

    if blue_blocks == 0 and gray_blocks == 4:
        return 0
    if blue_blocks == 0 and gray_blocks == 2:
        return 1
    if blue_blocks == 0 and gray_blocks == 1:
        return 1
    if blue_blocks == 1 and gray_blocks == 1:
        return 1
    if blue_blocks == 1 and gray_blocks == 3:
        return 1
    if blue_blocks == 2 and gray_blocks == 1:
        return 1
    if blue_blocks == 2 and gray_blocks == 2:
        return 2
    if blue_blocks == 2 and gray_blocks == 0:
        return 2
    if blue_blocks == 3 and gray_blocks == 1:
        return 3
    if blue_blocks == 3 and gray_blocks == 0:
        return 3
    if blue_blocks == 0 and gray_blocks == 0:
        return 4
    if blue_blocks == 1 and gray_blocks == 0:
        return 4
    return None

# --------------------------------
def main():
    while True:
        energy_1p = get_energy_from_roi(ROI_1P)
        energy_2p = get_energy_from_roi(ROI_2P)
        print(f"1P_energy={energy_1p}, 2P_energy={energy_2p}")
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass