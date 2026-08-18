# snapshot_once.py
from mss import mss
import numpy as np
import cv2
import os

save_dir = r"G:\god\pycharm\PythonProject\test\data\images"
os.makedirs(save_dir, exist_ok=True)

with mss() as sct:
    region1 = {"left":  96, "top": 92, "width": 60, "height": 16}
    img1 = sct.grab(region1)
    cv2.imwrite(os.path.join(save_dir, "1.png"), np.array(img1)[:, :, :3])

    region2 = {"left": 790, "top": 92, "width": 60, "height": 16}
    img2 = sct.grab(region2)
    cv2.imwrite(os.path.join(save_dir, "2.png"), np.array(img2)[:, :, :3])

print("已保存")