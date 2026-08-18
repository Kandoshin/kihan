from mss import mss
import numpy as np
import cv2
import os
import time
import random

save_dir = r"G:\god\pycharm\PythonProject\test\data\images"
os.makedirs(save_dir, exist_ok=True)

# 当前时间戳后 6 位 + 3 位随机数
def make_name():
    tail = str(int(time.time() * 1000))[-6:]          # 后 6 位
    rand = random.randint(0, 999)                     # 随机 0-999
    return f"{tail}{rand:03d}"                        # 共 9 位，如 123456789

with mss() as sct:
    regions = [
        {"left": 30, "top": 48,"width": 70, "height": 70},#1p头像
        {"left": 845, "top": 48, "width": 70, "height": 70},#2p头像
        {"left": 824, "top": 151, "width": 56, "height": 56},#己方通灵
        {"left": 824, "top": 238, "width": 56, "height": 56},#己方密卷
        {"left": 726, "top": 505, "width": 30, "height": 30},#skill1
        {"left": 738, "top": 400, "width": 30, "height": 30},#skill2
        {"left": 624, "top": 506, "width": 30, "height": 30},#stand

         {"left": 0, "top": 46,
          "width": 945, "height": 535}#game_region
    ]

    for region in regions:
        img = sct.grab(region)
        name = make_name()
        cv2.imwrite(os.path.join(save_dir, f"{name}.png"), np.array(img)[:, :, :3])

print("已保存")