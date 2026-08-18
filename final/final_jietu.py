from mss import mss
import numpy as np
import cv2
import os
import time
import random
import keyboard   # pip install keyboard

save_dir = r"G:\god\pycharm\PythonProject\test\data\images"
os.makedirs(save_dir, exist_ok=True)

# 生成文件名：时间后 6 位 + 3 位随机
def make_name():
    tail = str(int(time.time() * 1000))[-6:]
    rand = random.randint(0, 999)
    return f"{tail}{rand:03d}"

with mss() as sct:
    regions = [
        {"left": 30, "top": 48, "width": 70, "height": 70},  # 1p头像
        {"left": 845, "top": 48, "width": 70, "height": 70},  # 2p头像
        # {"left": 824, "top": 151, "width": 56, "height": 56},  # 己方通灵
        # {"left": 824, "top": 238, "width": 56, "height": 56},  # 己方密卷
        # {"left": 726, "top": 505, "width": 30, "height": 30},  # skill1
        # {"left": 738, "top": 400, "width": 30, "height": 30},  # skill2
        # {"left": 624, "top": 506, "width": 30, "height": 30},  # stand
        {"left": 0, "top": 46, "width": 945, "height": 535},  # game_region
        # {"left": 428,   "top": 83,  "width": 88, "height": 66} #time
        # {"left": 99, "top": 78, "width": 175, "height": 9},#p1_blood
        # {"left": 672, "top": 78, "width": 175, "height": 9},#p2_blood

    ]

    print("开始循环截图：每 30 秒自动截一次；按 C 键立即截图一次。Ctrl+C 退出。")

    last_auto = time.time()  # 上一次自动截图时间

    try:
        while True:
            now = time.time()

            # 1. 定时自动截图c
            if now - last_auto >= 30:
                for region in regions:
                    img = sct.grab(region)
                    file_name = os.path.join(save_dir, f"{make_name()}.png")
                    cv2.imwrite(file_name, np.array(img)[:, :, :3])
                print("[自动] 已截图")
                last_auto = now

            # 2. 按 C 键手动截图（非阻塞）
            # if keyboard.is_pressed('c'):
            #     for region in regions:
            #         img = sct.grab(region)
            #         file_name = os.path.join(save_dir, f"manual_{make_name()}.png")
            #         cv2.imwrite(file_name, np.array(img)[:, :, :3])
            #     print("[手动] 已截图")
                # 防止连按：等待 0.3 秒再检测
                time.sleep(0.3)

            # 3. 降低 CPU 占用
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("已停止截图")