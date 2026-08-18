import cv2
import numpy as np
import mss
import time
import os

# 1. 读取基准 Hu 矩（同一数字多条 Hu 全部归到 idx）
hu_file = r'G:\god\pycharm\PythonProject\test\time\hutxt\allhu.txt'
base_hu = {}                          # idx -> [hu1, hu2, ...]
with open(hu_file, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx_str, vec_str = line.split(':', 1)
        idx = int(idx_str)
        vec = np.fromstring(vec_str.strip('()'), sep=',', dtype=np.float64)
        base_hu.setdefault(idx, []).append(vec)

# 2. 截图区域
region = {"left": 428, "top": 83, "width": 88, "height": 66}
hsv_low  = np.array([0,   0,   0])
hsv_high = np.array([0,  70, 255])

# 3. 实时循环
print('开始实时匹配（至多两块白色区域合并计算），按 Ctrl+C 退出...')
with mss.mss() as sct:
    try:
        while True:
            # 截屏 → BGR
            img = np.array(sct.grab(region))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # HSV 过滤 → 白色区域
            hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_low, hsv_high)
            white = cv2.bitwise_and(img, img, mask=mask)

            # 灰度 → 找轮廓
            gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                print('无白色区域', end='\r')
                continue

            # 按面积从大到小排序，最多取前两块
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

            # 把这些轮廓合并到同一张 mask 上
            mask_merge = np.zeros_like(gray)
            cv2.drawContours(mask_merge, contours, -1, 255, thickness=cv2.FILLED)

            # 对合并后的整体计算 Hu 矩
            M = cv2.moments(mask_merge)
            live_hu = cv2.HuMoments(M).flatten()

            # 在基准库中找最近邻
            best_num = min(
                base_hu,
                key=lambda num: min(np.linalg.norm(hu - live_hu)
                                    for hu in base_hu[num])
            )
            print(f'最相似: {best_num}  | 当前Hu: {live_hu.tolist()}')
            # print(f'\r最相似: {best_idx}')
            time.sleep(0.1)

    except KeyboardInterrupt:
        print('\n已退出')