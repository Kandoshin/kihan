import cv2
import numpy as np
import os

# ------------- 基本路径 -------------
base_dir = r'G:\god\pycharm\PythonProject\test\time'
src_path = os.path.join(base_dir, 'images', '50.png')
save_dir = base_dir

# 如果目录不存在就创建（理论上已存在，保险）
os.makedirs(save_dir, exist_ok=True)

# ------------- 1. 读取并 HSV 过滤 -------------
img_bgr = cv2.imread(src_path)
if img_bgr is None:
    raise FileNotFoundError(f'无法读取图像: {src_path}')

hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
lower = (0, 0, 0)
upper = (0, 70, 255)
mask = cv2.inRange(hsv, lower, upper)      # 单通道 mask
filtered = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

cv2.imwrite(os.path.join(save_dir, '00_hsv_mask.png'), mask)
cv2.imwrite(os.path.join(save_dir, '01_filtered.png'), filtered)

# ------------- 2. 灰度 & 二值 -------------
gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

cv2.imwrite(os.path.join(save_dir, '02_gray.png'), gray)
cv2.imwrite(os.path.join(save_dir, '03_binary.png'), binary)

# ------------- 3. 轮廓 -------------
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print('未检测到任何轮廓，无法计算 Hu 矩。')
else:
    largest = max(contours, key=cv2.contourArea)

    # 为了可视化，复制一份彩色图
    contour_vis = filtered.copy()
    cv2.drawContours(contour_vis, [largest], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(save_dir, '04_contours.png'), contour_vis)

    # ------------- 4. Hu 矩 -------------
    M = cv2.moments(largest)
    hu = cv2.HuMoments(M).flatten()

    # 保存数值
    hu_path = os.path.join(save_dir, '05_hu_moments.txt')
    with open(hu_path, 'w') as f:
        for i, val in enumerate(hu, 1):
            f.write(f'Hu[{i}] = {val:.6e}\n')

    print('Hu 矩已写入:', hu_path)

print('所有中间结果已保存到:', save_dir)