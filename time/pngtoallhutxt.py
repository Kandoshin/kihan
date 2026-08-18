import cv2
import numpy as np
import os

# 1. 路径
IMAGE_DIR   = r'G:\god\pycharm\PythonProject\test\time\images'
OUTPUT_FILE = r'G:\god\pycharm\PythonProject\test\time\hutxt\allhu.txt'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 2. HSV 范围（与实时脚本一致）
hsv_low  = np.array([0,   0,   0])
hsv_high = np.array([0,  70, 255])

# 3. 收集所有 Hu 特征
all_lines = []

for idx in range(61):
    img_path = os.path.join(IMAGE_DIR, f'{idx}.png')
    if not os.path.exists(img_path):
        print(f'{idx}.png not found, skipped.')
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f'Cannot read {idx}.png, skipped.')
        continue

    # HSV 过滤
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, hsv_low, hsv_high)
    white = cv2.bitwise_and(img, img, mask=mask)

    # 灰度 → 找轮廓
    gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 无白色区域时跳过
    if not contours:
        print(f'{idx}.png 无白色区域，跳过')
        continue

    # 至多两块合并
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    merge_mask = np.zeros_like(gray)
    cv2.drawContours(merge_mask, contours, -1, 255, thickness=cv2.FILLED)

    # 计算整体 Hu 矩
    M  = cv2.moments(merge_mask)
    hu = cv2.HuMoments(M).flatten()

    # 写入列表
    hu_str = ','.join(f'{v:.8e}' for v in hu)
    all_lines.append(f'{idx}:({hu_str})')

# 4. 一次性写入 allhu.txt
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_lines))

print("全部处理完成 ✅，已生成", OUTPUT_FILE)