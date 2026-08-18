# blood_meter.py
import numpy as np
from mss import mss
import cv2

MONITOR = {"left": 675, "top": 80, "width": 169, "height": 6}
HSV_LOW  = np.array([0,   0, 251])
HSV_HIGH = np.array([179, 255, 255])

sct = mss()
while True:
    roi  = np.array(sct.grab(MONITOR))[:, :, :3]          # BGR
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)            # 白亮区域

    row1 = mask[0].sum() // 255          # 第一行白色像素个数
    row6 = mask[5].sum() // 255          # 第六行白色像素个数

    if row1 > 82:
        blood = round(row1 / 164, 6)
    else:
        blood = round(row6 / 164, 6)

    print(f"blood:{blood:.6f}")