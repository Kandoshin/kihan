import cv2
import numpy as np
from mss import mss

# ROI 坐标
ROI = {"left": 428, "top": 83,
               "width": 88, "height": 66}

# 实时滑条回调（啥也不干，只为刷新）
def nothing(x):
    pass

# 创建窗口和 6 个滑条
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.createTrackbar('H_low',  'mask', 0, 180, nothing)
cv2.createTrackbar('H_high', 'mask', 180, 180, nothing)
cv2.createTrackbar('S_low',  'mask', 0, 255, nothing)
cv2.createTrackbar('S_high', 'mask', 255, 255, nothing)
cv2.createTrackbar('V_low',  'mask', 0, 255, nothing)
cv2.createTrackbar('V_high', 'mask', 255, 255, nothing)

sct = mss()

while True:
    # 1. 截屏
    img = np.array(sct.grab(ROI))[:, :, :3]  # BGRA → BGR
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. 读滑条
    h1 = cv2.getTrackbarPos('H_low',  'mask')
    h2 = cv2.getTrackbarPos('H_high', 'mask')
    s1 = cv2.getTrackbarPos('S_low',  'mask')
    s2 = cv2.getTrackbarPos('S_high', 'mask')
    v1 = cv2.getTrackbarPos('V_low',  'mask')
    v2 = cv2.getTrackbarPos('V_high', 'mask')

    # 3. 阈值二值化并显示
    mask = cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
    cv2.imshow('mask', mask)

    # 4. 按 q 退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 打印最终阈值
print("最终阈值：")
print("H_low, H_high =", h1, h2)
print("S_low, S_high =", s1, s2)
print("V_low, V_high =", v1, v2)

cv2.destroyAllWindows()