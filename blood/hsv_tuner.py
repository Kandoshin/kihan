# hsv_tuner_resizable.py
import cv2
import numpy as np
from mss import mss

MONITOR = {"left": 672, "top": 90, "width": 175, "height": 6}
SCALE   = 4          # 放大倍数，自己改

g = {'h_low': 0, 's_low': 0, 'v_low': 251,
     'h_high': 179, 's_high': 255, 'v_high': 255}

def nothing(x):
    pass

# 创建两个可缩放窗口
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('roi',  cv2.WINDOW_NORMAL)

for (name, val) in [('H_low', g['h_low']), ('S_low', g['s_low']), ('V_low', g['v_low']),
                    ('H_high', g['h_high']), ('S_high', g['s_high']), ('V_high', g['v_high'])]:
    cv2.createTrackbar(name, 'mask', val, 255 if 'S' in name or 'V' in name else 179, nothing)

sct = mss()
print('窗口可拖拽缩放，按 q 退出')
while True:
    roi = np.array(sct.grab(MONITOR))[:, :, :3]          # BGR
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    hl = cv2.getTrackbarPos('H_low', 'mask')
    sl = cv2.getTrackbarPos('S_low', 'mask')
    vl = cv2.getTrackbarPos('V_low', 'mask')
    hh = cv2.getTrackbarPos('H_high', 'mask')
    sh = cv2.getTrackbarPos('S_high', 'mask')
    vh = cv2.getTrackbarPos('V_high', 'mask')

    mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh]))

    # 放大后显示
    roi_show  = cv2.resize(roi,  (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    mask_show = cv2.resize(mask, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)

    cv2.imshow('roi',  roi_show)
    cv2.imshow('mask', mask_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('最终阈值：')
        print(f'HSV_LOW  = np.array([{hl}, {sl}, {vl}])')
        print(f'HSV_HIGH = np.array([{hh}, {sh}, {vh}])')
        break

cv2.destroyAllWindows()