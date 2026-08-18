import cv2
import numpy as np
import mss
import ctypes
import time

# -------------------------
# 全局配置
# -------------------------
ctypes.windll.shcore.SetProcessDpiAwareness(2)

REGION = {"left": 624, "top": 506, "width": 30, "height": 30}

LOWER_YELLOW = np.array([20, 150, 150])
UPPER_YELLOW = np.array([30, 255, 255])

MEDIAN_KERNEL = 3
MORPH_KERNEL = (3, 3)
MIN_CONTOUR_AREA = 15
TOTAL_PIXELS = REGION['width'] * REGION['height']

# 预分配
buf_size = (REGION['height'], REGION['width'])
hsv_buf   = np.empty(buf_size + (3,), np.uint8)
mask_buf  = np.empty(buf_size, np.uint8)
edges_buf = np.empty(buf_size, np.uint8)
valid_mask = np.empty(buf_size, np.uint8)
final_buf  = np.empty(buf_size, np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)

# -------------------------
# 功能函数
# -------------------------
def process_image(img):
    cv2.cvtColor(img, cv2.COLOR_BGR2HSV, dst=hsv_buf)
    cv2.inRange(hsv_buf, LOWER_YELLOW, UPPER_YELLOW, dst=mask_buf)
    cv2.medianBlur(mask_buf, MEDIAN_KERNEL, dst=mask_buf)

    cv2.morphologyEx(mask_buf, cv2.MORPH_CLOSE, kernel, dst=mask_buf, iterations=1)
    cv2.morphologyEx(mask_buf, cv2.MORPH_OPEN, kernel, dst=mask_buf, iterations=1)

    edges_buf[:] = cv2.Canny(mask_buf, 50, 150)
    contours, _ = cv2.findContours(edges_buf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_mask.fill(0)
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
            cv2.drawContours(valid_mask, [cnt], -1, 255, -1)

    cv2.bitwise_and(mask_buf, valid_mask, dst=final_buf)
    return final_buf

def compute_features(binary_img):
    white_ratio = np.count_nonzero(binary_img) / TOTAL_PIXELS
    M = cv2.moments(binary_img, binaryImage=True)
    hu = cv2.HuMoments(M).flatten()
    return white_ratio + np.sum(np.abs(hu))

# -------------------------
# 主循环
# -------------------------
with mss.mss() as sct:
    while True:
        loop_start = time.perf_counter()
        raw = sct.grab(REGION)
        img = np.frombuffer(raw.rgb, np.uint8).reshape((REGION['height'], REGION['width'], 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        processed = process_image(img)
        value = compute_features(processed)
        print(f"{value:.6f}")
        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0, 0.5 - elapsed))