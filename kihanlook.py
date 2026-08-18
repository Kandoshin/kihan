# -*- coding:utf-8 -*-
import cv2
import numpy as np
import time
import threading
import mss
import ctypes
from typing import Optional

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

# ---------------- 能量 ----------------
ROI_1P = {"left": 90, "top": 95, "width": 50, "height": 1}
ROI_2P = {"left": 774, "top": 97, "width": 50, "height": 1}
BLUE_LOW  = np.array([92, 46, 251])
BLUE_HIGH = np.array([93, 61, 255])
GRAY_LOW  = np.array([109, 187, 98])
GRAY_HIGH = np.array([109, 188, 99])

def count_blocks(mask: np.ndarray) -> int:
    row = mask[0]
    padded = np.empty(row.size + 2, dtype=np.uint8)
    padded[0], padded[-1] = 0, 0
    padded[1:-1] = row
    return np.count_nonzero((padded[1:] > 0) & (padded[:-1] == 0))

def get_energy_from_roi(roi: dict, s) -> Optional[int]:
    hsv = cv2.cvtColor(np.array(s.grab(roi))[:, :, :3], cv2.COLOR_BGR2HSV)
    blue = count_blocks(cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH))
    gray = count_blocks(cv2.inRange(hsv, GRAY_LOW, GRAY_HIGH))
    if blue == 0 and gray == 4: return 0
    if blue == 0 and gray in (2, 1): return 1
    if blue == 1 and gray in (1, 3): return 1
    if blue == 2 and gray in (1, 2, 0): return 2
    if blue == 3 and gray <= 1: return 3
    if blue <= 1 and gray == 0: return 4
    return None

# ---------------- 血条 ----------------
MONITOR_1P = {"left": 88, "top": 76, "width": 175, "height": 9}
MONITOR_2P = {"left": 652, "top": 76, "width": 175, "height": 9}
HSV_LOW  = np.array([0, 200, 200])
HSV_HIGH = np.array([10, 255, 255])
KERNEL   = np.ones((3, 3), np.uint8)

def get_health(monitor: dict, s) -> float:
    hsv  = cv2.cvtColor(np.array(s.grab(monitor))[:, :, :3], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    row_counts = np.sum(mask[2:7] > 0, axis=1)
    max_val, min_val = row_counts.max(), row_counts.min()
    use_val = min_val if max_val > 81 else max_val
    return max(0.0, min(use_val / 166.0, 1.0))

# ---------------- 技能 CD ----------------
REGIONS = {
    "skill1": {"left": 726, "top": 505, "width": 30, "height": 30},
    "skill2": {"left": 738, "top": 400, "width": 30, "height": 30},
    "scroll": {"left": 839, "top": 252, "width": 30, "height": 30},
    "summon": {"left": 838, "top": 164, "width": 30, "height": 30},
    "stand":  {"left": 624, "top": 506, "width": 30, "height": 30},
}
TARGET_FPS = 10
THRESHOLD_MAP = {
    0.000000: 0, 2.903482: 0, 0.259047: 0, 0.692047: 1, 0.509911: 2,
    0.523316: 2, 0.523235: 2, 0.516876: 2, 0.513039: 2, 0.502968: 3,
    0.506050: 3, 0.469562: 3, 0.498380: 3, 0.497247: 3, 0.357581: 4,
    0.341202: 4, 0.365344: 4, 0.363567: 4, 0.358684: 4, 0.465936: 5,
    0.489101: 5, 0.481943: 5, 0.475570: 5, 0.469939: 5, 0.373683: 6,
    0.384621: 6, 0.389688: 6, 0.382677: 6, 0.379296: 6, 0.528576: 7,
    0.572295: 7, 0.585411: 7, 0.570425: 7, 0.536675: 7, 0.405508: 8,
    0.408693: 8, 0.409066: 8, 0.413025: 8, 0.397974: 8, 0.378639: 9,
    0.388742: 9, 0.381717: 9, 0.374345: 9, 0.508753: 10, 0.539525: 10,
    0.566447: 10, 0.534097: 10, 0.434794: 10, 0.491334: 10, 0.610137: 10,
    0.499651: 10, 0.526047: 10, 0.527882: 10, 0.528053: 10, 0.534475: 11,
    0.545355: 12, 0.557203: 12, 0.545415: 12, 0.559972: 13, 0.569395: 13,
    0.559329: 13, 0.516304: 14, 0.519440: 14, 0.529235: 14, 0.534320: 15,
    0.540876: 15, 0.542930: 15, 0.542850: 15, 0.454560: 15, 0.416632: 15,
    0.534498: 15, 0.496655: 16, 0.525640: 16, 0.583409: 17, 0.554847: 17,
    0.507827: 18, 0.512660: 18, 0.522608: 19, 0.640027: 20, 0.583391: 21,
    0.658315: 22, 0.672285: 23
}  # 待补充

LOWER_YELLOW = np.array([20, 150, 150])
UPPER_YELLOW = np.array([30, 255, 255])

buf_size   = (30, 30)
TOTAL_PIX  = 900
hsv_buf    = np.empty(buf_size + (3,), np.uint8)
mask_buf   = np.empty(buf_size, np.uint8)
edges_buf  = np.empty(buf_size, np.uint8)
valid_mask = np.empty(buf_size, np.uint8)
final_buf  = np.empty(buf_size, np.uint8)
kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def map_ratio_to_number(ratio: float) -> Optional[int]:
    return THRESHOLD_MAP.get(round(ratio, 6))

def process_skill(img: np.ndarray) -> Optional[int]:
    cv2.cvtColor(img, cv2.COLOR_BGR2HSV, dst=hsv_buf)
    cv2.inRange(hsv_buf, LOWER_YELLOW, UPPER_YELLOW, dst=mask_buf)
    cv2.medianBlur(mask_buf, 3, dst=mask_buf)
    cv2.morphologyEx(mask_buf, cv2.MORPH_CLOSE, kernel, dst=mask_buf)
    cv2.morphologyEx(mask_buf, cv2.MORPH_OPEN,  kernel, dst=mask_buf)
    cv2.Canny(mask_buf, 50, 150, edges_buf)
    contours, _ = cv2.findContours(edges_buf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_mask.fill(0)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 15:
            cv2.drawContours(valid_mask, [cnt], -1, 255, -1)
    cv2.bitwise_and(mask_buf, valid_mask, dst=final_buf)
    white_ratio = np.count_nonzero(final_buf) / TOTAL_PIX
    hu = cv2.HuMoments(cv2.moments(final_buf, True)).ravel()
    return map_ratio_to_number(white_ratio + np.abs(hu).sum())

# ---------------- 通用循环 ----------------
def run_loop(name: str, func, rois=None, fps=None):
    dt = 1.0 / fps if fps else 0
    with mss.mss() as s:
        while True:
            start = time.perf_counter()
            if rois is None:
                print(f"[{name}] {func(s)}")
            elif isinstance(rois, dict):  # 多 ROI
                result = {k: func(r, s) for k, r in rois.items()}
                print(f"[{name}] {result}")
            else:  # 两个 ROI
                r1, r2 = (func(r, s) for r in rois)
                print(f"[{name}] 1P={r1}  2P={r2}")
            if dt:
                elapsed = time.perf_counter() - start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

# ---------------- 启动 ----------------
if __name__ == "__main__":
    threading.Thread(target=run_loop, args=("Energy", get_energy_from_roi, [ROI_1P, ROI_2P]), daemon=True).start()
    threading.Thread(target=run_loop, args=("Health", get_health, [MONITOR_1P, MONITOR_2P]), daemon=True).start()
    threading.Thread(target=run_loop, args=("SkillCD", lambda roi, s: process_skill(np.array(s.grab(roi))[:, :, :3]), REGIONS, TARGET_FPS), daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
