import cv2
import numpy as np
import mss
import time
import ctypes

ctypes.windll.shcore.SetProcessDpiAwareness(2)

REGION_SKILL1 = {"left": 726, "top": 505, "width": 30, "height": 30}
REGION_SKILL2 = {"left": 738, "top": 400, "width": 30, "height": 30}
REGION_SCROLL = {"left": 839, "top": 252, "width": 30, "height": 30}
REGION_SUMMON = {"left": 838, "top": 164, "width": 30, "height": 30}
REGION_STAND = {"left": 624, "top": 506, "width": 30, "height": 30}
TARGET_FPS = 10

THRESHOLD_MAP = {
    0.000000: 0,
    2.903482: 0,
    0.259047: 0,
    0.692047: 1,
    0.509911: 2,
    0.523316: 2,
    0.523235: 2,
    0.516876: 2,
    0.513039: 2,
    0.502968: 3,
    0.506050: 3,
    0.469562: 3,
    0.498380: 3,
    0.497247: 3,
    0.357581: 4,
    0.341202: 4,
    0.365344: 4,
    0.363567: 4,
    0.358684: 4,
    0.465936: 5, 0.489101: 5,
    0.481943: 5,
    0.475570: 5,
    0.469939: 5,
    0.373683: 6,
    0.384621: 6,
    0.389688: 6,
    0.382677: 6,
    0.379296: 6, 0.528576: 7,
    0.572295: 7,
    0.585411: 7,
    0.570425: 7,
    0.536675: 7, 0.405508: 8,
    0.408693: 8,
    0.409066: 8,
    0.413025: 8,
    0.397974: 8,
    0.378639: 9,
    0.388742: 9,
    0.381717: 9,
    0.374345: 9, 0.508753: 10, 0.539525: 10, 0.566447: 10, 0.534097: 10,
    0.434794: 10, 0.491334: 10,
    0.610137: 10,
    0.499651: 10,
    0.526047: 10,
    0.527882: 10,
    0.528053: 10,
    0.534475: 11,
    0.545355: 12,
    0.557203: 12,
    0.545415: 12,
    0.559972: 13,
    0.569395: 13,
    0.559329: 13,
    0.516304: 14,
    0.519440: 14,
    0.529235: 14,
    0.534320: 15,
    0.540876: 15,
    0.542930: 15,
    0.542850: 15,
    0.454560: 15,
    0.416632: 15,
    0.534498: 15,
    0.496655: 16,
    0.525640: 16,
    0.583409: 17,
    0.554847: 17,
    0.507827: 18,
    0.512660: 18,
    0.522608: 19,
    0.640027: 20,
    0.583391: 21,
    0.658315: 22,
    0.672285: 23

}

LOWER_YELLOW = np.array([20, 150, 150])
UPPER_YELLOW = np.array([30, 255, 255])

buf_size = (REGION_SKILL1['height'], REGION_SKILL1['width'])
TOTAL_PIXELS = REGION_SKILL1['width'] * REGION_SKILL1['height']

hsv_buf   = np.empty(buf_size + (3,), np.uint8)
mask_buf  = np.empty(buf_size, np.uint8)
edges_buf = np.empty(buf_size, np.uint8)
valid_mask = np.empty(buf_size, np.uint8)
final_buf  = np.empty(buf_size, np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def map_ratio_to_number(ratio):
    """
    只在 THRESHOLD_MAP 的 key 中精确匹配 ratio；
    不存在则返回 None。
    """
    return THRESHOLD_MAP.get(ratio, None)

def process_image(img):
    cv2.cvtColor(img, cv2.COLOR_BGR2HSV, dst=hsv_buf)
    cv2.inRange(hsv_buf, LOWER_YELLOW, UPPER_YELLOW, dst=mask_buf)
    cv2.medianBlur(mask_buf, 3, dst=mask_buf)
    cv2.morphologyEx(mask_buf, cv2.MORPH_CLOSE, kernel, dst=mask_buf, iterations=1)
    cv2.morphologyEx(mask_buf, cv2.MORPH_OPEN, kernel, dst=mask_buf, iterations=1)
    edges_buf[:] = cv2.Canny(mask_buf, 50, 150)
    contours, _ = cv2.findContours(edges_buf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_mask.fill(0)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 15:
            cv2.drawContours(valid_mask, [cnt], -1, 255, -1)
    cv2.bitwise_and(mask_buf, valid_mask, dst=final_buf)
    white_ratio = np.count_nonzero(final_buf) / TOTAL_PIXELS
    M = cv2.moments(final_buf, binaryImage=True)
    hu = cv2.HuMoments(M).flatten()
    return map_ratio_to_number(round(white_ratio + np.sum(np.abs(hu)), 6))

def main():
    target_dt = 1.0 / TARGET_FPS
    with mss.mss() as sct:
        while True:
            start = time.perf_counter()
            # 区域1
            raw1 = sct.grab(REGION_SKILL1)
            img1 = np.frombuffer(raw1.rgb, np.uint8).reshape(buf_size + (3,))
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            cd1 = process_image(img1)
            # 区域2
            raw2 = sct.grab(REGION_SKILL2)
            img2 = np.frombuffer(raw2.rgb, np.uint8).reshape(buf_size + (3,))
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            cd2 = process_image(img2)
            #scroll
            raw3 = sct.grab(REGION_SCROLL)
            img3 = np.frombuffer(raw3.rgb, np.uint8).reshape(buf_size + (3,))
            img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
            cd3 = process_image(img3)
            #summon
            raw4 = sct.grab(REGION_SUMMON)
            img4 = np.frombuffer(raw4.rgb, np.uint8).reshape(buf_size + (3,))
            img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2BGR)
            cd4 = process_image(img4)
            #stand
            raw5 = sct.grab(REGION_STAND)
            img5 = np.frombuffer(raw5.rgb, np.uint8).reshape(buf_size + (3,))
            img5 = cv2.cvtColor(img5, cv2.COLOR_RGB2BGR)
            cd5 = process_image(img5)
            # 输出
            print(f"skill1cd:{cd1}")
            print(f"skill2cd:{cd2}")
            print(f"scrollcd:{cd3}")
            print(f"summoncd:{cd4}")
            print(f"standcd:{cd5}")
            # 帧率控制
            elapsed = time.perf_counter() - start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

if __name__ == "__main__":
    main()