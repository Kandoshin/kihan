# realtime_detect_fixed.py
import cv2
import torch
import numpy as np
import win32gui, win32ui, win32con, win32api   # pip install pywin32

# ========= 1. 配置 =========
DATA_YAML   = r"G:\god\yolov5\yolov5-7.0\yolov5-7.0\data\narutot.yaml"
MODEL_PATH  = r"G:\god\yolov5\yolov5-7.0\yolov5-7.0\best.pt"
GAME_REGION = {"left": 0, "top": 46, "width": 945, "height": 535}   # ROI
CONF_THRES  = 0.6
IOU_THRES   = 0.45
DEVICE      = 'cuda:0'

# ========= 2. 加载模型 =========
model = torch.hub.load(
    r'G:\god\yolov5\yolov5-7.0\yolov5-7.0',
    'custom',
    path=MODEL_PATH,
    source='local',
    device=DEVICE
)
model.conf = CONF_THRES
model.iou  = IOU_THRES

# ========= 3. 截屏函数 =========
def grab_screen(region):
    hwin = win32gui.GetDesktopWindow()
    left, top, width, height = region['left'], region['top'], region['width'], region['height']
    hwindc = win32gui.GetWindowDC(hwin)
    srcdc  = win32ui.CreateDCFromHandle(hwindc)
    memdc  = srcdc.CreateCompatibleDC()
    bmp    = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)
    srcdc.DeleteDC(); memdc.DeleteDC(); win32gui.ReleaseDC(hwin, hwindc); win32gui.DeleteObject(bmp.GetHandle())
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 返回 BGR

# ========= 4. 主循环 =========
print('按 q 退出...')
cv2.namedWindow('YOLOv5 Realtime', cv2.WINDOW_NORMAL)  # 可调窗口大小

# 固定比例
target_aspect = GAME_REGION["width"] / GAME_REGION["height"]

while True:
    frame_bgr = grab_screen(GAME_REGION)

    # 修正颜色空间 (BGR -> RGB)，保证和训练一致
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 用 RGB 输入 YOLO
    results = model(frame_rgb, size=640)

    # 画框
    for *xyxy, conf, cls in results.pred[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_bgr, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 保持比例缩放
    win_w, win_h = 1280, int(1280 / target_aspect)  # 默认 1280 宽，高度按比例
    resized = cv2.resize(frame_bgr, (win_w, win_h), interpolation=cv2.INTER_LINEAR)

    cv2.imshow('YOLOv5 Realtime', resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
