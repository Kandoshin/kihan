# realtime_detect_no_show.py
import cv2
import numpy as np
import torch
import time
import win32gui, win32ui, win32con, win32api   # pip install pywin32

# ========= 1. 基础配置 =========
DATA_YAML   = r"G:\god\yolov5\yolov5-7.0\yolov5-7.0\data\narutot.yaml"
MODEL_PATH  = r"G:\god\yolov5\yolov5-7.0\yolov5-7.0\best.pt"
GAME_REGION = {"left": 0, "top": 46, "width": 945, "height": 535}

CONF_THRES  = 0.6
IOU_THRES   = 1
DEVICE      = 'cuda:0'

# 六个目标变量，初始化 4 个空四元组
p = b = r = s = wm = rm = [(0,0,0,0)] * 4

# ========= 2. 加载 YOLOv5 模型 =========
model = torch.hub.load(
    r'G:\god\yolov5\yolov5-7.0\yolov5-7.0',
    'custom',
    path=MODEL_PATH,
    source='local',
    device=DEVICE
)
model.conf = CONF_THRES
model.iou  = IOU_THRES
model.eval()

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
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# ========= 4. 主循环 =========
print("开始检测，按 Ctrl+C 退出")
try:
    while True:
        tic = time.time()

        # 1) 截屏
        frame_bgr = grab_screen(GAME_REGION)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 2) 推理
        results = model(frame_rgb, size=640)
        preds = results.pred[0]                   # [x1, y1, x2, y2, conf, cls]

        # 3) 按类别分组
        boxes = {k: [] for k in ['p', 'b', 'r', 's', 'wm', 'rm']}
        for *xyxy, conf, cls_int in preds:
            cls_name = model.names[int(cls_int)]
            if cls_name in boxes:
                x1, y1, x2, y2 = map(float, xyxy)
                x_c = (x1 + x2) / 2
                y_c = (y1 + y2) / 2
                w   = x2 - x1
                h   = y2 - y1
                boxes[cls_name].append((conf, x_c, y_c, w, h))

        # 4) 填充变量
        for key in boxes:
            lst = boxes[key]
            lst.sort(reverse=True)          # 置信度降序
            top4 = lst[:4]                  # 最多 4 个
            # 转四元组
            tups = [(x, y, w, h) for _, x, y, w, h in top4]
            # 不足 4 个补零
            tups += [(0, 0, 0, 0)] * (4 - len(tups))
            globals()[key] = tups           # 动态写回变量

        # 5) 打印
        print(f"p:{p} b:{b} r:{r} s:{s} wm:{wm} rm:{rm}")

        # 6) 频率控制 0.1 s
        elapsed = time.time() - tic
        time.sleep(max(0, 0.1 - elapsed))

except KeyboardInterrupt:
    print("\n已退出")