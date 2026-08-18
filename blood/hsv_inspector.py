import cv2
import numpy as np
from mss import mss
import time

# ============= 配置区域 =============
# 你的血条区域
MONITOR = {"left": 672, "top": 90, "width": 175, "height": 6}
SCALE = 10  # 放大 10 倍，方便鼠标看清每个像素

# 全局变量存储鼠标位置
mouse_x, mouse_y = 0, 0


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        # 点击时打印详细信息
        print(f"【点击】坐标 ({x // SCALE}, {y // SCALE}) -> HSV: {param[y // SCALE, x // SCALE]}")


def analyze_area(hsv_img):
    """统计区域内的颜色范围，排除黑色背景"""
    # 提取三个通道
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    # 过滤掉亮度极低（接近黑色）的像素，避免它们干扰最小值统计
    # 假设 V > 40 才是有效像素（根据你的血条调整，一般血条都是发光的）
    mask = v > 40

    if not np.any(mask):
        print("⚠️ 画面太暗，没有检测到有效像素！")
        return

    # 只统计有效区域
    valid_h = h[mask]
    valid_s = s[mask]
    valid_v = v[mask]

    print("\n" + "=" * 40)
    print("📊 当前画面统计 (已忽略黑色背景):")
    print(f"🔴 H (色相) 范围: [{valid_h.min()} - {valid_h.max()}] \t(平均: {int(valid_h.mean())})")
    print(f"🟢 S (饱和) 范围: [{valid_s.min()} - {valid_s.max()}] \t(平均: {int(valid_s.mean())})")
    print(f"🔵 V (亮度) 范围: [{valid_v.min()} - {valid_v.max()}] \t(平均: {int(valid_v.mean())})")
    print("-" * 40)
    print("💡 建议配置:")
    print(
        f"HSV_LOW  = np.array([{max(0, valid_h.min() - 5)}, {max(0, valid_s.min() - 10)}, {max(0, valid_v.min() - 10)}])")
    print(f"HSV_HIGH = np.array([{min(179, valid_h.max() + 5)}, 255, 255])")
    print("=" * 40 + "\n")


# 初始化窗口
cv2.namedWindow('HSV Inspector', cv2.WINDOW_NORMAL)
sct = mss()

print(f"监测区域: {MONITOR}")
print("🔍 移动鼠标查看像素值")
print("⌨️  按 'S' 键：自动统计当前所有像素的范围")
print("⌨️  按 'Q' 键：退出")

while True:
    # 1. 截图
    img_bgr = np.array(sct.grab(MONITOR))[:, :, :3]

    # 2. 转换 HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 3. 放大显示 (BGR用于人眼观察)
    img_zoom = cv2.resize(img_bgr, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)

    # 4. 获取鼠标指向的像素值
    # 注意：鼠标坐标是放大后的，需要除以 SCALE 映射回原图
    orig_x, orig_y = mouse_x // SCALE, mouse_y // SCALE

    # 边界保护
    orig_x = np.clip(orig_x, 0, MONITOR['width'] - 1)
    orig_y = np.clip(orig_y, 0, MONITOR['height'] - 1)

    pixel_hsv = img_hsv[orig_y, orig_x]
    pixel_bgr = img_bgr[orig_y, orig_x]

    # 5. 在画面上绘制信息
    # 绘制一个十字准星
    cv2.drawMarker(img_zoom, (mouse_x, mouse_y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # 准备显示的文字
    text_hsv = f"HSV: {pixel_hsv}"
    text_bgr = f"BGR: {pixel_bgr}"

    # 在图片下方添加黑色区域显示文字
    info_panel = np.zeros((60, img_zoom.shape[1], 3), dtype=np.uint8)
    cv2.putText(info_panel, text_hsv, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # 黄色字
    cv2.putText(info_panel, text_bgr, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)  # 灰色字

    # 拼接
    final_display = np.vstack((img_zoom, info_panel))

    cv2.imshow('HSV Inspector', final_display)

    # 设置鼠标回调 (必须在 imshow 之后或窗口创建后)
    cv2.setMouseCallback('HSV Inspector', mouse_callback, img_hsv)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        analyze_area(img_hsv)
        # 暂停一下让用户看到控制台输出
        # time.sleep(0.5)

cv2.destroyAllWindows()