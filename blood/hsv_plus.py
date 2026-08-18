import cv2
import numpy as np
from mss import mss

# ============= 配置区域 =============
MONITOR = {"left": 672, "top": 91, "width": 176, "height": 5}
SCALE = 4  # 图像放大倍数

# 初始阈值 (你之前代码中的默认值)
g = {'h_low': 0, 's_low': 0, 'v_low': 251,
     'h_high': 179, 's_high': 255, 'v_high': 255}


def nothing(x):
    pass


# 创建两个可缩放窗口
cv2.namedWindow('mask_debug', cv2.WINDOW_NORMAL)
cv2.namedWindow('roi', cv2.WINDOW_NORMAL)

# 创建滑动条
for (name, val) in [('H_low', g['h_low']), ('S_low', g['s_low']), ('V_low', g['v_low']),
                    ('H_high', g['h_high']), ('S_high', g['s_high']), ('V_high', g['v_high'])]:
    # S和V的最大值是255，H是179
    max_val = 255 if 'S' in name or 'V' in name else 179
    cv2.createTrackbar(name, 'mask_debug', val, max_val, nothing)

sct = mss()
print(f"监测区域: {MONITOR}")
print('窗口可拖拽缩放，按 q 退出')

while True:
    # 1. 截图与转换
    roi = np.array(sct.grab(MONITOR))[:, :, :3]  # BGR
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 2. 获取当前滑块值
    hl = cv2.getTrackbarPos('H_low', 'mask_debug')
    sl = cv2.getTrackbarPos('S_low', 'mask_debug')
    vl = cv2.getTrackbarPos('V_low', 'mask_debug')
    hh = cv2.getTrackbarPos('H_high', 'mask_debug')
    sh = cv2.getTrackbarPos('S_high', 'mask_debug')
    vh = cv2.getTrackbarPos('V_high', 'mask_debug')

    # 3. 生成掩码 (Mask)
    mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh]))

    # 4. === 核心新增：统计第一行和最后一行像素 ===
    # mask 是 0 和 255 的矩阵，除以 255 得到像素个数
    # mask[0] 是第一行， mask[-1] 是最后一行
    row_top_cnt = int(mask[0].sum() // 255)
    row_bot_cnt = int(mask[-1].sum() // 255)

    # 5. 放大图像以便观察
    roi_show = cv2.resize(roi, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    mask_show = cv2.resize(mask, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)

    # 6. === 绘制数据显示面板 ===
    # 因为图片太扁（放大后高度可能只有24px），我们在 mask 下方拼一个黑底面板写字

    # 转成 BGR 方便显示彩色文字
    mask_display = cv2.cvtColor(mask_show, cv2.COLOR_GRAY2BGR)

    # 创建一个信息面板 (高度 80px，宽度与 mask_display 一致)
    panel_h = 80
    panel = np.zeros((panel_h, mask_display.shape[1], 3), dtype=np.uint8)

    # 写入文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 第一行数据 (绿色)
    cv2.putText(panel, f"Row 1 (Top): {row_top_cnt}", (10, 30), font, 0.7, (0, 255, 0), 2)
    # 最后一行数据 (红色)
    cv2.putText(panel, f"Row {MONITOR['height']} (Bot): {row_bot_cnt}", (10, 65), font, 0.7, (0, 0, 255), 2)

    # 垂直拼接：Mask在上，信息面板在下
    final_debug_img = np.vstack((mask_display, panel))

    # 7. 显示
    cv2.imshow('roi', roi_show)
    cv2.imshow('mask_debug', final_debug_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('\n=== 最终调试结果 ===')
        print(f'HSV_LOW  = np.array([{hl}, {sl}, {vl}])')
        print(f'HSV_HIGH = np.array([{hh}, {sh}, {vh}])')
        print(f'Final Row 1 Count: {row_top_cnt}')
        print(f'Final Row {MONITOR["height"]} Count: {row_bot_cnt}')
        break

cv2.destroyAllWindows()