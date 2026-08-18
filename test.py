import cv2
import numpy as np
import mss
import time

# ============= 🎯 调试目标区域 =============
# 你想调试的那个区域 (1p头像)
TARGET_REGION = {"left": 790, "top": 102, "width": 60, "height": 16}

# 窗口设置
WINDOW_NAME = "Region Debugger"
SCALE = 2  # 放大倍数，方便看清像素细节


def debug_region():
    print(f"🔍 开始调试区域: {TARGET_REGION}")
    print("--------------------------------------------------")
    print(f"📍 Left: {TARGET_REGION['left']}, Top: {TARGET_REGION['top']}")
    print(f"📏 Width: {TARGET_REGION['width']}, Height: {TARGET_REGION['height']}")
    print("--------------------------------------------------")
    print("按 'q' 键退出")

    with mss.mss() as sct:
        while True:
            # 1. 截图
            # mss.grab() 需要的格式是 {"top": int, "left": int, "width": int, "height": int}
            # 你的字典正好符合这个格式
            try:
                img_bgra = np.array(sct.grab(TARGET_REGION))
                img_bgr = img_bgra[:, :, :3]  # 去掉 Alpha 通道
            except Exception as e:
                print(f"截图失败: {e}")
                break

            # 2. 放大显示 (可选)
            if SCALE != 1:
                img_show = cv2.resize(img_bgr, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
            else:
                img_show = img_bgr

            # 3. 绘制准星 (辅助对齐)
            h, w = img_show.shape[:2]
            cx, cy = w // 2, h // 2
            # 绿色十字线
            cv2.line(img_show, (cx, 0), (cx, h), (0, 255, 0), 1)
            cv2.line(img_show, (0, cy), (w, cy), (0, 255, 0), 1)

            # 4. 显示
            cv2.imshow(WINDOW_NAME, img_show)

            # 5. 退出控制
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("✅ 调试结束")


if __name__ == "__main__":
    debug_region()