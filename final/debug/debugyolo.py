import os
import sys
import time
import numpy as np
import cv2
import mss
import torch
import random

# ============= 添加模块搜索路径 =============
# 将 game_config.py 所在的目录添加到系统路径中
# 这样 Python 才能找到并导入它
sys.path.append(r"G:\god\pycharm\PythonProject\test\a_simple")

# ============= 导入项目配置 =============
# 直接使用 game_config 中的配置，确保与训练环境一致
from game_config import (
    YOLO_PATH, MODEL_PATH, GAME_REGION,
    CONF_THRES, IOU_THRES, TARGET_CLASSES,
    IMG_SIZE
)

# ============= 全局开关 =============
# 1: 开启自动保存 (当检测到置信度低于 0.8 的目标时保存原图)
# 0: 关闭自动保存
AUTOSAVE = 0

# ============= 图片保存配置 =============
# 这里的路径根据你的实际需求修改
SAVE_DIR = r"G:\god\pycharm\PythonProject\test\data\images"

# 显示窗口分辨率 (保持原比例或固定大小)
DISPLAY_WIDTH = GAME_REGION["width"]
DISPLAY_HEIGHT = GAME_REGION["height"]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 类别颜色映射 (BGR格式)
# p:人物, b:自身蓝框, r:敌人红框, s:技能, wm:白字伤害, rm:红字伤害
CLASS_COLORS = {
    'p': (0, 255, 0),  # 绿色
    'b': (255, 0, 0),  # 蓝色 (OpenCV是BGR)
    'r': (0, 0, 255),  # 红色
    's': (255, 0, 255),  # 紫色
    'wm': (200, 200, 200),  # 灰白色
    'rm': (0, 69, 255),  # 橙红色 (为了和 r 区分)
}

# 全局变量：记录上次自动保存的时间
last_autosave_time = 0


def init_yolo_model():
    """初始化 YOLO 模型"""
    try:
        if not os.path.exists(YOLO_PATH):
            print(f"警告：YOLO 路径不存在: {YOLO_PATH}")
            return None
        if not os.path.exists(MODEL_PATH):
            print(f"警告：模型文件不存在: {MODEL_PATH}")
            return None

        print(f"正在加载YOLO模型: {MODEL_PATH} ...")
        # 加载本地自定义模型
        yolo = torch.hub.load(YOLO_PATH, 'custom', path=MODEL_PATH, source='local', device=DEVICE)

        # 如果开启自动保存，我们稍微降低阈值以便观察那些“边缘”样本
        # 如果是纯观察模式，使用配置文件的阈值
        yolo.conf = 0.5 if AUTOSAVE else CONF_THRES
        yolo.iou = IOU_THRES

        yolo.eval()
        print(f"✅ YOLO 模型加载成功，运行在 {DEVICE}")
        return yolo
    except Exception as e:
        print(f"❌ YOLO 模型加载失败: {e}")
        return None


def generate_filename():
    """生成符合要求的文件名：z + yyMMddHHmmss + 3位随机数.jpg"""
    # 1. 生成时间戳 (两位年份%y + 月日时分秒)
    timestamp = time.strftime("%y%m%d%H%M%S")
    # 2. 生成3位随机数 (100-999)
    random_suffix = random.randint(100, 999)
    # 3. 拼接: z + timestamp + random + .jpg
    return f"z{timestamp}{random_suffix}.jpg"


def save_image(img_data, reason="manual"):
    """保存图片"""
    try:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        filename = generate_filename()
        full_path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(full_path, img_data)

        if reason == "manual":
            print(f"💾 [手动保存] {filename}")
        else:
            print(f"📸 [自动保存] 低置信度样本: {filename}")

    except Exception as e:
        print(f"❌ 保存失败: {e}")


def process_yolo_detection(yolo_model, full_img):
    """
    执行推理并解析结果
    """
    global last_autosave_time

    if yolo_model is None:
        return [], None

    # 1. 裁剪游戏区域
    g = GAME_REGION
    game_crop = full_img[g['top']:g['top'] + g['height'], g['left']:g['left'] + g['width']]

    # 2. 模型推理
    # 注意：这里转换颜色空间 BGR -> RGB
    results = yolo_model(cv2.cvtColor(game_crop, cv2.COLOR_BGR2RGB), size=320)

    detections = []
    low_conf_detected = False

    # 3. 解析结果
    # results.pred[0] 是 (n, 6) 的张量: x1, y1, x2, y2, conf, cls
    for *xyxy, conf, cls in results.pred[0]:
        conf_val = float(conf)
        cls_id = int(cls)
        cls_name = yolo_model.names[cls_id]

        # 过滤掉不在我们关注列表里的类别（如果有的话）
        if cls_name not in TARGET_CLASSES:
            continue

        # 记录检测结果
        x1, y1, x2, y2 = map(int, xyxy)
        detections.append({
            'class': cls_name,
            'conf': conf_val,
            'bbox': (x1, y1, x2, y2)
        })

        # 检查低置信度 (用于自动保存逻辑)
        # 这里的阈值 0.8 是你认为“高质量”的界限
        if conf_val < 0.8:
            low_conf_detected = True

    # 4. 自动保存逻辑
    if AUTOSAVE == 1 and low_conf_detected:
        current_time = time.time()
        # 冷却时间 1.5 秒
        if current_time - last_autosave_time > 1.5:
            save_image(game_crop, reason="auto")
            last_autosave_time = current_time

    return detections, game_crop


def draw_detections(image, detections):
    """绘制可视化结果"""
    if image is None: return None

    draw_img = image.copy()

    # 统计信息
    counts = {k: 0 for k in TARGET_CLASSES}

    for det in detections:
        cls_name = det['class']
        conf = det['conf']
        x1, y1, x2, y2 = det['bbox']

        counts[cls_name] = counts.get(cls_name, 0) + 1

        # 获取颜色，默认灰色
        color = CLASS_COLORS.get(cls_name, (128, 128, 128))

        # 1. 画框
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)

        # 2. 画标签背景和文字
        label = f"{cls_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (t_w, t_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 标签背景条
        cv2.rectangle(draw_img, (x1, y1 - t_h - 5), (x1 + t_w, y1), color, -1)
        # 标签文字 (白色)
        cv2.putText(draw_img, label, (x1, y1 - 3), font, font_scale, (255, 255, 255), thickness)

    # 在左上角绘制统计摘要
    info_y = 30
    for cls, count in counts.items():
        if count > 0:
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            cv2.putText(draw_img, f"{cls}: {count}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            info_y += 25

    return draw_img


class YoloDebugger:
    def __init__(self):
        self.yolo_model = None
        self.sct = None
        self.running = False

    def run(self):
        print("\n=== YOLO 实时调试器启动 ===")
        print(f"目标类别: {TARGET_CLASSES}")
        print(f"自动保存: {'✅ 开启' if AUTOSAVE else '❌ 关闭'}")
        print("-" * 30)

        # 初始化
        try:
            self.sct = mss.mss()
            self.yolo_model = init_yolo_model()
            if self.yolo_model is None: return
        except Exception as e:
            print(f"初始化失败: {e}")
            return

        # 创建窗口
        cv2.namedWindow('YOLO Debug', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Debug', DISPLAY_WIDTH, DISPLAY_HEIGHT)

        self.running = True
        paused = False
        frame_count = 0
        start_time = time.time()

        print("\n操作指南:")
        print(" [SPACE] 暂停/继续")
        print(" [S]     手动保存当前帧")
        print(" [Q]     退出程序")
        print("-" * 30)

        try:
            while self.running:
                if not paused:
                    # 1. 抓屏
                    full_img = np.array(self.sct.grab(self.sct.monitors[0]))[..., :3]

                    # 2. 检测
                    detections, game_crop = process_yolo_detection(self.yolo_model, full_img)

                    # 3. 绘制
                    if game_crop is not None:
                        display_img = draw_detections(game_crop, detections)

                        # 显示 FPS
                        cur_time = time.time()
                        fps = frame_count / (cur_time - start_time) if frame_count > 0 else 0
                        cv2.putText(display_img, f"FPS: {fps:.1f}", (DISPLAY_WIDTH - 120, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        cv2.imshow('YOLO Debug', display_img)
                        frame_count += 1

                # 按键监听
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"状态: {'⏸️ 暂停' if paused else '▶️ 继续'}")
                elif key == ord('s'):
                    if 'game_crop' in locals() and game_crop is not None:
                        save_image(game_crop, reason="manual")

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            if self.sct: self.sct.close()
            cv2.destroyAllWindows()
            print("退出调试")


if __name__ == "__main__":
    YoloDebugger().run()