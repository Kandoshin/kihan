import os
import json
import numpy as np

# ============= 1. 动态路径配置 (输出文件保存在当前目录) =============
# 获取当前文件所在目录 (即 a 目录)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 训练输出文件 (自动保存在 a 下)
PPO_MODEL_PATH = os.path.join(CURRENT_DIR, "ppo_realtime.zip")
TRAINING_METADATA_PATH = os.path.join(CURRENT_DIR, "training_metadata.json")

# [新增] TensorBoard 日志目录
TENSORBOARD_LOG_DIR = os.path.join(CURRENT_DIR, "logs")

# ============= 2. 外部依赖路径 (指向你原来的资源位置) =============
# 请确保这些路径是存在的，不需要移动它们
YOLO_PATH = r"G:\god\yolov5\yolov5-7.0\yolov5-7.0"
MODEL_PATH = r"G:\god\yolov5\yolov5-7.0\yolov5-7.0\best.pt"

# CNN 模型相关路径
CNN_SAVE_DIR = r"G:\god\pycharm\PythonProject\test\cnn\models"
CNN_WEIGHT_PATH = os.path.join(CNN_SAVE_DIR, "best.pth")
CLASS_MAP_PATH = os.path.join(CNN_SAVE_DIR, "class_to_idx.json")

# ============= 3. 游戏区域与坐标 =============
GAME_REGION = {"left": 0, "top": 46, "width": 945, "height": 535}

# [精准坐标] 来自 detect_all_blood.py
MONITOR_1P = {"left": 98, "top": 91, "width": 176, "height": 5}
MONITOR_2P = {"left": 672, "top": 91, "width": 176, "height": 5}

# ============= 4. 视觉参数 =============
CONF_THRES, IOU_THRES = 0.8, 0.8
TARGET_CLASSES = {'p', 'b', 'r', 's'}
IMG_SIZE = 70

# [精准阈值] 来自 hsv_inspector 测量结果
# 过滤掉黑色背景，只保留血条颜色
HSV_LOW = np.array([0, 152, 43])
HSV_HIGH = np.array([49, 255, 255])

# ============= 5. 动作映射 (修改版) =============
# Move Head (Discrete 9)
MOVE_MAP = {
    0: set(), 1: {'w'}, 2: {'s'}, 3: {'a'}, 4: {'d'},
    5: {'w', 'a'}, 6: {'w', 'd'}, 7: {'a', 's'}, 8: {'s', 'd'},
}

# Skill Head (Discrete 7 -> 1 + 6 Skills)
# 移除了 u, o, l
# 新顺序: 1:j, 2:k, 3:i, 4:q, 5:e, 6:space
SKILL_MAP = {
    0: None,
    1: 'j',
    2: 'k',
    3: 'i',
    4: 'q',
    5: 'e',
    6: 'space'
}

KEYS = ['w', 's', 'a', 'd', 'j', 'k', 'i', 'q', 'e', 'space']

# ============= 6. CNN 区域定义 =============
cnn_regions = [
    {"left": 30, "top": 58, "width": 70, "height": 70, "name": "1p"},
    {"left": 845, "top": 58, "width": 70, "height": 70, "name": "2p"},
    {"left": 824, "top": 151, "width": 56, "height": 56, "name": "summon"},
    {"left": 824, "top": 238, "width": 56, "height": 56, "name": "scroll"},
    {"left": 624, "top": 506, "width": 30, "height": 30, "name": "stand"},
    {"left": 726, "top": 505, "width": 30, "height": 30, "name": "skill1"},
    {"left": 738, "top": 400, "width": 30, "height": 30, "name": "skill2"},
    {"left": 96, "top": 92, "width": 60, "height": 16, "name": "1p_energy"},
    {"left": 790, "top": 92, "width": 60, "height": 16, "name": "2p_energy"},
    {"left": 428, "top": 83, "width": 88, "height": 66, "name": "time"},
]

# ============= 7. 加载 Class Map =============
if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
else:
    class_to_idx = {"none": 0}

idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)