import os
import json
import time
import random
import queue
import numpy as np
import cv2
import mss
import torch
import gym
import pyautogui
import keyboard
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces
from PIL import Image
from torchvision import transforms, models
from typing import Tuple
from torch.distributions import Bernoulli as TorchBernoulli

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import BernoulliDistribution

import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
pyautogui.PAUSE = 0

# ---------- 路径/超参 ----------
YOLO_PATH = r"G:\god\yolov5\yolov5-7.0\yolov5-7.0"
MODEL_PATH = r"G:\god\yolov5\yolov5-7.0\yolov5-7.0\best.pt"
GAME_REGION = {"left": 0, "top": 46, "width": 945, "height": 535}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CONF_THRES, IOU_THRES = 0.6, 1
TARGET_CLASSES = {'p', 'b', 'r', 's', 'wm', 'rm'}

PPO_MODEL_PATH = r"G:\god\pycharm\PythonProject\test\final\ppo_realtime.zip"

# ---------- CNN 相关 ----------
img_size = 70
save_dir = r"G:\god\pycharm\PythonProject\test\cnn\models"
class_map_path = os.path.join(save_dir, "class_to_idx.json")

# 添加安全检查
if os.path.exists(class_map_path):
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
else:
    print(f"警告：找不到 {class_map_path}，使用默认映射")
    class_to_idx = {"none": 0}  # 默认映射

idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)
cnn_weight_path = os.path.join(save_dir, "best.pth")


def init_cnn_model():
    """在当前进程中初始化 CNN 模型"""
    try:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

        if os.path.exists(cnn_weight_path):
            sd = torch.load(cnn_weight_path, map_location=DEVICE)
            model.load_state_dict(sd, strict=True)
            print(f"成功加载 CNN 权重: {cnn_weight_path}")
        else:
            print(f"警告：找不到 CNN 权重文件 {cnn_weight_path}")

        model.to(DEVICE).eval()

        embedding_layer = nn.Embedding(num_classes, 16).to(DEVICE)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        return model, embedding_layer, transform
    except Exception as e:
        print(f"CNN 模型初始化失败: {e}")
        return None, None, None


# ---------- 屏幕/血条区域 ----------
cnn_regions = [
    {"left": 30, "top": 48, "width": 70, "height": 70, "name": "1p"},
    {"left": 845, "top": 48, "width": 70, "height": 70, "name": "2p"},
    {"left": 824, "top": 151, "width": 56, "height": 56, "name": "summon"},
    {"left": 824, "top": 238, "width": 56, "height": 56, "name": "scroll"},
    {"left": 624, "top": 506, "width": 30, "height": 30, "name": "stand"},
    {"left": 726, "top": 505, "width": 30, "height": 30, "name": "skill1"},
    {"left": 738, "top": 400, "width": 30, "height": 30, "name": "skill2"},
    {"left": 96, "top": 92, "width": 60, "height": 16, "name": "1p_energy"},
    {"left": 790, "top": 92, "width": 60, "height": 16, "name": "2p_energy"},
    {"left": 428, "top": 83, "width": 88, "height": 66, "name": "time"},
]


# ---------- 简化的 CNN 处理器（使用线程池而非多进程）----------
class ThreadCNNProcessor:
    """使用线程池而非多进程来避免句柄问题"""

    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self.executor = None
        self.cnn_model = None
        self.embedding_layer = None
        self.transform = None

    def start(self):
        """启动线程池并初始化模型"""
        try:
            self.cnn_model, self.embedding_layer, self.transform = init_cnn_model()
            if self.cnn_model is None:
                print("警告：CNN 模型初始化失败，将使用默认值")
                return False

            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            print(f"CNN 处理器启动成功，使用 {self.max_workers} 个线程")
            return True
        except Exception as e:
            print(f"CNN 处理器启动失败: {e}")
            return False

    def stop(self):
        """关闭线程池"""
        if self.executor:
            self.executor.shutdown(wait=True)
            print("CNN 处理器已关闭")

    def convert_predictions_to_features(self, pred_ids):
        """转换预测结果为特征"""
        try:
            cnn_vecs = []
            for i, pred_id in enumerate(pred_ids):
                if i >= len(cnn_regions):
                    break

                label = idx_to_class.get(int(pred_id), "none")
                region_name = cnn_regions[i]["name"]

                if region_name in {"1p", "2p"}:
                    if label == "none":
                        cnn_vecs.extend([0.0] * 16)
                    else:
                        vec = self.embedding_layer(torch.tensor([pred_id], device=DEVICE)).squeeze(
                            0).detach().cpu().numpy()
                        cnn_vecs.extend(vec.tolist())
                elif region_name in {"stand", "skill1", "skill2"}:
                    cnn_vecs.append(int(label) / 60.0 if label.isdigit() else 0.0)
                elif region_name in {"summon", "scroll"}:
                    if label.isdigit():
                        val = int(label) / 60.0
                        cnn_vecs.extend([val] * 16)
                    elif label == "none":
                        cnn_vecs.extend([0.0] * 16)
                    else:
                        vec = self.embedding_layer(torch.tensor([pred_id], device=DEVICE)).squeeze(
                            0).detach().cpu().numpy()
                        cnn_vecs.extend(vec.tolist())
                elif region_name in {"1p_energy", "2p_energy"}:
                    cnn_vecs.append(int(label) / 4.0 if label.isdigit() else 0.0)
                elif region_name == "time":
                    if label == "none":
                        cnn_vecs.append(-1.0)
                    elif label.isdigit():
                        val = int(label)
                        cnn_vecs.append(round(val / 60.0, 4))
                    else:
                        cnn_vecs.append(-1.0)
            return cnn_vecs
        except Exception as e:
            print(f"转换预测结果失败: {e}")
            return [0.0] * self.get_expected_dim()

    def get_expected_dim(self):
        """计算预期的特征维度"""
        dim = 0
        for reg in cnn_regions:
            name = reg["name"]
            if name in {"1p", "2p"}:
                dim += 16
            elif name in {"summon", "scroll"}:
                dim += 16
            elif name in {"stand", "skill1", "skill2"}:
                dim += 1
            elif name in {"1p_energy", "2p_energy"}:
                dim += 1
            elif name == "time":
                dim += 1
        return dim

    def predict(self, crop_images, timeout=2):
        """预测图像特征"""
        if self.cnn_model is None or self.executor is None:
            return [0.0] * self.get_expected_dim()

        try:
            future = self.executor.submit(self._predict_sync, crop_images)
            return future.result(timeout=timeout)
        except Exception as e:
            print(f"CNN 预测失败: {e}")
            return [0.0] * self.get_expected_dim()

    def _predict_sync(self, crop_images):
        """同步预测函数"""
        try:
            tensors = []
            for crop in crop_images:
                img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                tensors.append(self.transform(pil_img))

            if not tensors:
                return [0.0] * self.get_expected_dim()

            batch_tensor = torch.stack(tensors).to(DEVICE)
            with torch.no_grad():
                preds = torch.argmax(self.cnn_model(batch_tensor), dim=1).cpu().numpy()

            return self.convert_predictions_to_features(preds)
        except Exception as e:
            print(f"同步预测失败: {e}")
            return [0.0] * self.get_expected_dim()


# ---------- YOLO 初始化 ----------
def init_yolo_model():
    """初始化 YOLO 模型"""
    try:
        if not os.path.exists(YOLO_PATH):
            print(f"警告：YOLO 路径不存在: {YOLO_PATH}")
            return None
        if not os.path.exists(MODEL_PATH):
            print(f"警告：模型文件不存在: {MODEL_PATH}")
            return None

        yolo = torch.hub.load(YOLO_PATH, 'custom', path=MODEL_PATH, source='local', device=DEVICE)
        yolo.conf, yolo.iou = CONF_THRES, IOU_THRES
        yolo.eval()
        print("YOLO 模型加载成功")
        return yolo
    except Exception as e:
        print(f"YOLO 模型加载失败: {e}")
        return None


# ---------- 血条区域 ----------
MONITOR_1P = {"left": 99, "top": 78, "width": 175, "height": 9}
MONITOR_2P = {"left": 672, "top": 78, "width": 175, "height": 9}
HSV_LOW, HSV_HIGH = np.array([0, 200, 200]), np.array([10, 255, 255])
KERNEL = np.ones((3, 3), np.uint8)


def normalize_tuple(x, y, w, h, W, H):
    return (round(x / W, 4), round(y / H, 4), round(w / W, 4), round(h / H, 4))


def get_health(crop, last):
    """获取血量百分比"""
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
        row_counts = np.sum(mask[2:7] > 0, axis=1)
        if len(row_counts) == 0:
            return last
        max_val = int(np.max(row_counts))
        return round(max(0, min(max_val / 166.0, 1.0)) * 100, 4)
    except Exception as e:
        print(f"获取血量失败: {e}")
        return last


# ---------- 观测维度计算 ----------
KEYS = ['w', 's', 'a', 'd', 'j', 'k', 'l', 'i', 'u', 'o', 'q', 'e', 'space']
YOLO_DIM = len(TARGET_CLASSES) * 4 * 10
BASE_DIM = 2

# 重新计算 CNN 维度
CNN_DIM = 0
for reg in cnn_regions:
    name = reg["name"]
    if name in {"1p", "2p"}:
        CNN_DIM += 16
    elif name in {"summon", "scroll"}:
        CNN_DIM += 16
    elif name in {"stand", "skill1", "skill2"}:
        CNN_DIM += 1
    elif name in {"1p_energy", "2p_energy"}:
        CNN_DIM += 1
    elif name == "time":
        CNN_DIM += 1

OBS_DIM = BASE_DIM + YOLO_DIM + CNN_DIM
print(f"观测维度: BASE_DIM={BASE_DIM}, YOLO_DIM={YOLO_DIM}, CNN_DIM={CNN_DIM}, 总计={OBS_DIM}")

# ---------- 全局 CNN 处理器实例 ----------
_global_cnn_processor = None


def get_global_cnn_processor():
    """获取全局 CNN 处理器"""
    global _global_cnn_processor
    if _global_cnn_processor is None:
        _global_cnn_processor = ThreadCNNProcessor()
        _global_cnn_processor.start()
    return _global_cnn_processor


# ---------- 简化的游戏环境 ----------
class SimplifiedRealtimeGameEnv(gym.Env):
    """简化的实时游戏环境，避免复杂的多进程通信"""

    def __init__(self, env_id: int = 0):
        super().__init__()
        self.env_id = env_id
        self.action_space = spaces.MultiBinary(len(KEYS))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

        # 初始化屏幕捕获
        try:
            self.sct = mss.mss()
        except Exception as e:
            print(f"屏幕捕获初始化失败: {e}")
            self.sct = None

        # 初始化模型
        self.yolo_model = init_yolo_model()
        self.cnn_processor = get_global_cnn_processor()

        # 状态变量
        self.last_health = [100.0, 100.0]
        self.start_counter = 0
        self.end_counter_time = 0
        self.end_counter_p1 = 0
        self.end_counter_p2 = 0
        self.bblood = 1.0
        self.rblood = 1.0
        self.assign_flag = None
        self.init_bblood = 1.0
        self.init_rblood = 1.0
        self._latest_boxes = {}

        # 人类输入控制（仅主环境）
        self.human_action = [0] * len(KEYS)
        self.last_human_time = 0
        self.human_control_duration = 2.0
        self.human_mode = False

        # FPS 控制
        self.last_step_time = 0
        self.min_step_interval = 0.01  # ~13fps

        print(f"[Env {env_id}] 初始化完成")
        self._game_started = False
        self._episode_done = False
    def check_human_input(self):
        """检查人类输入（仅主环境）"""
        if self.env_id != 0:
            return False

        now = time.time()
        try:
            pressed = [int(keyboard.is_pressed(k)) for k in KEYS]
            if any(pressed):
                self.human_action = pressed
                self.last_human_time = now
                self.human_mode = True
                return True
        except Exception as e:
            print(f"检查人类输入失败: {e}")
            return False

        if self.human_mode and (now - self.last_human_time > self.human_control_duration):
            self.human_mode = False
            print("【提示】人类控制模式结束，AI接管")
        return self.human_mode

    def _get_obs(self) -> Tuple[np.ndarray, float, float, float, set]:
        """获取观测数据"""
        # FPS 控制
        now = time.time()
        elapsed = now - self.last_step_time
        if elapsed < self.min_step_interval:
            time.sleep(self.min_step_interval - elapsed)
        self.last_step_time = time.time()

        # 获取屏幕图像
        if self.sct is None:
            # 返回默认观测
            return self._get_default_obs()

        try:
            full_img = np.array(self.sct.grab(self.sct.monitors[0]))[..., :3]
        except Exception as e:
            print(f"屏幕捕获失败: {e}")
            return self._get_default_obs()

        # YOLO 检测
        yolo_flat, detected_classes = self._process_yolo(full_img)

        # 血量检测
        p1_blood, p2_blood = self._process_health(full_img)

        # CNN 处理
        cnn_vecs, time_val = self._process_cnn(full_img)

        # 更新血量分配
        self._update_blood_assignment(time_val, detected_classes, p1_blood, p2_blood)

        # 构造观测
        base_obs = [self.rblood, self.bblood] + yolo_flat
        obs = np.array(base_obs + cnn_vecs, dtype=np.float32)

        return obs, time_val, self.rblood, self.bblood, detected_classes

    def _get_default_obs(self):
        """获取默认观测（当模型加载失败时）"""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        return obs, -1.0, 1.0, 1.0, set()

    def _process_yolo(self, full_img):
        """处理 YOLO 检测"""
        try:
            if self.yolo_model is None:
                return [0.0] * YOLO_DIM, set()

            g = GAME_REGION
            game_crop = full_img[g['top']:g['top'] + g['height'], g['left']:g['left'] + g['width']]
            results = self.yolo_model(cv2.cvtColor(game_crop, cv2.COLOR_BGR2RGB), size=320)

            class_to_idx_yolo = {'p': 0, 's': 1, 'r': 2, 'b': 3, 'wm': 4, 'rm': 5}
            boxes = {k: [] for k in TARGET_CLASSES}
            detected_classes = set()

            for *xyxy, conf, cls in results.pred[0]:
                cls_name = self.yolo_model.names[int(cls)]
                if cls_name in boxes:
                    detected_classes.add(cls_name)
                    x1, y1, x2, y2 = map(float, xyxy)
                    x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2
                    norm_box = normalize_tuple(x_c, y_c, x2 - x1, y2 - y1, g['width'], g['height'])
                    one_hot = [0.0] * 6
                    one_hot[class_to_idx_yolo[cls_name]] = float(conf)
                    boxes[cls_name].append(list(norm_box) + one_hot)

            yolo_flat = []
            for cls in sorted(TARGET_CLASSES):
                cls_boxes = boxes[cls][:4]
                while len(cls_boxes) < 4:
                    cls_boxes.append([0.0] * 10)
                for b in cls_boxes:
                    yolo_flat.extend(b)
            # 在 _process_yolo 的最后，return 之前加
            self._latest_boxes = boxes  # 新增
            return yolo_flat, detected_classes
        except Exception as e:
            print(f"YOLO 处理失败: {e}")
            return [0.0] * YOLO_DIM, set()

    def _process_health(self, full_img):
        """处理血量检测"""
        try:
            crop1 = full_img[MONITOR_1P['top']:MONITOR_1P['top'] + MONITOR_1P['height'],
                    MONITOR_1P['left']:MONITOR_1P['left'] + MONITOR_1P['width']]
            crop2 = full_img[MONITOR_2P['top']:MONITOR_2P['top'] + MONITOR_2P['height'],
                    MONITOR_2P['left']:MONITOR_2P['left'] + MONITOR_2P['width']]

            p1_blood = get_health(crop1, self.last_health[0]) / 100.0
            p2_blood = get_health(crop2, self.last_health[1]) / 100.0
            self.last_health = [p1_blood * 100, p2_blood * 100]

            return p1_blood, p2_blood
        except Exception as e:
            print(f"血量检测失败: {e}")
            return 1.0, 1.0

    def _process_cnn(self, full_img):
        """处理 CNN 特征提取"""
        try:
            crops = []
            for reg in cnn_regions:
                crop = full_img[reg["top"]:reg["top"] + reg["height"], reg["left"]:reg["left"] + reg["width"]]
                crops.append(crop)

            cnn_vecs = self.cnn_processor.predict(crops)
            time_val = cnn_vecs[-1] if cnn_vecs else -1.0

            return cnn_vecs, time_val
        except Exception as e:
            print(f"CNN 处理失败: {e}")
            return [0.0] * CNN_DIM, -1.0

    def _update_blood_assignment(self, time_val, detected_classes, p1_blood, p2_blood):
        """更新血量分配：根据 b 的横坐标中点决定蓝方在左还是右"""
        if time_val >= 0.999:  # 只在倒计时 1.0 做一次判定
            self.init_bblood = self.bblood
            self.init_rblood = self.rblood

            # 找到所有类别为 b 的检测结果
            b_boxes = []
            for cls, boxes in self._latest_boxes.items():  # 需要先把 YOLO 结果缓存到 self._latest_boxes
                if cls == 'b':
                    b_boxes.extend(boxes)

            if b_boxes:  # 至少检测到一个 b
                # 计算 b 的平均中心 x（归一化 0~1）
                avg_x = sum(b[0] for b in b_boxes) / len(b_boxes)
                # 映射到实际像素：GAME_REGION 宽度 945
                pixel_x = avg_x * GAME_REGION['width']
                if pixel_x > 415:  # b 在右半屏
                    self.assign_flag = 'right'
                else:  # b 在左半屏
                    self.assign_flag = 'left'
                if self.env_id == 0:
                    print(f"【判定】b 在 {'右' if self.assign_flag == 'right' else '左'}半屏")
            else:  # 没检测到 b，默认右边蓝
                self.assign_flag = 'right'

        # 根据 assign_flag 映射血量
        if self.assign_flag == 'right':
            self.bblood = p2_blood
            self.rblood = p1_blood
        else:  # 'left'
            self.bblood = p1_blood
            self.rblood = p2_blood

    def reset(self):
        """重置环境：每次都强制等到下一轮倒计时 1.0"""
        self._reset_state()  # 清内部状态
        self._episode_done = False  # 清闸

        self.start_counter = 0
        while True:
            obs, t, _, _, _ = self._get_obs()

            # 倒计时接近 1.0 且连续 5 帧
            if abs(t - 1.0) < 0.01:
                self.start_counter += 1
            else:
                self.start_counter = 0

            if self.start_counter >= 5:
                if self.env_id == 0:
                    print("【提示】新一轮游戏开始")
                return obs

            time.sleep(0.01)

    def _reset_state(self):
        """重置状态变量"""
        self.start_counter = 0
        self.end_counter_time = 0
        self.end_counter_p1 = 0
        self.end_counter_p2 = 0
        self.bblood = 1.0
        self.rblood = 1.0
        self.assign_flag = None

    def step(self, action):
        """执行一步"""
        current_time = time.time()
        # ↓↓↓ 新增：如果本局已结束，直接返回空奖励并保持 done=True ↓↓↓
        if self._episode_done:
            obs = self._get_obs()[0]
            return obs, 0.0, True, {}
        human_override = self.check_human_input()
        obs, t, p1, p2, _ = self._get_obs()

        # 仅主环境执行按键
        if self.env_id == 0:
            if human_override or self.human_mode:
                if human_override:
                    action = self.human_action
                remaining = self.human_control_duration - (current_time - self.last_human_time)
                print(f"【人类控制中】剩余时间: {remaining:.1f}s")
            else:
                self._execute_actions(action)

        # 计算奖励
        reward = self._calculate_reward()

        # 检查游戏结束
        done = self._check_done(t, p1, p2)
        if done:
            self._episode_done = True   # ↓↓↓ 立闸，阻断后续奖励

        return obs, reward, done, {}

    def _execute_actions(self, action):
        """执行按键动作"""
        try:
            for i, pressed in enumerate(action):
                if pressed:
                    pyautogui.keyDown(KEYS[i])
                else:
                    pyautogui.keyUp(KEYS[i])
        except Exception as e:
            print(f"按键操作异常: {e}")

    def _calculate_reward(self):
        """计算奖励"""
        delta_b = (self.init_bblood - self.bblood) * 100
        delta_r = (self.init_rblood - self.rblood) * 100
        reward = delta_r - delta_b

        if abs(reward) > 1e-6 and self.env_id == 0:
            print(f"[Reward] {reward:+.1f} (Δr={delta_r:.1f}, Δb={delta_b:.1f})")

        return reward

    def _check_done(self, t, p1, p2):
        """检查游戏是否结束"""
        # 时间结束
        if abs(t) < 1e-3:
            self.end_counter_time += 1
        else:
            self.end_counter_time = 0

        # 玩家血量为0
        if p1 <= 0:
            self.end_counter_p1 += 1
        else:
            self.end_counter_p1 = 0

        if p2 <= 0:
            self.end_counter_p2 += 1
        else:
            self.end_counter_p2 = 0

        done = (self.end_counter_time >= 5 or
                self.end_counter_p1 >= 5 or
                self.end_counter_p2 >= 5)

        if done:
            if self.env_id == 0:
                print("【环境】对局结束")
                self.release_all_keys()
            self.end_counter_time = self.end_counter_p1 = self.end_counter_p2 = 0

        return done

    def release_all_keys(self):
        """释放所有按键"""
        if self.env_id == 0 and not self.human_mode:
            try:
                for k in KEYS:
                    pyautogui.keyUp(k)
            except Exception as e:
                print(f"释放按键失败: {e}")

    def close(self):
        """关闭环境"""
        try:
            self.release_all_keys()
        except:
            pass
        if self.sct:
            try:
                self.sct.close()
            except:
                pass


# ---------- 自定义策略 ----------
class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, features_dim)
        )
    def forward(self, obs): return self.net(obs)

policy_kwargs = dict(
    features_extractor_class=CustomExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

class BernoulliPolicy(ActorCriticPolicy):
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        logits = self.action_net(latent_pi)
        torch_dist = TorchBernoulli(logits=logits)
        sb3_dist = BernoulliDistribution(torch.sigmoid(logits))
        sb3_dist.distribution = torch_dist
        return sb3_dist

# ---------- 自定义 PPO，加入 imitation loss ----------
class CustomPPO(PPO):
    def __init__(self, *args, imitation_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.imitation_weight = imitation_weight
    def train(self) -> None:
        self.policy.train()
        clip_range = self.clip_range(self._current_progress_remaining) if callable(self.clip_range) else self.clip_range
        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, rollout_data.actions
                )
                adv = rollout_data.advantages
                if self.normalize_advantage:
                    adv = (adv - adv.mean())/(adv.std()+1e-8)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(rollout_data.returns, values.flatten())
                entropy_loss = -torch.mean(entropy)

                # 这里只能从主环境拿 demo，如果你需要，可加共享 demo 队列
                imitation_loss = 0.0
                try:
                    if hasattr(self.env, "envs") and len(self.env.envs) > 0:
                        env0 = self.env.envs[0]
                        demo = getattr(env0, "demo_buffer", None)
                        if demo and len(demo) > 0:
                            k = min(len(demo), max(1, int(self.batch_size)))
                            sample = random.sample(demo, k)
                            obs_demo = torch.tensor([d[0] for d in sample], dtype=torch.float32, device=self.device)
                            act_demo = torch.tensor([d[1] for d in sample], dtype=torch.float32, device=self.device)
                            dist = self.policy.get_distribution(obs_demo)
                            log_probs = dist.log_prob(act_demo).sum(dim=1)
                            imitation_loss = -torch.mean(log_probs)
                except Exception:
                    imitation_loss = 0.0

                loss = policy_loss + self.vf_coef*value_loss + self.ent_coef*entropy_loss + self.imitation_weight*imitation_loss
                self.policy.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

# ---------- env 工厂，传入可 picklable 句柄 ----------
def make_env(env_id, req_queue, res_dict, cnn_timeout=0.2):
    def _init():
        return ParallelRealtimeGameEnv(env_id=env_id, req_queue=req_queue, res_dict=res_dict, cnn_timeout=cnn_timeout)
    return _init

# ---------- 主训练 ----------
if __name__ == "__main__":
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("DEVICE:", DEVICE)
    mp.set_start_method("spawn", force=True)  # Windows 下显式指定

    # ✅ 使用 DummyVecEnv（线程内并行），不再需要 CNNBatchProcessor
    num_envs = 4  # 可根据 CPU 和 GPU 能力调整
    print(f"【提示】创建 {num_envs} 个线程内并行环境…")

    # 修复闭包陷阱
    env_fns = [lambda _i=i: SimplifiedRealtimeGameEnv(env_id=_i) for i in range(num_envs)]
    env = DummyVecEnv(env_fns)

    # 构建/加载模型
    if os.path.exists(PPO_MODEL_PATH):
        try:
            print("【提示】加载已有模型（CustomPPO + BernoulliPolicy）")
            model = CustomPPO.load(PPO_MODEL_PATH, env=env, custom_objects={"policy_class": BernoulliPolicy})
        except Exception as e:
            print(f"【提示】加载失败，新建模型。原因：{e}")
            model = CustomPPO(BernoulliPolicy, env, policy_kwargs=policy_kwargs,
                              imitation_weight=0.5, verbose=1,
                              n_steps=512, batch_size=256, learning_rate=3e-4)
    else:
        print("【提示】新建模型")
        model = CustomPPO(BernoulliPolicy, env, policy_kwargs=policy_kwargs,
                          imitation_weight=0.5, verbose=1,
                          n_steps=512, batch_size=256, learning_rate=3e-4)

    try:
        print("【提示】开始并行训练…")
        print(f"【信息】每轮采集: {512 * num_envs} 步")
        model.learn(total_timesteps=100000)
        print("【提示】训练完成")
    except KeyboardInterrupt:
        print("【提示】手动中断，准备保存…")
    finally:
        try:
            model.save(PPO_MODEL_PATH)
            print(f"【提示】模型已保存到 {PPO_MODEL_PATH}")
        except Exception as e:
            print(f"【错误】保存模型失败：{e}")
        try:
            env.close()
        except Exception:
            pass
        try:
            cnn_service.stop()
        except Exception:
            pass
        print("【提示】清理完成，退出。")

__all__ = ["SimplifiedRealtimeGameEnv"]

#