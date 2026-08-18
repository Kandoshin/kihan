import os
import time
import numpy as np
import cv2
import mss
import torch
import gymnasium as gym
import pyautogui
import multiprocessing as mp
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

# ============= [调试开关] =============
# 设置为 True 以打印详细的奖励变化和原因
PRINT_REWARD_DETAILS = True
# =====================================

# ============= 模块导入 =============
# 确保 game_config.py 也在 b_imitation 目录下，这样路径配置就是本地的
from game_config import (
    GAME_REGION, TARGET_CLASSES, MONITOR_1P, MONITOR_2P,
    MOVE_MAP, SKILL_MAP, KEYS, TRAINING_METADATA_PATH, PPO_MODEL_PATH,
    cnn_regions, TENSORBOARD_LOG_DIR
)
from vision_processor import (
    get_global_cnn_processor, init_yolo_model, get_health, normalize_tuple
)
from train_utils import (
    TrainingMetadata, EpisodeTrackerCallback, CustomPPO, CustomExtractor
)

pyautogui.PAUSE = 0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 计算 Observation Dimension
YOLO_DIM = len(TARGET_CLASSES) * 4 * 8
BASE_DIM = 2
CNN_DIM = 0
for reg in cnn_regions:
    name = reg["name"]
    if name in {"1p", "2p", "summon", "scroll"}:
        CNN_DIM += 16
    elif name in {"stand", "skill1", "skill2", "1p_energy", "2p_energy", "time"}:
        CNN_DIM += 1
OBS_DIM = BASE_DIM + YOLO_DIM + CNN_DIM

# ============= 全局 YOLO 单例 (防止多环境显存爆炸) =============
_GLOBAL_YOLO_MODEL = None


def get_global_yolo_model():
    global _GLOBAL_YOLO_MODEL
    if _GLOBAL_YOLO_MODEL is None:
        _GLOBAL_YOLO_MODEL = init_yolo_model()
    return _GLOBAL_YOLO_MODEL


class EnhancedRealtimeGameEnv(gym.Env):
    """主游戏环境类"""

    def __init__(self, env_id: int = 0):
        super().__init__()
        self.env_id = env_id
        # 动作空间: [Move(9), Skill1...Skill6(2...2)]
        # 修改：移除了 u, o, l，现在只有 6 个技能键
        self.action_space = spaces.MultiDiscrete([9, 2, 2, 2, 2, 2, 2])
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

        try:
            self.sct = mss.mss()
        except:
            self.sct = None

        # 使用全局单例，避免重复加载
        self.yolo_model = get_global_yolo_model()
        self.cnn_processor = get_global_cnn_processor()

        # 状态初始化
        self._reset_internal_state()
        self._key_control_enabled = True
        self.last_output_time = time.time()
        print(f"[Env {env_id}] 初始化完成")

    def _reset_internal_state(self):
        self.start_counter = 0
        # 结束判定计数器
        self.end_counter_time = 0

        # [新增] 步数计数器，用于统计每局时长
        self.episode_steps = 0

        # 死亡检测计数器 (x-181)
        self.end_counter_1p_dead = 0
        self.end_counter_2p_dead = 0

        # 状态记录
        self.prev_time_val = -10.0  # 上一帧时间值
        self.bblood = 1.00
        self.rblood = 1.00
        self.assign_flag = None
        # 记录上次打印的 assign_flag，防止刷屏
        self.last_printed_assign_flag = None

        self.init_bblood = 1.00
        self.init_rblood = 1.00
        self._latest_boxes = {}
        self.prev_bblood = 1.00
        self.prev_rblood = 1.00
        self.bp_energy = 0.0
        self.rp_energy = 0.0
        self.currently_pressed_keys = set()
        # 注意：prev_action 长度也要相应调整，虽然这里只是初始化
        self.prev_action = (0, np.zeros(6, dtype=int))
        self.cumulative_reward = 0.00
        self.game_end_time = None
        self.last_step_time = 0

    def log(self, *args):
        if self.env_id == 0:
            print(*args)
            self.last_output_time = time.time()

    def step(self, action):
        self.episode_steps += 1  # 步数+1

        # 动作处理
        if isinstance(action, (list, np.ndarray)):
            move_idx = int(action[0])
            skill_vec = np.array(action[1:], dtype=int).flatten()
        else:
            move_idx, skill_vec = 0, np.zeros(6, dtype=int)

        full_action = np.concatenate([[move_idx], skill_vec])

        if self._episode_done:
            return self._get_default_obs()[0], 0.0, True, False, {}

        # 获取观测 (unpack 增加 raw_ids)
        obs, t, p1, p2, _, cnn_vecs, raw_ids = self._get_obs()

        # 动作掩码与执行
        masked_action = self._apply_action_masks(full_action, obs)
        if self.env_id == 0 and self._key_control_enabled:
            self._execute_actions(masked_action)

        # 奖励计算与状态更新
        # [注意] 这里返回的 reward 是瞬时奖励
        reward = self._calculate_reward(raw_ids)

        self.prev_action = full_action

        # 结束判定 (传入 raw_ids 用于判断 id=181)
        terminated = self._check_done(t, p1, p2, cnn_vecs, raw_ids)
        if terminated: self._episode_done = True

        return obs, reward, terminated, False, {}

    def _get_obs(self):
        now = time.time()
        if now - self.last_step_time < 0.01: time.sleep(0.01 - (now - self.last_step_time))
        self.last_step_time = time.time()

        # 错误情况处理：多返回一个空列表
        if self.sct is None: return self._get_default_obs() + ([0.0] * CNN_DIM, [])

        try:
            full_img = np.array(self.sct.grab(self.sct.monitors[0]))[..., :3]
            # YOLO
            yolo_flat, detected_classes = self._process_yolo(full_img)
            # Health
            p1_blood, p2_blood = self._process_health(full_img)
            # CNN (这里会接收 (vecs, ids) 元组)
            cnn_res = self._process_cnn(full_img)
            # 兼容处理，防止 process_cnn 报错返回错误格式
            if isinstance(cnn_res, tuple) and len(cnn_res) == 3:
                cnn_vecs, time_val, raw_ids = cnn_res
            else:
                cnn_vecs, time_val, raw_ids = [], -1.0, []

            # 更新状态
            self._update_blood_assignment(time_val, detected_classes, p1_blood, p2_blood)
            if cnn_vecs:
                if len(cnn_vecs) >= 3: self.bp_energy = cnn_vecs[-3]
                if len(cnn_vecs) >= 2: self.rp_energy = cnn_vecs[-2]

            obs = np.array([self.rblood, self.bblood] + yolo_flat + cnn_vecs, dtype=np.float32)
            # 向上层多返回一个 raw_ids
            return obs, time_val, self.rblood, self.bblood, detected_classes, cnn_vecs, raw_ids
        except Exception as e:
            print(f"Obs error: {e}")
            return self._get_default_obs() + ([0.0] * CNN_DIM, [])

    def _get_default_obs(self):
        return np.zeros(OBS_DIM, dtype=np.float32), -1.0, 1.0, 1.0, set()

    def _process_yolo(self, full_img):
        if self.yolo_model is None: return [0.0] * YOLO_DIM, set()
        try:
            g = GAME_REGION
            game_crop = full_img[g['top']:g['top'] + g['height'], g['left']:g['left'] + g['width']]
            results = self.yolo_model(cv2.cvtColor(game_crop, cv2.COLOR_BGR2RGB), size=320)

            boxes = {k: [] for k in TARGET_CLASSES}
            detected = set()
            class_to_idx_yolo = {'p': 0, 's': 1, 'r': 2, 'b': 3}

            for *xyxy, conf, cls in results.pred[0]:
                cname = self.yolo_model.names[int(cls)]
                if cname in boxes:
                    detected.add(cname)
                    x1, y1, x2, y2 = map(float, xyxy)
                    norm = normalize_tuple((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, g['width'], g['height'])
                    one_hot = [0.0] * 4
                    one_hot[class_to_idx_yolo[cname]] = float(conf)
                    boxes[cname].append(list(norm) + one_hot)

            yolo_flat = []
            for cls in sorted(TARGET_CLASSES):
                cls_boxes = boxes[cls][:4]
                while len(cls_boxes) < 4: cls_boxes.append([0.0] * 8)
                for b in cls_boxes: yolo_flat.extend(b)

            self._latest_boxes = boxes
            return yolo_flat, detected
        except:
            return [0.0] * YOLO_DIM, set()

    def _process_health(self, full_img):
        try:
            c1 = full_img[MONITOR_1P['top']:MONITOR_1P['top'] + MONITOR_1P['height'],
                 MONITOR_1P['left']:MONITOR_1P['left'] + MONITOR_1P['width']]
            c2 = full_img[MONITOR_2P['top']:MONITOR_2P['top'] + MONITOR_2P['height'],
                 MONITOR_2P['left']:MONITOR_2P['left'] + MONITOR_2P['width']]
            return get_health(c1), get_health(c2)
        except:
            return 1.0, 1.0

    def _process_cnn(self, full_img):
        try:
            crops = [full_img[r["top"]:r["top"] + r["height"], r["left"]:r["left"] + r["width"]] for r in cnn_regions]
            # [修改] 现在 predict 返回 (vecs, raw_ids)
            vecs, raw_ids = self.cnn_processor.predict(crops, self.assign_flag)
            time_val = vecs[-1] if vecs else -1.0
            # 返回三样东西
            return vecs, time_val, raw_ids
        except:
            return [0.0] * CNN_DIM, -1.0, []

    def _update_blood_assignment(self, time_val, detected, p1, p2):
        if time_val >= 0.999:
            self.init_bblood, self.init_rblood = self.bblood, self.rblood
            b_boxes = []
            for c, boxes in self._latest_boxes.items():
                if c == 'b': b_boxes.extend(boxes)

            if b_boxes:
                avg_x = sum(b[0] for b in b_boxes) / len(b_boxes)
                self.assign_flag = 'right' if (avg_x * GAME_REGION['width']) > 415 else 'left'

                # [修改] 仅当判定发生变化时才打印
                if self.env_id == 0:
                    if self.assign_flag != self.last_printed_assign_flag:
                        self.log(f"【判定】b 在 {'右' if self.assign_flag == 'right' else '左'}半屏")
                        self.last_printed_assign_flag = self.assign_flag
            else:
                self.assign_flag = 'right'

        if self.assign_flag == 'right':
            self.bblood, self.rblood = p2, p1
        else:
            self.bblood, self.rblood = p1, p2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal_state()
        self._episode_done = False

        while True:
            # reset 时不需要 cnn_vecs/raw_ids
            obs, t, _, _, _, _, _ = self._get_obs()
            if abs(t - 1.0) < 0.01:
                self.start_counter += 2
            else:
                self.start_counter = 0

            if self.start_counter >= 5:
                self.log("【提示】新一轮游戏开始")
                # 重置所有计数器
                self.end_counter_time = 0
                self.end_counter_1p_dead = 0
                self.end_counter_2p_dead = 0
                self.end_counter_1p_none = 0
                self.end_counter_2p_none = 0
                self.prev_time_val = t
                return obs, {}
            time.sleep(0.01)

    def _apply_action_masks(self, action, obs):
        # 简化版掩码逻辑
        move, skill = int(action[0]), np.array(action[1:], dtype=int)

        # SKILL_MAP 映射参考:
        # 0:None, 1:j(idx0), 2:k(idx1), 3:i(idx2), 4:q(idx3), 5:e(idx4), 6:space(idx5)

        # 能量检查 (bp_energy 是最新的)
        # Space(替身) 是第6个技能，数组下标 5
        if self.bp_energy == 0.0: skill[5] = 0

        # CNN CD检查 (硬编码索引，基于 OBS 结构)
        # obs[cnn_start + 64] -> stand (space)
        # obs[cnn_start + 65] -> skill1 (j)
        # obs[cnn_start + 66] -> skill2 (i)
        cnn_start = BASE_DIM + YOLO_DIM

        def safe_get(idx):
            return obs[cnn_start + idx] if 0 <= cnn_start + idx < len(obs) else 0.0

        if safe_get(64) != 0.0: skill[5] = 0  # space (idx 5)
        if safe_get(65) != 0.0: skill[0] = 0  # j (idx 0)
        if safe_get(66) != 0.0: skill[2] = 0  # i (idx 2)

        return np.concatenate([[move], skill])

    def _execute_actions(self, action):
        if not self._key_control_enabled: return
        try:
            desired = set(MOVE_MAP.get(int(action[0]), set()))
            for i, v in enumerate(action[1:], start=1):
                # i 从 1 开始，对应 SKILL_MAP 的 key
                if v == 1 and i < len(SKILL_MAP):
                    k = SKILL_MAP.get(i)
                    if k: desired.add(k)

            to_release = self.currently_pressed_keys - desired
            to_press = desired - self.currently_pressed_keys

            for k in to_release: pyautogui.keyUp(k)
            for k in to_press: pyautogui.keyDown(k)
            self.currently_pressed_keys = desired
        except Exception:
            pass

    def release_all_keys(self):
        try:
            for k in list(self.currently_pressed_keys): pyautogui.keyUp(k)
            self.currently_pressed_keys.clear()
        except:
            pass

    def _calculate_reward(self, raw_ids):
        """
        基于血量的纯粹奖励计算
        已移除: 能量奖励、死亡检测奖励、以及相关的 prev_bp_energy 状态更新
        """

        step_reward_change = 0.0
        reward_reasons = []

        # --- 1. 血量奖励 ---
        deltab = (self.prev_bblood - self.bblood)
        deltar = (self.prev_rblood - self.rblood)

        # 异常过滤
        valid_blood = True

        if (self.bblood == 0 and self.rblood == 0) or \
                (abs(deltab) > 0.3 or abs(deltar) > 0.3):
            valid_blood = False

        if valid_blood:
            blood_diff = deltar - deltab
            if abs(blood_diff) > 1e-6:
                step_reward_change += blood_diff
                reward_reasons.append(f"血量差 (敌{deltar:+.6f}, 我{deltab:+.6f})")

        self.prev_bblood = self.bblood
        self.prev_rblood = self.rblood

        # [已移除] 能量奖励逻辑
        # [已移除] 死亡检测奖励逻辑
        # [已移除] self.prev_bp_energy/prev_rp_energy 的更新

        # 计算累计奖励用于统计，但返回瞬时奖励给模型
        self.cumulative_reward += step_reward_change

        # 打印日志
        if PRINT_REWARD_DETAILS and self.env_id == 0 and abs(step_reward_change) > 1e-6:
            reason_str = " | ".join(reward_reasons)
            print(f"[Reward] {step_reward_change:+.6f} (Total: {self.cumulative_reward:.6f}) -> {reason_str}")
            self.last_output_time = time.time()

        return step_reward_change  # 返回瞬时奖励

    def _check_done(self, t, p1, p2, cnn_vecs, raw_ids):
        """
        游戏结束判定逻辑 (Strict):
        1. 时间归零 (t < 0.001) 连续 5 帧
        4. 1P 或 2P 头像识别为 'x' (id=181) 连续 5 帧 (代表死亡)
        5. 时间跳回 60 (t >= 0.98)，判定为新局开始 -> 结束当前局
        """
        end_reason = ""
        is_new_round = False

        # 4. 新局开始判定
        if t > 0.98 and self.prev_time_val < 0.98:
            is_new_round = True

        # 更新上一帧时间
        self.prev_time_val = t

        # 1. 时间归零判定
        if abs(t) < 1e-3:
            self.end_counter_time += 1
        else:
            self.end_counter_time = 0

        # 3. 死亡判定 (ID=181)
        if raw_ids and len(raw_ids) >= 2:
            id_1p = raw_ids[0]
            id_2p = raw_ids[1]

            if id_1p == 181:
                self.end_counter_1p_dead += 1
            else:
                self.end_counter_1p_dead = 0

            if id_2p == 181:
                self.end_counter_2p_dead += 1
            else:
                self.end_counter_2p_dead = 0
        else:
            self.end_counter_1p_dead = 0
            self.end_counter_2p_dead = 0

        # 综合判定
        done = False
        if is_new_round:
            done = True
            end_reason = f"New Round Detected (t={t:.6f})"
        elif self.end_counter_time >= 5:
            done = True
            end_reason = f"TimeZero ({self.end_counter_time})"
        elif self.end_counter_1p_dead >= 5:
            done = True
            if self.assign_flag == 'right':
                who = "ENEMY (1P)"
            else:
                who = "SELF (1P)"
            end_reason = f"{who} Dead (x-181) ({self.end_counter_1p_dead})"
        elif self.end_counter_2p_dead >= 5:
            done = True
            if self.assign_flag == 'right':
                who = "SELF (2P)"
            else:
                who = "ENEMY (2P)"
            end_reason = f"{who} Dead (x-181) ({self.end_counter_2p_dead})"

        if done:
            if self.game_end_time is None: self.game_end_time = time.time()
            self.log(f"【环境】对局结束 | 原因: {end_reason}")
            self.release_all_keys()
        return done

    def close(self):
        self.release_all_keys()
        if self.sct: self.sct.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    metadata = TrainingMetadata(TRAINING_METADATA_PATH)

    print(
        f"\n{'=' * 30}\n📊 训练统计: 轮数={metadata.data['total_episodes']}, 步数={metadata.data['total_timesteps']}\n{'=' * 30}\n")

    # 打印统计信息的精度调整
    stats = metadata.get_stats()
    print(f"   - 近期平均奖励: {stats['avg_reward']:.6f}")

    # [关键修改] 只使用 1 个环境
    env = DummyVecEnv([lambda: EnhancedRealtimeGameEnv(0)])

    policy_kwargs = dict(features_extractor_class=CustomExtractor, features_extractor_kwargs=dict(features_dim=256))

    # [关键修改] 模型加载逻辑
    if os.path.exists(PPO_MODEL_PATH):
        print(f"【提示】发现断点续训模型: {PPO_MODEL_PATH}")
        try:
            model = CustomPPO.load(PPO_MODEL_PATH, env=env, custom_objects={"policy_class": ActorCriticPolicy},
                                   tensorboard_log=TENSORBOARD_LOG_DIR)
            print("【提示】成功加载 PPO 模型，继续训练")
        except Exception as e:
            print(f"【警告】模型加载失败: {e}")
            model = None
    else:
        model = None

    if model is None:
        imit_model_path = os.path.join(os.path.dirname(PPO_MODEL_PATH), "ppo_imitation.zip")
        if os.path.exists(imit_model_path):
            print(f"【提示】发现模仿学习预训练模型: {imit_model_path}")
            try:
                model = CustomPPO.load(imit_model_path, env=env, custom_objects={"policy_class": ActorCriticPolicy},
                                       tensorboard_log=TENSORBOARD_LOG_DIR)
                print("【提示】成功加载模仿学习模型")
            except Exception as e:
                print(f"【警告】模仿学习模型加载失败: {e}")
                model = None
        else:
            print(f"【提示】未发现模仿学习模型 ({imit_model_path})，跳过")

    if model is None:
        print("【提示】创建全新模型开始训练")
        model = CustomPPO(ActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1, n_steps=512,
                          batch_size=256, learning_rate=3e-4,
                          tensorboard_log=TENSORBOARD_LOG_DIR)

    try:
        # [关键修复] 传入 env 到 callback
        model.learn(total_timesteps=100000, callback=EpisodeTrackerCallback(metadata, env=env))
        metadata.save()
    except KeyboardInterrupt:
        print("【提示】手动中断")
    finally:
        model.save(PPO_MODEL_PATH)
        metadata.save()
        env.close()
        print("【提示】保存退出")