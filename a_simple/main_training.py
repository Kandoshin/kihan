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

# 引入配置
from game_config import (
    GAME_REGION, TARGET_CLASSES,
    MOVE_MAP, SKILL_MAP, TRAINING_METADATA_PATH, PPO_MODEL_PATH,
    cnn_regions, TENSORBOARD_LOG_DIR,
    NONE_ID, DEATH_ID
)
from vision_processor import (
    get_global_cnn_processor, init_yolo_model, normalize_tuple
)
from train_utils import (
    TrainingMetadata, EpisodeTrackerCallback, CustomPPO, CustomExtractor
)

# ============= [调试开关] =============
# 设置为 True 以打印详细的奖励变化和原因
PRINT_REWARD_DETAILS = True

# [新增] 设置为 True 以开启帧率计算和打印 (每10秒打印一次)
PRINT_FPS = True

pyautogui.PAUSE = 0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ============= 维度计算 (复习) =============
# IDs(2) + B/R_Box(8) + P_Boxes(16) + S_Boxes(16) + WM/RM(2) + Scalars(6) = 50
OBS_DIM = 50

# Scalars 在 OBS 数组的最后 6 位，顺序如下 (来自 vision_processor.py):
# [Time, Stand, Skill1, Skill2, Summon, Scroll]
# Index:
# Time: -6
# Stand (Space/替身): -5
# Skill1 (J): -4
# Skill2 (I): -3
# Summon: -2
# Scroll: -1

_GLOBAL_YOLO_MODEL = None


def get_global_yolo_model():
    global _GLOBAL_YOLO_MODEL
    if _GLOBAL_YOLO_MODEL is None:
        _GLOBAL_YOLO_MODEL = init_yolo_model()
    return _GLOBAL_YOLO_MODEL


class EnhancedRealtimeGameEnv(gym.Env):
    def __init__(self, env_id: int = 0):
        super().__init__()
        self.env_id = env_id
        # [9移动, 6技能]
        # Skill Map: 0:None, 1:j, 2:k, 3:i, 4:q, 5:e, 6:space
        self.action_space = spaces.MultiDiscrete([9, 2, 2, 2, 2, 2, 2])
        self.observation_space = spaces.Box(low=-1.0, high=999.0, shape=(OBS_DIM,), dtype=np.float32)

        try:
            self.sct = mss.mss()
        except:
            self.sct = None

        self.yolo_model = get_global_yolo_model()
        self.cnn_processor = get_global_cnn_processor()

        self._reset_internal_state()
        self.last_episode_reward = 0.0  # [新增] 专门用于存储上一局的最终得分，不受reset影响
        self._key_control_enabled = True
        self._episode_done = False

        # [FPS功能] 初始化 FPS 计数器
        self.fps_start_time = time.time()
        self.fps_frame_count = 0

        print(f"[Env {env_id}]环境初始化完成")

    def _reset_internal_state(self):
        self.start_counter = 0
        self.end_counter_time = 0
        self.episode_steps = 0
        self.end_counter_1p_dead = 0
        self.end_counter_2p_dead = 0
        self.assign_flag = 'right'
        self.prev_time_val = -1.0
        self.cumulative_reward = 0.0
        self.game_end_time = None
        self.last_step_time = 0
        self.currently_pressed_keys = set()

    def step(self, action):
        self.episode_steps += 1

        # [FPS功能] 帧率计算逻辑
        if PRINT_FPS and self.env_id == 0:
            self.fps_frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            if elapsed_time >= 10.0:  # 每10秒打印一次
                fps = self.fps_frame_count / elapsed_time
                print(f"fps: {fps:.2f}")
                # 重置计数器
                self.fps_start_time = current_time
                self.fps_frame_count = 0

        if self._episode_done:
            self.release_all_keys()
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, {}

        if isinstance(action, (list, np.ndarray)):
            move_idx = int(action[0])
            skill_vec = np.array(action[1:], dtype=int).flatten()
        else:
            move_idx, skill_vec = 0, np.zeros(6, dtype=int)

        obs, t_val, raw_ids, wm_val, rm_val = self._get_obs()

        full_action = np.concatenate([[move_idx], skill_vec])
        masked_action = self._apply_action_masks(full_action, obs)

        masked_move = masked_action[0]
        masked_skill = masked_action[1:]

        if self.env_id == 0 and self._key_control_enabled:
            self._execute_actions(masked_move, masked_skill)

        reward = 0.0
        if wm_val > 0.5: reward += 0.01
        if rm_val > 0.5: reward -= 0.01

        self.cumulative_reward += reward

        terminated = self._check_done(t_val, raw_ids)
        if terminated:
            self._episode_done = True
            # [关键修改] 在被 reset 归零前，把总分备份到这个变量里
            self.last_episode_reward = self.cumulative_reward
            self.release_all_keys()

        if PRINT_REWARD_DETAILS and abs(reward) > 0 and self.env_id == 0:
            print(
                f"[Step {self.episode_steps}] Reward: {reward:+.2f} | Total: {self.cumulative_reward:+.2f} ")
            # f"[Step {self.episode_steps}] Reward: {reward:+.2f} | Total: {self.cumulative_reward:+.2f} (WM:{wm_val:.0f} RM:{rm_val:.0f})")

        return obs, reward, terminated, False, {}

    def _get_obs(self):
        now = time.time()
        if now - self.last_step_time < 0.01: time.sleep(0.01 - (now - self.last_step_time))
        self.last_step_time = time.time()

        if self.sct is None: return np.zeros(OBS_DIM, dtype=np.float32), -1.0, [], 0, 0

        full_img = np.array(self.sct.grab(self.sct.monitors[0]))[..., :3]
        yolo_data = self._process_yolo(full_img)

        id_1p, id_2p, scalars, raw_ids = self.cnn_processor.predict(
            [full_img[r["top"]:r["top"] + r["height"], r["left"]:r["left"] + r["width"]] for r in cnn_regions]
        )
        t_val = scalars[0]

        # [修改] 仅在开局时间 t=1.0 时判断 assign_flag
        # 避免对局中途因角色换位导致数据跳变
        if t_val == 1.0:
            b_box = yolo_data.get('b', [0, 0, 0, 0])
            if b_box[2] > 0:
                new_flag = 'right' if (b_box[0] * GAME_REGION['width']) > 415 else 'left'

                # 如果判定发生了改变，打印日志
                if new_flag != self.assign_flag:
                    if self.env_id == 0:
                        print(f"【判定】当前角色位置为:{new_flag} ")
                    self.assign_flag = new_flag

        if self.assign_flag == 'left':
            final_ids = [float(id_1p), float(id_2p)]
        else:
            final_ids = [float(id_2p), float(id_1p)]

        b_coords = yolo_data.get('b', [0.0] * 4)
        r_coords = yolo_data.get('r', [0.0] * 4)

        p_flat = []
        for box in yolo_data.get('p', []): p_flat.extend(box)
        while len(p_flat) < 16: p_flat.append(0.0)

        s_flat = []
        for box in yolo_data.get('s', []): s_flat.extend(box)
        while len(s_flat) < 16: s_flat.append(0.0)

        wm_val = float(yolo_data.get('wm', 0))
        rm_val = float(yolo_data.get('rm', 0))

        obs_list = final_ids + b_coords + r_coords + p_flat + s_flat + [wm_val, rm_val] + scalars
        return np.array(obs_list, dtype=np.float32), t_val, raw_ids, wm_val, rm_val

    def _process_yolo(self, full_img):
        data = {'p': [], 's': [], 'b': [0.0] * 4, 'r': [0.0] * 4, 'wm': 0, 'rm': 0}
        if self.yolo_model is None: return data

        try:
            g = GAME_REGION
            crop = full_img[g['top']:g['top'] + g['height'], g['left']:g['left'] + g['width']]

            with torch.no_grad():
                results = self.yolo_model(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), size=320)

            for *xyxy, conf, cls in results.pred[0]:
                cname = self.yolo_model.names[int(cls)]
                threshold = 0.5 if cname in ['wm', 'rm'] else 0.8

                if conf < threshold: continue

                x1, y1, x2, y2 = map(float, xyxy)
                norm = normalize_tuple((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, g['width'], g['height'])

                if cname in ['b', 'r']:
                    if data[cname][2] == 0: data[cname] = list(norm)
                elif cname in ['wm', 'rm']:
                    data[cname] = 1
                elif cname in ['p', 's']:
                    if len(data[cname]) < 4:
                        data[cname].append(list(norm))
            return data
        except:
            return data

    def _apply_action_masks(self, action, obs):
        move = int(action[0])
        skill = np.array(action[1:], dtype=int)

        THRESHOLD = 0.001

        if obs[-5] > THRESHOLD: skill[5] = 0
        if obs[-4] > THRESHOLD: skill[0] = 0
        if obs[-3] > THRESHOLD: skill[2] = 0
        if obs[-2] > THRESHOLD: skill[3] = 0
        if obs[-1] > THRESHOLD: skill[4] = 0

        return np.concatenate([[move], skill])

    def _execute_actions(self, move_idx, skill_vec):
        if not self._key_control_enabled: return
        try:
            desired = set(MOVE_MAP.get(int(move_idx), set()))
            for i, v in enumerate(skill_vec, start=1):
                if v == 1:
                    k = SKILL_MAP.get(i)
                    if k: desired.add(k)

            to_release = self.currently_pressed_keys - desired
            to_press = desired - self.currently_pressed_keys

            for k in to_release: pyautogui.keyUp(k)
            for k in to_press: pyautogui.keyDown(k)
            self.currently_pressed_keys = desired
        except:
            pass

    def release_all_keys(self):
        try:
            for k in list(self.currently_pressed_keys): pyautogui.keyUp(k)
            self.currently_pressed_keys.clear()
        except:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal_state()
        self._episode_done = False

        print("【等待】正在等待对局开始")
        self.release_all_keys()

        wait_steps = 0
        while True:
            obs, t, _, _, _ = self._get_obs()

            if t == 1:
                self.start_counter += 1
            else:
                self.start_counter = 0

            if self.start_counter >= 5:
                print(f"【提示】新一轮游戏开始")
                self.prev_time_val = t
                self.end_counter_time = 0
                self.end_counter_1p_dead = 0
                self.end_counter_2p_dead = 0
                return obs, {}

            wait_steps += 1
            if wait_steps % 100 == 0:
                print(f"waiting... t={t:.2f}")

            time.sleep(0.05)

    def _check_done(self, t, raw_ids):
        if abs(t) < 1e-3:
            self.end_counter_time += 1
        else:
            self.end_counter_time = 0

        id_1p = raw_ids[0] if len(raw_ids) > 0 else NONE_ID
        id_2p = raw_ids[1] if len(raw_ids) > 1 else NONE_ID

        if DEATH_ID != -1:
            if id_1p == DEATH_ID:
                self.end_counter_1p_dead += 1
            else:
                self.end_counter_1p_dead = 0

            if id_2p == DEATH_ID:
                self.end_counter_2p_dead += 1
            else:
                self.end_counter_2p_dead = 0
        else:
            self.end_counter_1p_dead = 0
            self.end_counter_2p_dead = 0

        done = False
        reason = ""

        if self.end_counter_time >= 5:
            done = True
            reason = "time over"
        elif self.end_counter_1p_dead >= 5:
            done = True
            # [修正] 结合 assign_flag 判断 1P 是敌是友
            if self.assign_flag == 'left':
                reason = "1p dead (我方阵亡)"  # 我在左边，1P是我
            else:
                reason = "1p dead (敌方阵亡)"  # 我在右边，1P是敌
        elif self.end_counter_2p_dead >= 5:
            done = True
            # [修正] 结合 assign_flag 判断 2P 是敌是友
            if self.assign_flag == 'right':
                reason = "2p dead (我方阵亡)"  # 我在右边，2P是我
            else:
                reason = "2p dead (敌方阵亡)"  # 我在左边，2P是敌

        if done:
            if PRINT_REWARD_DETAILS and self.env_id == 0:
                print(f"【对局结束】reason: {reason}")
            return True
        return False

    def close(self):
        self.release_all_keys()
        if self.sct: self.sct.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    metadata = TrainingMetadata(TRAINING_METADATA_PATH)
    env = DummyVecEnv([lambda: EnhancedRealtimeGameEnv(0)])

    policy_kwargs = dict(features_extractor_class=CustomExtractor, features_extractor_kwargs=dict(features_dim=256))

    if os.path.exists(PPO_MODEL_PATH):
        print(f"加载模型: {PPO_MODEL_PATH}")
        model = CustomPPO.load(PPO_MODEL_PATH, env=env, custom_objects={"policy_class": ActorCriticPolicy})
    else:
        print("创建新模型")
        model = CustomPPO(ActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1,
                          batch_size=256, learning_rate=3e-4, tensorboard_log=TENSORBOARD_LOG_DIR)

    try:
        model.learn(total_timesteps=100000, callback=EpisodeTrackerCallback(metadata, env=env))
    except KeyboardInterrupt:
        print("手动停止")
    finally:
        model.save(PPO_MODEL_PATH)
        metadata.save()
        env.close()