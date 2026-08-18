import os
import time
import numpy as np
import cv2
import mss
import torch
import gymnasium as gym
import keyboard  # 用于监听键盘
import threading
import re
from stable_baselines3.common.vec_env import DummyVecEnv

# 导入复用的模块
from game_config import (
    GAME_REGION, TARGET_CLASSES, MONITOR_1P, MONITOR_2P,
    MOVE_MAP, SKILL_MAP, KEYS, cnn_regions
)
from vision_processor import (
    get_global_cnn_processor, init_yolo_model, get_health, normalize_tuple
)
from main_training import EnhancedRealtimeGameEnv

# ============= 配置 =============
SAVE_DIR = "recorded_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# 采样频率控制 (每秒采样多少次，建议与训练时的 FPS 接近)
SAMPLE_RATE = 15


def get_next_filename():
    """
    自动查找下一个可用的文件名 (expert_data1.npz, expert_data2.npz...)
    """
    files = os.listdir(SAVE_DIR)
    max_idx = 0
    pattern = re.compile(r"expert_data(\d+)\.npz")

    for f in files:
        match = pattern.match(f)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx

    return os.path.join(SAVE_DIR, f"expert_data{max_idx + 1}.npz")


def get_current_action_vector():
    """
    监听键盘，将当前按下的键转换为 MultiDiscrete 动作向量
    [Move(0-8), Skill1(0/1), ..., Skill9(0/1)]
    """
    # 1. 判断移动 (W, A, S, D) -> 0-8
    pressed = {k for k in ['w', 'a', 's', 'd'] if keyboard.is_pressed(k)}

    move_idx = 0
    # 简单的反向映射逻辑
    for idx, key_set in MOVE_MAP.items():
        if idx == 0: continue
        if key_set == pressed:
            move_idx = idx
            break

    if move_idx == 0 and pressed:
        if 'w' in pressed:
            move_idx = 1
        elif 's' in pressed:
            move_idx = 2
        elif 'a' in pressed:
            move_idx = 3
        elif 'd' in pressed:
            move_idx = 4

    # 2. 判断技能 (J, K, L, I, U, O, Q, E, Space)
    skill_vec = []
    for i in range(1, 10):
        key_name = SKILL_MAP[i]
        if keyboard.is_pressed(key_name):
            skill_vec.append(1)
        else:
            skill_vec.append(0)

    # 组合：[move_idx, s1, s2, ..., s9]
    return np.array([move_idx] + skill_vec, dtype=int)


def save_buffer(observations, actions):
    """保存当前缓冲区数据"""
    if len(observations) > 0:
        filename = get_next_filename()
        print(f"\n💾 正在保存片段数据 ({len(observations)} 帧)...")
        np.savez_compressed(
            filename,
            obs=np.array(observations),
            actions=np.array(actions)
        )
        print(f"✅ 已保存至: {filename}")
    else:
        print("\n⚠️ 当前缓冲区为空，忽略保存")


def record_data():
    print("正在初始化环境...")
    env = EnhancedRealtimeGameEnv(0)

    # 关闭环境内部的 AI 按键控制，只做观测
    env._key_control_enabled = False

    observations = []
    actions = []

    print("\n" + "=" * 40)
    print("🔴 录制准备就绪！")
    print("请切换到游戏窗口开始操作。")
    print("----------------------------------------")
    print("按 'T'   键: 开始录制 / 保存并开始下一段")
    print("按 'G'   键: 放弃当前段 / 重新开始下一段")
    print("按 'ESC' 键: 退出程序 (不保存当前未提交数据)")
    print("----------------------------------------")
    print("=" * 40 + "\n")

    is_recording = False
    step_interval = 1.0 / SAMPLE_RATE

    try:
        while True:
            loop_start = time.time()

            # === 控制逻辑 ===
            if keyboard.is_pressed('t'):
                if not is_recording:
                    # 状态 1: 未录制 -> 开始录制
                    is_recording = True
                    print(f"\n🔴 [开始] 录制已启动...")
                else:
                    # 状态 2: 录制中 -> 保存并重启录制
                    print(f"\n💾 [保存并继续] 正在保存上一段数据...")
                    save_buffer(observations, actions)

                    # 清空缓冲区，立即开始下一段
                    observations = []
                    actions = []
                    print(f"🔴 [重置] 已开始新一段录制...")

                # 简单防抖
                time.sleep(0.5)

            if keyboard.is_pressed('g'):
                if is_recording:
                    print(f"\n🗑️ [放弃并重置] 丢弃当前 {len(observations)} 帧数据...")
                    observations = []
                    actions = []
                    print(f"🔴 [重置] 已开始新一段录制...")
                else:
                    print("\n⚠️ 未在录制中，按 G 无效")

                time.sleep(0.5)

            if keyboard.is_pressed('esc'):
                print("\n❌ 用户退出 (当前未保存的数据将被丢弃)")
                break

            # === 录制逻辑 ===
            if is_recording:
                # 1. 获取当前观测 (Obs)
                full_obs_tuple = env._get_obs()
                obs = full_obs_tuple[0]

                # 2. 获取当前玩家操作 (Action)
                action = get_current_action_vector()

                # 3. 存入缓冲区
                observations.append(obs)
                actions.append(action)

                # 打印状态 (每100帧)
                if len(observations) % 100 == 0:
                    print(f"  -> 当前片段已录制: {len(observations)} 帧 | 动作: {action}")

            # 帧率控制
            elapsed = time.time() - loop_start
            if elapsed < step_interval:
                time.sleep(step_interval - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("录制程序已退出。")


if __name__ == "__main__":
    record_data()