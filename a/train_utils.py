import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class TrainingMetadata:
    """管理训练元数据"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = self.load()

    def load(self) -> dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'total_episodes': 0, 'total_timesteps': 0, 'last_save_time': None, 'training_history': []}

    def save(self):
        try:
            self.data['last_save_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"元数据保存失败: {e}")

    def update_episode(self, reward: float, timesteps: int):
        self.data['total_episodes'] += 1
        self.data['total_timesteps'] += timesteps
        self.data['training_history'].append({
            'episode': self.data['total_episodes'],
            'reward': float(reward),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        if len(self.data['training_history']) > 100:
            self.data['training_history'] = self.data['training_history'][-100:]

    def get_stats(self) -> dict:
        avg = np.mean([h['reward'] for h in self.data['training_history']]) if self.data['training_history'] else 0.0
        return {
            'total_episodes': self.data['total_episodes'],
            'total_timesteps': self.data['total_timesteps'],
            'avg_reward': avg
        }


class EpisodeTrackerCallback:
    """
    回调函数：
    1. 追踪训练进度
    2. 自动保存元数据
    3. 检查 'stop' 文件实现软停止
    4. 对局结束时保存模型 & 记录真实 Episode 奖励
    """

    # [修复] 这里增加了 env=None 参数，解决了你的 TypeError 报错
    def __init__(self, metadata: TrainingMetadata, env=None, check_freq: int = 1000):
        self.metadata = metadata
        self.env = env  # 保存 env 引用
        self.last_save_episode = 0
        self._step_count = 0
        self.check_freq = check_freq
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.stop_file_path = os.path.join(self.current_dir, "stop")
        self.model_save_path = os.path.join(self.current_dir, "ppo_realtime.zip")

    def __call__(self, locals_dict, globals_dict):
        self._step_count += 1

        # 1. 软停止检查
        if os.path.exists(self.stop_file_path):
            print(f"\n🛑 检测到停止信号文件: {self.stop_file_path}")
            print("🛑 正在优雅停止训练并保存模型...")
            try:
                os.remove(self.stop_file_path)
            except:
                pass
            return False

        # 2. 检测对局结束 (Done)
        dones = locals_dict.get("dones")
        if dones is not None and np.any(dones):
            # 尝试获取环境的 cumulative_reward
            try:
                # 获取训练环境 (优先从 locals 获取，如果失败则使用 self.env)
                training_env = locals_dict.get("self").env
                if training_env is None:
                    training_env = self.env

                # 获取第一个环境实例的 cumulative_reward 属性
                if training_env:
                    episode_rewards = training_env.get_attr("cumulative_reward")
                    real_reward = episode_rewards[0]  # 取第一个环境的奖励

                    # 更新元数据
                    self.metadata.update_episode(reward=real_reward, timesteps=0)

                    print(f"💾 [Episode End] 对局结束 (Score: {real_reward:.2f}) -> 自动保存")

                    # 强制保存
                    model = locals_dict.get("self")
                    if model:
                        model.save(self.model_save_path)
                        self.metadata.save()
            except Exception as e:
                # 这种错误通常不影响训练继续，所以只打印警告
                # print(f"⚠️ 获取环境奖励或保存失败: {e}")
                pass

        # 3. 每1000步打印一次进度
        if self._step_count % 1000 == 0:
            self.metadata.data['total_timesteps'] += 1000

            if self.metadata.data['total_episodes'] - self.last_save_episode >= 10:
                self.last_save_episode = self.metadata.data['total_episodes']
                print(
                    f"\n📈 [Stats] 累计步数: {self.metadata.data['total_timesteps']}, 完成对局: {self.metadata.data['total_episodes']}")

        return True


class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, features_dim)
        )

        def init_skill_bias(module):
            if isinstance(module, nn.Linear) and module.bias is not None:
                # 初始化 bias 为负数，抑制初期乱按技能
                nn.init.constant_(module.bias, -2.0)

        self.net.apply(init_skill_bias)

    def forward(self, obs):
        return self.net(obs)


class CustomPPO(PPO):
    pass