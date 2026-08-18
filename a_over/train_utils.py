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
    4. [修正] 每局结束(Done)时，从环境读取真实的累计奖励并保存
    """

    def __init__(self, metadata: TrainingMetadata, env, check_freq: int = 1000):
        self.metadata = metadata
        self.env = env  # [新增] 保存环境引用，以便直接读取内部状态
        self.last_save_episode = 0
        self._step_count = 0
        self.check_freq = check_freq
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.stop_file_path = os.path.join(self.current_dir, "stop")
        self.model_save_path = os.path.join(self.current_dir, "ppo_realtime.zip")

    def __call__(self, locals_dict, globals_dict):
        self._step_count += 1

        # 1. 检查软停止信号 (Soft Stop)
        if os.path.exists(self.stop_file_path):
            print(f"\n🛑 检测到停止信号文件: {self.stop_file_path}")
            print("🛑 正在优雅停止训练并保存模型...")
            try:
                os.remove(self.stop_file_path)
            except:
                pass
            return False

        # 2. [关键修正] 检测对局结束信号 (Done) 并进行准确统计
        dones = locals_dict.get("dones")
        if dones is not None and np.any(dones):
            # 获取模型实例
            model = locals_dict.get("self")

            # [核心修复] 从环境实例直接读取本局的累计奖励
            # self.env 是 DummyVecEnv，self.env.envs[0] 才是我们的 EnhancedRealtimeGameEnv
            try:
                real_env = self.env.envs[0]
                episode_reward = real_env.cumulative_reward
                episode_steps = getattr(real_env, 'episode_steps', 0)

                # 更新元数据
                self.metadata.update_episode(reward=episode_reward, timesteps=episode_steps)

                print(
                    f"📊 [Episode {self.metadata.data['total_episodes']}] 结束 | 得分: {episode_reward:.2f} | 步数: {episode_steps}")

                # 强制保存
                if model:
                    model.save(self.model_save_path)
                self.metadata.save()

            except Exception as e:
                print(f"⚠️ 统计更新/保存失败: {e}")

            # 打印近期统计
            if self.metadata.data['total_episodes'] - self.last_save_episode >= 10:
                self.last_save_episode = self.metadata.data['total_episodes']
                stats = self.metadata.get_stats()
                print(f"\n📈 [Stats] 近期平均分: {stats['avg_reward']:.2f} (Total Eps: {stats['total_episodes']})")

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
                nn.init.constant_(module.bias, -2.0)

        self.net.apply(init_skill_bias)

    def forward(self, obs):
        return self.net(obs)


class CustomPPO(PPO):
    pass