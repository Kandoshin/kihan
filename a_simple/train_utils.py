import os
import json
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import gc  # [修改] 导入垃圾回收模块


class TrainingMetadata:
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
        except Exception:
            pass

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
    def __init__(self, metadata: TrainingMetadata, env=None, check_freq: int = 1000):
        self.metadata = metadata
        self.env = env
        self.last_save_episode = 0
        self._step_count = 0
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.stop_file_path = os.path.join(self.current_dir, "stop")
        self.model_save_path = os.path.join(self.current_dir, "ppo_realtime.zip")

    def __call__(self, locals_dict, globals_dict):
        self._step_count += 1
        if os.path.exists(self.stop_file_path):
            print(f"\n🛑 检测到停止信号")
            try:
                os.remove(self.stop_file_path)
            except:
                pass
            return False

        dones = locals_dict.get("dones")
        if dones is not None and np.any(dones):
            try:
                training_env = locals_dict.get("self").env
                if training_env is None: training_env = self.env
                if training_env:
                    episode_rewards = training_env.get_attr("cumulative_reward")
                    real_reward = episode_rewards[0]
                    self.metadata.update_episode(reward=real_reward, timesteps=0)

                    # [修改] 增加保存成功的提示，并明确保存操作
                    model = locals_dict.get("self")
                    if model:
                        model.save(self.model_save_path)
                        self.metadata.save()
                        print(f"💾 [End] Score: {real_reward:.2f} | 模型已自动保存")
            except Exception as e:
                print(f"⚠️ 保存模型或更新元数据失败: {e}")

            # [修改] 对局结束，强制回收内存
            gc.collect()

        if self._step_count % 1000 == 0:
            self.metadata.data['total_timesteps'] += 1000
            print(f"📈 Steps: {self.metadata.data['total_timesteps']}")

        return True


class CustomExtractor(BaseFeaturesExtractor):
    """
    [核心重构]
    输入 OBS 维度 50:
    - [0]: Self ID (Int)
    - [1]: Enemy ID (Int)
    - [2:50]: Scalars (48个浮点数)

    处理逻辑:
    1. 提取 ID -> Embedding -> 16维向量 * 2 = 32维
    2. 提取 Scalars -> 保持 48维
    3. 拼接 -> 32 + 48 = 80维 -> 传入全连接层
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # 定义 Embedding 层
        # 假设最大 ID 不超过 200 (num_classes)，向量维度设为 16
        self.embedding = nn.Embedding(num_embeddings=200, embedding_dim=16)

        # 计算拼接后的总维度
        # ID Embedding (2 * 16 = 32) + Scalars (50 - 2 = 48) = 80
        concat_dim = 32 + (observation_space.shape[0] - 2)

        self.net = nn.Sequential(
            nn.Linear(concat_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, features_dim)
        )

        def init_bias(module):
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, -2.0)

        self.net.apply(init_bias)

    def forward(self, obs):
        # 1. 切片提取 ID (前2个)
        # 注意: obs 是 float 类型，Embedding 需要 long 类型索引
        # 必须确保 ID 在 [0, 199] 范围内，使用 clamp 防止越界崩溃
        ids = obs[:, 0:2].long()
        ids = torch.clamp(ids, 0, 199)

        # 2. 切片提取 标量数据 (从第3个开始到最后)
        scalars = obs[:, 2:]

        # 3. ID 向量化
        # (Batch, 2) -> (Batch, 2, 16) -> Flatten -> (Batch, 32)
        embedded_vecs = self.embedding(ids).flatten(1)

        # 4. 拼接
        combined = torch.cat([embedded_vecs, scalars], dim=1)

        # 5. 前向传播
        return self.net(combined)


class CustomPPO(PPO):
    pass