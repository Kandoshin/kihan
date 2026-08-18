import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

# 导入原有配置
from game_config import PPO_MODEL_PATH
from main_training import EnhancedRealtimeGameEnv, CustomExtractor
from train_utils import CustomPPO

# ============= 配置 =============
DATA_DIR = "recorded_data"  # 数据目录
SAVE_MODEL_PATH = "ppo_imitation.zip"  # 预训练模型保存文件名
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# [配置] 数据清洗：保留静止帧的比例 (0.0 ~ 1.0)
KEEP_NOOP_RATIO = 0.2


def load_all_data(data_dir):
    """加载目录下所有符合 expert_data*.npz 模式的文件，并进行清洗"""
    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return None, None

    # 筛选文件
    files = [f for f in os.listdir(data_dir) if f.startswith('expert_data') and f.endswith('.npz')]
    if not files:
        print(f"❌ 在 {data_dir} 中未找到 expert_data*.npz 文件")
        return None, None

    print(f"📂 找到 {len(files)} 个数据文件，正在加载并清洗...")

    # --- 统计变量 ---
    stats = {
        'raw_total': 0, 'raw_active': 0, 'raw_static': 0,
        'clean_total': 0, 'clean_active': 0, 'clean_static': 0
    }

    all_obs = []
    all_actions = []

    for f in files:
        path = os.path.join(data_dir, f)
        try:
            data = np.load(path)
            if len(data['obs']) > 0:
                obs = data['obs']
                acts = data['actions']

                # 遍历每一帧进行过滤
                for i in range(len(obs)):
                    action = acts[i]
                    # 判断是否为静止帧 (Move=0 且 所有Skill=0)
                    is_noop = (action[0] == 0) and (np.sum(action[1:]) == 0)

                    # --- 原始数据统计 ---
                    stats['raw_total'] += 1
                    if is_noop:
                        stats['raw_static'] += 1
                    else:
                        stats['raw_active'] += 1

                    # --- 清洗逻辑 ---
                    keep = False
                    if is_noop:
                        # 静止帧：按概率保留
                        if random.random() < KEEP_NOOP_RATIO:
                            keep = True
                    else:
                        # 动作帧：全部保留
                        keep = True

                    if keep:
                        all_obs.append(obs[i])
                        all_actions.append(acts[i])
                        # --- 清洗后数据统计 ---
                        stats['clean_total'] += 1
                        if is_noop:
                            stats['clean_static'] += 1
                        else:
                            stats['clean_active'] += 1

                print(f"  - ✅ 已处理: {f} (原: {len(obs)})")
            else:
                print(f"  - ⚠️ 跳过空文件: {f}")
        except Exception as e:
            print(f"  - ❌ 加载失败 {f}: {e}")

    if not all_obs:
        print("❌ 没有加载到任何有效数据")
        return None, None

    # --- 打印统计报告 ---
    print("\n" + "=" * 40)
    print("📊 数据清洗统计报告")
    print("-" * 40)
    print(f"【清洗前】总帧数: {stats['raw_total']}")
    print(f"   ├─ 动作帧: {stats['raw_active']} ({stats['raw_active'] / stats['raw_total'] * 100:.1f}%)")
    print(f"   └─ 静止帧: {stats['raw_static']} ({stats['raw_static'] / stats['raw_total'] * 100:.1f}%)")
    print("-" * 40)
    print(f"【清洗后】总帧数: {stats['clean_total']}")
    print(f"   ├─ 动作帧: {stats['clean_active']} ({stats['clean_active'] / stats['clean_total'] * 100:.1f}%)")
    print(f"   └─ 静止帧: {stats['clean_static']} ({stats['clean_static'] / stats['clean_total'] * 100:.1f}%)")
    print("-" * 40)
    print(f"📉 丢弃数据: {stats['raw_total'] - stats['clean_total']} 帧")
    print("=" * 40 + "\n")

    return np.array(all_obs), np.array(all_actions)


def train_bc():
    # 1. 加载数据
    obs_np, act_np = load_all_data(DATA_DIR)

    if obs_np is None: return

    obs_data = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
    act_data = torch.tensor(act_np, dtype=torch.long).to(DEVICE)

    dataset = TensorDataset(obs_data, act_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 初始化环境
    print("初始化模型结构...")
    env = DummyVecEnv([lambda: EnhancedRealtimeGameEnv(0)])
    policy_kwargs = dict(features_extractor_class=CustomExtractor, features_extractor_kwargs=dict(features_dim=256))

    if os.path.exists(SAVE_MODEL_PATH):
        print(f"🔄 加载旧模型微调: {SAVE_MODEL_PATH}")
        # custom_objects 确保加载正确的策略类
        model = CustomPPO.load(SAVE_MODEL_PATH, env=env, device=DEVICE,
                               custom_objects={"policy_class": ActorCriticPolicy})
    else:
        print("✨ 创建新模型")
        model = CustomPPO(ActorCriticPolicy, env, policy_kwargs=policy_kwargs, device=DEVICE)

    policy = model.policy
    policy.to(DEVICE)
    policy.train()
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # 3. 训练
    print(f"🚀 开始模仿学习训练 (Epochs: {EPOCHS})...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for obs_batch, act_batch in dataloader:
            optimizer.zero_grad()
            dist = policy.get_distribution(obs_batch)
            log_prob = dist.log_prob(act_batch)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # 4. 保存
    model.save(SAVE_MODEL_PATH)
    print(f"✅ 模型已保存至: {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    train_bc()