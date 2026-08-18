import os
import time
import numpy as np
import torch
import pyautogui
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

# 导入配置和环境
from game_config import PPO_MODEL_PATH
from main_training import EnhancedRealtimeGameEnv, CustomExtractor
from train_utils import CustomPPO

# ============= 配置 =============
# 这里指定你要测试的模型路径
# 如果你想测试刚才的模仿学习模型，请把 'ppo_imitation.zip' 写在这里
MODEL_TO_TEST = "ppo_imitation.zip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def play():
    print(f"正在初始化环境...")
    # 创建一个单环境用于推理
    env = DummyVecEnv([lambda: EnhancedRealtimeGameEnv(0)])

    # 检查模型是否存在
    if not os.path.exists(MODEL_TO_TEST):
        print(f"❌ 找不到模型文件: {MODEL_TO_TEST}")
        return

    print(f"正在加载模型: {MODEL_TO_TEST} ...")

    # 加载模型 (注意：不需要 tensorboard_log，因为只是推理)
    try:
        model = CustomPPO.load(
            MODEL_TO_TEST,
            env=env,
            device=DEVICE,
            custom_objects={"policy_class": ActorCriticPolicy}
        )
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print("\n" + "=" * 40)
    print("🎮 AI 对战模式已启动！")
    print("请切换到游戏窗口。")
    print("按 Ctrl+C 停止。")
    print("=" * 40 + "\n")

    obs = env.reset()

    try:
        while True:
            # 1. AI 决策
            # deterministic=True 让 AI 输出概率最大的动作（更稳定）
            # deterministic=False 让 AI 按概率采样（更具多样性，但可能偶尔犯傻）
            # 对于格斗游戏，通常推荐 True，或者 False 以增加不可预测性
            action, _states = model.predict(obs, deterministic=False)

            # 2. 执行动作
            obs, rewards, dones, info = env.step(action)

            # 这里的 env.step 已经包含了 _execute_actions (按键操作)
            # 所以你不需要额外写代码，AI 已经在玩了。

            # 如果对局结束，env.step 会自动 reset，你不需要手动处理

            # 稍微加一点点延时防止 CPU 占用过高 (可选，根据实际 FPS 调整)
            # time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n🛑 用户停止")
    finally:
        env.close()
        print("环境已关闭")


if __name__ == "__main__":
    play()