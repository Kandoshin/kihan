import os
import time
import torch
import pyautogui
from stable_baselines3.common.policies import ActorCriticPolicy

# 导入配置和环境类
# 注意：这需要 main_training.py 在同一目录下
from game_config import PPO_MODEL_PATH
from main_training import EnhancedRealtimeGameEnv
from train_utils import CustomPPO, CustomExtractor


def run_inference():
    print("=" * 40)
    print("start")
    print("=" * 40)

    # 1. 检查模型是否存在
    if not os.path.exists(PPO_MODEL_PATH):
        print(f"❌ 错误：未找到模型文件 {PPO_MODEL_PATH}")
        return

    env = EnhancedRealtimeGameEnv(env_id=0)

    print("正在加载决策模型")
    try:
        # custom_objects 是必须的，因为我们在训练时定义了特殊的 Policy
        model = CustomPPO.load(
            PPO_MODEL_PATH,
            env=env,
            custom_objects={"policy_class": ActorCriticPolicy}
        )
        print("决策模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 开始推理循环
    print("\n开始运行... (按 Ctrl+C 停止)")

    obs, _ = env.reset()

    try:
        while True:
            # 核心推理步骤：
            # deterministic=True 表示使用确定性策略（不探索，只选概率最大的动作）
            # 这通常能让 AI 在实战中表现更稳健
            action, _states = model.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)

            # 如果对局结束，重置环境
            if terminated or truncated:
                print("对局结束，重置环境")
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n手动停止")
    finally:
        # 清理工作
        env.close()
        print("程序已安全退出")


if __name__ == "__main__":
    # 设置 PyAutoGUI 的故障安全和暂停
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0

    run_inference()