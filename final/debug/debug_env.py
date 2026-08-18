import numpy as np
from debug import EnhancedRealtimeGameEnv

if __name__ == "__main__":
    env = EnhancedRealtimeGameEnv(env_id=0)

    obs, _ = env.reset()
    print("初始观测 shape:", obs.shape)

    for step in range(200):  # 运行200步
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, _, _ = env.step(action)

        # 打印调试信息
        print(f"Step {step+1:03d} | reward={reward:.3f} | "
              f"己方血量={env.bblood:.2f} | 敌方血量={env.rblood:.2f} | "
              f"己方能量={env.bp_energy:.2f} | 敌方能量={env.rp_energy:.2f}")

        if done:
            print("⚠️ 对局结束，重新reset")
            obs, _ = env.reset()

    env.close()
