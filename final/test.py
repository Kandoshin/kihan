from stable_baselines3 import PPO

model = PPO.load(r"G:\god\pycharm\PythonProject\test\final\ppo_realtime.zip")
print(model.num_timesteps)  # 训练过的总步数
