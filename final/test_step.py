import time, psutil, pynvml, torch as th
from really_finall import SimplifiedRealtimeGameEnv, CustomPPO, BernoulliPolicy, policy_kwargs
import numpy as np
pynvml.nvmlInit(); gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
env = SimplifiedRealtimeGameEnv(env_id=0); env.reset()
model = CustomPPO.load(r"G:\god\pycharm\PythonProject\test\final\ppo_realtime.zip",
                       env=env, device="cuda")
print("Grab | Vision | PPO | Step | CPU% RAM GPU VRAM")
while True:
    # 抓取
    t0 = time.perf_counter()
    img = np.array(env.sct.grab(env.sct.monitors[0]))[..., :3]  # <-- 加 np.array
    grab = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    yolo, _ = env._process_yolo(img); cnn, _ = env._process_cnn(img); vis = (time.perf_counter()-t0)*1000

    obs = th.tensor([env.rblood, env.bblood] + yolo + cnn, device="cuda")[None]
    t0 = time.perf_counter()
    with th.no_grad():
        act, *_ = model.policy(obs)
        ppo = (time.perf_counter()-t0)*1000

    t0 = time.perf_counter(); _, _, done, _ = env.step(act.cpu().numpy().squeeze()); step = (time.perf_counter()-t0)*1000
    cpu = psutil.cpu_percent(); ram = psutil.virtual_memory().percent; g = pynvml.nvmlDeviceGetUtilizationRates(gpu); vram = pynvml.nvmlDeviceGetMemoryInfo(gpu)
    print(f"{grab:5.1f} | {vis:6.1f} | {ppo:4.1f} | {step:5.1f} | {cpu:4.1f} {ram:4.1f} {g.gpu:2.0f} {100*vram.used/vram.total:4.1f}")