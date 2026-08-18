import cv2
import time
import numpy as np
import torch
import os
import sys

# 导入你的环境代码
# [修正] CNN_DIM, YOLO_DIM, BASE_DIM 实际上是在 main_training 中定义的
from main_training import EnhancedRealtimeGameEnv, CNN_DIM, YOLO_DIM, BASE_DIM


def draw_debug_overlay(img, obs, reward, detected_classes, p1_blood, p2_blood, raw_ids):
    """
    在画面上绘制调试信息，帮助可视化 Agent 到底看到了什么
    """
    debug_img = img.copy()
    h, w = debug_img.shape[:2]

    # 1. 绘制血量信息
    cv2.putText(debug_img, f"P1 HP: {p1_blood:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(debug_img, f"P2 HP: {p2_blood:.2f}", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 2. 绘制奖励 (关键!)
    color = (0, 255, 0) if reward > 0 else ((0, 0, 255) if reward < 0 else (200, 200, 200))
    cv2.putText(debug_img, f"Reward: {reward:.4f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 3. 绘制检测到的物体
    yolo_str = "YOLO: " + ", ".join(list(detected_classes)) if detected_classes else "YOLO: None"
    cv2.putText(debug_img, yolo_str, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 4. 绘制 CNN 原始 ID (判断死亡)
    if raw_ids and len(raw_ids) >= 2:
        id_str = f"CNN IDs: 1P={raw_ids[0]}, 2P={raw_ids[1]}"
        # ID 181 是死亡
        if raw_ids[0] == 181: id_str += " (1P DEAD)"
        if raw_ids[1] == 181: id_str += " (2P DEAD)"
        cv2.putText(debug_img, id_str, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return debug_img


def main():
    print("==========================================")
    print("   强化学习环境诊断工具 (Debug Mode)")
    print("   功能: 检查血量读取、奖励计算、视觉识别")
    print("   按 'q' 退出")
    print("==========================================")

    # 初始化环境
    try:
        env = EnhancedRealtimeGameEnv(env_id=0)
    except Exception as e:
        print(f"环境初始化失败: {e}")
        return

    obs, _ = env.reset()

    total_reward = 0.0

    try:
        while True:
            start_time = time.time()

            # 1. 获取观测数据 (直接调用内部函数以获取原始图像用于显示)
            # 注意：这里我们侵入式地获取 sct 的截图用于显示，实际 env.step 也会截一次，会有微小开销
            if env.sct:
                full_img = np.array(env.sct.grab(env.sct.monitors[0]))[..., :3]
                display_img = full_img.copy()
            else:
                display_img = np.zeros((600, 800, 3), dtype=np.uint8)

            # 2. 执行随机动作 (或者你可以改为不操作，只观察)
            # 动作格式: [Move(9), Skill1(2)...Skill9(2)]
            # 这里我们让 Agent 什么都不做，方便你手动操作游戏来测试奖励反馈
            action = np.zeros(10, dtype=int)
            # action[0] = 0 # No move

            # 3. 环境步进
            obs, reward, terminated, truncated, info = env.step(action)

            # 内部变量获取 (用于调试显示)
            p1_blood = env.rblood if env.assign_flag == 'right' else env.bblood  # 近似反推
            p2_blood = env.bblood if env.assign_flag == 'right' else env.rblood

            # 从 env._get_obs 的缓存中获取 detected_classes 等信息比较困难
            # 我们直接利用 step 返回的 obs 进行反向解析或依赖 env 的内部状态
            detected_classes = env._latest_boxes.keys() if hasattr(env, '_latest_boxes') else []

            # 尝试获取 CNN raw_ids (env 并没有直接存这个，除了在 _get_obs 的返回值里)
            # 为了调试，我们只能假设 env.step 内部计算正确。
            # 这里我们主要依赖 print 输出。

            total_reward += reward

            if reward != 0:
                print(f"✅ [Reward Triggered] {reward:.4f} | Total: {total_reward:.4f}")

            # 4. 绘制可视化
            debug_vis = draw_debug_overlay(
                display_img,
                obs,
                reward,
                detected_classes,
                env.bblood,  # 直接显示环境读取到的原始血量
                env.rblood,
                []  # 外部拿不到 raw_ids，除非修改 Env 代码，暂时置空
            )

            # 缩放以便显示
            scale = 0.6
            h, w = debug_vis.shape[:2]
            debug_vis = cv2.resize(debug_vis, (int(w * scale), int(h * scale)))

            cv2.imshow("Env Debugger", debug_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if terminated:
                print("🔄 Episode Done, Resetting...")
                env.reset()

            # 保持帧率
            # time.sleep(max(0, 0.05 - (time.time() - start_time)))

    except KeyboardInterrupt:
        print("停止调试")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()