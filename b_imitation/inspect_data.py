import numpy as np
import os

# ============= 配置 =============
# 修改为数据目录
DATA_DIR = "recorded_data"

# 动作映射说明 (根据 game_config.py)
MOVE_MAP_DESC = {
    0: "无移动", 1: "W (上)", 2: "S (下)", 3: "A (左)", 4: "D (右)",
    5: "WA (左上)", 6: "WD (右上)", 7: "AS (左下)", 8: "SD (右下)"
}
SKILL_MAP_DESC = [
    "J (普攻)", "K (跳跃)", "L (冲刺)", "I (技能1)",
    "U (技能2)", "O (奥义)", "Q (通灵)", "E (秘卷)", "Space (替身)"
]


def load_all_data(data_dir):
    """加载并合并目录下所有 expert_data*.npz 文件"""
    all_actions = []

    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return None

    # 筛选文件
    files = [f for f in os.listdir(data_dir) if f.startswith('expert_data') and f.endswith('.npz')]

    if not files:
        print(f"❌ 在 {data_dir} 中未找到数据文件")
        return None

    print(f"📂 正在扫描 {data_dir} ...")
    total_files = 0

    for f in files:
        path = os.path.join(data_dir, f)
        try:
            data = np.load(path)
            # 我们只需要 actions 来做统计，obs 太大就不全部加载到内存打印了
            if 'actions' in data and len(data['actions']) > 0:
                all_actions.append(data['actions'])
                total_files += 1
                print(f"  - ✅ {f}: {len(data['actions'])} 帧")
            else:
                print(f"  - ⚠️ {f}: 空文件或格式错误")
        except Exception as e:
            print(f"  - ❌ 加载失败 {f}: {e}")

    if not all_actions:
        return None

    # 合并所有动作数据
    combined_actions = np.concatenate(all_actions, axis=0)
    print(f"\n✅ 成功合并 {total_files} 个文件")
    return combined_actions


def inspect():
    actions = load_all_data(DATA_DIR)

    if actions is None:
        return

    total_frames = len(actions)
    print(f"\n📊 === 全局数据体检报告 ===")
    print(f"总样本量: {total_frames} 帧")
    # 按照 15 FPS 估算时长
    duration = total_frames / 15
    print(f"估算时长: {duration / 60:.1f} 分钟 ({duration:.1f} 秒)")

    # 1. 检查移动分布 (Action[0])
    print(f"\n🏃 === 移动操作分布 (仅看左手) ===")
    move_column = actions[:, 0]
    move_counts = {}

    # 统计每种移动指令
    unique, counts = np.unique(move_column, return_counts=True)
    move_stats = dict(zip(unique, counts))

    for val in range(9):  # 0-8
        count = move_stats.get(val, 0)
        percent = (count / total_frames) * 100
        desc = MOVE_MAP_DESC.get(val, f"未知({val})")

        # 简单的进度条显示
        bar = "█" * int(percent / 5)
        print(f"  - {desc:<8}: {count:>5} 帧 ({percent:>5.1f}%) {bar}")

    # 2. 检查技能按键 (Action[1] ~ Action[9])
    print(f"\n⚔️ === 技能按键统计 (仅看右手) ===")
    print(f"{'按键':<10} {'触发次数':<10} {'覆盖率':<10}")
    print("-" * 35)

    has_any_skill = False
    for i, skill_name in enumerate(SKILL_MAP_DESC):
        # 技能在 actions 里的索引是 i+1
        col_idx = i + 1
        press_count = np.sum(actions[:, col_idx])
        if press_count > 0:
            has_any_skill = True
            percent = (press_count / total_frames) * 100
            print(f"{skill_name:<10} {press_count:<10} {percent:>5.1f}%")

    if not has_any_skill:
        print("  ⚠️ 警告: 全程没有检测到任何技能按键！")

    # 3. [新增] 综合静止统计 (对齐 bc_train.py 的逻辑)
    print(f"\n💤 === 综合静止统计 (与训练清洗逻辑一致) ===")
    # 逻辑: 移动为0 且 所有技能均为0
    # actions shape: [N, 10] -> [move, s1...s9]
    move_part = actions[:, 0]
    skill_part = actions[:, 1:]

    # 计算每一帧按下的技能总数
    skill_sum = np.sum(skill_part, axis=1)

    # 真正的 No-Op: 没移动 且 没按技能
    true_no_op_mask = (move_part == 0) & (skill_sum == 0)
    true_no_op_count = np.sum(true_no_op_mask)
    active_count = total_frames - true_no_op_count

    print(f"  - 完全静止 (No-Op): {true_no_op_count} 帧 ({true_no_op_count / total_frames * 100:.1f}%)")
    print(f"    (说明: 既没有移动，也没有按任何技能键)")
    print(f"  - 有效操作 (Active): {active_count} 帧 ({active_count / total_frames * 100:.1f}%)")

    # 4. 总体诊断
    print(f"\n🩺 === 诊断结论 ===")

    # 这里使用真正的静止率来判断
    stop_ratio = true_no_op_count / total_frames

    if stop_ratio > 0.8:
        print(f"⚠️ 【注意】: 数据中 '完全静止' 的比例高达 {stop_ratio * 100:.1f}%。")
        print("   -> 这可能导致 AI 倾向于挂机。建议多录制一些激烈的战斗片段。")
    elif stop_ratio < 0.2:
        print(f"⚡ 【优秀】: 数据非常活跃！'完全静止' 比例仅 {stop_ratio * 100:.1f}%。")
    else:
        print(f"✅ 【正常】: 动静结合，数据分布比较合理。")

    if not has_any_skill:
        print("❌ 【严重】: 缺少技能数据，可能是录制时权限不足，请以管理员身份运行！")


if __name__ == "__main__":
    inspect()