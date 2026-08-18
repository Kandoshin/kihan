import torch
import zipfile
import tempfile
import os
import sys

MODEL_ZIP = r"G:\god\pycharm\PythonProject\test\final\ppo_realtime.zip"

def inspect_state_dict(state_dict, name_hint="policy"):
    print(f"\n========== {name_hint} 体检报告 ==========")
    total_params = 0
    nan_layers = []
    zero_layers = []
    init_like_layers = []

    for name, param in state_dict.items():
        numel = param.numel()
        total_params += numel
        mean = param.mean().item()
        std  = param.std().item()
        maxv = param.max().item()
        minv = param.min().item()

        # 异常值检查
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()

        # 是否像“未训练”：均值≈0 且 std 极小
        looks_init = abs(mean) < 1e-4 and std < 1e-3

        print(f"{name:<50}  shape={list(param.shape)}")
        print(f"{'':50}  mean={mean:+.6f}, std={std:.6f}, max={maxv:.3f}, min={minv:.3f}")

        if has_nan or has_inf:
            flag = "❌ NaN/Inf"
            nan_layers.append(name)
        elif looks_init:
            flag = "⚠️  像初始化"
            init_like_layers.append(name)
        elif maxv == 0 and minv == 0:
            flag = "⚠️  全零"
            zero_layers.append(name)
        else:
            flag = "✅ 正常"
        print(f"{'':50}  {flag}\n")

    print("---------- 小结 ----------")
    print(f"总参数量：{total_params:,}")
    if nan_layers:
        print(f"❌ 含 NaN/Inf 的层：{nan_layers}")
    if zero_layers:
        print(f"⚠️  全零层：{zero_layers}")
    if init_like_layers:
        print(f"⚠️  像未训练的层（{len(init_like_layers)} 个）")
    if not any([nan_layers, zero_layers, init_like_layers]):
        print("✅ 未发现明显异常，模型看起来已训练且未崩溃")

def main():
    if not os.path.isfile(MODEL_ZIP):
        print("❌ 模型 zip 不存在：", MODEL_ZIP)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zf:
            zf.extractall(tmpdir)

        policy_file   = os.path.join(tmpdir, "policy.pth")
        optimizer_file= os.path.join(tmpdir, "policy.optimizer.pth")

        if os.path.isfile(policy_file):
            inspect_state_dict(torch.load(policy_file, map_location="cpu"), "policy")
        else:
            print("❌ zip 内未找到 policy.pth")

        if os.path.isfile(optimizer_file):
            opt_state = torch.load(optimizer_file, map_location="cpu")
            # 只打印键，不打印具体数值（太大）
            print("\n---------- optimizer 状态键 ----------")
            for k in opt_state["state"].keys():
                print(k, "(含 momentum/grad 等)")

if __name__ == "__main__":
    main()