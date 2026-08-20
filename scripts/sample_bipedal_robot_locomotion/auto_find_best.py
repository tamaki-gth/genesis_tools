#!/usr/bin/env -S python3

import argparse
from pathlib import Path
import re
import yaml
from importlib import metadata

import torch

# --- rsl-rl ライブラリのバージョンチェック ---
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from sbr1_env import Sbr1Env


def eval_single_run(policy, env, target_dist=20.0, max_time=30.0):
    """1回の20m走を実行してタイム(秒)を計測する"""
    obs, _ = env.reset()
    dt = getattr(env, "dt", 0.01)
    max_steps = int(max_time / dt)
    
    with torch.no_grad():
        for step in range(max_steps):
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            
            # ロボットのX軸位置(進行距離)を取得
            if hasattr(env, "base_pos"):
                x_pos = env.base_pos[0, 0].item()
            elif hasattr(env, "robot"):
                x_pos = env.robot.get_pos()[0, 0].item()
            else:
                x_pos = infos.get("x_pos", 0.0)

            # 20mに到達したらタイムを確定して終了
            if x_pos >= target_dist:
                return step * dt
            
            # 途中で転倒(done)した場合は失敗判定
            if dones[0].item():
                return float("inf")
                
    return float("inf")


def main():
    parser = argparse.ArgumentParser(description="ログフォルダ内の全モデルを評価し最速のckptを特定する")
    parser.add_argument("-l", "--log_dir", type=str, required=True, help="評価対象のログフォルダパス")
    parser.add_argument("--target_dist", type=float, default=20.0, help="目標距離 (m)")
    parser.add_argument("--num_trials", type=int, default=3, help="1ckptあたりの試行回数 (平均を取る)")
    args = parser.parse_args()

    gs.init()

    log_dir = Path(args.log_dir).expanduser()
    cfg_file = log_dir / "cfgs.yaml"
    
    if not cfg_file.exists():
        print(f"エラー: 設定ファイルが見つかりません: {cfg_file}")
        return

    with open(cfg_file, "r") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = yaml.safe_load(f).values()

    # --- 【適合修正】評価時の目標速度を上限値(例: 3.6 m/s)に固定 ---
    if "lin_vel_x_range" in command_cfg:
        max_vel = command_cfg["lin_vel_x_range"][1]
        command_cfg["lin_vel_x_range"] = [max_vel, max_vel]
        print(f"★ 評価時の目標速度を最大値 {max_vel} m/s に固定しました")

    # 画面非表示 (show_viewer=False) で環境を作成 (高速化)
    env = Sbr1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False
    )

    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)

    # フォルダ内の model_*.pt を数値順に取得
    def get_ckpt_num(path):
        m = re.search(r"model_(\d+)\.pt", path.name)
        return int(m.group(1)) if m else -1

    ckpt_files = sorted([p for p in log_dir.glob("model_*.pt") if get_ckpt_num(p) >= 0], key=get_ckpt_num)

    if not ckpt_files:
        print(f"エラー: {log_dir} 内に model_*.pt が見つかりません。")
        return

    best_ckpt = None
    best_avg_time = float("inf")

    print(f"\n=================== 20m走 最速判定開始 ({args.num_trials}回平均) ===================")
    for ckpt_path in ckpt_files:
        ckpt_num = get_ckpt_num(ckpt_path)

        runner.load(str(ckpt_path))
        policy = runner.get_inference_policy(device=gs.device)
        
        times = []
        is_failed = False
        for _ in range(args.num_trials):
            t = eval_single_run(policy, env, target_dist=args.target_dist)
            if t == float("inf"):
                is_failed = True
                break
            times.append(t)

        if is_failed:
            print(f"CKPT: {ckpt_num:>7} | 転倒 または 未到達 (失敗)")
        else:
            avg_time = sum(times) / len(times)
            times_str = ", ".join([f"{t:.2f}s" for t in times])
            print(f"CKPT: {ckpt_num:>7} | 平均: {avg_time:.2f}秒 (内訳: {times_str})")

            if avg_time < best_avg_time:
                best_avg_time = avg_time
                best_ckpt = ckpt_num

    print("=================================================================================")
    if best_ckpt is not None:
        print(f"\n★ 【最速エピソード決定】")
        print(f"  ・チェックポイント番号: {best_ckpt}")
        print(f"  ・対象ファイル: model_{best_ckpt}.pt")
        print(f"  ・平均タイム: {best_avg_time:.2f} 秒\n")

        result_file = log_dir / "best_ckpt.txt"
        with open(result_file, "w") as f:
            f.write(str(best_ckpt))
        print(f"※ 結果を {result_file} に保存しました。")
    else:
        print("警告: 20mを安定して完走できるチェックポイントが見つかりませんでした。")


if __name__ == "__main__":
    main()