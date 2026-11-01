#!/usr/bin/env -S python3 -i

import argparse
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import genesis as gs

from t_sbr1_env import Sbr1Env
from sbr1_train import get_cfgs, get_train_cfg
from rsl_rl.runners import OnPolicyRunner


def save_velocity_plot(vel_history, episode_num, save_dir):
    """1エピソード分の速度グラフを作成して保存する関数"""
    if len(vel_history) == 0:
        return

    steps_range = np.arange(len(vel_history))
    plt.figure(figsize=(10, 5))
    plt.plot(
        steps_range,
        vel_history,
        color="tab:blue",
        linewidth=1.5,
        label=f"Ep {episode_num} Velocity",
    )
    plt.xlabel("Step")
    plt.ylabel("Velocity [m/s]")
    plt.title(f"Forward Velocity Over Time (Episode {episode_num})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"eval_velocity_ep{episode_num}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()  # メモリ解放のため非表示で閉じる
    print(f"エピソード {episode_num} のグラフを保存しました: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to model_XXXX.pt")
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=2000, help="Total steps to evaluate")
    parser.add_argument("--max_episodes", type=int, default=5, help="Max episodes to record")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(max_iterations=1)

    env = Sbr1Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        eval_mode=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)

    print(f"Loading model from: {args.model_path}")
    runner.load(args.model_path)
    policy = runner.get_inference_policy(device=gs.device)

    save_dir = os.path.expanduser("~/logs/sbr1_locomotion")
    os.makedirs(save_dir, exist_ok=True)

    current_ep_vel = []
    episode_count = 1
    total_step = 0

    obs, _ = env.reset()

    print(f"評価を開始します (最大 {args.max_episodes} エピソード)。Ctrl+C で途中終了可能。")

    try:
        with torch.no_grad():
            while total_step < args.max_steps and episode_count <= args.max_episodes:
                t_start = time.perf_counter()

                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                # 前進速度を記録
                current_vel = env.base_lin_vel[0, 0].item()
                current_ep_vel.append(current_vel)

                total_step += 1

                # 転倒・リセットが発生した場合
                if dones[0]:
                    print(f"\n[リセット検知] エピソード {episode_count} 終了 ({len(current_ep_vel)} ステップ)")
                    save_velocity_plot(current_ep_vel, episode_count, save_dir)

                    episode_count += 1
                    current_ep_vel = []  # 次のエピソード用にクリア

                # 制御周期同期 (sleep)
                t_elapsed = time.perf_counter() - t_start
                sleep_time = max(0.0, env.dt - t_elapsed)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n手動で停止しました。")

    # 途中終了（Ctrl+C 等）された場合でも、計測中のデータがあれば保存
    if len(current_ep_vel) > 0 and episode_count <= args.max_episodes:
        print(f"計測中のエピソード {episode_count} のデータを保存します...")
        save_velocity_plot(current_ep_vel, episode_count, save_dir)

    print("すべての評価・グラフ保存処理が完了しました。")


if __name__ == "__main__":
    main()