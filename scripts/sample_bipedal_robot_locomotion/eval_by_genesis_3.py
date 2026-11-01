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


def save_torque_plot(torque_history, episode_num, joint_name, save_dir):
    """1エピソード分の右股関節ピッチのトルクグラフを作成して保存する関数"""
    if len(torque_history) == 0:
        return

    torque_array = np.array(torque_history)  # [steps]
    steps_range = np.arange(len(torque_history))

    plt.figure(figsize=(10, 5))
    plt.plot(steps_range, torque_array, color="tab:red", linewidth=1.5, label=joint_name)

    plt.xlabel("Step")
    plt.ylabel("Torque [Nm]")
    plt.title(f"Right Hip Pitch Torque Over Time (Episode {episode_num})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"eval_right_hip_pitch_torque_ep{episode_num}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()  # メモリ解放
    print(f"エピソード {episode_num} のトルクグラフを保存しました: {save_path}")


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

    # --------------------------------------------------
    # hip_pitch_right_joint のインデックス特定
    # --------------------------------------------------
    target_name = "hip_pitch_right_joint"
    target_idx = None

    all_dof_names = getattr(env, "dof_names", None)
    if all_dof_names is not None:
        for idx, name in enumerate(all_dof_names):
            if name == target_name:
                target_idx = idx
                break
            # 完全一致しなかった場合の曖昧検索 (pitch かつ right かつ hip)
            if "hip" in name.lower() and "pitch" in name.lower() and ("right" in name.lower() or "r" in name.lower()):
                target_idx = idx

    if target_idx is None:
        print(f"警告: {target_name} が見つからなかったため、0番目の関節を使用します。")
        target_idx = 0
        joint_label = "Joint 0 (Default)"
    else:
        joint_label = all_dof_names[target_idx]
        print(f"対象関節を特定: {joint_label} (インデックス: {target_idx})")

    current_ep_torque = []
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

                # トルクの取得
                if hasattr(env, "torques"):
                    all_torques = env.torques[0].cpu().numpy()
                else:
                    target_pos = actions[0] * env.env_cfg["action_scale"] + env.default_dof_pos[0]
                    current_pos = env.dof_pos[0]
                    current_vel_dof = env.dof_vel[0]
                    kp = getattr(env, "p_gains", 100.0)
                    kd = getattr(env, "d_gains", 2.0)
                    all_torques = (kp * (target_pos - current_pos) - kd * current_vel_dof).cpu().numpy()

                # hip_pitch_right_joint の数値だけを取り出して追記
                current_ep_torque.append(all_torques[target_idx])

                total_step += 1

                # 転倒・リセット判定
                if dones[0]:
                    print(f"\n[リセット検知] エピソード {episode_count} 終了 ({len(current_ep_torque)} ステップ)")
                    save_torque_plot(current_ep_torque, episode_count, joint_label, save_dir)

                    episode_count += 1
                    current_ep_torque = []

                t_elapsed = time.perf_counter() - t_start
                sleep_time = max(0.0, env.dt - t_elapsed)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n手動で停止しました。")

    if len(current_ep_torque) > 0 and episode_count <= args.max_episodes:
        print(f"計測中のエピソード {episode_count} のデータを保存します...")
        save_torque_plot(current_ep_torque, episode_count, joint_label, save_dir)

    print("すべての処理が完了しました。")


if __name__ == "__main__":
    main()