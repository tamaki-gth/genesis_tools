#!/usr/bin/env -S python3 -i

import argparse
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
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
    plt.close()
    print(f"エピソード {episode_num} のトルクグラフを保存しました: {save_path}")


def save_videos(frames, episode_num, save_dir, base_fps=50, slow_ratio=0.2):
    """1エピソード分のフレームから「等速版」と「スロー版」のmp4を同時保存する関数"""
    if len(frames) == 0:
        return

    normal_path = os.path.join(save_dir, f"eval_ep{episode_num}_normal.mp4")
    slow_path = os.path.join(save_dir, f"eval_ep{episode_num}_slow_{slow_ratio}x.mp4")
    slow_fps = max(1, int(round(base_fps * slow_ratio)))

    imageio.mimsave(normal_path, frames, fps=base_fps)
    imageio.mimsave(slow_path, frames, fps=slow_fps)

    print(f"動画を保存しました（等速 {base_fps}fps）  : {normal_path}")
    print(f"動画を保存しました（スロー {slow_fps}fps）: {slow_path}")


def draw_grf_3axis(scene, pos, grf_3d, xy_scale, z_scale, radius=0.012):
    """地面反力を X(赤)・Y(緑)・Z(青) の3方向の矢印に分解して描画する関数"""
    fx, fy, fz = grf_3d

    # X軸方向（前後力 Fx）：赤
    if abs(fx) > 0.5:
        scene.draw_debug_arrow(
            pos=pos,
            vec=np.array([fx * xy_scale, 0.0, 0.0]),
            radius=radius,
            color=(1.0, 0.2, 0.2, 0.9),
        )

    # Y軸方向（左右力 Fy）：緑
    if abs(fy) > 0.5:
        scene.draw_debug_arrow(
            pos=pos,
            vec=np.array([0.0, fy * xy_scale, 0.0]),
            radius=radius,
            color=(0.2, 1.0, 0.2, 0.9),
        )

    # Z軸方向（垂直抗力 Fz）：青
    if abs(fz) > 1.0:
        scene.draw_debug_arrow(
            pos=pos,
            vec=np.array([0.0, 0.0, fz * z_scale]),
            radius=radius,
            color=(0.2, 0.5, 1.0, 0.9),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to model_XXXX.pt")
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=2000, help="Total steps to evaluate")
    parser.add_argument("--max_episodes", type=int, default=1, help="Max episodes to record")
    parser.add_argument("--slow_ratio", type=float, default=0.2, help="Slow motion speed ratio (e.g. 0.2 = 1/5 speed)")
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

    # 録画用カメラの取得（t_sbr1_env.py 側で scene.build() 前に追加されているカメラ）
    cam = getattr(env, "cam", None)
    if cam is not None:
        # ロボットの傾き（姿勢）にカメラが連動しないよう切り離す
        try:
            cam.detach()
        except Exception:
            pass
    if cam is None:
        print("【注意】env.cam が見つかりません。動画を保存するには t_sbr1_env.py の scene.build() 直前に self.cam を追加してください。")
    else:
        print("録画用カメラを検出しました（ビューアと同じ視点で等速・スロー動画を自動保存します）。")

    base_fps = max(1, int(round(1.0 / env.dt)))

    # 足部リンクのインデックスをセット
    left_foot_idx = 11   # ankle_roll_left_link
    right_foot_idx = 12  # ankle_roll_right_link
    print(f"足リンク番号設定 -> 左足: {left_foot_idx}, 右足: {right_foot_idx}")

    # hip_pitch_right_joint のインデックス特定
    target_name = "hip_pitch_right_joint"
    target_idx = None

    all_dof_names = getattr(env, "dof_names", None)
    if all_dof_names is not None:
        for idx, name in enumerate(all_dof_names):
            if name == target_name:
                target_idx = idx
                break
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
    current_ep_frames = []
    episode_count = 1
    total_step = 0

    obs, _ = env.reset()

    print(f"評価を開始します (最大 {args.max_episodes} エピソード)。Ctrl+C で途中終了可能。")
    print(f"--- Episode {episode_count} Start ---")
    print(f"Target Command (vx, vy, wz): {env.commands[0].cpu().numpy()}")

    # ロボットの各リンク質量と全質量を計算
    link_masses = np.array([getattr(link, 'mass', getattr(link, 'inertial_mass', 0.0)) for link in env.robot.links])
    total_mass = float(np.sum(link_masses))
    print(f"ロボット全質量: {total_mass:.2f} kg")

    # 矢印描画用のスケール設定
    GRF_XY_SCALE = 0.015    # 水平力(Fx, Fy)用: 20 N -> 長さ 0.30 m
    GRF_Z_SCALE  = 0.003    # 垂直抗力(Fz)用: 200 N -> 長さ 0.60 m
    ACC_ARROW_SCALE = 0.05  # 重心加速度用: 10 m/s^2 -> 長さ 0.50 m

    try:
        with torch.no_grad():
            while total_step < args.max_steps and episode_count <= args.max_episodes:
                t_start = time.perf_counter()

                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                # 地面反力を取得
                net_forces = env.robot.get_links_net_contact_force()
                left_grf_3d = net_forces[0, left_foot_idx].cpu().numpy()   # 左足 [Fx, Fy, Fz]
                right_grf_3d = net_forces[0, right_foot_idx].cpu().numpy() # 右足 [Fx, Fy, Fz]

                left_fx, left_fy, left_fz = left_grf_3d
                right_fx, right_fy, right_fz = right_grf_3d

                # 左右の地面反力を合算 [Fx, Fy, Fz]
                total_grf = left_grf_3d + right_grf_3d

                # GRFから3軸重心加速度 [ax, ay, az] (m/s^2) を計算
                com_ax = total_grf[0] / total_mass
                com_ay = total_grf[1] / total_mass
                com_az = (total_grf[2] / total_mass) - 9.81
                com_acc_3d = np.array([com_ax, com_ay, com_az])

                # =========================================================
                # ★ 3D空間上への矢印描画（録画より先に実行する）
                # =========================================================
                if hasattr(env, "scene"):
                    env.scene.clear_debug_objects()

                    links_pos = env.robot.get_links_pos()[0].cpu().numpy()
                    left_foot_pos = links_pos[left_foot_idx]
                    right_foot_pos = links_pos[right_foot_idx]

                    if total_mass > 0:
                        com_pos = np.sum(links_pos * link_masses[:, None], axis=0) / total_mass
                    else:
                        com_pos = env.base_pos[0].cpu().numpy()

                    # 1. 左足の地面反力（X:赤, Y:緑, Z:青）
                    draw_grf_3axis(env.scene, left_foot_pos, left_grf_3d, GRF_XY_SCALE, GRF_Z_SCALE)

                    # 2. 右足の地面反力（X:赤, Y:緑, Z:青）
                    draw_grf_3axis(env.scene, right_foot_pos, right_grf_3d, GRF_XY_SCALE, GRF_Z_SCALE)

                    # 3. 重心加速度（黄色矢印）
                    if np.linalg.norm(com_acc_3d) > 0.1:
                        env.scene.draw_debug_arrow(
                            pos=com_pos,
                            vec=com_acc_3d * ACC_ARROW_SCALE,
                            radius=0.018,
                            color=(1.0, 0.9, 0.1, 0.9),
                        )

                # =========================================================
                # ★ 地面と平行な目線で真横から追従するカメラ設定
                # =========================================================
                if cam is not None:
                    robot_pos = env.base_pos[0].cpu().numpy()

                    # カメラの高さ（地面からの目線の高さ [m]）
                    eye_height = 0.75
                    # ロボットの真横（Y方向）に離れる距離 [m]（-2.5 で左→右へ前進、+2.5 で右→左へ前進）
                    side_distance = -2.5

                    # カメラ位置と注視点の高さ(Z)を完全に一致させて地面と平行にする
                    cam_pos = np.array([
                        robot_pos[0],
                        robot_pos[1] + side_distance,
                        eye_height,
                    ])
                    cam_lookat = np.array([
                        robot_pos[0],
                        robot_pos[1],
                        eye_height,
                    ])

                    # up=(0, 0, 1) を指定し、カメラの水平（地面の水平）を絶対に傾けない
                    cam.set_pose(
                        pos=cam_pos,
                        lookat=cam_lookat,
                        up=(0.0, 0.0, 1.0),
                    )

                    render_out = cam.render()
                    rgb = render_out[0] if isinstance(render_out, tuple) else render_out
                    if rgb is not None:
                        current_ep_frames.append(np.asarray(rgb, dtype=np.uint8))
                print(f"Step {total_step:03d}")
                print(f"  左足 GRF [Fx, Fy, Fz]: {left_fx:6.2f}, {left_fy:6.2f}, {left_fz:6.2f} N")
                print(f"  右足 GRF [Fx, Fy, Fz]: {right_fx:6.2f}, {right_fy:6.2f}, {right_fz:6.2f} N")
                print(f" 重心加速度 [ax, ay, az]: [{com_ax:6.2f}, {com_ay:6.2f}, {com_az:6.2f}] m/s^2")

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

                current_ep_torque.append(all_torques[target_idx])

                total_step += 1

                # 転倒・リセット判定
                if dones[0]:
                    print(f"\n[リセット検知] エピソード {episode_count} 終了 ({len(current_ep_torque)} ステップ)")
                    save_torque_plot(current_ep_torque, episode_count, joint_label, save_dir)
                    save_videos(current_ep_frames, episode_count, save_dir, base_fps=base_fps, slow_ratio=args.slow_ratio)

                    episode_count += 1
                    current_ep_torque = []
                    current_ep_frames = []

                    if episode_count <= args.max_episodes:
                        print(f"\n--- Episode {episode_count} Start ---")
                        print(f"Target Command (vx, vy, wz): {env.commands[0].cpu().numpy()}")

                t_elapsed = time.perf_counter() - t_start
                sleep_time = max(0.0, env.dt - t_elapsed)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n手動で停止しました。")

    if len(current_ep_torque) > 0 and episode_count <= args.max_episodes:
        print(f"計測中のエピソード {episode_count} のデータを保存します...")
        save_torque_plot(current_ep_torque, episode_count, joint_label, save_dir)
        save_videos(current_ep_frames, episode_count, save_dir, base_fps=base_fps, slow_ratio=args.slow_ratio)

    print("すべての処理が完了しました。")


if __name__ == "__main__":
    main()