#!/usr/bin/env -S python3 -i

import argparse
import csv
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


def quat_to_euler_deg(quat_wxyz):
    """Genesisのクォータニオン [w, x, y, z] をオイラー角 [roll, pitch, yaw] (deg) に変換"""
    w, x, y, z = quat_wxyz

    # Roll (X軸回り)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y軸回り：正が前傾)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (Z軸回り)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])


def save_episode_timeseries(history, episode_num, save_dir):
    """横軸を秒数[s]に揃えたエピソード全体の同期グラフ(.png)と全計測データ(.csv)を保存する"""
    if len(history) == 0:
        return

    # 1. CSVファイルの書き出し（後から1周期のストライドやケイデンスを精密解析する用）
    csv_path = os.path.join(save_dir, f"eval_ep{episode_num}_timeseries.csv")
    fieldnames = list(history[0].keys())
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"エピソード {episode_num} の全時系列CSVを保存しました: {csv_path}")

    # 2. 同期グラフの描画（横軸：秒数 [s]）
    t = np.array([row["time_s"] for row in history])

    left_fz = np.array([row["left_fz"] for row in history])
    right_fz = np.array([row["right_fz"] for row in history])

    left_fx = np.array([row["left_fx"] for row in history])
    right_fx = np.array([row["right_fx"] for row in history])

    com_vx = np.array([row["com_vx"] for row in history])
    com_vz = np.array([row["com_vz"] for row in history])
    cmd_vx = np.array([row["cmd_vx"] for row in history])

    pitch_deg = np.array([row["pitch_deg"] for row in history])

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    # [1段目] 垂直地面反力 Fz（接地・離地・両脚支持期の確認）
    axes[0].plot(t, left_fz, color="tab:blue", linewidth=1.4, label="Left Foot Fz")
    axes[0].plot(t, right_fz, color="tab:red", linewidth=1.4, label="Right Foot Fz")
    axes[0].set_ylabel("Vertical GRF\nFz [N]")
    axes[0].set_title(f"Episode {episode_num} Whole-Body Locomotion Analysis (Time-Synced)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right")

    # [2段目] 前後地面反力 Fx（着地ブレーキ < 0 と蹴り出し推進 > 0 の確認）
    axes[1].axhline(0.0, color="gray", linewidth=0.9, linestyle=":")
    axes[1].plot(t, left_fx, color="tab:blue", linewidth=1.4, label="Left Foot Fx")
    axes[1].plot(t, right_fx, color="tab:red", linewidth=1.4, label="Right Foot Fx")
    axes[1].set_ylabel("Fore-Aft GRF\nFx [N]")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    # [3段目] 重心速度 vx, vz（加速期 vs 定常歩行期の判別）
    axes[2].plot(t, com_vx, color="tab:green", linewidth=1.6, label="CoM vx (Forward)")
    axes[2].plot(t, cmd_vx, color="black", linewidth=1.2, linestyle="--", label="Target cmd_vx")
    axes[2].plot(t, com_vz, color="tab:cyan", linewidth=1.2, alpha=0.85, label="CoM vz (Vertical)")
    axes[2].set_ylabel("CoM Velocity\n[m/s]")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right")

    # [4段目] 胴体前傾角度 Pitch（加速時の前傾と定常時の姿勢安定性）
    axes[3].axhline(0.0, color="gray", linewidth=0.9, linestyle=":")
    axes[3].plot(t, pitch_deg, color="tab:purple", linewidth=1.5, label="Trunk Pitch (+: Forward Lean)")
    axes[3].set_ylabel("Trunk Pitch\n[deg]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"eval_ep{episode_num}_overview_sec.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"エピソード {episode_num} の全体同期グラフを保存しました: {fig_path}")


def save_torque_plot(torque_history, time_history, episode_num, joint_name, save_dir):
    """1エピソード分の右股関節ピッチのトルクグラフ（横軸：秒数）を保存する関数"""
    if len(torque_history) == 0:
        return

    plt.figure(figsize=(10, 4.5))
    plt.plot(time_history, torque_history, color="tab:red", linewidth=1.5, label=joint_name)

    plt.xlabel("Time [s]")
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

def draw_grf_x(scene, pos, grf_3d, xy_scale, radius=0.014):
    """地面反力の X軸方向（前後力 Fx：赤）のみを描画する関数"""
    fx = grf_3d[0]
    if abs(fx) > 0.5:
        scene.draw_debug_arrow(
            pos=pos,
            vec=np.array([fx * xy_scale, 0.0, 0.0]),
            radius=radius,
            color=(1.0, 0.2, 0.2, 0.95),
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

    # 録画用カメラの取得と姿勢追従の切り離し
    cam = getattr(env, "cam", None)
    if cam is not None:
        try:
            cam.detach()
        except Exception:
            pass
        print("録画用カメラを検出しました（地面水平・真横アングルで等速・スロー動画を保存します）。")
    else:
        print("【注意】env.cam が見つかりません。動画を保存するには t_sbr1_env.py の scene.build() 直前に self.cam を追加してください。")

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
        target_idx = 0
        joint_label = "Joint 0 (Default)"
    else:
        joint_label = all_dof_names[target_idx]

    current_ep_torque = []
    current_ep_times = []
    current_ep_history = []
    current_ep_frames = []

    episode_count = 1
    ep_step = 0
    total_step = 0

    obs, _ = env.reset()

    print(f"評価を開始します (最大 {args.max_episodes} エピソード)。Ctrl+C で途中終了可能。")
    print(f"--- Episode {episode_count} Start ---")
    print(f"Target Command (vx, vy, wz): {env.commands[0].cpu().numpy()}")

    # ロボットの各リンク質量と全質量を計算
    link_masses = np.array([getattr(link, "mass", getattr(link, "inertial_mass", 0.0)) for link in env.robot.links])
    total_mass = float(np.sum(link_masses))
    print(f"ロボット全質量: {total_mass:.2f} kg / 制御周期 dt: {env.dt:.4f} s ({base_fps} Hz)")

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

                # 経過秒数 [s]
                time_s = ep_step * env.dt

                # 地面反力を取得
                net_forces = env.robot.get_links_net_contact_force()
                left_grf_3d = net_forces[0, left_foot_idx].cpu().numpy()
                right_grf_3d = net_forces[0, right_foot_idx].cpu().numpy()

                left_fx, left_fy, left_fz = left_grf_3d
                right_fx, right_fy, right_fz = right_grf_3d

                # 左右の地面反力を合算 [Fx, Fy, Fz]
                total_grf = left_grf_3d + right_grf_3d

                # GRFから3軸重心加速度 [ax, ay, az] (m/s^2) を計算
                com_ax = total_grf[0] / total_mass
                com_ay = total_grf[1] / total_mass
                com_az = (total_grf[2] / total_mass) - 9.81
                com_acc_3d = np.array([com_ax, com_ay, com_az])

                # 全リンク位置と重心（CoM）座標の算出
                links_pos = env.robot.get_links_pos()[0].cpu().numpy()
                left_foot_pos = links_pos[left_foot_idx]
                right_foot_pos = links_pos[right_foot_idx]

                if total_mass > 0:
                    com_pos = np.sum(links_pos * link_masses[:, None], axis=0) / total_mass
                else:
                    com_pos = env.base_pos[0].cpu().numpy()

                # ワールド座標系での重心速度 [vx, vy, vz] の算出
                if hasattr(env.robot, "get_links_vel") and total_mass > 0:
                    links_vel = env.robot.get_links_vel()[0].cpu().numpy()
                    com_vel = np.sum(links_vel * link_masses[:, None], axis=0) / total_mass
                elif hasattr(env.robot, "get_vel"):
                    com_vel = env.robot.get_vel()[0].cpu().numpy()
                else:
                    com_vel = env.base_lin_vel[0].cpu().numpy()

                # 胴体姿勢角（Roll, Pitch, Yaw [deg]）の算出
                base_quat = env.base_quat[0].cpu().numpy()
                roll_deg, pitch_deg, yaw_deg = quat_to_euler_deg(base_quat)

                # 目標コマンド [vx, vy, wz]
                cmd_vec = env.commands[0].cpu().numpy()

                # 3D空間上への矢印描画
                if hasattr(env, "scene"):
                    env.scene.clear_debug_objects()
                    # draw_grf_3axis(env.scene, left_foot_pos, left_grf_3d, GRF_XY_SCALE, GRF_Z_SCALE)
                    # draw_grf_3axis(env.scene, right_foot_pos, right_grf_3d, GRF_XY_SCALE, GRF_Z_SCALE)

                    draw_grf_x(env.scene, left_foot_pos, left_grf_3d, GRF_XY_SCALE) 
                    draw_grf_x(env.scene, right_foot_pos, right_grf_3d, GRF_XY_SCALE) 

                    # if np.linalg.norm(com_acc_3d) > 0.1:
                    #     env.scene.draw_debug_arrow(
                    #         pos=com_pos,
                    #         vec=com_acc_3d * ACC_ARROW_SCALE,
                    #         radius=0.018,
                    #         color=(1.0, 0.9, 0.1, 0.9),
                    #     )

                    # 3. 重心の前後加速度 ax のみ（黄色矢印）
                    # スケールを GRF_XY_SCALE * total_mass にすると、赤い矢印の合計長さと一致する
                    ax_scale = GRF_XY_SCALE * total_mass
                    if abs(com_ax) > 0.1:
                        env.scene.draw_debug_arrow(
                            pos=com_pos,
                            vec=np.array([com_ax * ax_scale, 0.0, 0.0]),
                            radius=0.018,
                            color=(1.0, 0.9, 0.1, 0.95),
                     )

                # 地面と平行な真横目線カメラの更新とキャプチャ
                if cam is not None:
                    robot_pos = env.base_pos[0].cpu().numpy()
                    eye_height = 0.75
                    side_distance = -2.5

                    cam_pos = np.array([robot_pos[0], robot_pos[1] + side_distance, eye_height])
                    cam_lookat = np.array([robot_pos[0], robot_pos[1], eye_height])

                    cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=(0.0, 0.0, 1.0))

                    render_out = cam.render()
                    rgb = render_out[0] if isinstance(render_out, tuple) else render_out
                    if rgb is not None:
                        current_ep_frames.append(np.asarray(rgb, dtype=np.uint8))

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
                current_ep_times.append(time_s)

                # 時系列レコードの蓄積
                current_ep_history.append({
                    "time_s": round(time_s, 4),
                    "step": ep_step,
                    "com_x": float(com_pos[0]),
                    "com_y": float(com_pos[1]),
                    "com_z": float(com_pos[2]),
                    "com_vx": float(com_vel[0]),
                    "com_vy": float(com_vel[1]),
                    "com_vz": float(com_vel[2]),
                    "cmd_vx": float(cmd_vec[0]),
                    "cmd_vy": float(cmd_vec[1]),
                    "cmd_wz": float(cmd_vec[2]),
                    "roll_deg": float(roll_deg),
                    "pitch_deg": float(pitch_deg),
                    "yaw_deg": float(yaw_deg),
                    "left_foot_x": float(left_foot_pos[0]),
                    "left_foot_z": float(left_foot_pos[2]),
                    "right_foot_x": float(right_foot_pos[0]),
                    "right_foot_z": float(right_foot_pos[2]),
                    "left_fx": float(left_fx),
                    "left_fy": float(left_fy),
                    "left_fz": float(left_fz),
                    "right_fx": float(right_fx),
                    "right_fy": float(right_fy),
                    "right_fz": float(right_fz),
                    "com_ax": float(com_ax),
                    "com_ay": float(com_ay),
                    "com_az": float(com_az),
                })

                print(
                    f"t={time_s:5.2f}s (Step {ep_step:03d}) | "
                    f"vx={com_vel[0]:5.2f} m/s | Pitch={pitch_deg:5.1f} deg | "
                    f"L_Fz={left_fz:6.1f} N, R_Fz={right_fz:6.1f} N"
                )

                ep_step += 1
                total_step += 1

                # 転倒・リセット判定
                if dones[0]:
                    print(f"\n[リセット検知] エピソード {episode_count} 終了 ({time_s:.2f} 秒 / {ep_step} ステップ)")
                    save_episode_timeseries(current_ep_history, episode_count, save_dir)
                    save_torque_plot(current_ep_torque, current_ep_times, episode_count, joint_label, save_dir)
                    save_videos(current_ep_frames, episode_count, save_dir, base_fps=base_fps, slow_ratio=args.slow_ratio)

                    episode_count += 1
                    ep_step = 0
                    current_ep_torque = []
                    current_ep_times = []
                    current_ep_history = []
                    current_ep_frames = []

                    if episode_count <= args.max_episodes:
                        print(f"\n--- Episode {episode_count} Start ---")
                        print(f"Target Command (vx, vy, wz): {env.commands[0].cpu().numpy()}")

                t_elapsed = time.perf_counter() - t_start
                sleep_time = max(0.0, env.dt - t_elapsed)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n手動で停止しました。")

    if len(current_ep_history) > 0 and episode_count <= args.max_episodes:
        print(f"計測中のエピソード {episode_count} のデータを保存します...")
        save_episode_timeseries(current_ep_history, episode_count, save_dir)
        save_torque_plot(current_ep_torque, current_ep_times, episode_count, joint_label, save_dir)
        save_videos(current_ep_frames, episode_count, save_dir, base_fps=base_fps, slow_ratio=args.slow_ratio)

    print("すべての処理が完了しました。")


if __name__ == "__main__":
    main()