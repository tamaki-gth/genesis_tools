#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from PIL import Image


def detect_one_cycle(df, search_start_s, foot="right", fz_thresh=15.0):
    """指定秒数以降で「基準足の接地 → 次の同側足の接地（2歩分）」を自動検出する"""
    fz_col = "right_fz" if foot == "right" else "left_fz"
    sub = df[df["time_s"] >= search_start_s].reset_index(drop=True)

    contact = (sub[fz_col].values > fz_thresh).astype(int)
    touchdowns = np.where(np.diff(contact) == 1)[0] + 1

    if len(touchdowns) < 2:
        raise ValueError(
            f"{search_start_s}秒以降で{foot}足の接地が2回見つかりませんでした。"
            "--start の秒数を早めるか、--auto_cycle を外して --start と --end を直接指定してください。"
        )

    idx_start = touchdowns[0]
    idx_end = touchdowns[1]
    t_start = float(sub.loc[idx_start, "time_s"])
    t_end = float(sub.loc[idx_end, "time_s"])
    return t_start, t_end


def compute_cycle_metrics(cycle_df, foot="right", fz_thresh=15.0):
    """1周期（2歩）の定量的指標（ストライド、ケイデンス、速度変化、前傾角、GRF）を算出"""
    t = cycle_df["time_s"].values
    duration = t[-1] - t[0]
    cadence = 2.0 / duration if duration > 0 else 0.0

    com_stride = float(cycle_df["com_x"].iloc[-1] - cycle_df["com_x"].iloc[0])
    foot_col = "right_foot_x" if foot == "right" else "left_foot_x"
    foot_stride = float(cycle_df[foot_col].iloc[-1] - cycle_df[foot_col].iloc[0])

    vx_start = float(cycle_df["com_vx"].iloc[0])
    vx_end = float(cycle_df["com_vx"].iloc[-1])
    delta_vx = vx_end - vx_start
    vx_mean = float(cycle_df["com_vx"].mean())

    pitch_mean = float(cycle_df["pitch_deg"].mean())
    pitch_min = float(cycle_df["pitch_deg"].min())
    pitch_max = float(cycle_df["pitch_deg"].max())

    max_left_fz = float(cycle_df["left_fz"].max())
    max_right_fz = float(cycle_df["right_fz"].max())
    max_propulsion_fx = float(np.maximum(cycle_df["left_fx"], cycle_df["right_fx"]).max())
    max_braking_fx = float(np.minimum(cycle_df["left_fx"], cycle_df["right_fx"]).min())

    left_contact = cycle_df["left_fz"].values > fz_thresh
    right_contact = cycle_df["right_fz"].values > fz_thresh
    double_support_ratio = float(np.mean(left_contact & right_contact) * 100.0)
    flight_ratio = float(np.mean((~left_contact) & (~right_contact)) * 100.0)

    return {
        "期間 [s]": f"{t[0]:.2f} -> {t[-1]:.2f}",
        "1周期時間 (2歩) [s]": round(duration, 3),
        "ケイデンス [steps/s]": round(cadence, 2),
        "ストライド長 (CoM) [m]": round(com_stride, 3),
        "ストライド長 (足部) [m]": round(foot_stride, 3),
        "開始時速度 vx [m/s]": round(vx_start, 3),
        "終了時速度 vx [m/s]": round(vx_end, 3),
        "速度変化 Δvx [m/s]": round(delta_vx, 3),
        "平均速度 vx [m/s]": round(vx_mean, 3),
        "平均前傾角 Pitch [deg]": round(pitch_mean, 2),
        "前傾角範囲 [deg]": f"{pitch_min:.1f} ~ {pitch_max:.1f}",
        "最大垂直反力 Fz (左/右) [N]": f"{max_left_fz:.1f} / {max_right_fz:.1f}",
        "最大推進力 Fx / ブレーキ Fx [N]": f"{max_propulsion_fx:.1f} / {max_braking_fx:.1f}",
        "両脚支持期割合 [%]": round(double_support_ratio, 1),
        "滞空期割合 [%]": round(flight_ratio, 1),
    }


def make_cycle_figure(cycle_df, current_time=None, fz_thresh=15.0):
    """1周期の同期グラフを生成（current_time指定時は現在時刻の縦線を描画）"""
    t = cycle_df["time_s"].values
    left_contact = cycle_df["left_fz"].values > fz_thresh
    right_contact = cycle_df["right_fz"].values > fz_thresh
    double_support = left_contact & right_contact

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 9.5), sharex=True)

    for ax in axes:
        ax.fill_between(
            t, 0, 1, where=double_support,
            color="gold", alpha=0.22, transform=ax.get_xaxis_transform(),
            label="Double Support" if ax == axes[0] else ""
        )
        if current_time is not None:
            ax.axvline(current_time, color="black", linewidth=2.0, linestyle="-")

    # 1段目: 垂直地面反力 Fz
    axes[0].plot(t, cycle_df["left_fz"], color="tab:blue", linewidth=1.8, label="Left Fz")
    axes[0].plot(t, cycle_df["right_fz"], color="tab:red", linewidth=1.8, label="Right Fz")
    axes[0].set_ylabel("Fz [N]")
    axes[0].set_title(f"Gait Cycle Detail ({t[0]:.2f}s - {t[-1]:.2f}s)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=8)

    # 2段目: 前後地面反力 Fx
    axes[1].axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    axes[1].plot(t, cycle_df["left_fx"], color="tab:blue", linewidth=1.8, label="Left Fx")
    axes[1].plot(t, cycle_df["right_fx"], color="tab:red", linewidth=1.8, label="Right Fx")
    axes[1].set_ylabel("Fx [N]")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=8)

    # 3段目: 重心速度 vx, vz
    axes[2].plot(t, cycle_df["com_vx"], color="tab:green", linewidth=2.0, label="CoM vx")
    axes[2].plot(t, cycle_df["cmd_vx"], color="black", linewidth=1.1, linestyle="--", label="Target vx")
    axes[2].plot(t, cycle_df["com_vz"], color="tab:cyan", linewidth=1.3, label="CoM vz")
    axes[2].set_ylabel("Vel [m/s]")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=8)

    # 4段目: 前傾角 Pitch
    axes[3].axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    axes[3].plot(t, cycle_df["pitch_deg"], color="tab:purple", linewidth=1.8, label="Pitch [deg]")
    axes[3].set_ylabel("Pitch [deg]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, linestyle="--", alpha=0.5)
    axes[3].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def export_cycle_videos(cycle_df, video_path, out_prefix, base_fps=50.0, slow_ratio=0.05):
    """該当区間の動画切り出し（スロー版）と、グラフ連動の同期解析動画をどちらも指定倍率のスローで生成"""
    if not os.path.exists(video_path):
        print(f"【注意】動画ファイルが見つからないため動画出力をスキップします: {video_path}")
        return

    reader = imageio.get_reader(video_path)
    total_video_frames = reader.count_frames()

    steps = cycle_df["step"].astype(int).values
    times = cycle_df["time_s"].values

    # 超スロー（例: 0.05倍速 -> 実質2.5fps）でもプレイヤーで滑らかにシーク・再生できるよう、
    # 同一フレームを複数回複製して 20fps 以上の標準フレームレートで書き出す
    effective_fps = base_fps * slow_ratio
    repeat_count = max(1, int(round(20.0 / effective_fps)))
    export_fps = effective_fps * repeat_count

    clipped_frames = []
    synced_frames = []

    print(
        f"動画切り出し＆グラフ同期動画を {slow_ratio}倍速 "
        f"(実質 {effective_fps:.2f} fps / 出力 {export_fps:.1f} fps) で生成中（全 {len(steps)} ステップ）..."
    )

    for step_idx, t_cur in zip(steps, times):
        if step_idx >= total_video_frames:
            break
        frame = reader.get_data(int(step_idx))

        # 現在時刻の縦線入りグラフを画像配列化
        fig = make_cycle_figure(cycle_df, current_time=t_cur)
        fig.canvas.draw()
        graph_rgba = np.asarray(fig.canvas.buffer_rgba())
        graph_rgb = graph_rgba[:, :, :3]
        plt.close(fig)

        # 動画フレームの高さに合わせてグラフ画像をリサイズし、横に連結
        vid_h, vid_w, _ = frame.shape
        g_h, g_w, _ = graph_rgb.shape
        new_g_w = int(round(g_w * (vid_h / g_h)))
        graph_resized = np.array(
            Image.fromarray(graph_rgb).resize((new_g_w, vid_h), Image.Resampling.BILINEAR)
        )

        combined = np.concatenate([frame, graph_resized], axis=1)

        # 指定スロー倍率になるようフレームを複製して追加
        for _ in range(repeat_count):
            clipped_frames.append(frame)
            synced_frames.append(combined)

    reader.close()

    if len(clipped_frames) > 0:
        clip_path = f"{out_prefix}_clip_slow_{slow_ratio}x.mp4"
        sync_path = f"{out_prefix}_sync_graph_{slow_ratio}x.mp4"

        imageio.mimsave(clip_path, clipped_frames, fps=export_fps)
        imageio.mimsave(sync_path, synced_frames, fps=export_fps)

        print(f"切り出しスロー動画を保存しました : {clip_path}")
        print(f"グラフ連動解析動画を保存しました : {sync_path}")


def main():
    parser = argparse.ArgumentParser(description="1周期歩行データの切り出し・詳細解析・動画同期ツール")
    parser.add_argument("--ep", type=int, default=1, help="エピソード番号（デフォルト: 1）")
    parser.add_argument("--start", type=float, required=True, help="開始時刻 [s]（例: 3.0）")
    parser.add_argument("--end", type=float, default=None, help="終了時刻 [s]（--auto_cycle未指定時に必須）")
    parser.add_argument("--auto_cycle", action="store_true", help="--start以降から基準足の1周期(2歩)を自動検出する")
    parser.add_argument("--foot", type=str, default="right", choices=["right", "left"], help="自動検出の基準足")
    parser.add_argument("--slow_ratio", type=float, default=0.05, help="スロー再生倍率（デフォルト: 0.05 = 1/20倍速）")
    parser.add_argument("--data_dir", type=str, default="~/logs/sbr1_locomotion", help="CSVと動画の保存フォルダ")
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    csv_path = os.path.join(data_dir, f"eval_ep{args.ep}_timeseries.csv")
    video_path = os.path.join(data_dir, f"eval_ep{args.ep}_normal.mp4")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)

    # CSVのtime_sから元のサンプリングレート（例: 50Hz）を自動計算
    dt_est = float(np.median(np.diff(df["time_s"].values)))
    base_fps = round(1.0 / dt_est) if dt_est > 0 else 50.0

    if args.auto_cycle:
        t_start, t_end = detect_one_cycle(df, args.start, foot=args.foot)
        print(f"[自動検出] {args.foot}足接地基準の1周期: {t_start:.3f}s -> {t_end:.3f}s")
    else:
        if args.end is None:
            raise ValueError("--auto_cycle を使わない場合は --end を指定してください。")
        t_start, t_end = args.start, args.end

    cycle_df = df[(df["time_s"] >= t_start) & (df["time_s"] <= t_end)].reset_index(drop=True)
    if len(cycle_df) < 2:
        raise ValueError("指定された区間のデータが少なすぎます。秒数を確認してください。")

    # 1. 定量指標の表示と保存
    metrics = compute_cycle_metrics(cycle_df, foot=args.foot)
    print("\n=== 1周期（2歩）解析サマリー ===")
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")
    print("================================\n")

    out_prefix = os.path.join(data_dir, f"cycle_ep{args.ep}_{t_start:.2f}s_to_{t_end:.2f}s")

    with open(f"{out_prefix}_metrics.txt", "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # 2. 静止画グラフ保存
    fig = make_cycle_figure(cycle_df)
    fig_path = f"{out_prefix}_graph.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"1周期詳細グラフを保存しました   : {fig_path}")

    # 3. 動画切り出し＆グラフ連動動画保存（どちらも指定スロー倍率）
    export_cycle_videos(cycle_df, video_path, out_prefix, base_fps=base_fps, slow_ratio=args.slow_ratio)


if __name__ == "__main__":
    main()