import os
import pandas as pd
import matplotlib.pyplot as plt

# CSVの読み込み先 兼 画像の保存先ディレクトリ (~/logs/sbr1_locomotion)
logs_dir = os.path.expanduser("~/logs/sbr1_locomotion")
os.makedirs(logs_dir, exist_ok=True)

# --------------------------------------------------
# 1. mean_reward (1〜3) を1枚のグラフにプロット
# --------------------------------------------------
plt.figure(figsize=(10, 6))

for i in range(1, 4):
    file_path = os.path.join(logs_dir, f"sbr1_locomotion_forward_{i}_mean_reward.csv")
    try:
        df = pd.read_csv(file_path)
        x_col = 'Step' if 'Step' in df.columns else df.columns[0]
        y_col = 'Value' if 'Value' in df.columns else df.columns[-1]
        
        plt.plot(df[x_col], df[y_col], label=f"forward_{i}")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {file_path}")

plt.title("Mean Reward Comparison (Forward 1-3)")
plt.xlabel("Step")
plt.ylabel("Mean Reward")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

save_path_mean = os.path.join(logs_dir, "mean_reward_comparison.png")
plt.savefig(save_path_mean, dpi=300)
print(f"保存完了: {save_path_mean}")
plt.show()

# --------------------------------------------------
# 2. rew_forward_vel (1〜3) を1枚のグラフにプロット
# --------------------------------------------------
plt.figure(figsize=(10, 6))

for i in range(1, 4):
    file_path = os.path.join(logs_dir, f"sbr1_locomotion_forward_{i}_rew_forward_vel.csv")
    try:
        df = pd.read_csv(file_path)
        x_col = 'Step' if 'Step' in df.columns else df.columns[0]
        y_col = 'Value' if 'Value' in df.columns else df.columns[-1]
        
        plt.plot(df[x_col], df[y_col], label=f"forward_{i}")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {file_path}")

plt.title("Forward Velocity Reward Comparison (Forward 1-3)")
plt.xlabel("Step")
plt.ylabel("Rew Forward Vel (m/s)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

save_path_vel = os.path.join(logs_dir, "rew_forward_vel_comparison.png")
plt.savefig(save_path_vel, dpi=300)
print(f"保存完了: {save_path_vel}")
plt.show()