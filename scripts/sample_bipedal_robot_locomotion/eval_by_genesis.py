#!/usr/bin/env -S python3 -i

import argparse
import os
import torch
import genesis as gs
import time

from t_sbr1_env import Sbr1Env
from sbr1_train import get_cfgs, get_train_cfg
from rsl_rl.runners import OnPolicyRunner

def main():
    parser = argparse.ArgumentParser()
    # 評価したいチェックポイント（モデル）のパスを指定
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to model_XXXX.pt")
    # 可視化のときは画面が見やすいよう台数を少なくする（例: 1台〜4台）
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    args = parser.parse_args()

    # Genesisを可視化（Viewer）有効で初期化
    gs.init(logging_level="warning")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(max_iterations=1)

    # 評価用環境の作成（show_viewer=True など環境クラスの引数があれば設定）
    env = Sbr1Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,  # GUIウィンドウを表示するオプション
        eval_mode=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)
    
    # 指定した学習済みモデルをロード
    print(f"Loading model from: {args.model_path}")
    runner.load(args.model_path)
    policy = runner.get_inference_policy(device=gs.device)

    # 推論ループ
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            t_start = time.perf_counter()
            # モデルから行動を取得して環境を進める
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            # 1ステップにかかった処理時間を計算
            t_elapsed = time.perf_counter() - t_start

            # env.dt から処理時間を引いた「残りの時間」だけ sleep
            sleep_time = max(0.0, env.dt - t_elapsed)
            time.sleep(sleep_time)  # 制御周期（例: 0.02秒）に合わせて待機

if __name__ == "__main__":
    main()