#!/bin/bash

SCRIPT_PATH=$(find $(rospack find genesis_tools) -name "t_sbr1_train.py")
# $HOME を使うことで /home/rl-user/logs/sbr1_locomotion 以下に確実に保存される
BASE_DIR="${HOME}/logs/sbr1_locomotion/auto_sat_trial"
MAX_ITERS=1

for i in $(seq 1 3); do
    LOG_PATH="${BASE_DIR}_${i}"
    echo "=========================================================="
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] Trial ${i} / 10 開始: ${LOG_PATH}"
    echo "=========================================================="

    PYTHONINSPECT="" python3 "${SCRIPT_PATH}" -l "${LOG_PATH}" --max_iterations ${MAX_ITERS}

    echo " [$(date '+%Y-%m-%d %H:%M:%S')] Trial ${i} / 10 完了"
    echo ""
done

echo "全10回の自動学習が完了しました！"