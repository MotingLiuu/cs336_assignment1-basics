SWEEP_ID=mutyuu/test1/f3tor4ib
TOTAL_RUNS=5

for i in $(seq 1 $TOTAL_RUNS)
do
    echo "▶️ [实验 ${i}/${TOTAL_RUNS}] 启动 wandb agent..."

    wandb agent --count=1 "${SWEEP_ID}"

    echo "🔄 [实验 ${i}/${TOTAL_RUNS}] 实验完成，开始同步..."

    # 查找最新创建的离线运行目录并同步它
    # ls -td 会按修改时间降序排序目录
    # head -n 1 会选取最新的一个
    LATEST_RUN_DIR=$(ls -td wandb/offline-run-* | head -n 1)

    if [ -n "${LATEST_RUN_DIR}" ]; then
        wandb sync "${LATEST_RUN_DIR}"
        echo "✅ [实验 ${i}/${TOTAL_RUNS}] 同步成功: ${LATEST_RUN_DIR}"
    else
        echo "⚠️ [实验 ${i}/${TOTAL_RUNS}] 未找到新的离线运行目录进行同步。"
    fi
    echo "--------------------------------------------------"
done

echo "${TOTAL_RUNS} 个实验已成功执行并同步"