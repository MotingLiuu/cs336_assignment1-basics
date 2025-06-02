#!/bin/bash
set -e

# --- 默认相对路径 ---
DEFAULT_SCRIPT_REL_PATH="experiments/bpe_train_TinyStories.py"
DEFAULT_REPORT_REL_PATH="viz_profile.html"
DEFAULT_INCLUDE_EXPERIMENTS_REL_PATH="./experiments"
DEFAULT_INCLUDE_LIB_REL_PATH="./cs336_basics" # 假设 cs336_basics 是一个目录

# --- 使用的相对路径 (可能被命令行参数覆盖) ---
SCRIPT_REL_PATH="$DEFAULT_SCRIPT_REL_PATH"
REPORT_REL_PATH="$DEFAULT_REPORT_REL_PATH"
# 注意：对于 include_files 的路径，我们暂时不让它们被命令行参数直接覆盖，
# 因为它们的正确性与 SCRIPT_REL_PATH 和库的固定位置有关。
# 如果需要更灵活，可以后续调整。

# --- 用法说明 ---
usage() {
    echo "Usage: $0 [SCRIPT_RELATIVE_PATH] [REPORT_RELATIVE_PATH]"
    echo "Example: $0 experiments/bpe_train_TinyStories.py viz_profile.html"
    exit 1
}

# --- 处理命令行可选参数 (仍然是相对路径) ---
if [ "$1" ]; then
    SCRIPT_REL_PATH="$1"
fi
if [ "$2" ]; then
    REPORT_REL_PATH="$2"
fi

# --- 将核心路径转换为绝对路径 ---
# realpath 会解析路径，处理掉 '..' 和 '.' 等，并给出绝对路径
# 获取脚本执行的当前目录的绝对路径
CWD_ABS="$(pwd)" # 或者用 CWD_ABS="$(realpath .)"

SCRIPT_ABS_PATH="$(realpath "$SCRIPT_REL_PATH")"
REPORT_ABS_PATH="$(realpath "$REPORT_REL_PATH")" # 输出报告的路径，绝对化可能更好管理

# 确保目录存在，否则 realpath 可能会报错，或者如果想让 realpath 创建最后一级之前的目录，可以先创建
# 对于 --include_files 的路径，也转换为绝对路径
# 假设它们是相对于脚本执行目录 (CWD_ABS) 的
INCLUDE_EXPERIMENTS_ABS_PATH="$(realpath "$DEFAULT_INCLUDE_EXPERIMENTS_REL_PATH")"
INCLUDE_LIB_ABS_PATH="$(realpath "$DEFAULT_INCLUDE_LIB_REL_PATH")"

if [ ! -f "$SCRIPT_ABS_PATH" ]; then
    echo "错误: 脚本 '$SCRIPT_ABS_PATH' (源自 '$SCRIPT_REL_PATH') 未找到。"
    exit 1
fi

# --- 检查 include 目录是否存在 (用绝对路径检查) ---
if [ ! -d "$INCLUDE_EXPERIMENTS_ABS_PATH" ]; then
    echo "错误: include 目录 '$INCLUDE_EXPERIMENTS_ABS_PATH' (源自 '$DEFAULT_INCLUDE_EXPERIMENTS_REL_PATH') 未找到。"
    exit 1
fi
if [ ! -d "$INCLUDE_LIB_ABS_PATH" ]; then
    echo "错误: include 目录 '$INCLUDE_LIB_ABS_PATH' (源自 '$DEFAULT_INCLUDE_LIB_REL_PATH') 未找到。"
    exit 1
fi


echo "📊 Profiling $SCRIPT_ABS_PATH..."
echo "Output report will be at: $REPORT_ABS_PATH"
echo "Tracing will include (absolute paths):"
echo "  - Experiments dir: $INCLUDE_EXPERIMENTS_ABS_PATH"
echo "  - Library dir:     $INCLUDE_LIB_ABS_PATH"

# --- 执行 viztracer，全部使用绝对路径 ---
PYTHONPATH="$CWD_ABS" python -m viztracer -o "$REPORT_ABS_PATH" \
    --include_files "$INCLUDE_EXPERIMENTS_ABS_PATH" \
    --include_files "$INCLUDE_LIB_ABS_PATH" \
    "$SCRIPT_ABS_PATH"

VIZTRACER_EXIT_CODE=$?
echo "VizTracer exit code: $VIZTRACER_EXIT_CODE"

if [ $VIZTRACER_EXIT_CODE -ne 0 ]; then
    echo "错误: VizTracer 执行失败。"
    if [ -f "$REPORT_ABS_PATH" ]; then
        echo "报告文件 '$REPORT_ABS_PATH' 的前20行内容:"
        head -n 20 "$REPORT_ABS_PATH"
    fi
    exit $VIZTRACER_EXIT_CODE
fi

# 检查报告文件大小
if [ -f "$REPORT_ABS_PATH" ]; then
    FILE_SIZE=$(stat -c%s "$REPORT_ABS_PATH") # Linux
    # FILE_SIZE=$(stat -f%z "$REPORT_ABS_PATH") # macOS - uncomment if needed
    echo "报告文件大小: $FILE_SIZE bytes"
    if [ "$FILE_SIZE" -lt 1024 ]; then # 假设有效的HTML报告至少有1KB
        echo "警告: 报告文件 '$REPORT_ABS_PATH' 非常小，可能没有包含有效的追踪数据。"
        echo "报告文件 '$REPORT_ABS_PATH' 的前10行内容:"
        head -n 10 "$REPORT_ABS_PATH"
    fi
else
    echo "错误: 未找到报告文件 '$REPORT_ABS_PATH'。"
fi

echo "✅ 完成。报告已保存至 $REPORT_ABS_PATH"