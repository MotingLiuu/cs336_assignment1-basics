#!/bin/bash

# Default script to run
SCRIPT_PATH="experiments/bpe_train_TinyStories.py"
# Default output report (文本)
REPORT_PATH="scalene_profile_cputime_TinyStories_test_vocab_20000_cpu.html"

# Usage message
usage() {
    echo "Usage: $0 [SCRIPT_PATH] [REPORT_PATH]"
    echo "Example: $0 experiments/bpe_train_TinyStories.py scalene_profile"
    exit 1
}

# Handle optional arguments
if [ "$1" ]; then
    SCRIPT_PATH="$1"
fi
if [ "$2" ]; then
    REPORT_PATH="$2"
fi

# Ensure scalene is installed
if ! command -v scalene &>/dev/null; then
    echo "❌ scalene not found. Install it with: pip install scalene"
    exit 1
fi

# Run the profiler (纯文本模式，没有 --html)
echo "📊 Profiling $SCRIPT_PATH..."
PYTHONPATH=. scalene --html --profile-all --cpu-only --outfile "$REPORT_PATH" "$SCRIPT_PATH"
echo "✅ Done. Report saved to $REPORT_PATH"
