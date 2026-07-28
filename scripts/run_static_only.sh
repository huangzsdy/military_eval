#!/bin/bash
# ============================================================
# 仅运行第一层静态评测（不需要 GPU，速度快）
# 用法: bash scripts/run_static_only.sh [config.yaml]
# ============================================================

set -e

CONFIG="${1:-config.yaml}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "军事数据集 · 静态质量评测（第一层）"
echo "============================================================"

PYTHON_BIN=$(which python3 || which python)

# 检查数据
DATA_PATH=$($PYTHON_BIN -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['data']['path'])")
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 数据文件不存在: $DATA_PATH"
    echo "请将你的 jsonl 数据放入正确路径后重试"
    exit 1
fi

echo "[INFO] 数据: $DATA_PATH"

# 安装最小依赖
$PYTHON_BIN -c "import torch, transformers, datasets, datasketch, numpy, matplotlib, pyyaml, tqdm" 2>/dev/null || {
    echo "[INFO] 安装依赖..."
    $PYTHON_BIN -m pip install -r requirements.txt
}

# 运行静态评测
$PYTHON_BIN -m src.static_eval --config "$CONFIG"

echo ""
echo "============================================================"
echo "✅ 静态评测完成！"
echo ""
echo "查看结果:"
echo "  指标: cat output/metrics/static_eval.json"
echo "  图表: ls output/figures/"
echo "============================================================"
