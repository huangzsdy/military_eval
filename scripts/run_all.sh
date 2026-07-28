#!/bin/bash
# ============================================================
# 一键运行全部评测
# 用法: bash scripts/run_all.sh [config.yaml]
# ============================================================

set -e

CONFIG="${1:-config.yaml}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "军事数据集评测框架"
echo "项目根目录: $PROJECT_ROOT"
echo "配置文件: $CONFIG"
echo "============================================================"

# 检查 Python
PYTHON_BIN=$(which python3 || which python)
echo "[INFO] Python: $PYTHON_BIN"

# 检查依赖
echo "[INFO] 检查依赖..."
$PYTHON_BIN -c "import torch, transformers, datasets, seqeval, evaluate; print('  ✅ 核心依赖已安装')" 2>/dev/null || {
    echo "  ⚠️ 部分依赖缺失，正在安装..."
    $PYTHON_BIN -m pip install -r requirements.txt
}

# 检查数据文件
DATA_PATH=$($PYTHON_BIN -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['data']['path'])")
echo "[INFO] 数据路径: $DATA_PATH"

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 数据文件不存在: $DATA_PATH"
    echo ""
    echo "请将你的 jsonl 数据放到正确路径，或编辑 $CONFIG 修改 data.path"
    echo ""
    echo "示例:"
    echo "  mkdir -p data/raw"
    echo "  cp /path/to/your/data.jsonl data/raw/military_data.jsonl"
    echo "  # 然后编辑 config.yaml 设置 data.path"
    exit 1
fi

echo ""
echo "============================================================"
echo "阶段 1/4: 静态质量评测"
echo "============================================================"
$PYTHON_BIN -m src.static_eval --config "$CONFIG"
echo "✅ 静态评测完成"
echo ""

echo "============================================================"
echo "阶段 2/4: 继续预训练"
echo "============================================================"
echo "⚠️ 此阶段需要 GPU，预计耗时较长"
echo "   如使用 DeepSpeed: deepspeed --num_gpus=N -m src.pretrain --config $CONFIG"
echo ""

# 检查是否有继续预训练产物
CKPT_DIR=$($PYTHON_BIN -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['output_dir'])")
if [ -f "$CKPT_DIR/TRAINING_DONE" ]; then
    echo "✅ 检测到继续预训练产物，跳过训练"
else
    echo "未检测到继续预训练产物，开始训练..."
    $PYTHON_BIN -m src.pretrain --config "$CONFIG" || {
        echo "❌ 继续预训练失败（可能需要更多 GPU 资源）"
        echo "   你可以手动运行后重新执行此脚本"
        echo "   或设置 use_lora: true 减少显存占用"
    }
fi
echo ""

echo "============================================================"
echo "阶段 3/4: 下游任务探针"
echo "============================================================"

# NER 探针
echo "--- 探针 1: 军事 NER (ND-NER) ---"
$PYTHON_BIN -m src.probe_ner --config "$CONFIG" || echo "⚠️ NER 探针跳过（数据未准备）"
echo ""

# 事件抽取探针
echo "--- 探针 2: 事件抽取 (CMNEE) ---"
$PYTHON_BIN -m src.probe_event --config "$CONFIG" || echo "⚠️ 事件抽取探针跳过（数据未准备）"
echo ""

# QA 探针
echo "--- 探针 3: 军事 QA ---"
$PYTHON_BIN -m src.probe_qa --config "$CONFIG" || echo "⚠️ QA 探针跳过"
echo ""

echo "============================================================"
echo "阶段 4/4: 生成评测报告"
echo "============================================================"
$PYTHON_BIN -m src.report --config "$CONFIG"
echo ""

echo "============================================================"
echo "✅ 全部评测流程完成！"
echo ""
echo "查看报告:"
echo "  cat output/reports/evaluation_report.md"
echo "  或浏览器打开 output/reports/evaluation_report.html"
echo ""
echo "查看指标:"
echo "  ls output/metrics/"
echo "============================================================"
