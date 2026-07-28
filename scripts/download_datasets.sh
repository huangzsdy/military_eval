#!/bin/bash
# ============================================================
# 下载公开军事 NLP 数据集
# - ND-NER: 军事命名实体识别
# - CMNEE:  军事文档级事件抽取
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/processed/nd_ner
mkdir -p data/processed/cmnee

echo "============================================================"
echo "下载公开军事 NLP 数据集"
echo "============================================================"

# ---- ND-NER ----
echo ""
echo "--- ND-NER (军事 NER) ---"
echo "GitHub: https://github.com/XinyanLi2016/ND-NER"
echo ""
echo "请手动执行以下步骤:"
echo "  1. git clone https://github.com/XinyanLi2016/ND-NER.git tmp_ndner"
echo "  2. 将 ND-NER/data/ 下的 train.txt/dev.txt/test.txt"
echo "     复制到 data/processed/nd_ner/"
echo "  3. 在 config.yaml 中设置:"
echo "     probe_ner.dataset_path: data/processed/nd_ner"
echo ""

# 尝试自动下载
if command -v git &> /dev/null; then
    read -p "是否尝试自动下载 ND-NER? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[INFO] 正在下载 ND-NER..."
        git clone https://github.com/XinyanLi2016/ND-NER.git tmp_ndner 2>/dev/null || {
            echo "❌ 下载失败（可能需要手动下载）"
        }
        if [ -d "tmp_ndner/data" ]; then
            cp tmp_ndner/data/*.txt data/processed/nd_ner/ 2>/dev/null || true
            echo "✅ ND-NER 数据已复制到 data/processed/nd_ner/"
            echo "   请检查文件格式并编辑 config.yaml"
        fi
    fi
fi

# ---- CMNEE ----
echo ""
echo "--- CMNEE (军事事件抽取) ---"
echo "GitHub: https://github.com/Mzzzhu/CMNEE"
echo "Google Drive: https://drive.google.com/drive/folders/1nfKiSsu88oBeykUSYm7NGn4Q50_2GPS1"
echo ""
echo "请手动执行以下步骤:"
echo "  1. 访问 https://github.com/Mzzzhu/CMNEE"
echo "  2. 按论文说明申请数据集（部分需填写申请表）"
echo "  3. 将数据放入 data/processed/cmnee/"
echo "  4. 在 config.yaml 中设置:"
echo "     probe_event.dataset_path: data/processed/cmnee"
echo ""

# ---- 完成 ----
echo "============================================================"
echo "数据准备完成后，运行:"
echo "  bash scripts/run_all.sh"
echo "============================================================"
