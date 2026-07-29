# 军事数据集评测框架

## English Version Below | 中文版在下面

---

# Military Dataset Evaluation Framework (English)

## Project Overview

This framework evaluates the quality of Chinese military text datasets (jsonl format, each line containing a `content` field) and their effectiveness for training military large language models.

## Three-Layer Evaluation System

### Layer 1: Static Quality Evaluation
Analyzes the dataset itself without model training.

| Module | Metrics |
|--------|---------|
| Basic Statistics | Document count, character count, token count, length distribution |
| n-gram Repetition | In-document / cross-document repetition rates |
| MinHash Deduplication | Near-duplicate detection |
| Domain Distribution | Category entropy (8 categories) |
| MAUVE Comparison | Distribution similarity vs reference corpus |
| Desensitization Scan | Sensitive info detection (unit numbers, equipment codes) |

### Layer 2: Downstream Task Probes
Trains models on your data and evaluates on public military NLP benchmarks.

| Probe Task | Dataset | Metrics | Evaluates |
|------------|---------|---------|-----------|
| Military NER | ND-NER (19 entity types) | entity-level P/R/F1 | Named entity recognition |
| Event Extraction | CMNEE (8 event types) | Type Acc, Arg R, Trigger F1 | Event understanding |
| Military QA | Auto-constructed QA set | ROUGE, BERTScore, LLM Judge | Question answering |

### Layer 3: Training Benefit Quantification
A/B ablation experiments to prove data effectiveness:
- **Group A (Baseline)**: Fine-tune base model directly on public data
- **Group B (Your Data)**: Continue pre-training with your data, then fine-tune

Compare metrics difference to quantify your data's value.

## Input & Output

### Input
```json
{"content": "中国人民解放军东部战区某部近日在台海方向组织了实战化联合演训...", "id": "001"}
```

### Output
```
output/
├── metrics/
│   ├── static_eval.json      # Static evaluation metrics
│   ├── probe_ner.json        # NER evaluation results
│   ├── probe_event.json      # Event extraction results
│   └── full_summary.json     # All metrics summary
├── figures/
│   ├── domain_distribution.png
│   └── ngram_repetition.png
└── reports/
    ├── evaluation_report.md   # Markdown report
    └── evaluation_report.html # HTML report
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your data
cp your_data.jsonl data/raw/military_data.jsonl

# 3. Edit config.yaml
vim config.yaml

# 4. Run evaluation
# Static only (fast):
bash scripts/run_static_only.sh

# Full pipeline (requires GPU):
bash scripts/run_all.sh
```

## Multi-Field Evaluation

If your dataset contains multiple fields (e.g., original text, QA pairs, Wikipedia-style rephrasing), you can use the multi-field evaluation system:

```bash
# Run multi-field evaluation
python -m src.multi_field_eval --config config.yaml
```

### Supported Field Types

| Field | Description | Evaluation |
|-------|-------------|------------|
| `content` | Original military text | Full evaluation (static + NER + event + QA) |
| `synthesized_content_QA` | QA pairs | QA-specific evaluation |
| `synthesized_Wikipedia-style_rephrasing` | Wiki-style rephrasing | Quality evaluation |

### Input Format

```json
{
  "id": "001",
  "content": "原始军事文本内容...",
  "synthesized_content_QA": "[{\"question\": \"问题\", \"answer\": \"答案\"}]",
  "synthesized_Wikipedia-style_rephrasing": "Wiki百科风格改写..."
}
```

## Directory Structure

```
military_eval/
├── config.yaml              # Global configuration
├── requirements.txt         # Python dependencies
├── README.md
├── data/
│   ├── raw/                 # Raw jsonl data
│   └── processed/           # Processed data / public datasets
├── src/
│   ├── __init__.py
│   ├── static_eval.py       # Layer 1: Static quality
│   ├── pretrain.py          # Continue pre-training
│   ├── probe_ner.py        # Probe 1: Military NER (ND-NER)
│   ├── probe_event.py      # Probe 2: Event extraction (CMNEE)
│   ├── probe_qa.py         # Probe 3: Military QA
│   ├── report.py           # Generate evaluation report
│   └── run_all.py          # Run all evaluations
├── scripts/
│   ├── run_all.sh          # Run all evaluations
│   └── run_static_only.sh  # Static evaluation only
└── output/
    ├── figures/             # Visualization charts
    ├── reports:            # Evaluation reports
    └── metrics:            # JSON metrics
```

## Detailed Evaluation Workflow

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: Your Military Dataset                  │
│                  (jsonl format, content field)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 1: Static Quality Evaluation                 │
│                   (Statistical analysis only)                   │
└─────────────────────────────────────────────────────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
   ┌──────────┐       ┌──────────┐       ┌──────────┐
   │ Scale    │       │ Dedup    │       │ Domain   │
   │ Stats    │       │ Detection│       │ Analysis │
   └──────────┘       └──────────┘       └──────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    static_eval.json

                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 2: Downstream Task Probes                   │
│    (Train models on your data, test on public benchmarks)     │
└─────────────────────────────────────────────────────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
   ┌──────────┐       ┌──────────┐       ┌──────────┐
   │   NER    │       │  Event   │       │    QA    │
   │  Probe   │       │  Probe   │       │  Probe   │
   └────┬─────┘       └────┬─────┘       └────┬─────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Fine-tune on   │ │ Fine-tune on   │ │ Generate       │
│ ND-NER         │ │ CMNEE         │ │ Answers        │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         ▼                   ▼                   ▼
    P/R/F1             Trigger F1          ROUGE/BERTScore
                                              (Optional)
                                                │
                                                ▼
                                           LLM Judge
                                           Quality Score

                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            Layer 3: Training Benefit Quantification            │
│                  (A/B ablation experiments)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: Static Quality Evaluation (Details)

| Module | What it does | Metrics |
|--------|--------------|---------|
| **Basic Statistics** | Count documents/chars/tokens | num_docs, avg_length, P50/P95/P99 |
| **n-gram Repetition** | Measure text repetition | in-doc / cross-doc repetition rate |
| **MinHash Dedup** | Detect near-duplicate docs | duplicate_rate |
| **Domain Distribution** | Analyze topic coverage | category entropy |
| **MAUVE** | Compare distribution with reference | similarity score (0-1) |
| **Desensitization** | Scan sensitive info | pattern matches |

### Layer 2: Probe Tasks (Details)

#### 2.1 NER Probe
```
Your Data → Continue Pre-training → Fine-tune → Test on ND-NER → P/R/F1
```

- **Dataset**: ND-NER (19 entity types)
- **Model**: Qwen2.5-7B + TokenClassification head
- **Metrics**: entity-level Precision/Recall/F1

#### 2.2 Event Extraction Probe
```
Your Data → Continue Pre-training → Fine-tune → Test on CMNEE → Trigger F1
```

- **Dataset**: CMNEE (8 event types: experiment/movement/deployment/support/accident/training/conflict/casualty)
- **Model**: Qwen2.5-7B + Dual heads (trigger + argument)
- **Metrics**: Type Acc, Type R, Arg R, Trigger F1

#### 2.3 QA Probe

**Option A: Auto Metrics (No Judge Model)**
```
Question → Generate Answer → Compare with Reference → ROUGE/BERTScore
```

- **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore

**Option B: LLM-as-Judge (Optional)**
```
Question + Answer → Judge Model (72B) → Quality Scores
```

- **Scores**: accuracy, completeness, relevance, military_expertise (1-10 each)

### Layer 3: A/B Experiment

| Group | Treatment | Evaluation |
|-------|-----------|------------|
| **A (Baseline)** | Fine-tune base model directly | NER/Event/QA metrics |
| **B (Experiment)** | Continue pre-train with your data, then fine-tune | NER/Event/QA metrics |

**Improvement** = (B - A) / A × 100%

## Models Used

| Model | Purpose | Usage |
|-------|---------|-------|
| **Qwen/Qwen2.5-7B** | Base model | Continue pre-training + fine-tuning |
| **Qwen/Qwen2.5-72B-Instruct** | Judge model | LLM-as-Judge (optional) |
| **LoRA (r=16)** | Efficient training | Reduce GPU memory |

## Configuration (config.yaml)

```yaml
data:
  path: data/raw/military_data.jsonl
  content_field: content
  max_docs: -1  # -1 means all documents

model:
  base_model: Qwen/Qwen2.5-7B
  output_dir: output/checkpoints
  use_lora: true

probe_ner:
  dataset_path: data/processed/nd_ner

probe_event:
  enabled: false  # Requires CMNEE dataset application
```

## Evaluation Metrics

### Static Quality
- Document/character/token counts
- n-gram repetition rates
- MinHash deduplication rate
- Domain entropy
- MAUVE distribution score

### NER (ND-NER)
- entity-level Precision/Recall/F1 (seqeval strict mode)
- per-type F1

### Event Extraction (CMNEE)
- Type Acc / Type R
- Arg R
- Co-ref R
- Trigger F1 / Argument F1

## Public Datasets

| Dataset | Description | Download |
|---------|-------------|----------|
| ND-NER | Military Named Entity Recognition (19 types) | GitHub: XinyanLi2016/ND-NER |
| CMNEE | Military Document-level Event Extraction (8 types) | GitHub: Mzzzhu/CMNEE (application required) |

## GPU Memory Requirements

| Model | Precision | Training VRAM | Inference VRAM |
|-------|-----------|---------------|-----------------|
| Qwen2.5-1.8B | fp16 | ~6GB | ~4GB |
| Qwen2.5-7B | fp16 | ~16GB | ~14GB |
| Qwen2.5-7B + LoRA | fp16 | ~8GB | ~14GB |
| Qwen2.5-72B | fp16 | ~160GB | ~150GB |

## Troubleshooting

### 1. CUDA Errors
If you encounter CUDA errors, try modifying the config:

```yaml
model:
  dtype: float16
  per_device_train_batch_size: 1
  max_seq_length: 1024
```

### 2. Out of Memory
- Enable LoRA: `use_lora: true`
- Reduce LoRA rank: `lora_rank: 8`
- Reduce batch size

### 3. AdamW Import Error
Import from torch instead:
```python
from torch.optim import AdamW
```

### 4. Model Download Failed
Use Chinese mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## License

This project is for research purposes. Please comply with data licensing terms.

---

# 军事数据集评测框架

## 项目简介

本框架用于评测中文军事文本数据集（jsonl 格式，每行含 `content` 字段）的质量及其对军事大模型训练的有效性。

## 三层评测体系

### 第一层：静态质量评测
无需训练模型，直接分析原始数据质量。

| 评测模块 | 评测指标 |
|----------|----------|
| 基础规模统计 | 文档数、字符数、token数、长度分布 |
| n-gram 重复率 | 文档内/文档间重复率 |
| MinHash 去重 | 近似重复文档检测 |
| 领域分布分析 | 8大类（战略/战术/装备/后勤/情报/训练/条令/指挥）熵值 |
| MAUVE 对比 | 与参考语料的分布相似度 |
| 脱敏合规扫描 | 番号、装备编号等敏感信息检测 |

### 第二层：下游任务探针
使用你的数据训练模型，在公开军事NLP基准上测试。

| 探针任务 | 数据集 | 评测指标 | 评估能力 |
|----------|--------|----------|----------|
| 军事NER | ND-NER（19类实体） | entity-level P/R/F1 | 命名实体识别 |
| 事件抽取 | CMNEE（8类事件） | Type Acc、Arg R、Trigger F1 | 事件理解 |
| 军事QA | 自动构建的QA集 | ROUGE、BERTScore、LLM Judge | 问答生成 |

### 第三层：训练收益量化

通过 **A/B 消融实验** 量化数据的实际价值：

- **A组（基线）**：直接用基座模型在公开数据上微调
- **B组（你的数据）**：用你的数据继续预训练后微调

对比两组指标的差异，量化你的数据的**提升幅度**。

## 输入与输出

### 输入格式
```json
{"content": "中国人民解放军东部战区某部近日在台海方向组织了实战化联合演训...", "id": "001"}
```

### 输出结构
```
output/
├── metrics/
│   ├── static_eval.json      # 静态评测指标
│   ├── probe_ner.json        # NER评测结果
│   ├── probe_event.json      # 事件抽取结果
│   └── full_summary.json     # 全部指标汇总
├── figures/
│   ├── domain_distribution.png   # 领域分布饼图
│   └── ngram_repetition.png      # 重复率柱状图
└── reports/
    ├── evaluation_report.md   # Markdown报告
    └── evaluation_report.html # HTML报告
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备你的数据
cp your_data.jsonl data/raw/military_data.jsonl

# 3. 编辑配置
vim config.yaml

# 4. 运行评测
# 仅静态评测（快速）:
bash scripts/run_static_only.sh

# 完整流程（需要GPU）:
bash scripts/run_all.sh
```

## 多字段评测

如果你的数据集包含多个字段（例如：原始文本、QA对、Wiki百科风格改写），可以使用多字段评测系统：

```bash
# 运行多字段评测
python -m src.multi_field_eval --config config.yaml
```

### 支持的字段类型

| 字段 | 说明 | 评测方式 |
|------|------|----------|
| `content` | 原始军事文本 | 完整评测（静态+NER+事件+QA） |
| `synthesized_content_QA` | QA问答对 | QA专项评测 |
| `synthesized_Wikipedia-style_rephrasing` | Wiki风格改写 | 改写质量评测 |

### 输入格式

```json
{
  "id": "001",
  "content": "原始军事文本内容...",
  "synthesized_content_QA": "[{\"question\": \"问题\", \"answer\": \"答案\"}]",
  "synthesized_Wikipedia-style_rephrasing": "Wiki百科风格改写..."
}
```

## 目录结构

```
military_eval/
├── config.yaml              # 全局配置
├── requirements.txt         # Python 依赖
├── README.md
├── data/
│   ├── raw/                 # 原始 jsonl 数据
│   └── processed/           # 预处理后的数据 / 公开数据集
├── src/
│   ├── __init__.py
│   ├── static_eval.py       # 第一层：静态质量评测
│   ├── pretrain.py          # 继续预训练脚本
│   ├── probe_ner.py         # 探针任务1：军事 NER（ND-NER）
│   ├── probe_event.py       # 探针任务2：事件抽取（CMNEE）
│   ├── probe_qa.py          # 探针任务3：军事 QA
│   ├── report.py            # 第三层：生成评测报告
│   └── run_all.py           # 运行全部评测
├── scripts/
│   ├── run_all.sh           # 运行全部评测
│   └── run_static_only.sh   # 仅运行静态评测
└── output/
    ├── figures/             # 可视化图表
    ├── reports/             # 评测报告
    └── metrics/            # 量化指标 JSON
```

## 详细评测流程

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    输入：你的军事数据集                           │
│                  (jsonl格式，content字段)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  第一层：静态质量评测                             │
│                     (只做统计分析)                              │
└─────────────────────────────────────────────────────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
   ┌──────────┐       ┌──────────┐       ┌──────────┐
   │ 规模统计  │       │ 去重检测  │       │ 分布分析  │
   └──────────┘       └──────────┘       └──────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    static_eval.json

                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   第二层：下游任务探针                            │
│             (用你的数据训练模型，在公开基准上测试)               │
└─────────────────────────────────────────────────────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
   ┌──────────┐       ┌──────────┐       ┌──────────┐
   │   NER    │       │ 事件抽取  │       │   QA    │
   │  探针    │       │   探针    │       │   探针   │
   └────┬─────┘       └────┬─────┘       └────┬─────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ 在ND-NER上微调  │ │ 在CMNEE上微调  │ │ 生成回答       │
│ 并测试          │ │ 并测试          │ │ + 指标计算     │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         ▼                   ▼                   ▼
    P/R/F1             Trigger F1          ROUGE/BERTScore
                                              (可选)
                                                │
                                                ▼
                                           裁判模型评估
                                           质量评分

                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    第三层：训练收益量化                          │
│                     (A/B消融实验)                              │
└─────────────────────────────────────────────────────────────────┘
```

### 第一层：静态质量评测（详解）

| 模块 | 操作 | 指标 |
|------|------|------|
| **基础规模统计** | 统计文档/字符/token数 | 文档数、平均长度、P50/P95/P99 |
| **n-gram重复率** | 计算n元组重复程度 | 文档内/文档间重复率 |
| **MinHash去重** | 检测近似重复文档 | 重复率 |
| **领域分布分析** | 基于关键词匹配8大类 | 类别熵值 |
| **MAUVE对比** | 与参考语料对比分布 | 相似度分数(0-1) |
| **脱敏扫描** | 匹配敏感信息模式 | 匹配数量 |

### 第二层：探针任务（详解）

#### 2.1 NER探针
```
你的数据 → 继续预训练 → 微调(ND-NER) → 测试 → P/R/F1
```

- **数据集**：ND-NER（19类实体）
- **模型**：Qwen2.5-7B + TokenClassification头
- **指标**：entity-level Precision/Recall/F1

#### 2.2 事件抽取探针
```
你的数据 → 继续预训练 → 微调(CMNEE) → 测试 → Trigger F1
```

- **数据集**：CMNEE（8类事件：实验/机动/部署/支援/事故/演训/冲突/伤亡）
- **模型**：Qwen2.5-7B + 双头（触发词+论元）
- **指标**：Type Acc、Type R、Arg R、Trigger F1

#### 2.3 QA探针

**方式A：自动指标（不需要裁判模型）**
```
问题 → 模型生成回答 → 对比参考答案 → ROUGE/BERTScore
```

- **指标**：ROUGE-1、ROUGE-2、ROUGE-L、BERTScore

**方式B：LLM-as-Judge（可选）**
```
问题 + 回答 → 裁判模型(72B) → 质量评分
```

- **评分维度**：准确性、完整性、相关性、军事专业性（各1-10分）

### 第三层：A/B实验

| 组别 | 处理方式 | 评测任务 |
|------|----------|----------|
| **A组（基线）** | 基座模型直接微调 | NER/事件抽取/QA |
| **B组（实验组）** | 你的数据继续预训练后微调 | NER/事件抽取/QA |

**提升幅度** = (B组指标 - A组指标) / A组指标 × 100%

## 使用的模型

| 模型 | 用途 | 说明 |
|------|------|------|
| **Qwen/Qwen2.5-7B** | 基座模型 | 继续预训练+微调 |
| **Qwen/Qwen2.5-72B-Instruct** | 裁判模型 | LLM-as-Judge（可选） |
| **LoRA (r=16)** | 高效训练 | 降低显存占用 |

## 配置文件说明（config.yaml）

```yaml
data:
  path: data/raw/military_data.jsonl
  content_field: content
  max_docs: -1  # -1 表示全部

model:
  base_model: Qwen/Qwen2.5-7B
  output_dir: output/checkpoints
  use_lora: true

probe_ner:
  dataset_path: data/processed/nd_ner

probe_event:
  enabled: false  # CMNEE需要申请，暂不可用
```

## 评测指标说明

### 静态质量
- 文档数/字符数/token数
- n-gram 重复率（文档内/文档间）
- MinHash 去重率
- 类别熵值（领域分布均衡性）
- MAUVE 分布对比分数

### NER（ND-NER）
- entity-level Precision/Recall/F1（seqeval strict 模式）
- per-type F1

### 事件抽取（CMNEE）
- Type Acc / Type R
- Arg R
- Co-ref R
- Trigger F1 / Argument F1

### 训练收益
- A/B 消融对比表
- 与公开基线的对比

## 公开数据集

| 数据集 | 说明 | 下载方式 |
|--------|------|----------|
| ND-NER | 军事命名实体识别（19类实体） | GitHub: XinyanLi2016/ND-NER |
| CMNEE | 军事文档级事件抽取（8类事件） | GitHub: Mzzzhu/CMNEE（需申请） |

## 运行完整流程示例

```bash
# 下载公开数据集
bash scripts/download_datasets.sh

# 仅运行静态评测（快速，无需GPU）
python -m src.static_eval --config config.yaml

# 运行完整评测（需要GPU，约16GB显存）
python -m src.run_all --config config.yaml

# 多字段评测（需要配置 fields）
python -m src.multi_field_eval --config config.yaml
```

## 显存要求

| 模型 | 精度 | 训练显存 | 推理显存 |
|------|------|----------|----------|
| Qwen2.5-1.8B | fp16 | ~6GB | ~4GB |
| Qwen2.5-7B | fp16 | ~16GB | ~14GB |
| Qwen2.5-7B + LoRA | fp16 | ~8GB | ~14GB |
| Qwen2.5-72B | fp16 | ~160GB | ~150GB |

## 常见问题

### 1. CUDA 错误
如果遇到 `CUBA` 错误，尝试修改配置：

```yaml
model:
  dtype: float16  # 改用 float16
  per_device_train_batch_size: 1  # 减小 batch size
  max_seq_length: 1024  # 减小序列长度
```

### 2. 显存不足
- 启用 LoRA：`use_lora: true`
- 减小 LoRA rank：`lora_rank: 8`
- 减少 batch size

### 3. AdamW 导入错误
新版本 transformers 不包含 AdamW，请确保从 torch 导入：
```python
from torch.optim import AdamW
```

### 4. 模型下载失败
配置代理或使用国内镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 许可证

本项目仅用于研究目的。请遵守数据许可协议。

---

## 核心价值

回答核心问题：**"我的军事数据质量怎么样？用这些数据训练模型到底有没有用？"**
