"""
探针任务 1：军事命名实体识别 (NER)
====================================
使用 ND-NER 数据集（公开军事 NER 基准）。
评测指标: entity-level P/R/F1 via seqeval (strict 模式)

论文依据:
- ND-NER: ICONIP 2022, CC BY-SA 4.0
- SeqScore: 2021, 验证 seqeval strict 边界的正确性
- 实体级 F1 是 NER 领域标准指标（区别于 token 级 accuracy）

运行:
    python -m src.probe_ner --config config.yaml
"""

import argparse
import sys
import os
import json
from pathlib import Path
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer,
    Trainer, TrainingArguments,
    DataCollatorForTokenClassification,
)
import evaluate
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load_config, get_output_dirs, save_metrics, get_device_info


# ============================================================
# ND-NER 实体标签（19 类）
# ============================================================
ND_NER_LABELS = [
    "O",
    "B-COU", "I-COU",          # 国家
    "B-PER", "I-PER",          # 人物
    "B-TIM", "I-TIM",          # 时间
    "B-LOC", "I-LOC",          # 地理位置
    "B-MILORG", "I-MILORG",    # 军事组织与机构
    "B-MILFAC", "I-MILFAC",    # 军事设施
    "B-MILEVE", "I-MILEVE",    # 军事与政治事件
    "B-MILRAN", "I-MILRAN",    # 军衔或军职
    "B-ART", "I-ART",          # 火炮
    "B-EXP", "I-EXP",          # 爆炸物
    "B-AIR", "I-AIR",          # 航空器
    "B-NAV", "I-NAV",          # 舰船
    "B-MIS", "I-MIS",          # 导弹武器
    "B-SPA", "I-SPA",          # 天基装备
    "B-TNK", "I-TNK",          # 坦克装甲车
    "B-SML", "I-SML",          # 枪械与单兵装备
    "B-ELC", "I-ELC",          # 电磁网络设备
    "B-WMD", "I-WMD",          # 大规模杀伤性武器
    "B-NCW", "I-NCW",          # 新概念武器
]

# 简化为常用 6 大类（用于快速评测）
SIMPLIFIED_LABELS_6 = [
    "O",
    "B-LOC", "I-LOC",          # 地点
    "B-ORG", "I-ORG",          # 组织
    "B-PER", "I-PER",          # 人物
    "B-WPN", "I-WPN",          # 武器装备
    "B-EVE", "I-EVE",          # 事件
    "B-TIME", "I-TIME",        # 时间
]


def load_nd_ner_data(data_dir: str) -> dict:
    """
    加载 ND-NER 数据集。
    期望目录结构:
        data_dir/
            train.txt  (BIO 格式)
            dev.txt
            test.txt
    每行: token\tlabel  或  空行分隔文档
    """
    splits = {}
    for split in ["train", "dev", "test"]:
        path = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(path):
            print(f"[WARN] {path} 不存在，跳过")
            continue
        
        sentences = []
        labels = []
        tokens = []
        tags = []
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        sentences.append(tokens)
                        labels.append(tags)
                        tokens = []
                        tags = []
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    tokens.append(parts[0])
                    tags.append(parts[1])
                elif len(parts) == 1 and parts[0]:
                    # 可能空格分隔
                    sub = parts[0].split()
                    if len(sub) >= 2:
                        tokens.append(sub[0])
                        tags.append(sub[-1])
            
            if tokens:
                sentences.append(tokens)
                labels.append(tags)
        
        splits[split] = {"tokens": sentences, "labels": labels}
        print(f"  [ND-NER] {split}: {len(sentences)} 句")
    
    return splits


class NERDataset(Dataset):
    """将 token+label 列表转为模型输入"""
    
    def __init__(self, tokens_list, labels_list, tokenizer, label_list, max_length=256):
        self.tokens_list = tokens_list
        self.labels_list = labels_list
        self.tokenizer = tokenizer
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.id2label = {i: l for i, l in enumerate(label_list)}
        self.max_length = max_length
    
    def __len__(self):
        return len(self.tokens_list)
    
    def __getitem__(self, idx):
        tokens = self.tokens_list[idx][:self.max_length]
        labels = self.labels_list[idx][:self.max_length]
        
        # 字符级分词（中文）
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # 对齐标签到 word_ids
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word:
                # 取该词对应的标签
                if word_id < len(labels):
                    label_str = labels[word_id]
                    label_ids.append(self.label2id.get(label_str, 0))
                else:
                    label_ids.append(0)
            else:
                # 子词: 设为 -100（忽略）
                label_ids.append(-100)
            prev_word = word_id
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids),
        }


def compute_ner_metrics(eval_pred, id2label):
    """使用 seqeval 计算实体级 P/R/F1"""
    metric = evaluate.load("seqeval")
    
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    
    # 解码: id → label
    true_preds = []
    true_labels = []
    
    for pred_seq, label_seq in zip(preds, labels):
        pred_tags = []
        label_tags = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            pred_tags.append(id2label.get(int(p), "O"))
            label_tags.append(id2label.get(int(l), "O"))
        true_preds.append(pred_tags)
        true_labels.append(label_tags)
    
    results = metric.compute(predictions=true_preds, references=true_labels)
    
    # 提取关键指标
    output = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results.get("overall_accuracy", 0.0),
    }
    
    # 每类 F1
    per_type = {}
    for key, val in results.items():
        if key.startswith("overall_"):
            continue
        if isinstance(val, dict) and "f1" in val:
            per_type[key] = {
                "precision": val.get("precision", 0),
                "recall": val.get("recall", 0),
                "f1": val.get("f1", 0),
            }
    output["per_type_f1"] = per_type
    
    return output


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    dirs = get_output_dirs(config)
    
    if not config["probe_ner"]["enabled"]:
        print("[INFO] NER 探针已禁用 (config.yaml)")
        return
    
    print("=" * 60)
    print("探针任务 1：军事 NER（ND-NER）")
    print(f"GPU: {get_device_info()}")
    print("=" * 60)
    
    # 确定模型路径：优先用继续预训练后的模型
    ckpt_dir = config["model"]["output_dir"]
    if os.path.exists(os.path.join(ckpt_dir, "TRAINING_DONE")):
        model_name = ckpt_dir
        print(f"[INFO] 使用继续预训练后的模型: {model_name}")
    else:
        model_name = config["model"]["base_model"]
        print(f"[INFO] 使用基座模型（未检测到继续预训练产物）: {model_name}")
    
    # 加载 ND-NER 数据
    ner_config = config["probe_ner"]
    data_dir = ner_config["dataset_path"]
    
    if not data_dir or not os.path.exists(data_dir):
        print(f"\n[WARN] ND-NER 数据路径未配置或不存在: {data_dir}")
        print("请按以下步骤准备数据:")
        print("  1. git clone https://github.com/XinyanLi2016/ND-NER")
        print("  2. 将 data/ 目录下的 train.txt/dev.txt/test.txt 放入")
        print("     data/processed/nd_ner/ 目录")
        print("  3. 在 config.yaml 中设置 probe_ner.dataset_path")
        print("\n跳过 NER 探针。")
        return
    
    print(f"[INFO] 加载 ND-NER 数据: {data_dir}")
    splits = load_nd_ner_data(data_dir)
    
    if "train" not in splits or "test" not in splits:
        print("[ERROR] ND-NER 需要 train.txt 和 test.txt")
        return
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 标签体系
    label_list = ND_NER_LABELS  # 可用 SIMPLIFIED_LABELS_6 简化
    num_labels = len(label_list)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    
    print(f"[INFO] 标签体系: {num_labels} 个标签（{len(set(l[2:] for l in label_list if l != 'O'))} 类实体）")
    
    # 数据集
    train_dataset = NERDataset(
        splits["train"]["tokens"], splits["train"]["labels"],
        tokenizer, label_list
    )
    eval_dataset = NERDataset(
        splits["test"]["tokens"], splits["test"]["labels"],
        tokenizer, label_list
    )
    
    # 模型
    print(f"[INFO] 加载 TokenClassification 模型: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
    )
    
    # 训练参数
    output_dir = os.path.join(dirs["checkpoints"], "ner_ft")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=ner_config["fine_tune_epochs"],
        per_device_train_batch_size=ner_config["batch_size"],
        per_device_eval_batch_size=ner_config["batch_size"],
        learning_rate=ner_config["learning_rate"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=4,
    )
    
    # DataCollator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_ner_metrics(p, id2label),
    )
    
    # 训练 + 评测
    print("\n[INFO] 开始 NER 微调...")
    trainer.train()
    
    print("\n[INFO] 在测试集上评测...")
    eval_results = trainer.evaluate()
    
    # 详细评测（含 per-type F1）
    predictions = trainer.predict(eval_dataset)
    detailed = compute_ner_metrics(predictions, id2label)
    
    all_results = {
        "dataset": "ND-NER",
        "model": model_name,
        "eval_results": eval_results,
        "detailed_metrics": detailed,
    }
    
    # 打印
    print("\n" + "=" * 50)
    print("ND-NER 评测结果")
    print("=" * 50)
    print(f"  Overall Precision: {detailed['overall_precision']:.4f}")
    print(f"  Overall Recall:    {detailed['overall_recall']:.4f}")
    print(f"  Overall F1:        {detailed['overall_f1']:.4f}")
    print(f"  Overall Accuracy:  {detailed['overall_accuracy']:.4f}")
    print("\n  Per-type F1:")
    for ent_type, scores in sorted(detailed["per_type_f1"].items()):
        print(f"    {ent_type:15s}: P={scores['precision']:.4f} "
              f"R={scores['recall']:.4f} F1={scores['f1']:.4f}")
    
    # 保存
    metrics_path = os.path.join(dirs["metrics"], "probe_ner.json")
    save_metrics(all_results, metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="军事 NER 探针 (ND-NER)")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
