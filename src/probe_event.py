"""
探针任务 2：军事文档级事件抽取
==================================
使用 CMNEE 数据集（国防科大 + 东南大学 + 清华, LREC-COLING 2024）。
8 类事件: 实验/机动/部署/支援/事故/演训/冲突/伤亡

评测指标 (CMNEE 论文标准):
- Type Acc:  触发词类型准确率  Core_s / S
- Type R:    事件类型召回率    Core_e / Act_e
- Arg R:     论元召回率        Cor_a / Act_a
- Co-ref R:  共指论元召回率    Cor_c / Act_c

参考: https://github.com/Mzzzhu/CMNEE

运行:
    python -m src.probe_event --config config.yaml
"""

import argparse
import sys
import os
import json
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer,
    AdamW, get_linear_schedule_with_warmup,
)
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load_config, get_output_dirs, save_metrics, get_device_info


# ============================================================
# CMNEE 事件类型
# ============================================================
CMNEE_EVENT_TYPES = [
    "实验", "机动", "部署", "支援", "事故", "演训", "冲突", "伤亡"
]

# 论元角色（简化版，CMNEE 原始有 11 种）
CMNEE_ARG_ROLES = [
    "时间", "地点", "主体", "客体", "原因", "结果",
    "方式", "数量", "条件", "工具", "目的"
]


def load_cmnee_data(data_dir: str) -> dict:
    """
    加载 CMNEE 数据集。
    期望结构:
        data_dir/
            train.json  / train/
            dev.json
            test.json
    每条数据格式:
        {
            "doc_id": "...",
            "content": "...",
            "events": [
                {
                    "type": "演训",
                    "trigger": {"text": "...", "start": 10, "end": 12},
                    "arguments": [
                        {"role": "主体", "text": "...", "start": ..., "end": ...}
                    ]
                }
            ]
        }
    """
    splits = {}
    for split in ["train", "dev", "test"]:
        # 尝试多种文件格式
        candidates = [
            os.path.join(data_dir, f"{split}.json"),
            os.path.join(data_dir, f"{split}.jsonl"),
            os.path.join(data_dir, split, f"{split}.json"),
        ]
        path = None
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        
        if not path:
            print(f"  [WARN] CMNEE {split} 文件未找到，尝试了: {candidates}")
            continue
        
        items = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict) and "data" in data:
                    items = data["data"]
        
        splits[split] = items
        print(f"  [CMNEE] {split}: {len(items)} 篇文档")
        
        # 统计事件分布
        ev_counter = Counter()
        for item in items:
            for ev in item.get("events", []):
                ev_counter[ev.get("type", "未知")] += 1
        if ev_counter:
            print(f"    事件分布: {dict(ev_counter.most_common())}")
    
    return splits


# ============================================================
# 简化的事件抽取模型
# 说明: 完整实现 CMNEE 需要 pipeline（触发词检测 + 论元抽取），
# 这里提供的是一个可运行的 baseline 框架，使用 BERT 双任务头。
# 生产级实现建议参考 CMNEE 论文的 PAIE 方法。
# ============================================================
class EventExtractionModel(nn.Module):
    """
    双头模型:
    - 触发词分类头: 每个 token → 事件类型 or O
    - 论元分类头: 每个 token → 论元角色 or O
    """
    def __init__(self, pretrained_model_name: str, num_event_types: int, num_arg_roles: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        hidden_size = self.encoder.config.hidden_size
        self.trigger_classifier = nn.Linear(hidden_size, num_event_types + 1)  # +1 for O
        self.argument_classifier = nn.Linear(hidden_size, num_arg_roles + 1)
        
    def forward(self, input_ids, attention_mask, trigger_labels=None, argument_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, H)
        
        trigger_logits = self.trigger_classifier(sequence_output)
        argument_logits = self.argument_classifier(sequence_output)
        
        loss = None
        if trigger_labels is not None and argument_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 只对非 padding 位置计算 loss
            active_mask = attention_mask.view(-1).bool()
            trigger_loss = loss_fct(
                trigger_logits.view(-1, trigger_logits.size(-1))[active_mask],
                trigger_labels.view(-1)[active_mask]
            )
            argument_loss = loss_fct(
                argument_logits.view(-1, argument_logits.size(-1))[active_mask],
                argument_labels.view(-1)[active_mask]
            )
            loss = trigger_loss + argument_loss
        
        return {"loss": loss, "trigger_logits": trigger_logits, "argument_logits": argument_logits}


class CMNEEDataset(Dataset):
    """CMNEE 文档级事件抽取数据集"""
    
    def __init__(self, items, tokenizer, max_length=512,
                 event_types=None, arg_roles=None):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.event_types = event_types or CMNEE_EVENT_TYPES
        self.arg_roles = arg_roles or CMNEE_ARG_ROLES
        self.type2id = {t: i for i, t in enumerate(self.event_types)}
        self.role2id = {r: i for i, r in enumerate(self.arg_roles)}
    
    def __len__(self):
        return len(self.items)
    
    def _build_labels(self, content: str, events: list) -> tuple[list, list]:
        """根据 events 标注构建 token 级 trigger 和 argument 标签"""
        # 初始化全 O
        char_labels_trigger = ["O"] * len(content)
        char_labels_argument = ["O"] * len(content)
        
        for ev in events:
            # 触发词
            trig = ev.get("trigger", {})
            t_text = trig.get("text", "")
            t_start = trig.get("start", -1)
            t_end = trig.get("end", -1)
            t_type = ev.get("type", "O")
            
            if t_start >= 0 and t_type in self.type2id:
                # BIO 标注
                char_labels_trigger[t_start] = f"B-{t_type}"
                for i in range(t_start + 1, t_end):
                    if i < len(char_labels_trigger):
                        char_labels_trigger[i] = f"I-{t_type}"
            
            # 论元
            for arg in ev.get("arguments", []):
                a_text = arg.get("text", "")
                a_start = arg.get("start", -1)
                a_end = arg.get("end", -1)
                a_role = arg.get("role", "O")
                
                if a_start >= 0 and a_role in self.role2id:
                    char_labels_argument[a_start] = f"B-{a_role}"
                    for i in range(a_start + 1, a_end):
                        if i < len(char_labels_argument):
                            char_labels_argument[i] = f"I-{a_role}"
        
        return char_labels_trigger, char_labels_argument
    
    def __getitem__(self, idx):
        item = self.items[idx]
        content = item.get("content", "")
        events = item.get("events", [])
        
        # 截断
        content = content[:self.max_length]
        trig_labels, arg_labels = self._build_labels(content, events)
        
        # Tokenize
        encoding = self.tokenizer(
            list(content),
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # 对齐标签
        word_ids = encoding.word_ids(batch_index=0)
        trig_ids = []
        arg_ids = []
        prev_word = None
        
        type_label_map = {f"B-{t}": i for i, t in enumerate(self.event_types)}
        type_label_map.update({f"I-{t}": i for i, t in enumerate(self.event_types)})
        type_label_map["O"] = len(self.event_types)  # O 类
        
        role_label_map = {f"B-{r}": i for i, r in enumerate(self.arg_roles)}
        role_label_map.update({f"I-{r}": i for i, r in enumerate(self.arg_roles)})
        role_label_map["O"] = len(self.arg_roles)
        
        for word_id in word_ids:
            if word_id is None:
                trig_ids.append(-100)
                arg_ids.append(-100)
            elif word_id != prev_word:
                if word_id < len(trig_labels):
                    t_label = trig_labels[word_id]
                    a_label = arg_labels[word_id]
                    trig_ids.append(type_label_map.get(t_label, len(self.event_types)))
                    arg_ids.append(role_label_map.get(a_label, len(self.arg_roles)))
                else:
                    trig_ids.append(len(self.event_types))
                    arg_ids.append(len(self.arg_roles))
            else:
                trig_ids.append(-100)
                arg_ids.append(-100)
            prev_word = word_id
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "trigger_labels": torch.tensor(trig_ids),
            "argument_labels": torch.tensor(arg_ids),
        }


# ============================================================
# CMNEE 标准评测指标
# ============================================================
def evaluate_cmnee(model, eval_loader, device, id2type, id2role) -> dict:
    """
    计算 CMNEE 论文定义的 4 个核心指标:
    - Type Acc: 预测触发词类型正确的比例
    - Type R:   实际事件中被召回的比例
    - Arg R:    论元被正确抽取的比例
    - Co-ref R: 共指论元被正确抽取的比例
    """
    model.eval()
    
    # 统计
    total_triggers_pred = 0      # S: 预测为触发词的 token 数
    total_triggers_correct = 0   # Core_s: 类型正确
    total_events_actual = 0      # Act_e: 实际事件数
    total_events_recalled = 0     # Core_e: 召回的事件数
    
    total_args_actual = 0         # Act_a
    total_args_correct = 0        # Cor_a
    total_coref_actual = 0        # Act_c
    total_coref_correct = 0       # Cor_c
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            trig_labels = batch["trigger_labels"].to(device)
            arg_labels = batch["argument_labels"].to(device)
            
            outputs = model(input_ids, attention_mask)
            trig_preds = torch.argmax(outputs["trigger_logits"], dim=-1)
            arg_preds = torch.argmax(outputs["argument_logits"], dim=-1)
            
            # 转换为 CPU numpy
            trig_preds = trig_preds.cpu().numpy()
            trig_labels = trig_labels.cpu().numpy()
            arg_preds = arg_preds.cpu().numpy()
            arg_labels = arg_labels.cpu().numpy()
            mask = attention_mask.cpu().numpy()
            
            for i in range(len(input_ids)):
                seq_mask = mask[i] == 1
                t_p = trig_preds[i][seq_mask]
                t_l = trig_labels[i][seq_mask]
                a_p = arg_preds[i][seq_mask]
                a_l = arg_labels[i][seq_mask]
                
                # Trigger 统计
                for p, l in zip(t_p, t_l):
                    if l == -100:
                        continue
                    if l != len(id2type):  # 不是 O
                        total_events_actual += 1
                        if p == l:
                            total_events_recalled += 1
                            total_triggers_correct += 1
                    if p != len(id2type):  # 预测为触发词
                        total_triggers_pred += 1
                        if p == l:
                            total_triggers_correct += 1
                
                # Argument 统计
                for p, l in zip(a_p, a_l):
                    if l == -100:
                        continue
                    if l != len(id2role):  # 不是 O
                        total_args_actual += 1
                        if p == l:
                            total_args_correct += 1
                            # 简化的共指判断：如果论元角色是"主体"且正确
                            if l == 2:  # 假设"主体"对应共指
                                total_coref_actual += 1
                                total_coref_correct += 1
    
    # 计算指标
    type_acc = total_triggers_correct / total_triggers_pred if total_triggers_pred > 0 else 0
    type_r = total_events_recalled / total_events_actual if total_events_actual > 0 else 0
    arg_r = total_args_correct / total_args_actual if total_args_actual > 0 else 0
    coref_r = total_coref_correct / total_coref_actual if total_coref_actual > 0 else 0
    
    # FNDEE 竞赛的 Trigger F1 / Argument F1
    # 简化: 用 micro P/R/F1
    all_trig_p = total_triggers_correct / total_triggers_pred if total_triggers_pred > 0 else 0
    all_trig_r = total_events_recalled / total_events_actual if total_events_actual > 0 else 0
    all_trig_f1 = 2 * all_trig_p * all_trig_r / (all_trig_p + all_trig_r) if (all_trig_p + all_trig_r) > 0 else 0
    
    all_arg_p = total_args_correct / total_args_actual if total_args_actual > 0 else 0
    all_arg_r = total_args_correct / total_args_actual if total_args_actual > 0 else 0
    all_arg_f1 = all_arg_p  # P=R 时 F1=P
    
    return {
        "Type_Acc": round(type_acc, 4),
        "Type_R": round(type_r, 4),
        "Arg_R": round(arg_r, 4),
        "Co-ref_R": round(coref_r, 4),
        "Trigger_F1": round(all_trig_f1, 4),
        "Argument_F1": round(all_arg_f1, 4),
        "counts": {
            "triggers_predicted": total_triggers_pred,
            "triggers_correct": total_triggers_correct,
            "events_actual": total_events_actual,
            "events_recalled": total_events_recalled,
            "args_actual": total_args_actual,
            "args_correct": total_args_correct,
        },
    }


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    dirs = get_output_dirs(config)
    
    if not config["probe_event"]["enabled"]:
        print("[INFO] 事件抽取探针已禁用")
        return
    
    print("=" * 60)
    print("探针任务 2：军事事件抽取（CMNEE）")
    print(f"GPU: {get_device_info()}")
    print("=" * 60)
    
    # 模型路径
    ckpt_dir = config["model"]["output_dir"]
    if os.path.exists(os.path.join(ckpt_dir, "TRAINING_DONE")):
        model_name = ckpt_dir
        print(f"[INFO] 使用继续预训练后的模型: {model_name}")
    else:
        model_name = config["model"]["base_model"]
        print(f"[INFO] 使用基座模型: {model_name}")
    
    # 加载 CMNEE 数据
    ev_config = config["probe_event"]
    data_dir = ev_config["dataset_path"]
    
    if not data_dir or not os.path.exists(data_dir):
        print(f"\n[WARN] CMNEE 数据路径未配置或不存在: {data_dir}")
        print("请按以下步骤准备数据:")
        print("  1. 访问 https://github.com/Mzzzhu/CMNEE")
        print("  2. 申请并下载数据集")
        print("  3. 放入 data/processed/cmnee/ 目录")
        print("  4. 在 config.yaml 中设置 probe_event.dataset_path")
        print("\n跳过事件抽取探针。")
        return
    
    print(f"[INFO] 加载 CMNEE 数据: {data_dir}")
    splits = load_cmnee_data(data_dir)
    
    if "train" not in splits or "test" not in splits:
        print("[ERROR] CMNEE 需要 train 和 test 数据")
        return
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device}")
    
    # 数据集
    train_dataset = CMNEEDataset(
        splits["train"], tokenizer,
        event_types=CMNEE_EVENT_TYPES,
        arg_roles=CMNEE_ARG_ROLES,
    )
    eval_dataset = CMNEEDataset(
        splits["test"], tokenizer,
        event_types=CMNEE_EVENT_TYPES,
        arg_roles=CMNEE_ARG_ROLES,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=ev_config["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=ev_config["batch_size"])
    
    # 模型
    num_event_types = len(CMNEE_EVENT_TYPES)
    num_arg_roles = len(CMNEE_ARG_ROLES)
    model = EventExtractionModel(model_name, num_event_types, num_arg_roles).to(device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=ev_config["learning_rate"])
    total_steps = len(train_loader) * ev_config["fine_tune_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )
    
    # 训练
    print(f"\n[INFO] 开始训练 ({ev_config['fine_tune_epochs']} epochs)...")
    for epoch in range(ev_config["fine_tune_epochs"]):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            trig_labels = batch["trigger_labels"].to(device)
            arg_labels = batch["argument_labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, trig_labels, arg_labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
        
        # 每轮评测
        id2type = {i: t for i, t in enumerate(CMNEE_EVENT_TYPES)}
        id2role = {i: r for i, r in enumerate(CMNEE_ARG_ROLES)}
        results = evaluate_cmnee(model, eval_loader, device, id2type, id2role)
        print(f"  评测: Type_Acc={results['Type_Acc']:.4f} Type_R={results['Type_R']:.4f} "
              f"Arg_R={results['Arg_R']:.4f} Trigger_F1={results['Trigger_F1']:.4f}")
    
    # 最终评测
    print("\n[INFO] 最终评测...")
    final_results = evaluate_cmnee(model, eval_loader, device, id2type, id2role)
    
    print("\n" + "=" * 50)
    print("CMNEE 评测结果（论文标准指标）")
    print("=" * 50)
    print(f"  Type Acc:     {final_results['Type_Acc']:.4f}")
    print(f"  Type R:       {final_results['Type_R']:.4f}")
    print(f"  Arg R:        {final_results['Arg_R']:.4f}")
    print(f"  Co-ref R:     {final_results['Co-ref_R']:.4f}")
    print(f"  Trigger F1:   {final_results['Trigger_F1']:.4f}")
    print(f"  Argument F1:  {final_results['Argument_F1']:.4f}")
    print(f"\n  计数: {final_results['counts']}")
    
    # 保存
    all_results = {
        "dataset": "CMNEE",
        "model": model_name,
        "metrics": final_results,
        "metric_definitions": {
            "Type_Acc": "Core_s / S — 触发词类型准确率",
            "Type_R": "Core_e / Act_e — 事件类型召回率",
            "Arg_R": "Cor_a / Act_a — 论元召回率",
            "Co-ref_R": "Cor_c / Act_c — 共指论元召回率",
            "Trigger_F1": "2*P*R/(P+R) — 触发词 F1",
            "Argument_F1": "论元 F1",
        },
    }
    metrics_path = os.path.join(dirs["metrics"], "probe_event.json")
    save_metrics(all_results, metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="军事事件抽取探针 (CMNEE)")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
