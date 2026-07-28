"""
继续预训练脚本
=================================
使用你的军事 jsonl 数据继续预训练基座模型。
支持 DeepSpeed + LoRA（显存不足时）或全参数训练。

运行:
    python -m src.pretrain --config config.yaml
    # 或
    deepspeed --num_gpus=4 -m src.pretrain --config config.yaml
"""

import argparse
import sys
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load_config, load_jsonl, get_output_dirs, get_device_info


class MilitaryTextDataset(Dataset):
    """将 jsonl 的 content 字段转为 causal LM 训练样本"""
    
    def __init__(self, data: list[dict], tokenizer, max_length: int, content_field: str):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [item[content_field] for item in data if content_field in item]
        print(f"[INFO] 构建数据集: {len(self.texts)} 条文本")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 截断过长文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        # Causal LM: labels = input_ids（padding 位置设为 -100）
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    dirs = get_output_dirs(config)
    
    print("=" * 60)
    print("继续预训练")
    print(f"GPU: {get_device_info()}")
    print("=" * 60)
    
    # 加载数据
    data = load_jsonl(
        config["data"]["path"],
        content_field=config["data"]["content_field"],
        max_docs=config["data"]["max_docs"],
    )
    print(f"[INFO] 加载 {len(data)} 条文档用于继续预训练")
    
    # 加载 tokenizer 和模型
    model_name = config["model"]["base_model"]
    print(f"[INFO] 加载基座模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.bfloat16 if config["model"]["dtype"] == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    
    # LoRA 配置
    if config["model"]["use_lora"]:
        lora_config = LoraConfig(
            r=config["model"]["lora_rank"],
            lora_alpha=config["model"]["lora_alpha"],
            lora_dropout=config["model"]["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 数据集
    train_dataset = MilitaryTextDataset(
        data, tokenizer,
        max_length=config["model"]["max_seq_length"],
        content_field=config["data"]["content_field"],
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM
    )
    
    # 训练参数
    output_dir = config["model"]["output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["model"]["num_train_epochs"],
        per_device_train_batch_size=config["model"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["model"]["gradient_accumulation_steps"],
        learning_rate=config["model"]["learning_rate"],
        warmup_ratio=config["model"]["warmup_ratio"],
        logging_steps=config["model"]["logging_steps"],
        save_steps=config["model"]["save_steps"],
        save_total_limit=3,
        bf16=(config["model"]["dtype"] == "bfloat16"),
        fp16=(config["model"]["dtype"] == "float16"),
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\n[INFO] 开始继续预训练...")
    trainer.train()
    
    # 保存
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n[INFO] 模型已保存到 {output_dir}")
    
    # 标记训练完成
    done_flag = os.path.join(output_dir, "TRAINING_DONE")
    with open(done_flag, "w") as f:
        f.write("ok")
    print(f"[INFO] 训练完成标记: {done_flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="军事数据继续预训练")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
