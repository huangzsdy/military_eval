"""
军事数据集评测框架 · 工具模块
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str = "config.yaml") -> dict:
    """加载并解析 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_jsonl(path: str, content_field: str = "content", max_docs: int = -1) -> list[dict]:
    """
    加载 jsonl 文件，返回 list of dict。
    每行格式: {"content": "...", ...}
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs > 0 and i >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] 第 {i+1} 行 JSON 解析失败，跳过")
                continue
            if content_field not in item:
                print(f"[WARN] 第 {i+1} 行缺少字段 '{content_field}'，跳过")
                continue
            data.append(item)
    return data


def get_output_dirs(config: dict) -> dict:
    """确保输出目录存在并返回路径字典"""
    dirs = {
        "checkpoints": config["model"]["output_dir"],
        "reports": config["report"]["output_dir"],
        "figures": config["report"]["figures_dir"],
        "metrics": config["report"]["metrics_dir"],
    }
    for name, path in dirs.items():
        Path(path).mkdir(parents=True, exist_ok=True)
    return dirs


def save_metrics(metrics: dict, path: str):
    """保存指标到 JSON 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 指标已保存: {path}")


def get_device_info() -> dict:
    """获取当前 GPU 信息（torch 不可用时优雅降级）"""
    try:
        import torch
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["total_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
        return info
    except ImportError:
        return {"cuda_available": False, "device_count": 0, "note": "torch not installed"}
