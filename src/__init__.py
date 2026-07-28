"""
军事数据集评测框架 · 工具模块
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Optional


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


def load_jsonl_multi_field(path: str, fields: list[str], max_docs: int = -1) -> dict[str, list[dict]]:
    """
    加载 jsonl 文件，支持多字段。
    
    Args:
        path: jsonl文件路径
        fields: 需要提取的字段列表，如 ["content", "synthesized_content_QA", "synthesized_Wikipedia-style_rephrasing"]
        max_docs: 最大文档数，-1表示全部
    
    Returns:
        dict: {字段名: [{记录}, ...], ...}
    """
    # 初始化每个字段的数据列表
    data_by_field = {field: [] for field in fields}
    
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
            
            # 为每个字段提取数据
            for field in fields:
                if field in item:
                    data_by_field[field].append(item)
                else:
                    print(f"[WARN] 第 {i+1} 行缺少字段 '{field}'")
    
    return data_by_field


def load_qa_pairs_from_field(data: list[dict], qa_field: str, qa_format: str = "json") -> list[dict]:
    """
    从QA字段加载问答对。
    
    Args:
        data: 原始数据列表
        qa_field: QA字段名
        qa_format: "json" 或 "text"
    
    Returns:
        list[dict]: QA对列表 [{"question": "...", "answer": "..."}, ...]
    """
    qa_pairs = []
    
    for item in data:
        if qa_field not in item:
            continue
            
        qa_content = item[qa_field]
        
        if qa_format == "json":
            # 尝试解析JSON格式的QA对
            try:
                qa_data = json.loads(qa_content)
                if isinstance(qa_data, list):
                    for qa in qa_data:
                        if "question" in qa and "answer" in qa:
                            qa_pairs.append({
                                "question": qa["question"],
                                "answer": qa["answer"],
                                "source_doc_id": item.get("id", "")
                            })
                elif isinstance(qa_data, dict):
                    if "question" in qa_data and "answer" in qa_data:
                        qa_pairs.append({
                            "question": qa_data["question"],
                            "answer": qa_data["answer"],
                            "source_doc_id": item.get("id", "")
                        })
            except json.JSONDecodeError:
                # 如果不是JSON，尝试文本解析
                pass
        else:
            # 文本格式: question: xxx answer: xxx
            # 暂时不支持
            pass
    
    return qa_pairs


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


def get_enabled_fields(config: dict) -> list[str]:
    """获取所有启用的字段名"""
    fields_config = config.get("fields", {})
    enabled = []
    for field_name, field_config in fields_config.items():
        if isinstance(field_config, dict) and field_config.get("enabled", False):
            enabled.append(field_name)
    return enabled
