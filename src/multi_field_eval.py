"""
多字段评测入口
===================================
针对数据集中的不同字段，使用不同的评测策略。

支持的字段类型:
- content: 原始文本 → 完整评测（静态+NER+事件+QA）
- synthesized_content_QA: QA对 → QA专项评测
- synthesized_Wikipedia-style_rephrasing: Wiki改写 → 质量评测

运行:
    python -m src.multi_field_eval --config config.yaml
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    load_config, load_jsonl, load_jsonl_multi_field, 
    get_output_dirs, save_metrics, get_device_info, get_enabled_fields
)
from src.static_eval import (
    compute_basic_stats, compute_ngram_repetition, compute_minhash_dedup,
    compute_domain_distribution, compute_mauve, desensitize_scan
)


def eval_single_field(field_name: str, field_data: list[dict], config: dict, dirs: dict) -> dict:
    """
    对单个字段进行评测
    
    Args:
        field_name: 字段名
        field_data: 该字段的数据列表
        config: 配置
        dirs: 输出目录
    
    Returns:
        dict: 评测结果
    """
    print(f"\n{'='*60}")
    print(f"评测字段: {field_name}")
    print(f"数据量: {len(field_data)} 条")
    print(f"{'='*60}")
    
    # 获取字段配置
    field_config = config.get("fields", {}).get(field_name, {})
    
    # 提取文本内容
    texts = []
    for item in field_data:
        if field_name in item:
            texts.append(item[field_name])
    
    print(f"[INFO] 提取到 {len(texts)} 条文本")
    
    results = {
        "field_name": field_name,
        "description": field_config.get("description", ""),
        "num_documents": len(texts),
    }
    
    # ===== 1. 静态质量评测 =====
    if field_config.get("static_eval", True):
        print(f"\n--- {field_name}: 静态质量评测 ---")
        
        # 1.1 基础统计
        basic_stats, token_counts = compute_basic_stats(field_data, None, config)
        results["basic_stats"] = basic_stats
        print(f"  文档数: {basic_stats['num_documents']}")
        
        # 1.2 n-gram 重复率
        ngram_rep = compute_ngram_repetition(texts, config["static_eval"]["ngram_sizes"])
        results["ngram_repetition"] = ngram_rep
        
        # 1.3 MinHash 去重
        minhash_result = compute_minhash_dedup(texts)
        results["minhash_dedup"] = minhash_result
        print(f"  重复率: {minhash_result['duplicate_rate']:.4f}")
        
        # 1.4 领域分布（仅对content字段）
        if field_name == "content":
            domain_dist = compute_domain_distribution(texts)
            results["domain_distribution"] = domain_dist
            print(f"  领域熵: {domain_dist['normalized_entropy']:.4f}")
        
        # 1.5 MAUVE（如果有参考语料）
        # 暂时跳过，需要配置reference_path
        results["mauve"] = {"note": "需要配置reference_path进行MAUVE对比"}
        
        # 1.6 脱敏扫描
        des_result = desensitize_scan(texts, config)
        results["desensitize"] = des_result
    
    # ===== 2. 字段特定的专项评测 =====
    probe_tasks = field_config.get("probe_tasks", [])
    
    if "qa_direct" in probe_tasks:
        # QA字段专项评测
        print(f"\n--- {field_name}: QA专项评测 ---")
        qa_format = field_config.get("qa_format", "json")
        from src import load_qa_pairs_from_field
        qa_pairs = load_qa_pairs_from_field(field_data, field_name, qa_format)
        results["qa_evaluation"] = {
            "num_qa_pairs": len(qa_pairs),
            "note": "QA评测需要调用 probe_qa 模块进行生成和评估"
        }
        print(f"  QA对数量: {len(qa_pairs)}")
    
    if "quality" in probe_tasks or "coherence" in probe_tasks:
        # Wiki改写质量评测
        print(f"\n--- {field_name}: 改写质量评测 ---")
        results["quality_evaluation"] = {
            "note": "改写质量评测需要使用LLM进行评估"
        }
    
    if field_name == "content":
        # Content字段的完整评测（包括探针）
        print(f"\n--- {field_name}: 完整评测（需要模型训练）---")
        results["probe_note"] = "NER/事件抽取/QA探针需要运行 python -m src.probe_ner 等命令"
    
    return results


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    dirs = get_output_dirs(config)
    
    print("=" * 60)
    print("多字段评测系统")
    print(f"GPU 信息: {get_device_info()}")
    print("=" * 60)
    
    # 获取启用的字段
    enabled_fields = get_enabled_fields(config)
    print(f"\n[INFO] 启用的字段: {enabled_fields}")
    
    if not enabled_fields:
        print("[ERROR] 没有启用的字段，请检查 config.yaml 中的 fields 配置")
        return
    
    # 加载数据（一次加载所有数据）
    data_path = config["data"]["path"]
    max_docs = config["data"]["max_docs"]
    
    # 获取所有需要的字段
    all_results = {}
    
    for field_name in enabled_fields:
        field_config = config.get("fields", {}).get(field_name, {})
        
        # 加载该字段的数据
        field_data = load_jsonl(
            data_path, 
            content_field=field_name, 
            max_docs=max_docs
        )
        
        if not field_data:
            print(f"[WARN] 字段 '{field_name}' 没有数据，跳过")
            continue
        
        # 评测该字段
        field_results = eval_single_field(field_name, field_data, config, dirs)
        all_results[field_name] = field_results
    
    # 保存结果
    metrics_path = os.path.join(dirs["metrics"], "multi_field_eval.json")
    save_metrics(all_results, metrics_path)
    
    print("\n" + "=" * 60)
    print("多字段评测完成！")
    print(f"结果保存到: {metrics_path}")
    print("=" * 60)
    
    # 打印摘要
    print("\n评测摘要:")
    print("-" * 40)
    for field_name, results in all_results.items():
        print(f"\n【{field_name}】")
        print(f"  文档数: {results.get('num_documents', 'N/A')}")
        if "basic_stats" in results:
            bs = results["basic_stats"]
            print(f"  Token数: {bs.get('total_tokens', 'N/A')}")
        if "minhash_dedup" in results:
            mr = results["minhash_dedup"]
            print(f"  重复率: {mr.get('duplicate_rate', 'N/A'):.4f}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多字段评测系统")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
