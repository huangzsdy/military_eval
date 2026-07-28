"""
第一层：静态质量评测
=================================
- 规模统计（文档数 / 字符数 / token 数）
- n-gram 重复率
- MinHash 去重检测
- 领域分布熵
- MAUVE 分布对比（vs 参考军事语料）
- 脱敏合规扫描

运行:
    python -m src.static_eval --config config.yaml
"""

import argparse
import sys
import os
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    load_config, load_jsonl, get_output_dirs, save_metrics, get_device_info
)


# ============================================================
# 1. 基础规模统计
# ============================================================
def compute_basic_stats(data: list[dict], tokenizer, config: dict) -> dict:
    """计算文档数、字符数、token 数、平均长度等"""
    texts = [item[config["data"]["content_field"]] for item in data]
    
    num_docs = len(texts)
    char_counts = [len(t) for t in texts]
    total_chars = sum(char_counts)
    
    # token 统计（优先用 tokenizer，不可用时用字符数估算）
    token_counts = []
    if tokenizer is not None:
        print("[INFO] 正在统计 token 数...")
        for text in tqdm(texts, desc="Tokenizing"):
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_counts.append(len(tokens))
            except Exception:
                token_counts.append(len(text))  # fallback: 字符数
    else:
        print("[INFO] 无 tokenizer，使用字符数作为 token 估算")
        token_counts = char_counts
    
    total_tokens = sum(token_counts)
    
    stats = {
        "num_documents": num_docs,
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "avg_chars_per_doc": round(total_chars / num_docs, 2) if num_docs else 0,
        "avg_tokens_per_doc": round(total_tokens / num_docs, 2) if num_docs else 0,
        "median_tokens_per_doc": int(np.median(token_counts)),
        "max_tokens_per_doc": int(max(token_counts)) if token_counts else 0,
        "min_tokens_per_doc": int(min(token_counts)) if token_counts else 0,
        "tokens_p50": int(np.percentile(token_counts, 50)),
        "tokens_p95": int(np.percentile(token_counts, 95)),
        "tokens_p99": int(np.percentile(token_counts, 99)),
    }
    return stats, token_counts


# ============================================================
# 2. n-gram 重复率
# ============================================================
def compute_ngram_repetition(texts: list[str], ngram_sizes: list[int]) -> dict:
    """
    计算文档内和文档间的 n-gram 重复率。
    参考: Lee et al. 2022 "Deduplicating Training Data Makes Language Models Better"
    """
    results = {}
    for n in ngram_sizes:
        # 文档内重复
        in_doc_reps = []
        # 文档间重复（全局 n-gram 集合）
        global_ngrams = Counter()
        doc_ngrams_list = []
        
        for text in tqdm(texts, desc=f"Computing {n}-gram"):
            chars = list(text)
            if len(chars) < n:
                continue
            ngrams = [tuple(chars[i:i+n]) for i in range(len(chars)-n+1)]
            ng_counter = Counter(ngrams)
            doc_ngrams_list.append(set(ngrams))
            global_ngrams.update(ngrams)
            
            total = len(ngrams)
            dup = sum(c - 1 for c in ng_counter.values() if c > 1)
            in_doc_reps.append(dup / total if total > 0 else 0)
        
        # 文档间重复率：重复的 ngram 占全部的比例
        dup_ngrams = sum(1 for c in global_ngrams.values() if c > 1)
        total_unique = len(global_ngrams)
        cross_doc_rate = dup_ngrams / total_unique if total_unique > 0 else 0
        
        results[f"ngram_{n}"] = {
            f"in_document_repetition_rate": round(float(np.mean(in_doc_reps)), 4),
            "cross_document_repetition_rate": round(cross_doc_rate, 4),
            "unique_ngrams": total_unique,
            "total_ngram_occurrences": sum(global_ngrams.values()),
            "most_common": [
                {"ngram": "".join(k), "count": v}
                for k, v in global_ngrams.most_common(20)
            ],
        }
        print(f"  [{n}-gram] 文档内重复率: {results[f'ngram_{n}']['in_document_repetition_rate']:.4f}, "
              f"文档间重复率: {results[f'ngram_{n}']['cross_document_repetition_rate']:.4f}")
    return results


# ============================================================
# 3. MinHash 去重
# ============================================================
def compute_minhash_dedup(texts: list[str], num_perm: int = 128, threshold: float = 0.8) -> dict:
    """
    MinHash + LSH 近重复检测。
    参考: Broder 1997 "On the resemblance and containment of documents"
    实现: datasketch MinHashLSH
    """
    from datasketch import MinHash, MinHashLSH
    
    print(f"[INFO] MinHash 去重检测 (num_perm={num_perm}, threshold={threshold})...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    unique_docs = 0
    duplicate_docs = 0
    duplicate_pairs = 0
    
    for i, text in enumerate(tqdm(texts, desc="MinHash")):
        m = MinHash(num_perm=num_perm)
        for char in text:
            m.update(char.encode("utf-8"))
        
        # 查询是否有相似文档
        result = lsh.query(m)
        if len(result) > 0:
            duplicate_docs += 1
            duplicate_pairs += len(result)
        else:
            unique_docs += 1
            lsh.insert(str(i), m)
    
    total = len(texts)
    dup_rate = duplicate_docs / total if total > 0 else 0
    
    return {
        "total_documents": total,
        "unique_documents": unique_docs,
        "near_duplicate_documents": duplicate_docs,
        "duplicate_rate": round(dup_rate, 4),
        "approx_duplicate_pairs": duplicate_pairs,
        "threshold": threshold,
        "num_permutations": num_perm,
    }


# ============================================================
# 4. 领域分布分析（基于关键词聚类）
# ============================================================
DOMAIN_KEYWORDS = {
    "战略": ["战略", "地缘政治", "国家安全", "大战略", "威慑", "联盟", "博弈"],
    "战术": ["战术", "作战", "攻防", "突击", "包围", "迂回", "伏击", "穿插"],
    "装备": ["导弹", "战斗机", "驱逐舰", "坦克", "潜艇", "雷达", "卫星", "无人机",
             "火炮", "枪械", "装甲车", "预警机", "航母", "核潜艇"],
    "后勤": ["后勤", "补给", "运输", "仓储", "维修", "保障", "油料", "弹药"],
    "情报": ["情报", "侦察", "监视", "电子战", "信号情报", "图像情报", "人力情报"],
    "训练": ["训练", "演习", "演练", "考核", "比武", "教范", "教程"],
    "条令": ["条令", "条例", "规定", "纲要", "准则", "规范", "章程"],
    "指挥": ["指挥", "控制", "通信", "协同", "OODA", "决策", "态势"],
}


def compute_domain_distribution(texts: list[str]) -> dict:
    """
    基于关键词匹配计算领域分布。
    输出类别熵值（熵越高越均衡）。
    """
    domain_counts = Counter()
    doc_domains = []
    
    for text in tqdm(texts, desc="Domain classification"):
        matched = set()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    matched.add(domain)
                    break
        if not matched:
            matched = {"其他"}
        doc_domains.append(matched)
        for d in matched:
            domain_counts[d] += 1
    
    total = len(texts)
    distribution = {
        d: {"count": c, "ratio": round(c / total, 4)}
        for d, c in domain_counts.most_common()
    }
    
    # 计算类别熵
    probs = np.array([c / total for c in domain_counts.values()])
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = float(np.log2(len(domain_counts)))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        "distribution": distribution,
        "total_docs_classified": total,
        "num_domains": len(domain_counts),
        "entropy": round(entropy, 4),
        "max_entropy": round(max_entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "interpretation": "分布均衡" if normalized_entropy > 0.8 else
                         "分布较均衡" if normalized_entropy > 0.6 else
                         "分布集中",
    }


# ============================================================
# 5. MAUVE 分布对比
# ============================================================
def compute_mauve(texts: list[str], reference_texts: list[str], config: dict) -> dict:
    """
    MAUVE 评分：量化你的语料 vs 参考军事语料的分布相似度。
    参考: Pillutla et al. NeurIPS 2021 "MAUVE: Measuring the Gap Between
          Neural Text and Human Text Distributions"
    """
    try:
        import mauve
    except ImportError:
        print("[WARN] mauve-text 未安装，跳过 MAUVE 评测。pip install mauve-text")
        return {"error": "mauve not installed", "score": None}
    
    print("[INFO] 正在计算 MAUVE 分数...")
    
    # MAUVE 需要 text + tokenize 函数
    # 对中文，我们用字符级 tokenize
    def char_tokenize(text):
        return list(text[:512])  # 截断避免过长
    
    # 降采样（MAUVE 计算量大）
    max_n = min(10000, len(texts), len(reference_texts))
    texts_sample = texts[:max_n]
    ref_sample = reference_texts[:max_n]
    
    try:
        out = mauve.compute_mauve(
            p_text=texts_sample,
            q_text=ref_sample,
            p_tokenizer=char_tokenize,
            q_tokenizer=char_tokenize,
            device_id=-1,          # 用 CPU（中文分词 CPU 即可）
            max_text_length=512,
            verbose=False,
        )
        score = float(out.mauve)
    except Exception as e:
        print(f"[WARN] MAUVE 计算失败: {e}")
        return {"error": str(e), "score": None}
    
    return {
        "mauve_score": round(score, 4),
        "num_samples": max_n,
        "interpretation": "接近参考语料" if score > 0.9 else
                         "较接近参考语料" if score > 0.7 else
                         "分布差异较大",
        "reference": config["static_eval"]["reference_corpus"],
    }


# ============================================================
# 6. 脱敏合规扫描
# ============================================================
def desensitize_scan(texts: list[str], config: dict) -> dict:
    """扫描敏感信息：番号、坐标、装备编号等"""
    if not config["static_eval"]["desensitize"]["enabled"]:
        return {"enabled": False}
    
    patterns = config["static_eval"]["desensitize"]["patterns"]
    results = Counter()
    matched_examples = {}
    
    for text in tqdm(texts, desc="Desensitize scan"):
        for pat in patterns:
            matches = re.findall(pat, text)
            if matches:
                results[pat] += len(matches)
                if pat not in matched_examples:
                    matched_examples[pat] = matches[:5]
    
    total = len(texts)
    docs_with_sensitive = 0
    for text in texts:
        for pat in patterns:
            if re.search(pat, text):
                docs_with_sensitive += 1
                break
    
    return {
        "enabled": True,
        "total_documents": total,
        "documents_with_sensitive": docs_with_sensitive,
        "sensitive_doc_ratio": round(docs_with_sensitive / total, 4) if total else 0,
        "pattern_matches": dict(results),
        "examples": matched_examples,
    }


# ============================================================
# 7. 可视化
# ============================================================
def plot_distributions(metrics: dict, output_dir: str):
    """生成可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams["font.family"] = "WenQuanYi Micro Hei"
    plt.rcParams["axes.unicode_minus"] = False
    
    # 图1: 领域分布饼图
    if "domain_distribution" in metrics:
        dom = metrics["domain_distribution"]["distribution"]
        labels = list(dom.keys())
        sizes = [v["count"] for v in dom.values()]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("军事数据集 · 领域分布")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "domain_distribution.png"), dpi=150)
        plt.close(fig)
    
    # 图2: Token 长度分布直方图
    if "basic_stats" in metrics:
        # 这里需要原始 token_counts，从 metrics 里读不到，画图在数据收集阶段做
        pass
    
    # 图3: n-gram 重复率对比
    if "ngram_repetition" in metrics:
        ng = metrics["ngram_repetition"]
        ns = list(ng.keys())
        in_doc = [v["in_document_repetition_rate"] for v in ng.values()]
        cross_doc = [v["cross_document_repetition_rate"] for v in ng.values()]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(ns))
        width = 0.35
        ax.bar([i - width/2 for i in x], in_doc, width, label="文档内重复率")
        ax.bar([i + width/2 for i in x], cross_doc, width, label="文档间重复率")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{n}" for n in ns])
        ax.set_ylabel("重复率")
        ax.set_title("n-gram 重复率")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "ngram_repetition.png"), dpi=150)
        plt.close(fig)
    
    print(f"[INFO] 图表已保存到 {output_dir}")


# ============================================================
# 主流程
# ============================================================
def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    dirs = get_output_dirs(config)
    
    print("=" * 60)
    print("第一层：静态质量评测")
    print(f"GPU 信息: {get_device_info()}")
    print("=" * 60)
    
    # 加载数据
    data = load_jsonl(
        config["data"]["path"],
        content_field=config["data"]["content_field"],
        max_docs=config["data"]["max_docs"],
    )
    texts = [item[config["data"]["content_field"]] for item in data]
    print(f"[INFO] 加载 {len(texts)} 条文档")
    
    # 加载 tokenizer（可选）
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        print(f"[INFO] 加载 tokenizer: {config['model']['tokenizer']}")
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["tokenizer"], trust_remote_code=True
        )
    except Exception as e:
        print(f"[WARN] tokenizer 加载失败 ({e})，将使用字符级统计")
    
    all_metrics = {}
    
    # 1. 基础统计
    print("\n--- 1. 基础规模统计 ---")
    basic_stats, token_counts = compute_basic_stats(data, tokenizer, config)
    all_metrics["basic_stats"] = basic_stats
    for k, v in basic_stats.items():
        print(f"  {k}: {v}")
    
    # 保存 token 长度用于画图
    token_counts_path = os.path.join(dirs["metrics"], "token_counts.json")
    save_metrics({"token_counts": token_counts}, token_counts_path)
    
    # 2. n-gram 重复率
    print("\n--- 2. n-gram 重复率 ---")
    ngram_rep = compute_ngram_repetition(texts, config["static_eval"]["ngram_sizes"])
    all_metrics["ngram_repetition"] = ngram_rep
    
    # 3. MinHash 去重
    print("\n--- 3. MinHash 去重检测 ---")
    minhash_result = compute_minhash_dedup(texts)
    all_metrics["minhash_dedup"] = minhash_result
    for k, v in minhash_result.items():
        print(f"  {k}: {v}")
    
    # 4. 领域分布
    print("\n--- 4. 领域分布分析 ---")
    domain_dist = compute_domain_distribution(texts)
    all_metrics["domain_distribution"] = domain_dist
    print(f"  熵值: {domain_dist['entropy']} / {domain_dist['max_entropy']} "
          f"(归一化: {domain_dist['normalized_entropy']})")
    print(f"  判定: {domain_dist['interpretation']}")
    for d, info in domain_dist["distribution"].items():
        print(f"    {d}: {info['count']} ({info['ratio']*100:.1f}%)")
    
    # 5. MAUVE（可选，需要参考语料）
    print("\n--- 5. MAUVE 分布对比 ---")
    ref_path = config["static_eval"].get("reference_path", "")
    if ref_path and os.path.exists(ref_path):
        # 加载参考语料
        ref_data = load_jsonl(ref_path, content_field="text", max_docs=20000)
        ref_texts = [item.get("text", "") for item in ref_data]
        mauve_result = compute_mauve(texts, ref_texts, config)
    else:
        print(f"  [INFO] 参考语料路径未配置或不存在，跳过 MAUVE。"
              f"请在 config.yaml 中设置 static_eval.reference_path")
        mauve_result = {"error": "no reference corpus", "score": None}
    all_metrics["mauve"] = mauve_result
    if mauve_result.get("score") is not None:
        print(f"  MAUVE: {mauve_result['score']} ({mauve_result['interpretation']})")
    
    # 6. 脱敏扫描
    print("\n--- 6. 脱敏合规扫描 ---")
    des_result = desensitize_scan(texts, config)
    all_metrics["desensitize"] = des_result
    if des_result.get("enabled"):
        print(f"  含敏感信息文档数: {des_result['documents_with_sensitive']} / "
              f"{des_result['total_documents']} "
              f"({des_result['sensitive_doc_ratio']*100:.2f}%)")
        for pat, cnt in des_result["pattern_matches"].items():
            print(f"    模式 '{pat}': {cnt} 处匹配")
    
    # 保存全部指标
    metrics_path = os.path.join(dirs["metrics"], "static_eval.json")
    save_metrics(all_metrics, metrics_path)
    
    # 可视化
    plot_distributions(all_metrics, dirs["figures"])
    
    print("\n" + "=" * 60)
    print("第一层评测完成！结果已保存到:")
    print(f"  - 指标: {metrics_path}")
    print(f"  - 图表: {dirs['figures']}")
    print("=" * 60)
    
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="军事数据集 · 静态质量评测")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
