"""
第三层：评测报告生成
============================
汇总所有探针结果，生成：
- Markdown 报告（人类可读）
- HTML 报告（可视化丰富）
- JSON 指标（机器可读）

运行:
    python -m src.report --config config.yaml
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load_config, get_output_dirs


def load_metrics_file(path: str) -> dict:
    """加载 JSON 指标文件"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_markdown_report(config: dict, metrics_dir: str) -> str:
    """生成 Markdown 格式报告"""
    lines = []
    lines.append("# 军事数据集有效性评估报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**数据路径**: `{config['data']['path']}`")
    lines.append(f"**基座模型**: `{config['model']['base_model']}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # ===== 第一层：静态质量 =====
    lines.append("## 第一层：静态质量评测")
    lines.append("")
    static = load_metrics_file(os.path.join(metrics_dir, "static_eval.json"))
    
    if static.get("basic_stats"):
        bs = static["basic_stats"]
        lines.append("### 1.1 基础规模统计")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 文档数 | {bs.get('num_documents', 'N/A'):,} |")
        lines.append(f"| 总字符数 | {bs.get('total_characters', 'N/A'):,} |")
        lines.append(f"| 总 Token 数 | {bs.get('total_tokens', 'N/A'):,} |")
        lines.append(f"| 平均字符/文档 | {bs.get('avg_chars_per_doc', 'N/A')} |")
        lines.append(f"| 平均 Token/文档 | {bs.get('avg_tokens_per_doc', 'N/A')} |")
        lines.append(f"| Token P50 | {bs.get('tokens_p50', 'N/A')} |")
        lines.append(f"| Token P95 | {bs.get('tokens_p95', 'N/A')} |")
        lines.append(f"| Token P99 | {bs.get('tokens_p99', 'N/A')} |")
        lines.append("")
    
    if static.get("ngram_repetition"):
        lines.append("### 1.2 n-gram 重复率")
        lines.append("")
        lines.append("| n-gram | 文档内重复率 | 文档间重复率 | 唯一 n-gram 数 |")
        lines.append("|--------|-------------|-------------|----------------|")
        for key, val in static["ngram_repetition"].items():
            n = key.replace("ngram_", "")
            lines.append(
                f"| {n} | {val['in_document_repetition_rate']:.4f} "
                f"| {val['cross_document_repetition_rate']:.4f} "
                f"| {val['unique_ngrams']:,} |"
            )
        lines.append("")
    
    if static.get("minhash_dedup"):
        md = static["minhash_dedup"]
        lines.append("### 1.3 MinHash 去重")
        lines.append("")
        lines.append(f"- 总文档数: {md.get('total_documents', 'N/A')}")
        lines.append(f"- 近重复文档数: {md.get('near_duplicate_documents', 'N/A')}")
        lines.append(f"- 重复率: {md.get('duplicate_rate', 'N/A')}")
        lines.append(f"- 近似重复对数: {md.get('approx_duplicate_pairs', 'N/A')}")
        lines.append(f"- 阈值: {md.get('threshold', 'N/A')}")
        lines.append("")
    
    if static.get("domain_distribution"):
        dd = static["domain_distribution"]
        lines.append("### 1.4 领域分布")
        lines.append("")
        lines.append(f"- 类别熵: {dd.get('entropy', 'N/A')} / {dd.get('max_entropy', 'N/A')}")
        lines.append(f"- 归一化熵: {dd.get('normalized_entropy', 'N/A')}")
        lines.append(f"- 判定: **{dd.get('interpretation', 'N/A')}**")
        lines.append("")
        lines.append("| 领域 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        for d, info in dd.get("distribution", {}).items():
            lines.append(f"| {d} | {info['count']:,} | {info['ratio']*100:.1f}% |")
        lines.append("")
    
    if static.get("mauve") and static["mauve"].get("score") is not None:
        mv = static["mauve"]
        lines.append("### 1.5 MAUVE 分布对比")
        lines.append("")
        lines.append(f"- 参考语料: {mv.get('reference', 'N/A')}")
        lines.append(f"- MAUVE 分数: **{mv.get('score', 'N/A')}**")
        lines.append(f"- 判定: {mv.get('interpretation', 'N/A')}")
        lines.append("")
    
    if static.get("desensitize") and static["desensitize"].get("enabled"):
        ds = static["desensitize"]
        lines.append("### 1.6 脱敏合规扫描")
        lines.append("")
        lines.append(f"- 含敏感信息文档: {ds.get('documents_with_sensitive', 'N/A')} / {ds.get('total_documents', 'N/A')}")
        lines.append(f"- 占比: {ds.get('sensitive_doc_ratio', 'N/A')*100:.2f}%")
        if ds.get("pattern_matches"):
            lines.append("- 匹配模式:")
            for pat, cnt in ds["pattern_matches"].items():
                lines.append(f"  - `{pat}`: {cnt} 处")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # ===== 第二层：探针任务 =====
    lines.append("## 第二层：下游任务探针评测")
    lines.append("")
    
    # NER
    ner = load_metrics_file(os.path.join(metrics_dir, "probe_ner.json"))
    if ner.get("detailed_metrics"):
        dm = ner["detailed_metrics"]
        lines.append("### 2.1 军事 NER（ND-NER 数据集）")
        lines.append("")
        lines.append(f"**模型**: `{ner.get('model', 'N/A')}`")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| Precision | {dm.get('overall_precision', 'N/A'):.4f} |")
        lines.append(f"| Recall | {dm.get('overall_recall', 'N/A'):.4f} |")
        lines.append(f"| **F1** | **{dm.get('overall_f1', 'N/A'):.4f}** |")
        lines.append(f"| Accuracy | {dm.get('overall_accuracy', 'N/A'):.4f} |")
        lines.append("")
        
        if dm.get("per_type_f1"):
            lines.append("**Per-type F1:**")
            lines.append("")
            lines.append("| 实体类型 | Precision | Recall | F1 |")
            lines.append("|----------|-----------|--------|-----|")
            for ent, sc in sorted(dm["per_type_f1"].items()):
                lines.append(f"| {ent} | {sc['precision']:.4f} | {sc['recall']:.4f} | {sc['f1']:.4f} |")
            lines.append("")
    
    # Event Extraction
    ev = load_metrics_file(os.path.join(metrics_dir, "probe_event.json"))
    if ev.get("metrics"):
        m = ev["metrics"]
        lines.append("### 2.2 军事事件抽取（CMNEE 数据集）")
        lines.append("")
        lines.append(f"**模型**: `{ev.get('model', 'N/A')}`")
        lines.append("")
        lines.append("| 指标 | 数值 | 定义 |")
        lines.append("|------|------|------|")
        lines.append(f"| Type Acc | {m.get('Type_Acc', 'N/A'):.4f} | 触发词类型准确率 |")
        lines.append(f"| Type R | {m.get('Type_R', 'N/A'):.4f} | 事件类型召回率 |")
        lines.append(f"| Arg R | {m.get('Arg_R', 'N/A'):.4f} | 论元召回率 |")
        lines.append(f"| Co-ref R | {m.get('Co-ref_R', 'N/A'):.4f} | 共指论元召回率 |")
        lines.append(f"| **Trigger F1** | **{m.get('Trigger_F1', 'N/A'):.4f}** | 触发词 F1 |")
        lines.append(f"| Argument F1 | {m.get('Argument_F1', 'N/A'):.4f} | 论元 F1 |")
        lines.append("")
    
    # QA
    qa = load_metrics_file(os.path.join(metrics_dir, "probe_qa.json"))
    if qa.get("auto_metrics"):
        am = qa["auto_metrics"]
        js = qa.get("judge_scores", {})
        lines.append("### 2.3 军事问答（QA）")
        lines.append("")
        lines.append(f"**模型**: `{qa.get('model', 'N/A')}`")
        lines.append(f"**裁判模型**: `{qa.get('judge_model', 'N/A')}`")
        lines.append(f"**问题数**: {qa.get('num_questions', 'N/A')}")
        lines.append("")
        lines.append("**自动指标:**")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| ROUGE-1 | {am.get('ROUGE-1', 'N/A')} |")
        lines.append(f"| ROUGE-2 | {am.get('ROUGE-2', 'N/A')} |")
        lines.append(f"| ROUGE-L | {am.get('ROUGE-L', 'N/A')} |")
        lines.append(f"| BERTScore F1 | {am.get('BERTScore_F1', 'N/A')} |")
        lines.append("")
        lines.append("**LLM-as-Judge 评分:**")
        lines.append("")
        lines.append("| 维度 | 平均分 |")
        lines.append("|------|--------|")
        for k in ["accuracy", "completeness", "relevance", "military_expertise", "overall"]:
            if k in js:
                lines.append(f"| {k} | {js[k]} |")
        lines.append("")
        
        if qa.get("by_question_type"):
            lines.append("**按问题类型:**")
            lines.append("")
            lines.append("| 类型 | ROUGE-1 | ROUGE-L | BERTScore F1 |")
            lines.append("|------|---------|---------|---------------|")
            for t, tm in qa["by_question_type"].items():
                lines.append(f"| {t} | {tm.get('ROUGE-1', 'N/A')} | {tm.get('ROUGE-L', 'N/A')} | {tm.get('BERTScore_F1', 'N/A')} |")
            lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # ===== 第三层：消融对比 =====
    lines.append("## 第三层：训练收益量化（消融实验）")
    lines.append("")
    lines.append("> ⚠️ 此表需手动填入 A 组（基线）和 B 组（你的数据）的对比结果")
    lines.append("")
    lines.append("| 任务 | 指标 | A组（基线） | B组（你的数据） | 提升幅度 |")
    lines.append("|------|------|------------|----------------|----------|")
    lines.append("| 军事 NER | F1 | TBD | TBD | TBD |")
    lines.append("| 事件抽取 | Trigger F1 | TBD | TBD | TBD |")
    lines.append("| 事件抽取 | Argument F1 | TBD | TBD | TBD |")
    lines.append("| 军事 QA | ROUGE-L | TBD | TBD | TBD |")
    lines.append("| 军事 QA | BERTScore F1 | TBD | TBD | TBD |")
    lines.append("| 军事 QA | Judge Overall | TBD | TBD | TBD |")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # ===== 合规声明 =====
    lines.append("## 合规与脱敏声明")
    lines.append("")
    lines.append("- **数据来源**: [请填写]")
    lines.append("- **数据分级**: 内部级 / 公开级")
    lines.append("- **脱敏处理**: 已通过正则 + NER 扫描")
    lines.append("- **使用权限**: 仅限内部团队模型训练")
    lines.append("- **禁止行为**: 禁止用于自主武器、火力打击等有害场景")
    lines.append("- **许可协议**: [请填写，建议 CC BY-NC 4.0 或自定义研究协议]")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## 引用")
    lines.append("")
    lines.append("```bibtex")
    lines.append("@dataset{military_corpus_2025,")
    lines.append("  title={中文军事数据集有效性评估报告},")
    lines.append("  author={[请填写]},")
    lines.append("  year={2025},")
    lines.append("  publisher={内部团队}")
    lines.append("}")
    lines.append("```")
    lines.append("")
    
    return "\n".join(lines)


def generate_html_report(markdown_content: str, output_path: str):
    """
    将 Markdown 转为简单 HTML 报告。
    实际部署建议用 mkdocs 或 pandoc 生成更美观的版本。
    """
    # 简单实现：直接嵌入 markdown 文本 + 基础样式
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>军事数据集有效性评估报告</title>
<style>
body {{
    font-family: 'WenQuanYi Micro Hei', 'Microsoft YaHei', sans-serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
    line-height: 1.8;
    color: #333;
}}
h1, h2, h3 {{ color: #1a1a2e; }}
h1 {{ border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
h2 {{ border-left: 4px solid #0f3460; padding-left: 12px; margin-top: 40px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: left; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
pre {{ background: #f5f5f5; padding: 16px; border-radius: 6px; overflow-x: auto; }}
blockquote {{ border-left: 4px solid #ffc107; background: #fff9e6; padding: 10px 16px; }}
</style>
</head>
<body>
<pre>{markdown_content}</pre>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    dirs = get_output_dirs(config)
    metrics_dir = dirs["metrics"]
    
    print("=" * 60)
    print("第三层：生成评测报告")
    print("=" * 60)
    
    # 生成 Markdown
    md_content = generate_markdown_report(config, metrics_dir)
    
    # 保存 Markdown
    md_path = os.path.join(dirs["reports"], "evaluation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[INFO] Markdown 报告: {md_path}")
    
    # 保存 HTML
    html_path = os.path.join(dirs["reports"], "evaluation_report.html")
    generate_html_report(md_content, html_path)
    print(f"[INFO] HTML 报告: {html_path}")
    
    # 汇总 JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_path": config["data"]["path"],
        "base_model": config["model"]["base_model"],
        "static_eval": load_metrics_file(os.path.join(metrics_dir, "static_eval.json")),
        "probe_ner": load_metrics_file(os.path.join(metrics_dir, "probe_ner.json")),
        "probe_event": load_metrics_file(os.path.join(metrics_dir, "probe_event.json")),
        "probe_qa": load_metrics_file(os.path.join(metrics_dir, "probe_qa.json")),
    }
    summary_path = os.path.join(metrics_dir, "full_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 汇总 JSON: {summary_path}")
    
    print("\n" + "=" * 60)
    print("✅ 评测报告生成完成！")
    print(f"   Markdown: {md_path}")
    print(f"   HTML:     {html_path}")
    print(f"   JSON:     {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成评测报告")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
