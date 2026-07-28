"""
一键运行全部评测（Python 版入口）
依次执行: 静态评测 → 继续预训练 → NER 探针 → 事件抽取探针 → QA 探针 → 报告生成

运行:
    python -m src.run_all --config config.yaml
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load_config


def run_step(name: str, module: str, config_path: str) -> bool:
    """运行一个评测步骤"""
    print("\n" + "=" * 60)
    print(f"▶ {name}")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", module, "--config", config_path],
        cwd=str(Path(__file__).parent.parent),
    )
    
    if result.returncode != 0:
        print(f"\n⚠️ {name} 执行失败 (returncode={result.returncode})")
        print("   你可以修复问题后单独重跑该步骤:")
        print(f"   python -m {module} --config {config_path}")
        return False
    else:
        print(f"\n✅ {name} 完成")
        return True


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    
    print("=" * 60)
    print("军事数据集评测框架 · 全流程")
    print(f"数据: {config['data']['path']}")
    print(f"基座: {config['model']['base_model']}")
    print("=" * 60)
    
    steps = [
        ("第一层：静态质量评测", "src.static_eval"),
        ("继续预训练", "src.pretrain"),
        ("探针1：军事 NER (ND-NER)", "src.probe_ner"),
        ("探针2：事件抽取 (CMNEE)", "src.probe_event"),
        ("探针3：军事 QA", "src.probe_qa"),
        ("生成评测报告", "src.report"),
    ]
    
    results = []
    for name, module in steps:
        ok = run_step(name, module, config_path)
        results.append((name, ok))
        
        # 继续预训练失败不致命，后续探针可用基座模型
        if not ok and module == "src.pretrain":
            print("   继续预训练跳过，后续探针将使用基座模型")
            continue
        # 探针失败也不致命
        if not ok and module.startswith("src.probe"):
            continue
        # 报告生成必须成功
        if not ok and module == "src.report":
            print("❌ 报告生成失败")
            break
    
    # 汇总
    print("\n" + "=" * 60)
    print("评测流程汇总")
    print("=" * 60)
    for name, ok in results:
        status = "✅" if ok else "⚠️"
        print(f"  {status} {name}")
    
    print("\n报告位置: output/reports/evaluation_report.md")
    print("指标位置: output/metrics/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一键运行全部评测")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
