"""
军事数据集评测框架 · 模块入口
"""

import argparse
import sys
from pathlib import Path

# 确保项目根在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load_config


def main():
    parser = argparse.ArgumentParser(
        description="军事数据集评测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 一键运行全部评测
  python -m src.run_all --config config.yaml
  
  # 仅运行静态评测
  python -m src.static_eval --config config.yaml
  
  # 仅运行 NER 探针
  python -m src.probe_ner --config config.yaml
  
  # 生成报告
  python -m src.report --config config.yaml
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    print("请指定要运行的模块:")
    print("  python -m src.static_eval   # 第一层：静态质量评测")
    print("  python -m src.pretrain      # 继续预训练")
    print("  python -m src.probe_ner     # 探针1：军事 NER")
    print("  python -m src.probe_event   # 探针2：事件抽取")
    print("  python -m src.probe_qa      # 探针3：军事 QA")
    print("  python -m src.report        # 生成评测报告")
    print("")
    print("或一键运行: bash scripts/run_all.sh")


if __name__ == "__main__":
    main()
