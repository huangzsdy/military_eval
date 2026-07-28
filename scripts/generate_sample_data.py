"""
生成示例军事数据（用于测试评测框架）
实际使用时替换为你的真实数据。
"""

import json
import random
import os

random.seed(42)

# 示例军事文本片段
SAMPLE_TEXTS = [
    "中国人民解放军东部战区某部近日在台海方向组织了实战化联合演训。演训期间，多型战机梯次编队出动，与水面舰艇、岸导部队协同行动，重点演练了对海突击、制空作战和联合封控等课目。",
    "歼-20 隐身战斗机是我国自主研制的第五代制空战斗机，具备高隐身性、高态势感知和高机动性能力。该机采用双发、全动双垂尾布局，配备有源相控阵雷达和先进电子战系统。",
    "东风-21D 中程弹道导弹被誉为'航母杀手'，射程可达 1500 公里以上，具备对大型水面舰艇实施精确打击的能力。该导弹采用公路机动发射方式，提高了生存能力和反应速度。",
    "在现代联合作战中，指挥控制系统是连接各军兵种的神经中枢。C4ISR 系统集指挥、控制、通信、计算机、情报、监视和侦察于一体，是夺取信息优势的关键装备。",
    "两栖登陆作战是最复杂的军事行动之一。从诺曼底登陆到硫磺岛战役，两栖作战历来伴随着巨大风险。现代两栖作战强调'超视距登陆'，利用气垫登陆艇和直升机实施垂直包围。",
    "电子战已成为现代战争的'第五维战场'。从俄乌冲突中的 GPS 干扰到中东地区的无人机频谱对抗，电磁频谱的争夺直接影响着制空权和制海权的归属。",
    "055 型驱逐舰是我国自主设计建造的新一代万吨级大型驱逐舰，排水量超过 12000 吨。该舰装备有 112 单元垂直发射系统，可发射防空、反潜、对陆攻击等多种导弹。",
    "军事后勤保障是战争胜负的隐形之手。海湾战争中，美军每天消耗物资达 40 万吨，后勤补给线长达 12000 公里。现代战争的后勤已从'粮草先行'演变为'算法先行'。",
    "非对称作战是弱势一方对抗强势对手的典型战法。从越南战争中的丛林游击到现代的网络攻防，非对称作战的核心在于扬长避短、避实击虚。",
    "OODA 环（观察-判断-决策-行动）是军事决策的基础模型。在信息化战争中，谁能更快完成 OODA 循环，谁就能掌握战场主动权。'观察窗'的争夺已成为决定胜负的关键。",
    "特种部队在现代战争中扮演着'尖刀'角色。从海豹突击队击毙本·拉登到俄罗斯阿尔法小组的人质营救，特种作战强调精兵、快速、精确、出其不意。",
    "核威慑理论是现代战略学的核心议题。'相互确保摧毁'（MAD）理论指出，当双方都具备二次核打击能力时，核战争就变得'不可赢'，从而维持了战略稳定。",
    "无人机蜂群作战是近年来军事技术的热点。2020 年纳卡冲突中，阿塞拜疆使用土耳其制 TB-2 无人机大量摧毁亚美尼亚装甲目标，展示了'无人机主导战场'的新范式。",
    "《孙子兵法》提出'知己知彼，百战不殆'。在现代情报战中，信号情报（SIGINT）、图像情报（IMINT）和人力情报（HUMINT）构成三大支柱，为决策提供信息基础。",
    "坦克自一战索姆河首次亮相以来，始终是陆战核心装备。从 T-34 到 M1A2，坦克设计始终在火力、防护和机动三者间寻求平衡，现代主战坦克重量普遍超过 60 吨。",
]

# 扩展模板（用于生成更多数据）
EXTEND_TEMPLATES = [
    "在最近的{military_exercise}中，{unit}展示了强大的{capability}，特别是在{aspect}方面取得了突破性进展。",
    "{weapon_system}的服役标志着我军{capability}迈上新台阶。该系统的核心优势在于{advantage}，能够在{scenario}中发挥决定性作用。",
    "根据最新条令规定，{operation_type}必须遵循{principle}的基本原则。各级指挥员应当{action}，确保{objective}的实现。",
    "在{war}期间，{side}采用了{tactic}战术，取得了显著效果。战后分析表明，{key_factor}是决定胜负的关键因素。",
    "{country}的{military_branch}近期进行了{reform_type}改革，重点提升{capability}。改革后的部队在{test_scenario}中表现优异。",
    "{operation_type}是现代军队的核心能力之一。通过{action}，指挥员能够有效{objective}。",
    "在{scenario}中，{weapon_system}展现了卓越的{advantage}，为{side}赢得了战场主动权。",
]

# 词汇表
UNITS = ["某合成旅", "某陆航旅", "某潜艇支队", "某防空旅", "某特战大队", "某电子对抗团"]
WEAPONS = ["东风-17 高超音速导弹", "歼-16D 电子战机", "直-20 通用直升机", "红旗-22 防空导弹", "翼龙-3 无人机"]
CAPABILITIES = ["体系作战能力", "精确打击能力", "全域机动能力", "网络攻防能力", "联合作战能力"]
ASPECTS = ["信息融合", "指挥协同", "火力分配", "态势感知", "后勤保障"]
ADVANTAGES = ["超远射程", "高机动性", "强突防能力", "全天候作战", "智能化指挥"]
SCENARIOS = ["远海防卫", "高原山地作战", "城市巷战", "电磁对抗环境", "夜间突袭行动"]
TACTICS = ["钳形攻势", "中心开花", "蛙跳战术", "火力覆盖", "分割包围"]
PRINCIPLES = ["集中优势兵力", "速战速决", "灵活机动", "攻防兼备"]
WARS = ["海湾战争", "科索沃战争", "阿富汗战争", "叙利亚内战", "俄乌冲突"]
SIDES = ["美军", "俄军", "以军", "乌克兰军队", "阿塞拜疆军队"]
KEY_FACTORS = ["制空权", "后勤保障", "情报优势", "指挥效率", "士气"]
COUNTRIES = ["中国", "美国", "俄罗斯", "印度", "日本"]
BRANCHES = ["陆军", "海军", "空军", "火箭军", "战略支援部队"]
REFORMS = ["合成化", "模块化", "智能化", "无人化", "网络中心化"]
ACTIONS = ["强化态势感知", "优化指挥流程", "整合火力资源", "提升通信保障"]
OBJECTIVES = ["作战效能最大化", "减少附带损伤", "确保任务完成", "维护己方优势"]


def generate_text() -> str:
    """随机生成一条军事文本"""
    if random.random() < 0.5:
        return random.choice(SAMPLE_TEXTS)
    else:
        template = random.choice(EXTEND_TEMPLATES)
        return template.format(
            military_exercise=random.choice(["联合演习", "实弹演练", "跨区机动", "对抗训练"]),
            unit=random.choice(UNITS),
            capability=random.choice(CAPABILITIES),
            aspect=random.choice(ASPECTS),
            weapon_system=random.choice(WEAPONS),
            advantage=random.choice(ADVANTAGES),
            scenario=random.choice(SCENARIOS),
            operation_type=random.choice(["联合作战", "体系破击", "精确打击", "信息对抗"]),
            tactic=random.choice(TACTICS),
            principle=random.choice(PRINCIPLES),
            war=random.choice(WARS),
            side=random.choice(SIDES),
            key_factor=random.choice(KEY_FACTORS),
            country=random.choice(COUNTRIES),
            military_branch=random.choice(BRANCHES),
            reform_type=random.choice(REFORMS),
            test_scenario=random.choice(SCENARIOS),
            action=random.choice(ACTIONS),
            objective=random.choice(OBJECTIVES),
        )


def main():
    output_path = "data/raw/military_data.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    num_docs = 5000  # 生成 5000 条示例数据
    print(f"[INFO] 生成 {num_docs} 条示例军事数据 → {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(num_docs):
            text = generate_text()
            # 每条数据附带一些元数据
            item = {
                "content": text,
                "id": f"mil_doc_{i:06d}",
                "source": "synthetic_sample",  # 标记是合成的
                "domain": random.choice(["战略", "战术", "装备", "后勤", "情报", "训练", "条令", "指挥"]),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ 示例数据生成完成: {output_path}")
    print(f"   {num_docs} 条文档")
    print(f"")
    print(f"⚠️ 这是合成示例数据，仅用于测试框架流程。")
    print(f"   实际评测时请替换为你的真实军事数据。")


if __name__ == "__main__":
    main()
