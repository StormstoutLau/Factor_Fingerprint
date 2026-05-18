# -*- coding: utf-8 -*-
"""
Factor_Fingerprint 模块演示脚本

演示因子指纹提取、分类和语义理解功能。
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'f:/Coding')

from Factor_Fingerprint import (
    FactorFingerprinter,
    AdaptiveFactorClassifier,
    FactorSemanticUnderstanding,
    extract_from_text,
)


def generate_test_data(n_periods=120, n_stocks=50, ar1_strength=0.7):
    """生成测试数据"""
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', periods=n_periods, freq='ME')
    stocks = [f'STOCK_{i:03d}' for i in range(n_stocks)]

    data = pd.DataFrame(index=dates, columns=stocks)
    for col in stocks:
        np.random.seed(ord(col[-1]))
        eps = np.random.normal(0, 1, n_periods)
        y = np.zeros(n_periods)
        for t in range(1, n_periods):
            y[t] = ar1_strength * y[t-1] + eps[t]
        data[col] = y

    return data


def demo_fingerprint():
    """演示因子指纹提取"""
    print("=" * 80)
    print("因子指纹提取演示")
    print("=" * 80)

    fingerprinter = FactorFingerprinter()

    # 生成不同AR(1)强度的因子
    test_cases = [
        ("静态因子(高AR)", 0.85),
        ("混合因子(中AR)", 0.60),
        ("动态因子(低AR)", 0.20),
    ]

    for name, ar1 in test_cases:
        data = generate_test_data(ar1_strength=ar1)
        fp = fingerprinter.extract_fingerprint(data)

        print(f"\n{name}:")
        print(f"  AR(1)中位数: {fp.ar1_median:.4f}")
        print(f"  秩自相关: {fp.rank_autocorr:.4f}")
        print(f"  半衰期: {fp.half_life:.2f}")
        print(f"  静态-动态得分: {fp.sd_score:.4f}")
        print(f"  复杂度需求: {fp.complexity_need:.4f}")


def demo_classifier():
    """演示因子分类"""
    print("\n" + "=" * 80)
    print("因子分类演示")
    print("=" * 80)

    fingerprinter = FactorFingerprinter()
    classifier = AdaptiveFactorClassifier()

    test_cases = [
        ("静态因子", 0.85),
        ("混合因子", 0.60),
        ("动态因子", 0.20),
    ]

    for name, ar1 in test_cases:
        data = generate_test_data(ar1_strength=ar1)
        fp = fingerprinter.extract_fingerprint(data)
        result = classifier.classify(fp)

        print(f"\n{name}:")
        print(f"  硬分类: {result.primary_type.value}")
        print(f"  主类型概率: {result.primary_prob:.2%}")
        if result.secondary_type:
            print(f"  次类型: {result.secondary_type.value} ({result.secondary_prob:.2%})")
        print(f"  置信度: {result.confidence:.2%}")
        print(f"  是否硬分类: {result.is_hard}")


def demo_semantic():
    """演示语义理解"""
    print("\n" + "=" * 80)
    print("因子构造语义理解演示")
    print("=" * 80)

    test_texts = [
        "动量因子定义为过去12个月扣除最近1个月后的累积收益率",
        "使用ROE的ttm值与行业平均ROE的比值作为质量因子",
        "基于20日换手率与120日平均换手率的比值构建流动性因子",
        "反转因子是过去1周的日收益率的相反数",
        "市值因子取对数市值，PB使用最新财报的账面价值除以总市值",
    ]

    for text in test_texts:
        print(f"\n文本: {text}")
        result = extract_from_text(text)

        print(f"  推断类别: {result.category or 'unknown'} "
              f"(置信度: {result.category_confidence:.2f})")
        print(f"  总体置信度: {result.overall_confidence:.2f}")
        print(f"  时间窗口: {result.lookback_windows}")
        print(f"  数据来源: {result.data_sources or '无'}")
        print(f"  语义角色: {list(result.semantic_roles.keys())}")


def demo_integration():
    """演示完整流程：语义理解 → 指纹提取 → 分类"""
    print("\n" + "=" * 80)
    print("完整流程演示：语义理解 → 指纹提取 → 分类")
    print("=" * 80)

    # 1. 语义理解
    text = "动量因子定义为过去12个月扣除最近1个月后的累积收益率"
    semantic_result = extract_from_text(text)
    print(f"\n1. 语义理解:")
    print(f"   文本: {text}")
    print(f"   推断类别: {semantic_result.category}")
    print(f"   预期稳定性: {semantic_result.knowledge_graph_matches[0]['typical_stability'] if semantic_result.knowledge_graph_matches else 'unknown'}")

    # 2. 指纹提取（模拟数据）
    data = generate_test_data(ar1_strength=0.6)
    fingerprinter = FactorFingerprinter()
    fp = fingerprinter.extract_fingerprint(data)
    print(f"\n2. 指纹提取:")
    print(f"   AR(1)中位数: {fp.ar1_median:.4f}")
    print(f"   静态-动态得分: {fp.sd_score:.4f}")

    # 3. 分类
    classifier = AdaptiveFactorClassifier()
    classification = classifier.classify(fp)
    print(f"\n3. 分类结果:")
    print(f"   硬分类: {classification['hard_class'].value}")
    print(f"   软分类: 静态={classification['soft_prob']['static']:.2%}, "
          f"混合={classification['soft_prob']['mixed']:.2%}, "
          f"动态={classification['soft_prob']['dynamic']:.2%}")

    # 4. 一致性检查
    print(f"\n4. 一致性检查:")
    semantic_category = semantic_result.category
    data_category = classification['hard_class'].value
    if semantic_category == data_category:
        print(f"   ✅ 语义先验与数据后验一致: {semantic_category}")
    else:
        print(f"   ⚠️ 不一致: 语义={semantic_category}, 数据={data_category}")
        print(f"      可能原因: 因子构造与表现不符，或数据周期不足")


if __name__ == '__main__':
    demo_fingerprint()
    demo_classifier()
    demo_semantic()
    demo_integration()

    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
