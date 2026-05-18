# Factor Fingerprint

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT">
  <img src="https://img.shields.io/badge/Status-Stable-brightgreen?style=for-the-badge" alt="Status: Stable">
</p>

**因子指纹与自适应分类系统** —— 基于时序稳定性、截面稳定性与语义理解，为量化因子生成唯一"指纹"，实现因子类型的智能识别与差异化处理。

> **核心能力**：指纹提取 → 自适应分类 → 语义-统计融合 → 迁移监测

---

## 目录

- [核心特性](#核心特性)
- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [模块详解](#模块详解)
- [学术依据](#学术依据)
- [与 Factor_Decoupler 的协同](#与-factor_decoupler-的协同)
- [API 参考](#api-参考)
- [文件结构](#文件结构)
- [版本信息](#版本信息)

---

## 核心特性

### 1. 多维度因子指纹提取

| 维度 | 指标 | 说明 |
|------|------|------|
| **时序稳定性** | AR(1)中位数、秩自相关、半衰期 | 衡量因子排序的持续性 |
| **截面稳定性** | 偏度/峰度标准差、JS散度 | 衡量截面分布的稳定性 |
| **综合衍生** | 静态-动态得分、复杂度需求 | 综合判断因子类型 |

### 2. 自适应分类系统

```
AR(1) > 0.80  →  STATIC (静态因子：价值、盈利、质量)
AR(1) < 0.40  →  DYNAMIC (动态因子：反转、波动率变化)
0.40 ~ 0.80   →  MIXED (混合因子：动量)
```

支持**硬分类**（阈值判断）与**软分类**（sigmoid概率加权）两种模式。

### 3. 语义-统计融合

- **冷启动信任语义**：数据不足时，语义分析是唯一可靠依据
- **数据充足信任统计**：统计指纹反映因子本质
- **冲突时保守降级**：语义与统计不一致时，采用最保守的混合管道

### 4. 迁移监测

持续监测因子指纹变化，当因子类型发生迁移时发出预警，支持软过渡权重机制。

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    FactorFingerprinter                       │
│                    (因子指纹提取器)                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ 时序稳定性   │  │ 截面稳定性   │  │ 综合衍生指标         │ │
│  │ AR(1)       │  │ 偏度/峰度   │  │ 静态-动态得分        │ │
│  │ 秩自相关     │  │ JS散度      │  │ 复杂度需求           │ │
│  │ 半衰期      │  │ 覆盖率      │  │                     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         └─────────────────┼────────────────────┘            │
│                           ↓                                  │
│              ┌─────────────────────────┐                     │
│              │   FactorFingerprint      │                     │
│              │   (指纹向量)              │                     │
│              └────────────┬────────────┘                     │
└───────────────────────────┼─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 AdaptiveFactorClassifier                     │
│                 (自适应分类器)                                 │
├─────────────────────────────────────────────────────────────┤
│  硬分类: 阈值判断 → STATIC / DYNAMIC / MIXED                │
│  软分类: sigmoid概率 → 静态概率 / 混合概率 / 动态概率          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              SemanticStatisticalFusion                       │
│              (语义-统计融合)                                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ SemanticPrior│  │ 统计指纹     │  │ ConflictArbitrator │ │
│  │ (语义先验)   │  │ (数据后验)   │  │ (冲突仲裁)          │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         └─────────────────┼────────────────────┘            │
│                           ↓                                  │
│              ┌─────────────────────────┐                     │
│              │   ArbitratedResult       │                     │
│              │   (融合分类结果)          │                     │
│              └─────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/StormstoutLau/factor_pipeline.git
cd factor_pipeline/Factor_Fingerprint

# 安装依赖
pip install numpy pandas scipy jieba
```

### 基础用法

```python
import pandas as pd
from Factor_Fingerprint import (
    FactorFingerprinter,
    AdaptiveFactorClassifier,
    FactorType,
)

# 准备因子数据 (T x N): 时间 x 股票
factor_data = pd.DataFrame(...)

# 1. 提取指纹
fingerprinter = FactorFingerprinter()
fingerprint = fingerprinter.extract_fingerprint(factor_data)

print(f"AR(1)中位数: {fingerprint.ar1_median:.4f}")
print(f"静态-动态得分: {fingerprint.sd_score:.4f}")

# 2. 分类
classifier = AdaptiveFactorClassifier()
result = classifier.classify(fingerprint)

print(f"硬分类: {result['hard_class'].value}")
print(f"软分类概率: 静态={result['soft_prob']['static']:.2%}")
```

### 语义-统计融合

```python
from Factor_Fingerprint import (
    SemanticStatisticalFusion,
    SemanticPrior,
)

# 构造语义先验
semantic = SemanticPrior.from_description(
    name="动量因子",
    description="过去12个月扣除最近1个月后的累积收益率",
    lookback_windows=[12, 1],
)

# 融合分类
fusion = SemanticStatisticalFusion()
result = fusion.classify(
    fingerprint=fingerprint,
    semantic_prior=semantic,
    data_coverage=0.8,  # 数据覆盖率
)

print(f"融合分类: {result.final_type.value}")
print(f"置信度: {result.confidence:.2%}")
```

### 运行演示

```bash
python demo.py
```

---

## 模块详解

### core/fingerprint.py — 因子指纹提取

| 类/函数 | 说明 |
|---------|------|
| `FactorFingerprinter` | 指纹提取器主类 |
| `FactorFingerprint` | 指纹数据命名元组 |
| `FactorType` | 因子类型枚举 (STATIC/DYNAMIC/MIXED) |
| `FingerprintConfig` | 指纹提取配置 |

**指纹指标说明**：

| 指标 | 类型 | 说明 | 静态因子典型值 | 动态因子典型值 |
|------|------|------|--------------|--------------|
| `ar1_median` | 时序 | AR(1)系数中位数 | > 0.80 | < 0.40 |
| `rank_autocorr` | 时序 | 截面秩自相关 | > 0.95 | < 0.80 |
| `half_life` | 时序 | 自相关系数半衰期 | > 10期 | < 3期 |
| `skewness_std` | 截面 | 偏度标准差 | < 0.05 | > 0.15 |
| `js_divergence_mean` | 截面 | JS散度均值 | < 0.05 | > 0.15 |
| `sd_score` | 综合 | 静态-动态得分 | > 0.70 | < 0.30 |

**阈值设计依据**：

| 指标 | 阈值 | 依据类型 | 理论基础 |
|------|------|---------|---------|
| **AR(1)** | > 0.80 / < 0.40 | **核心分类依据** | Box-Jenkins 时间序列分析：0.80 为"强自相关"临界值，0.40 为"弱自相关"临界值 |
| **秩自相关** | > 0.95 / < 0.80 | 辅助验证 | Spearman 秩相关：> 0.95 表示截面排序几乎不变，< 0.80 表示排序变化显著 |
| **半衰期** | > 10期 / < 3期 | 辅助指标 | 信息衰减理论：自相关衰减至 0.5 所需的滞后阶数，10期以上表示极度持久 |
| **偏度标准差** | < 0.05 / > 0.15 | 截面稳定性 | 分布形态稳定性：各期偏度波动小表示截面分布稳定 |
| **JS散度** | < 0.05 / > 0.15 | 截面稳定性 | Jensen-Shannon 信息论：相邻两期截面分布的相似度，0.05 以下几乎相同 |
| **SD得分** | > 0.70 / < 0.30 | 综合指标 | 加权合成：AR(1) 35% + 秩自相关 30% + 半衰期 20% + 水平/差分 IC 比 15% |

**关键设计要点**：

1. **AR(1) 是唯一硬分类依据**：代码中硬分类仅使用 AR(1) 阈值（0.80 / 0.40），其他指标用于软分类概率和综合评估
2. **阈值具有经验性**：基于金融因子典型特征设定，非严格统计检验临界值
3. **完全可配置**：所有阈值可通过 `ClassificationConfig` 调整，适应不同市场

**金融直觉解释**：

- **静态因子**（PB、ROE）：公司基本面变化缓慢，AR(1) 通常 > 0.80，截面排序稳定（秩自相关 > 0.95）
- **动态因子**（反转、换手率变化）：信号衰减快，AR(1) 通常 < 0.40，截面排序变化大（秩自相关 < 0.80）
- **混合因子**（动量）：介于两者之间，0.40 ~ 0.80

### core/classifier.py — 自适应分类

| 类 | 说明 |
|----|------|
| `AdaptiveFactorClassifier` | 自适应分类器 |
| `ClassificationConfig` | 分类阈值配置 |
| `ClassificationResult` | 分类结果 |

**分类阈值**（可配置）：

```python
config = ClassificationConfig(
    static_ar1_threshold=0.80,   # AR(1) > 0.80 → 静态
    dynamic_ar1_threshold=0.40,  # AR(1) < 0.40 → 动态
    soft_boundary=True,          # 启用软边界
    sigmoid_steepness=10.0,      # sigmoid陡峭度
)
```

### core/semantic.py — 语义理解

| 类/函数 | 说明 |
|---------|------|
| `FactorSemanticUnderstanding` | 语义理解系统主类 |
| `FinancialTokenizer` | 金融增强分词器 |
| `FinancialKnowledgeGraph` | 金融知识图谱 |
| `SemanticMatcher` | 语义相似度匹配器 |
| `extract_from_text` | 从文本提取因子元数据 |

**支持的语义分析**：
- 因子名称识别（动量、价值、质量等）
- 时间窗口提取（"过去12个月"、"20日"）
- 数据来源识别（财报、行情、另类数据）
- 构造方式分类（累积收益、比值、排名等）

### core/semantic_fusion.py — 语义-统计融合

| 类 | 说明 |
|----|------|
| `SemanticStatisticalFusion` | 融合分类主类 |
| `SemanticPrior` | 语义先验分布 |
| `BayesianFactorClassifier` | 贝叶斯分类器 |
| `ConflictArbitrator` | 冲突仲裁引擎 |
| `ArbitratedResult` | 仲裁结果 |

**冲突处理策略**：

| 数据覆盖率 | 信任偏向 | 冲突处理方式 |
|-----------|---------|------------|
| < 30% | 语义优先 | 采用语义分类，标记低置信度 |
| 30% ~ 70% | 加权融合 | 贝叶斯后验 = 先验 × 似然 |
| > 70% | 统计优先 | 采用统计分类，语义辅助验证 |

### core/monitor.py — 迁移监测

| 类 | 说明 |
|----|------|
| `FactorFingerprintMonitor` | 指纹监测器 |
| `MonitorConfig` | 监测配置 |
| `MigrationAlert` | 迁移预警 |

---

## 学术依据

### 经典基础

- **Barra 多因子模型** — 因子分类与风险分解的标准框架
- **Carhart (1997)** — 四因子模型中的动量因子定义
- **Fama-French (1993, 2015)** — 价值、规模、盈利、投资因子的经典定义

### 前沿方法关联

| 方法 | 代表文献 | 与当前实现的关联 |
|------|---------|----------------|
| **因子模型** | Bai & Ng (2002), *Econometrica* | 指纹指标可扩展至因子载荷分析 |
| **交互固定效应** | Bai (2009), *Econometrica* | 截面稳定性分析的理论基础 |
| **文本分析** | Loughran & McDonald (2011), *JFE* | 金融文本分析的词典方法 |
| **知识图谱** | 金融语义网络 | 因子构造规则的图表示 |

---

## 与 Factor_Decoupler 的协同

Factor_Fingerprint 与 Factor_Decoupler 构成完整的因子处理流水线：

```
Factor_Fingerprint                Factor_Decoupler
─────────────────                ─────────────────
提取指纹 → 分类因子                    差异化处理
    STATIC  ─────────────────────→  跳过解耦（保留截面排序）
    DYNAMIC ─────────────────────→  三重中性化（提取新息）
    MIXED   ─────────────────────→  条件性解耦
```

详细协同设计参见 [factor_pipeline/pipelines_v2.py](../factor_pipeline/pipelines_v2.py)。

---

## 与 Factor_Imputer 的区别

### 定位差异

| 维度 | Factor_Imputer_v2.0 | Factor_Fingerprint |
|------|---------------------|-------------------|
| **核心目标** | 选择最优**插补策略** | 选择最优**解耦策略** |
| **类型体系** | 业务语义（财务/估值/技术/宏观） | 投资逻辑（静态/动态/混合） |
| **检测依据** | 分布形态 + 缺失模式 | 时序持续性 + 截面稳定性 |
| **理论基础** | 启发式规则 | 时间序列分析 + 信息论 |
| **输出用途** | 插补方法参数（分组方式、权重） | 解耦/中性化决策 |
| **语义融合** | 关键词匹配（计划中） | 完整的语义-统计贝叶斯融合 |
| **迁移监测** | 无 | 有（持续监测类型变化） |

### 类型体系对比

**Factor_Imputer 的业务分类**（用于插补）：
- `financial` — 财务因子（ROE、营收等）→ 行业分组中位数插补
- `valuation` — 估值因子（PE、PB 等）→ 行业分组中位数插补
- `growth` — 成长因子（增长率）→ KNN/滚动插补
- `technical` — 技术因子（MACD、RSI）→ 前向填充
- `quality` — 质量因子 → 行业分组插补
- `risk` — 风险因子 → 市值分组插补
- `macro` — 宏观因子 → 前向填充/滚动窗口

**Factor_Fingerprint 的投资逻辑分类**（用于解耦）：
- `STATIC` — 静态因子（PB、ROE）→ **跳过解耦**，保留截面排序
- `DYNAMIC` — 动态因子（反转、换手率变化）→ **三重中性化**，提取时序新息
- `MIXED` — 混合因子（动量）→ **条件性解耦**

### 互补关系

```
数据输入
    ↓
[Factor_Imputer]  ← 业务分类（财务/估值/技术）
    ↓
缺失值插补（截面中位数/ffill/滚动窗口）
    ↓
[Factor_Fingerprint]  ← 投资逻辑分类（静态/动态/混合）
    ↓
解耦决策（跳过/三重中性化/条件解耦）
    ↓
纯净信号输出
```

**类比**：
- Factor_Imputer 的因子类型 = **医院科室分类**（内科/外科/儿科）→ 决定治疗方式（插补方法）
- Factor_Fingerprint 的因子类型 = **疾病性质分类**（急性/慢性/混合型）→ 决定用药策略（解耦策略）

两者**互补而非替代**，共同构成完整的因子预处理流水线。

---

## API 参考

### FactorFingerprinter

| 方法 | 签名 | 说明 |
|------|------|------|
| `extract_fingerprint` | `(data: pd.DataFrame) -> FactorFingerprint` | 提取单因子指纹 |
| `batch_extract` | `(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, FactorFingerprint]` | 批量提取 |

### AdaptiveFactorClassifier

| 方法 | 签名 | 说明 |
|------|------|------|
| `classify` | `(fp: FactorFingerprint) -> Dict` | 分类单因子 |
| `batch_classify` | `(fps: Dict[str, FactorFingerprint]) -> Dict[str, Dict]` | 批量分类 |

### SemanticStatisticalFusion

| 方法 | 签名 | 说明 |
|------|------|------|
| `classify` | `(fingerprint, semantic_prior, data_coverage) -> ArbitratedResult` | 融合分类 |

---

## 文件结构

```
Factor_Fingerprint/
├── __init__.py                 # 包入口，导出核心类
├── core/
│   ├── __init__.py             # 核心模块入口
│   ├── fingerprint.py          # 因子指纹提取器
│   ├── classifier.py           # 自适应分类器
│   ├── semantic.py             # 语义理解系统
│   ├── semantic_fusion.py      # 语义-统计融合
│   └── monitor.py              # 迁移监测
├── tests/
│   └── test_semantic_statistical_fusion.py  # 融合测试
├── docs/
│   ├── factor_prior_semantic_analysis.md    # 语义分析文档
│   ├── factor_rule_based_fingerprint.md     # 规则指纹文档
│   ├── nlp_rule_extraction_analysis.md      # NLP规则提取
│   └── nlp_semantic_improvement.md          # 语义改进
├── demo.py                     # 演示脚本
└── README.md                   # 本文档
```

---

## 版本信息

- **版本**: v1.0.0
- **Python**: 3.10+
- **依赖**: numpy, pandas, scipy, jieba
- **构建日期**: 2026.05.17
- **状态**: STABLE
- **许可证**: MIT
