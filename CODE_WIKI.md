# Factor_Fingerprint Code Wiki

[中文](CODE_WIKI.md) | [English](CODE_WIKI_EN.md)

> 因子指纹与自适应分类系统 — 完整代码文档
> 版本: v1.0.0 | 构建日期: 2026-05-17 | 许可: MIT

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [文件结构](#3-文件结构)
4. [核心模块详解](#4-核心模块详解)
   - [4.1 fingerprint.py — 因子指纹提取器](#41-fingerprintpy--因子指纹提取器)
   - [4.2 classifier.py — 自适应分类器](#42-classifierpy--自适应分类器)
   - [4.3 semantic.py — 语义理解系统](#43-semanticpy--语义理解系统)
   - [4.4 semantic_fusion.py — 语义-统计融合](#44-semantic_fusionpy--语义-统计融合)
   - [4.5 monitor.py — 迁移监测器](#45-monitorpy--迁移监测器)
5. [关键数据流](#5-关键数据流)
6. [模块依赖关系](#6-模块依赖关系)
7. [外部依赖](#7-外部依赖)
8. [运行方式](#8-运行方式)
9. [测试](#9-测试)
10. [设计原则](#10-设计原则)
11. [与外部系统的协同](#11-与外部系统的协同)
12. [API 快速参考](#12-api-快速参考)

---

## 1. 项目概述

**Factor_Fingerprint** 是一个量化因子指纹提取与自适应分类系统。它基于时序稳定性、截面稳定性与语义理解，为每个量化因子生成唯一"指纹"向量，实现因子类型的智能识别与差异化处理。

### 核心能力

```
指纹提取 → 自适应分类 → 语义-统计融合 → 迁移监测 → 健康度监测
```

### 因子分类体系

| 类型 | 含义 | 典型代表 | AR(1) 特征 |
|------|------|---------|-----------|
| `STATIC` | 静态因子 | PB, ROE, 市值 | > 0.80 |
| `DYNAMIC` | 动态因子 | 反转, 换手率变化 | < 0.40 |
| `MIXED` | 混合因子 | 动量, 成长 | 0.40 ~ 0.80 |
| `UNKNOWN` | 无法分类 | — | NaN |

### 设计哲学

- **数据驱动自适应**: 因子管道由指纹指标自动决定，无需人工干预
- **前瞻偏差防护**: 指纹在扩展窗口上计算，无未来信息泄露
- **先验知识融合**: 构造规则语义分析 + 统计指纹，贝叶斯融合
- **中间状态追踪**: 指纹历史可追溯，类型迁移可监测

---

## 2. 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                       Factor_Fingerprint 系统架构                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  FactorFingerprinter                       │   │
│  │                  (因子指纹提取器)                            │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  时序稳定性指标               截面稳定性指标                 │   │
│  │  ├─ AR(1)中位数              ├─ 偏度标准差                  │   │
│  │  ├─ 秩自相关                 ├─ 峰度标准差                  │   │
│  │  ├─ 波动率聚集检验           ├─ JS散度均值                  │   │
│  │  ├─ 半衰期                  ├─ 缺失率变异系数               │   │
│  │  └─ 水平/差分 IC 比         └─ 因子覆盖率                  │   │
│  │                                                              │   │
│  │  综合衍生指标: SD得分 · 复杂度需求 · 信噪比估计               │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               AdaptiveFactorClassifier                     │   │
│  │               (自适应分类器)                                 │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  硬分类: AR(1) > 0.80 → STATIC | < 0.40 → DYNAMIC        │   │
│  │  软分类: sigmoid 概率加权 → 静态/混合/动态概率分布          │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   ┌──────────────────┐    ┌──────────────────────────┐   │   │
│  │   │  semantic.py      │    │  semantic_fusion.py       │   │   │
│  │   │  语义理解系统       │───▶│  语义-统计融合            │   │   │
│  │   │                   │    │                           │   │   │
│  │   │  FinancialTokenizer│    │  SemanticPrior           │   │   │
│  │   │  SemanticRoleLabeler│   │  BayesianFactorClassifier│   │   │
│  │   │  FinancialKnowledgeGraph│ ConflictArbitrator       │   │   │
│  │   │  SemanticMatcher   │    │  SemanticStatisticalFusion│  │   │
│  │   └──────────────────┘    └──────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               FactorFingerprintMonitor                     │   │
│  │               (迁移监测器)                                   │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  多时间尺度: 快速(1期) · 标准(3期) · 长期(6期)             │   │
│  │  平滑过渡: 指数衰减权重, 避免硬切换                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               FactorHealthMonitor                          │   │
│  │               (健康度监测器)                                 │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  五维评估: 拥挤度(0.25) · 效能(0.35) · 容量(0.15)          │   │
│  │           · 衰减(0.15) · 体制敏感性(0.10)                 │   │
│  │  四级警报: HEALTHY · WATCH · WARNING · CRITICAL           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 文件结构

```
Factor_Fingerprint/
├── __init__.py                          # 包入口, 导出所有核心类 (v1.0.0)
├── demo.py                              # 演示脚本 (5个场景)
├── README.md                            # 项目说明文档
├── core/
│   ├── fingerprint.py                   # 因子指纹提取器 (523行)
│   ├── classifier.py                    # 自适应分类器 (249行)
│   ├── semantic.py                      # 语义理解系统 (423行)
│   ├── semantic_fusion.py               # 语义-统计融合 (536行)
│   ├── monitor.py                       # 迁移监测器 (428行)
│   └── health.py                        # 健康度监测器 (~1300行)
├── tests/
│   ├── test_semantic_statistical_fusion.py  # 融合系统测试 (281行)
│   ├── test_health_phase2.py               # 效能指标测试 (28 tests)
│   ├── test_health_phase3.py               # 衰减指标测试 (19 tests)
│   ├── test_health_phase4.py               # 拥挤度指标测试 (16 tests)
│   ├── test_health_phase5.py               # 容量指标测试 (11 tests)
│   ├── test_health_phase6.py               # 体制敏感性测试 (11 tests)
│   └── test_health_phase7.py               # 综合评分+集成测试 (25 tests)
└── docs/
    ├── factor_prior_semantic_analysis.md    # 先验语义分析价值论证
    ├── factor_rule_based_fingerprint.md     # 规则指纹设计文档
    ├── nlp_rule_extraction_analysis.md      # NLP规则提取方案分析
    ├── nlp_semantic_improvement.md          # 语义理解提升方案
    └── health_module_implementation_plan.md # 健康度模块实施计划
```

---

## 4. 核心模块详解

### 4.1 fingerprint.py — 因子指纹提取器

**文件路径**: [core/fingerprint.py](file:///f:/Coding/Factor_Fingerprint/core/fingerprint.py)

#### 4.1.1 枚举类型

**`FactorType`** (Enum) — 因子类型枚举

| 成员 | 值 | 含义 |
|------|-----|------|
| `STATIC` | `"static"` | 静态因子: 高自相关, 排序稳定 |
| `DYNAMIC` | `"dynamic"` | 动态因子: 低自相关, 新息主导 |
| `MIXED` | `"mixed"` | 混合因子: 介于两者之间 |
| `UNKNOWN` | `"unknown"` | 无法分类 |

#### 4.1.2 数据类

**`FactorFingerprint`** (NamedTuple) — 因子指纹向量, 包含 13 个指标

```
┌─────────────────────────────────────────────────────────────────┐
│                    FactorFingerprint 字段                        │
├─────────────────┬───────────────────────────────────────────────┤
│  时序稳定性指标   │                                               │
│  ar1_median      │ float  AR(1)系数中位数 (核心指标)             │
│  rank_autocorr   │ float  截面秩自相关 (Spearman)                │
│  vol_clustering_pvalue │ float  波动率聚集Ljung-Box p值          │
│  half_life       │ float  自相关系数半衰期 (衰减至0.5的期数)     │
│  level_diff_ic_ratio │ float  水平vs差分IC比                     │
├─────────────────┼───────────────────────────────────────────────┤
│  截面稳定性指标   │                                               │
│  skewness_std    │ float  各期截面偏度标准差                      │
│  kurtosis_std    │ float  各期截面峰度标准差                      │
│  js_divergence_mean │ float  相邻两期JS散度均值                   │
│  missing_cv      │ float  缺失率变异系数                          │
│  coverage_ratio  │ float  因子覆盖率                              │
├─────────────────┼───────────────────────────────────────────────┤
│  综合衍生指标     │                                               │
│  sd_score        │ float  静态-动态倾向得分 (0=动态, 1=静态)      │
│  complexity_need │ float  处理复杂度需求 (0=简单, 1=复杂)         │
│  snr_estimate    │ float  信噪比估计                              │
└─────────────────┴───────────────────────────────────────────────┘
```

**`FingerprintConfig`** (dataclass) — 指纹提取配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_window` | 24 | 最短计算窗口 (期数) |
| `decay_halflife` | 12 | 记忆衰退半衰期 |
| `min_obs_per_stock` | 12 | 每只股票最少有效观测数 |
| `min_stocks` | 10 | 最少股票数才计算中位数 |
| `min_cv_threshold` | 0.01 | 变异系数最小阈值 |
| `js_bins` | 20 | JS散度直方图分箱数 |
| `vol_cluster_lags` | 12 | 波动率聚集检验滞后阶数 |
| `ar1_max_lag` | 20 | 半衰期计算最大滞后阶数 |

#### 4.1.3 主类: `FactorFingerprinter`

**构造方法**: `__init__(config: Optional[FingerprintConfig] = None)`

**核心方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `extract_fingerprint` | `(factor_data: pd.DataFrame) -> FactorFingerprint` | 提取单个因子完整指纹 |
| `batch_extract` | `(factor_dict: Dict[str, pd.DataFrame]) -> Dict[str, FactorFingerprint]` | 批量提取多个因子指纹 |

**内部方法 (时序稳定性)**:

| 方法 | 说明 |
|------|------|
| `_compute_ar1_median` | 对每只股票拟合 AR(1) 模型, 取系数中位数。含三重筛选: 有效样本数、变异系数、最少股票数 |
| `_compute_rank_autocorr` | 当期与下期截面排序的 Spearman 相关系数均值, 指数加权 |
| `_test_volatility_clustering` | 对截面标准差平方序列进行 Ljung-Box 检验, 返回最小 p 值 |
| `_estimate_half_life` | 截面均值序列自相关衰减至 0.5 的滞后阶数, 线性插值 |
| `_compute_level_diff_ic_ratio` | 水平自相关 / 差分自相关, 上限截断为 10 |

**内部方法 (截面稳定性)**:

| 方法 | 说明 |
|------|------|
| `_compute_skewness_std` | 各期截面偏度值的标准差 |
| `_compute_kurtosis_std` | 各期截面峰度值的标准差 |
| `_compute_js_divergence_mean` | 相邻两期截面直方图的 Jensen-Shannon 散度均值 |
| `_compute_missing_cv` | 各期缺失率的标准差 / 均值 |
| `_compute_coverage_ratio` | 有效值样本数 / 总样本数的均值 |

**内部方法 (综合衍生)**:

| 方法 | 公式 | 说明 |
|------|------|------|
| `_derive_sd_score` | `0.35*AR(1) + 0.30*秩自相关 + 0.20*半衰期 + 0.15*IC比` | 加权合成, 各成分归一化到 [0,1] |
| `_derive_complexity_need` | `0.4*偏度波动 + 0.4*峰度波动 + 0.2*JS散度` | 分布越不稳定, 复杂度越高 |
| `_estimate_snr` | `abs(均值) / 标准差` | 截面均值序列的信噪比 |

**工具方法**:

| 方法 | 说明 |
|------|------|
| `_exponential_weights` | 生成指数衰减权重, 半衰期由 `decay_halflife` 控制 |
| `_manual_ljungbox` | 手动计算 Ljung-Box 统计量 (statsmodels 不可用时的回退) |

**使用示例**:

```python
fingerprinter = FactorFingerprinter(
    config=FingerprintConfig(min_window=24, decay_halflife=12)
)
fp = fingerprinter.extract_fingerprint(factor_data)  # pd.DataFrame, shape (T, N)
print(f"AR(1)={fp.ar1_median:.4f}, SD_Score={fp.sd_score:.4f}")
```

---

### 4.2 classifier.py — 自适应分类器

**文件路径**: [core/classifier.py](file:///f:/Coding/Factor_Fingerprint/core/classifier.py)

#### 4.2.1 数据类

**`ClassificationConfig`** (dataclass) — 分类器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `static_ar1_threshold` | 0.80 | AR(1) 高于此值 → STATIC |
| `dynamic_ar1_threshold` | 0.40 | AR(1) 低于此值 → DYNAMIC |
| `soft_boundary` | True | 是否启用软边界 (sigmoid) |
| `sigmoid_steepness` | 10.0 | sigmoid 陡峭度 |
| `mixed_zone_width` | 0.10 | 混合区半宽 |

**`ClassificationResult`** (dataclass) — 分类结果

| 字段 | 类型 | 说明 |
|------|------|------|
| `primary_type` | FactorType | 主分类类型 |
| `primary_prob` | float | 主类型概率 |
| `secondary_type` | Optional[FactorType] | 次类型 (可能为 None) |
| `secondary_prob` | float | 次类型概率 |
| `confidence` | float | 分类置信度 [0, 1] |
| `is_hard` | bool | 是否为硬分类 (置信度 > 0.8) |

#### 4.2.2 主类: `AdaptiveFactorClassifier`

**构造方法**: `__init__(config: Optional[ClassificationConfig] = None)`

**核心方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `classify` | `(fingerprint: FactorFingerprint) -> ClassificationResult` | 对单个因子进行分类 |
| `batch_classify` | `(fingerprints: Dict[str, FactorFingerprint]) -> Dict[str, ClassificationResult]` | 批量分类 |
| `get_classification_summary` | `(classifications: Dict[str, ClassificationResult]) -> pd.DataFrame` | 获取分类汇总表 |

**分类逻辑**:

```
classify(fingerprint)
  ├─ AR(1) 为 NaN → UNKNOWN
  ├─ _hard_classify(ar1)         # 硬分类: 阈值判断
  │   ├─ ar1 > 0.80 → STATIC
  │   ├─ ar1 < 0.40 → DYNAMIC
  │   └─ else → MIXED
  ├─ _soft_classify(ar1)         # 软分类: sigmoid 概率
  │   ├─ P(static) = sigmoid(k*(ar1 - 0.80))
  │   ├─ P(dynamic) = sigmoid(-k*(ar1 - 0.40))
  │   └─ P(mixed) = 1 - P(static) - P(dynamic)
  └─ _compute_confidence(ar1, hard_class)  # 置信度
      └─ 距离阈值越远, 置信度越高
```

**`_soft_classify` 详细逻辑**:

```
k = sigmoid_steepness (默认 10.0)
s_threshold = 0.80 (静态阈值)
d_threshold = 0.40 (动态阈值)

p_static  = 1 / (1 + exp(-k * (ar1 - s_threshold)))
p_dynamic = 1 / (1 + exp(k * (ar1 - d_threshold)))
p_mixed   = max(1 - p_static - p_dynamic, 0.0)

归一化: p_static + p_mixed + p_dynamic = 1.0
```

**`_compute_confidence` 逻辑**:

- STATIC: `confidence = min(|ar1 - 0.80| / 0.2, 1.0)`
- DYNAMIC: `confidence = min(|0.40 - ar1| / 0.2, 1.0)`
- MIXED: `confidence = min(min_distance_to_boundary / 0.2, 1.0)`

---

### 4.3 semantic.py — 语义理解系统

**文件路径**: [core/semantic.py](file:///f:/Coding/Factor_Fingerprint/core/semantic.py)

#### 4.3.1 分层架构

```
┌──────────────────────────────────────────────────────────────┐
│              语义理解系统分层架构                              │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: FinancialTokenizer  (金融增强分词器)                │
│     ├─ jieba 分词 + 金融自定义词典 (40+ 金融术语)              │
│     ├─ 词性标注 (金融术语/名词/动词/数量词/时间词)             │
│     └─ 数字+单位提取 (正则: "12个月", "20日")                 │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: SemanticRoleLabeler  (语义角色标注器)               │
│     ├─ 7 种语义关系: 时间范围/操作类型/数据来源/比较对象/      │
│     │   排序方式/修饰关系/否定关系                             │
│     └─ 正则表达式匹配                                         │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: FinancialKnowledgeGraph  (金融知识图谱)             │
│     ├─ 8 种因子实体: 动量/反转/价值/质量/成长/市值/流动性/情绪 │
│     ├─ 9 种指标实体: PE/PB/PS/ROE/ROA/毛利率/净利率/换手率/市值│
│     ├─ 8 种操作符实体: 除以/对数/累积/排名/相反数/差分/均值/标准差│
│     └─ 语义消歧: 基于名称和别名匹配打分                        │
├──────────────────────────────────────────────────────────────┤
│  Layer 4: SemanticMatcher  (语义相似度匹配器)                 │
│     ├─ 6 种因子模板: 动量/反转/价值/质量/成长/流动性           │
│     ├─ 关键词匹配 (回退方案)                                   │
│     └─ 返回 Top-K 匹配结果, 含类别和典型稳定性                  │
└──────────────────────────────────────────────────────────────┘
```

#### 4.3.2 数据模型

**`Token`** (dataclass) — 分词结果

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | str | 词文本 |
| `pos` | str | 词性标签 |
| `offset` | int | 偏移量 |
| `entity_type` | str | 实体类型 (金融术语/名词/动词...) |

**`FactorMeta`** (dataclass) — 因子构造元数据 (先验知识)

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | 因子名称 |
| `category` | Optional[str] | 因子类别 (momentum/value/quality/...) |
| `is_price_based` | bool | 是否基于价格数据 |
| `is_financial_report` | bool | 是否基于财报数据 |
| `is_analyst_data` | bool | 是否基于分析师数据 |
| `lookback_window` | Optional[float] | 回溯窗口 |
| `rebalance_freq` | str | 调仓频率 (monthly) |
| `known_issues` | List[str] | 已知问题 |
| `typical_stability` | Optional[str] | 预期稳定性 (static/dynamic/mixed) |
| `literature_support` | bool | 是否有文献支撑 |

**`ExtractedRule`** (dataclass) — 从文本中提取的规则

| 字段 | 类型 | 说明 |
|------|------|------|
| `category` | Optional[str] | 推断的因子类别 |
| `category_confidence` | float | 类别置信度 |
| `is_price_based` | bool | 是否基于价格 |
| `is_financial_report` | bool | 是否基于财报 |
| `is_analyst_data` | bool | 是否基于分析师数据 |
| `lookback_windows` | List[Tuple[float, str]] | 时间窗口列表 |
| `rebalance_freq` | str | 调仓频率 |
| `operators` | List[str] | 操作符列表 |
| `data_sources` | List[str] | 数据来源列表 |
| `semantic_roles` | Dict[str, List[str]] | 语义角色 |
| `knowledge_graph_matches` | List[Dict] | 知识图谱匹配结果 |
| `overall_confidence` | float | 总体置信度 |
| `inferred_factor_type` | Optional[str] | 推断的因子类型 |

#### 4.3.3 主类: `FactorSemanticUnderstanding`

**构造方法**: `__init__()` — 初始化所有 4 层组件

**核心方法**:

| 方法 | 说明 |
|------|------|
| `understand(text: str) -> ExtractedRule` | 4 层流水线: 分词 → 语义角色 → 知识图谱消歧 → 语义匹配 → 综合决策 |
| `_extract_candidates(tokens) -> List[str]` | 从分词结果中提取候选实体 (金融术语 + 名词) |
| `_make_decision(rule) -> ExtractedRule` | 综合决策: 知识图谱 > 语义匹配 > 基础规则, 计算总体置信度 |

**决策优先级**:

```
1. 知识图谱精确匹配 → category_confidence = 0.8
2. 语义相似度匹配 → category_confidence = similarity * 0.7
3. 基础规则 → 无类别
```

**总体置信度计算**:

```
score = category_confidence * 0.4
      + (0.2 if lookback_windows else 0)
      + (0.2 if data_sources else 0)
      + (0.2 if semantic_roles else 0)
overall_confidence = min(score, 1.0)
```

#### 4.3.4 便捷函数

| 函数 | 签名 | 说明 |
|------|------|------|
| `extract_from_text` | `(text: str) -> ExtractedRule` | 从文本提取规则, 自动创建系统实例 |
| `extract_to_meta` | `(text: str, name: str) -> FactorMeta` | 提取并转换为 FactorMeta |

---

### 4.4 semantic_fusion.py — 语义-统计融合

**文件路径**: [core/semantic_fusion.py](file:///f:/Coding/Factor_Fingerprint/core/semantic_fusion.py)

#### 4.4.1 融合设计哲学

```
冷启动 (数据 < 30%)  → 信任语义 (语义是唯一可靠依据)
观察期 (30% ~ 70%)   → 加权融合 (贝叶斯后验 = 先验 × 似然)
成熟期 (数据 > 70%)  → 信任统计 (统计指纹反映因子本质)
冲突时               → 保守降级 (采用最保守的混合管道)
```

#### 4.4.2 数据类

**`SemanticPrior`** (dataclass) — 语义先验

| 字段 | 类型 | 说明 |
|------|------|------|
| `expected_type` | FactorType | 语义推断的因子类型 |
| `prior_strength` | float | 先验强度 [0, 1] |
| `confidence` | float | 语义分析置信度 |

**类别到类型映射** (`CATEGORY_TO_TYPE`):

| 类别 | 因子类型 | 类别 | 因子类型 |
|------|---------|------|---------|
| `value` | STATIC | `quality` | STATIC |
| `size` | STATIC | `momentum` | MIXED |
| `growth` | MIXED | `reversal` | DYNAMIC |
| `liquidity` | DYNAMIC | `sentiment` | DYNAMIC |

**AR(1) 先验分布参数** (`AR1_PRIOR_MAP`):

| 类型 | 均值 | 标准差 |
|------|------|--------|
| STATIC | 0.85 | 0.10 |
| MIXED | 0.60 | 0.15 |
| DYNAMIC | 0.20 | 0.10 |
| UNKNOWN | 0.50 | 0.25 |

**关键方法**:

| 方法 | 说明 |
|------|------|
| `to_ar1_prior() -> Tuple[float, float]` | 将语义类型转换为 AR(1) 高斯先验 (均值, 标准差) |
| `from_extracted_rule(rule) -> SemanticPrior` | 从 ExtractedRule 创建先验, 计算 prior_strength |
| `from_description(description) -> SemanticPrior` | 从自然语言描述直接创建先验 (含语义分析) |

#### 4.4.3 冲突诊断类型

**`ConflictDiagnosis`** (Enum):

| 值 | 含义 |
|-----|------|
| `description_error` | 描述错误: 语义分析与实际构造不符 |
| `regime_change` | 市场体制变化: 因子性质发生结构性改变 |
| `impurity` | 构造不纯: 因子构造中混入了其他因子成分 |
| `noise` | 统计噪声: 数据不足导致统计结果不可靠 |
| `human_review` | 需要人工审查: 持续冲突无法自动解决 |

#### 4.4.4 主类: `BayesianFactorClassifier`

继承自 `AdaptiveFactorClassifier`, 增加语义先验支持。

**关键方法**:

| 方法 | 说明 |
|------|------|
| `classify_with_prior(fingerprint, prior) -> ClassificationResult` | 支持语义先验的贝叶斯分类 |
| `_estimate_data_weight(fingerprint) -> float` | 估计数据充足度权重 (基于 AR(1) 距离边界) |

**分类逻辑**:

```
classify_with_prior(fingerprint, prior)
  ├─ 无先验 → 退化为纯统计分类
  ├─ AR(1) 为 NaN 但先验存在 → 信任语义
  ├─ 统计与先验一致 → 提升置信度: confidence * (1 + prior_strength * 0.3)
  └─ 冲突:
      ├─ data_weight > 0.7 → 统计主导, 标记为软分类
      └─ data_weight ≤ 0.7 → 语义主导, 标记为软分类
```

#### 4.4.5 主类: `ConflictArbitrator`

**`ArbitratedResult`** — 扩展 `ClassificationResult`, 增加冲突信息:

| 新增字段 | 类型 | 说明 |
|---------|------|------|
| `conflict_reason` | Optional[str] | 冲突原因 |
| `diagnosis` | Optional[ConflictDiagnosis] | 诊断类型 |

**核心方法**:

| 方法 | 说明 |
|------|------|
| `arbitrate(semantic, statistical, data_sufficiency) -> ArbitratedResult` | 仲裁语义与统计的冲突 |

**仲裁策略**:

```
arbitrate(semantic, statistical, data_sufficiency)
  ├─ 一致 → _high_confidence_lock: confidence * 1.1, is_hard=True
  ├─ 不一致:
  │   ├─ data_sufficiency < 0.3 → _semantic_override: 语义优先, 低置信度
  │   ├─ 0.3 ≤ data_sufficiency < 0.7 → _fallback_to_mixed: 降级为 MIXED
  │   └─ data_sufficiency ≥ 0.7 → _alert_human_review: 统计主导, 标记人工审查
```

#### 4.4.6 主类: `SemanticStatisticalFusion`

端到端融合接口, 整合语义分析、统计指纹、贝叶斯分类和冲突仲裁。

**构造方法**: `__init__()` — 初始化 `FactorSemanticUnderstanding`, `BayesianFactorClassifier`, `ConflictArbitrator`

**核心方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `classify` | `(description: str, fingerprint: FactorFingerprint, data_months: int) -> ArbitratedResult` | 单因子融合分类 |
| `classify_batch` | `(descriptions, fingerprints, data_months) -> Dict[str, ArbitratedResult]` | 批量融合分类 |
| `_compute_data_sufficiency` | `(data_months, fingerprint) -> float` | 计算数据充足度 (长度 + 质量) |

**完整流水线**:

```
classify(description, fingerprint, data_months)
  Step 1: semantic_prior = SemanticPrior.from_description(description)
  Step 2: statistical_result = classifier.classify_with_prior(fingerprint, semantic_prior)
  Step 3: data_sufficiency = _compute_data_sufficiency(data_months, fingerprint)
  Step 4: 一致 → 高置信度锁定; 冲突 → ConflictArbitrator.arbitrate()
```

**数据充足度计算**:

```
length_score = min(data_months / 24, 1.0)           # 数据长度: 24个月为满分
quality_score = 0.5*AR(1)有效 + 0.3*秩自相关有效 + 0.2*SD_score有效
data_sufficiency = length_score * 0.6 + quality_score * 0.4
```

---

### 4.5 monitor.py — 迁移监测器

**文件路径**: [core/monitor.py](file:///f:/Coding/Factor_Fingerprint/core/monitor.py)

#### 4.5.1 数据类

**`MigrationAlert`** (NamedTuple) — 迁移警报

| 字段 | 类型 | 说明 |
|------|------|------|
| `factor_name` | str | 因子名称 |
| `from_type` | FactorType | 迁移前类型 |
| `to_type` | FactorType | 迁移后类型 |
| `level` | str | 警报级别: WARNING / INFO / CRITICAL |
| `window` | int | 监测窗口长度 |
| `timestamp` | datetime | 时间戳 |
| `recommendation` | str | 建议措施 |
| `fingerprint_distance` | float | 指纹距离 |

**`MonitorConfig`** (dataclass) — 监测器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `short_window` | 1 | 快速监测窗口 (期数) |
| `medium_window` | 3 | 标准监测窗口 (期数) |
| `long_window` | 6 | 长期监测窗口 (期数) |
| `short_threshold` | 0.3 | 快速警报阈值 (指纹距离) |
| `medium_threshold` | 0.2 | 标准警报阈值 |
| `long_threshold` | 0.15 | 长期警报阈值 |
| `migration_consecutive` | 3 | 连续变化期数触发迁移 |
| `enable_smooth_transition` | True | 是否启用平滑过渡 |

#### 4.5.2 主类: `FactorFingerprintMonitor`

**内部状态**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `fingerprint_history` | `Dict[str, List[FactorFingerprint]]` | 指纹历史: {因子名: [指纹序列]} |
| `classification_history` | `Dict[str, List[ClassificationResult]]` | 分类历史: {因子名: [分类结果序列]} |
| `alert_history` | `List[MigrationAlert]` | 警报历史 |

**核心方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `add_fingerprint` | `(factor_name, fingerprint)` | 添加新指纹到历史, 同步分类 |
| `check_type_migration` | `(factor_name, current_fingerprint) -> List[MigrationAlert]` | 检查类型迁移, 返回触发的警报 |
| `get_transition_weights` | `(factor_name, current_fingerprint) -> Dict[FactorType, float]` | 获取类型迁移时的过渡权重 |
| `get_migration_summary` | `(factor_name=None) -> pd.DataFrame` | 获取迁移摘要表 |
| `get_factor_stability_score` | `(factor_name) -> float` | 获取因子稳定性得分 [0, 1] |

**多时间尺度监测**:

```
check_type_migration(factor_name, current_fingerprint)
  ├─ add_fingerprint: 记录当前指纹 + 分类
  ├─ _check_short_migration:  1期剧烈变化 → WARNING (距离 > 0.3)
  ├─ _check_medium_migration: 3期趋势变化 → INFO (距离 > 0.2)
  └─ _check_long_migration:   6期结构性漂移 → INFO (距离 > 0.15, 主导类型变化)
```

**平滑过渡权重** (`get_transition_weights`):

```
- 最近3期类型一致 → 无迁移, 返回 {current_type: 1.0}
- 有迁移 → 指数衰减: 越近的类型权重越高 (decay=0.7)
  例: 最近5期类型为 [S, S, D, D, D] → 归一化后权重
```

**指纹距离** (`_compute_fingerprint_distance`):

```
使用归一化欧氏距离, 关键指标: ar1_median, rank_autocorr, skewness_std, kurtosis_std
distance = ||vec1 - vec2|| / sqrt(len)
```

**稳定性得分** (`get_factor_stability_score`):

```
基于最近6期分类历史的类型一致性
stability = max_count / 6
数据不足3期 → 0.5 (中性)
```

### 4.6 health.py — 因子健康度监测器

**文件路径**: [core/health.py](file:///f:/Coding/Factor_Fingerprint/core/health.py)

#### 4.6.1 设计理念

五维正交评估框架，独立计算各维度得分后加权融合，生成综合健康度报告：

```
拥挤度 (Crowding)    ── 0.25 ─┐
效能   (Efficacy)    ── 0.35 ─┤
容量   (Capacity)    ── 0.15 ─┼── 加权合成 → 健康分 [0-100] + 等级判定
衰减   (Decay)       ── 0.15 ─┤
体制   (Regime)      ── 0.10 ─┘
```

#### 4.6.2 枚举类型

**`HealthAlertLevel`** (Enum) — 健康度警报等级

| 成员 | 值 | 含义 |
|------|-----|------|
| `HEALTHY` | `"healthy"` | 健康: 所有指标正常 (得分 ≥ 60, 无警报) |
| `WATCH` | `"watch"` | 关注: 得分 < 60 但无超阈值警报 |
| `WARNING` | `"warning"` | 警告: 1-2 个维度超阈值 |
| `CRITICAL` | `"critical"` | 危急: ≥3 个维度超阈值 或 存在衰减趋势警报 |

#### 4.6.3 数据类

**`HealthAlert`** (NamedTuple) — 健康度单项警报

| 字段 | 类型 | 说明 |
|------|------|------|
| `metric_name` | str | 指标名称 |
| `metric_value` | float | 当前值 |
| `threshold` | float | 预警阈值 |
| `direction` | str | 超阈值方向: `'above'` / `'below'` |
| `level` | HealthAlertLevel | 警报等级 |
| `category` | str | 所属维度: `crowding`/`efficacy`/`capacity`/`decay`/`regime` |
| `recommendation` | str | 建议措施 |

**`HealthConfig`** (dataclass) — 健康度监测配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_data_periods` | 12 | 最少数据期数 |
| `min_stocks` | 10 | 最少股票数 |
| **权重** | | |
| `health_score_weights` | `{efficacy:0.35, crowding:0.25, capacity:0.15, decay:0.15, regime:0.10}` | 五维加权权重 |
| **拥挤度阈值** | | |
| `crowding_corr_threshold` | 0.70 | 配对相关性阈值 |
| `crowding_hhi_threshold` | 0.15 | 持仓集中度 HHI 阈值 |
| `crowding_turnover_threshold` | 0.50 | 年化换手率阈值 |
| `crowding_reversal_threshold` | 0.30 | 收益反转阈值 |
| **效能阈值** | | |
| `efficacy_icir_threshold` | 0.50 | IC IR 最低阈值 |
| `efficacy_ic_win_rate_threshold` | 0.55 | IC 胜率最低阈值 |
| **衰减阈值** | | |
| `decay_mk_trend_pvalue` | 0.05 | Mann-Kendall 趋势显著性 |
| `decay_long_short_ratio_threshold` | 0.50 | 多空收益衰减比阈值 |
| **容量阈值** | | |
| `capacity_effective_n_threshold` | 30 | 有效股票数阈值 |
| `capacity_top5_threshold` | 0.40 | Top5 集中度阈值 |

**`FactorHealthReport`** (dataclass) — 健康度综合报告

| 字段 | 类型 | 说明 |
|------|------|------|
| `factor_name` | str | 因子名称 |
| `health_score` | float | 综合健康分 [0, 100] |
| `health_level` | HealthAlertLevel | 健康等级 |
| `crowding_score` | float | 拥挤度得分 [0, 100] |
| `efficacy_score` | float | 效能得分 |
| `capacity_score` | float | 容量得分 |
| `decay_score` | float | 衰减得分 |
| `regime_score` | float | 体制敏感性得分 |
| `crowding_metrics` | Dict[str, float] | 拥挤度子指标 |
| `efficacy_metrics` | Dict[str, float] | 效能子指标 |
| `capacity_metrics` | Dict[str, float] | 容量子指标 |
| `decay_metrics` | Dict[str, float] | 衰减子指标 |
| `regime_metrics` | Dict[str, float] | 体制敏感性子指标 |
| `alerts` | List[HealthAlert] | 警报列表 |

#### 4.6.4 主类: `FactorHealthMonitor`

**构造方法**: `__init__(config: Optional[HealthConfig] = None)`

**内部状态**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `health_history` | `Dict[str, List[FactorHealthReport]]` | 健康度历史: {因子名: [报告序列]} |
| `alert_history` | `List[HealthAlert]` | 警报历史 |

**公开方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `evaluate_health` | `(factor_name, factor_data, returns_data, market_cap_data, volume_data) -> FactorHealthReport` | 评估单个因子健康度 |
| `evaluate_health_batch` | `(factor_dict, returns_data, ...) -> Dict[str, FactorHealthReport]` | 批量评估多个因子 |
| `get_health_summary` | `(factor_name=None) -> pd.DataFrame` | 获取健康度摘要表 |
| `get_health_trend` | `(factor_name, lookback=12) -> pd.DataFrame` | 获取最近 N 期健康度趋势 |

**私有方法 — 综合评分**:

| 方法 | 公式 | 说明 |
|------|------|------|
| `_compute_health_score` | `Σ(dim_score[k] × weight[k])` | 加权合成, clip 到 [0, 100] |
| `_determine_health_level` | 见等级判定逻辑 | 基于得分 + 警报维度数判定 |
| `_normalize_score` | `(value - worst) / (best - worst) × 100` | 线性映射到 [0, 100], NaN → 50 |

**私有方法 — 效能 (Efficacy, 权重 0.35)**:

| 方法 | 公式 | 说明 |
|------|------|------|
| `_compute_rank_ic` | Spearman(factor_t, return_t+1) | 截面秩相关系数 |
| `_compute_ic_series` | 逐期计算 RankIC | 生成 IC 时间序列 |
| `_compute_ic_ir` | mean(IC) / std(IC, ddof=1) | IC 信息比 |
| `_compute_rolling_ic_mean` | 指数加权移动平均 IC | 近期 IC 均值 |
| `_compute_ic_win_rate` | count(IC > 0) / N | IC 正值比例 |
| `_compute_ic_autocorr` | lag-1 自相关 | IC 持续性 |

**私有方法 — 拥挤度 (Crowding, 权重 0.25)**:

| 方法 | 公式 | 说明 |
|------|------|------|
| `_compute_pairwise_corr_concentration` | mean(pairwise_corr) | 股票间因子值配对相关性 |
| `_compute_position_hhi` | `Σ(w_i²) / (Σw_i)²` | 持仓权重 Herfindahl-Hirschman 指数 |
| `_compute_turnover` | mean(abs(rank_t - rank_t-1)) / N | 排名变化率 |
| `_compute_return_reversal` | | 收益反转风险 (需 returns_data) |

**私有方法 — 容量 (Capacity, 权重 0.15)**:

| 方法 | 公式 | 说明 |
|------|------|------|
| `_compute_effective_n` | `1 / HHI` | 有效股票数 |
| `_compute_top5_concentration` | top5权重和 | 前5大持仓集中度 |
| `_compute_cap_weighted_concentration` | | 市值加权集中度 (需 market_cap_data) |

**私有方法 — 衰减 (Decay, 权重 0.15)**:

| 方法 | 公式 | 说明 |
|------|------|------|
| `_compute_mann_kendall` | S = Σ sign(x_j - x_i) | 非参数单调趋势检验, 返回 (tau, p_value) |
| `_compute_long_short_decay` | | 多空收益衰减比 (前1/3 vs 后1/3) |
| `_compute_rolling_ic_slope` | polyfit(IC, deg=1) | IC 序列线性回归斜率 |

**私有方法 — 体制敏感性 (Regime, 权重 0.10)**:

| 方法 | 公式 | 说明 |
|------|------|------|
| `_split_bull_bear` | 基于市场收益正负 | 划分牛熊市区间 |
| `_compute_bull_bear_ic_ratio` | IC_bull / IC_bear | 牛熊 IC 比 |
| `_compute_vol_conditional_ic_corr` | corr(IC, volatility) | 波动率条件 IC 相关性 |

#### 4.6.5 等级判定逻辑

```
_determine_health_level(score, alerts)
  ├─ WARNING/CRITICAL 警报 ≥ 3 个不同维度 → CRITICAL
  ├─ WARNING/CRITICAL 警报 ≥ 1 个维度 → WARNING
  ├─ 得分 < 60 → WATCH
  └─ 得分 ≥ 60 → HEALTHY
```

#### 4.6.6 使用示例

```python
from Factor_Fingerprint import FactorHealthMonitor, HealthConfig

monitor = FactorHealthMonitor(HealthConfig(min_data_periods=12))

# 单因子评估
report = monitor.evaluate_health(
    factor_name='momentum_12m1m',
    factor_data=factor_panel,      # pd.DataFrame, T×N
    returns_data=returns_panel,    # pd.DataFrame, T×N
    market_cap_data=mcap_panel,    # pd.DataFrame, T×N (optional)
)
print(f"健康分: {report.health_score:.1f}, 等级: {report.health_level.value}")

# 批量评估
results = monitor.evaluate_health_batch(
    {'momentum': data1, 'value': data2},
    returns_data=returns_panel,
    market_cap_data=mcap_panel,
)

# 趋势查询
trend = monitor.get_health_trend('momentum_12m1m', lookback=12)
```

---

## 5. 关键数据流

### 5.1 完整处理流水线

```
输入: 因子面板数据 (T × N) + 构造描述文本
  │
  ├─[1]─ FactorFingerprinter.extract_fingerprint(data)
  │      └─ 输出: FactorFingerprint (13个指标)
  │
  ├─[2]─ FactorSemanticUnderstanding.understand(description)
  │      └─ 输出: ExtractedRule (语义分析结果)
  │
  ├─[3]─ SemanticPrior.from_extracted_rule(rule)
  │      └─ 输出: SemanticPrior (先验分布)
  │
  ├─[4]─ BayesianFactorClassifier.classify_with_prior(fp, prior)
  │      └─ 输出: ClassificationResult (融合分类)
  │
  ├─[5]─ ConflictArbitrator.arbitrate(semantic, statistical, data_sufficiency)
  │      └─ 输出: ArbitratedResult (仲裁结果)
  │
  └─[6]─ FactorFingerprintMonitor.add_fingerprint(name, fp)
         └─ 输出: MigrationAlert[] (迁移警报, 若有)

  └─[7]─ FactorHealthMonitor.evaluate_health(name, data, returns, mcap)
         └─ 输出: FactorHealthReport (五维健康度报告)
```

### 5.2 数据流图

```
                    ┌───────────────────┐
                    │   因子面板数据 T×N  │
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │ FactorFingerprinter│
                    │  extract_fingerprint│
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │ FactorFingerprint  │
                    │ (13维指纹向量)     │
                    └──┬───────────┬────┘
                       │           │
          ┌────────────▼──┐   ┌───▼──────────────┐
          │classifier.py  │   │semantic_fusion.py │
          │AdaptiveFactor │   │SemanticStatistical│
          │Classifier     │   │Fusion             │
          │               │   │                    │
          │ .classify(fp) │   │ .classify(desc,   │
          │ → Classification│  │  fp, months)      │
          │   Result      │   │ → ArbitratedResult│
          └───────┬───────┘   └────────┬──────────┘
                  │                    │
          ┌───────▼────────────────────▼──────────┐
          │          FactorFingerprintMonitor      │
          │  .add_fingerprint(name, fp)            │
          │  .check_type_migration(name, fp)       │
          │  → MigrationAlert[]                    │
          └────────────────────────────────────────┘
```

---

## 6. 模块依赖关系

```
                    ┌──────────────────┐
                    │   __init__.py    │  (包入口)
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                  │
    ┌─────────▼────┐ ┌──────▼───────┐ ┌────────▼─────────┐
    │ fingerprint  │ │  classifier  │ │    semantic       │
    │   .py        │ │    .py       │ │     .py           │
    │              │◄┤              │ │                   │
    │ FactorType   │ │ imports      │ │ (独立, 无内部依赖) │
    │ FactorFinger │ │ FactorFinger │ │                   │
    │ print        │ │ print,       │ │ FinancialTokenizer│
    │              │ │ FactorType   │ │ SemanticRoleLabel │
    └──────┬───────┘ └──────┬───────┘ │ FinancialKnowledge│
           │                │         │ Graph             │
           │                │         │ SemanticMatcher   │
           │                │         └────────┬──────────┘
           │                │                  │
    ┌──────▼────────────────▼──────┐  ┌───────▼──────────┐
    │      semantic_fusion.py      │  │    monitor.py    │
    │                               │  │                  │
    │ imports: fingerprint,         │  │ imports:         │
    │   classifier, semantic        │  │  fingerprint,    │
    │                               │  │  classifier      │
    │ SemanticPrior                 │  │                  │
    │ BayesianFactorClassifier      │  │ FactorFingerprint│
    │   (extends AdaptiveFactor)    │  │ Monitor          │
    │ ConflictArbitrator            │  │                  │
    │ SemanticStatisticalFusion     │  │                  │
    └───────────────────────────────┘  └──────────────────┘

dependency层次:
  Layer 0: fingerprint.py (基础层, 无内部依赖)
  Layer 1: classifier.py (依赖 fingerprint), semantic.py (独立)
  Layer 2: semantic_fusion.py (依赖 fingerprint + classifier + semantic)
  Layer 3: monitor.py (依赖 fingerprint + classifier)
  Layer 4: health.py (依赖 fingerprint + classifier)
```

---

## 7. 外部依赖

### 7.1 必需依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `numpy` | — | 数值计算, 矩阵运算 |
| `pandas` | — | 面板数据处理, DataFrame |
| `scipy` | — | 统计检验 (Spearman, chi2, Jensen-Shannon) |
| `jieba` | — | 中文分词 (语义理解模块) |

### 7.2 可选依赖

| 包名 | 用途 | 回退方案 |
|------|------|---------|
| `statsmodels` | Ljung-Box 检验 (`acorr_ljungbox`) | 手动实现 (`_manual_ljungbox`) |

### 7.3 安装命令

```bash
pip install numpy pandas scipy jieba
```

---

## 8. 运行方式

### 8.1 运行演示脚本

```bash
cd Factor_Fingerprint
python demo.py
```

演示脚本包含 5 个场景:

1. **`demo_fingerprint()`** — 指纹提取: 生成 3 种不同 AR(1) 强度的模拟数据, 提取并展示指纹指标
2. **`demo_classifier()`** — 因子分类: 对 3 种因子进行分类, 展示硬分类和软分类结果
3. **`demo_semantic()`** — 语义理解: 对 5 段因子构造描述进行语义分析
4. **`demo_integration()`** — 完整流程: 语义理解 → 指纹提取 → 分类 → 一致性检查
5. **`demo_health()`** — 健康度监测: 评估因子五维健康度, 展示综合评分和警报

### 8.2 运行测试

```bash
cd Factor_Fingerprint
python tests/test_semantic_statistical_fusion.py
```

测试覆盖:
- `TestSemanticPrior` (5 个测试): 语义先验创建、AR(1) 映射、从 ExtractedRule 创建
- `TestBayesianClassifier` (4 个测试): 无先验/一致先验/冲突先验分类
- `TestConflictArbitrator` (4 个测试): 一致/数据不足/观察期/持续冲突仲裁
- `TestSemanticStatisticalIntegration` (3 个测试): 端到端流水线、冷启动、冲突场景

### 8.3 编程接口使用

```python
import sys
sys.path.insert(0, 'f:/Coding')

from Factor_Fingerprint import (
    FactorFingerprinter,
    AdaptiveFactorClassifier,
    SemanticStatisticalFusion,
    SemanticPrior,
    FactorFingerprintMonitor,
    FactorType,
)

# 1. 指纹提取
fingerprinter = FactorFingerprinter()
fp = fingerprinter.extract_fingerprint(factor_data)  # pd.DataFrame, T×N

# 2. 纯统计分类
classifier = AdaptiveFactorClassifier()
result = classifier.classify(fp)

# 3. 语义-统计融合
fusion = SemanticStatisticalFusion()
arbitrated = fusion.classify(
    description="市盈率因子，基于最新财报数据",
    fingerprint=fp,
    data_months=36
)

# 4. 迁移监测
monitor = FactorFingerprintMonitor()
monitor.add_fingerprint('PB', fp_t1)
alerts = monitor.check_type_migration('PB', fp_t2)
```

---

## 9. 测试

**测试文件**: [tests/test_semantic_statistical_fusion.py](file:///f:/Coding/Factor_Fingerprint/tests/test_semantic_statistical_fusion.py)

**健康度测试文件** (6 个):
- [tests/test_health_phase2.py](file:///f:/Coding/Factor_Fingerprint/tests/test_health_phase2.py) — 效能指标 (28 tests)
- [tests/test_health_phase3.py](file:///f:/Coding/Factor_Fingerprint/tests/test_health_phase3.py) — 衰减指标 (19 tests)
- [tests/test_health_phase4.py](file:///f:/Coding/Factor_Fingerprint/tests/test_health_phase4.py) — 拥挤度指标 (16 tests)
- [tests/test_health_phase5.py](file:///f:/Coding/Factor_Fingerprint/tests/test_health_phase5.py) — 容量指标 (11 tests)
- [tests/test_health_phase6.py](file:///f:/Coding/Factor_Fingerprint/tests/test_health_phase6.py) — 体制敏感性 (11 tests)
- [tests/test_health_phase7.py](file:///f:/Coding/Factor_Fingerprint/tests/test_health_phase7.py) — 综合评分+集成 (25 tests)

**测试框架**: `pytest` (健康度), `unittest` (融合)

**测试类**:

| 测试类 | 测试数 | 覆盖范围 |
|--------|--------|---------|
| `TestSemanticPrior` | 5 | 先验创建、AR(1) 映射 (STATIC/DYNAMIC/MIXED)、从 ExtractedRule 创建 |
| `TestBayesianClassifier` | 4 | 无先验退化、一致先验提升置信度、数据不足信任语义、冲突时软分类 |
| `TestConflictArbitrator` | 4 | 一致高置信度、数据不足语义覆盖、观察期降级混合、持续冲突人工审查 |
| `TestSemanticStatisticalIntegration` | 3 | 完整流水线、冷启动、冲突场景 |

**运行方式**:

```bash
# 融合系统测试 (unittest)
python tests/test_semantic_statistical_fusion.py

# 健康度模块测试 (pytest, 全部 110 tests)
python -m pytest tests/test_health_phase*.py -v

# 仅运行特定阶段
python -m pytest tests/test_health_phase7.py -v
```

---

## 10. 设计原则

### 10.1 核心原则

1. **数据驱动自适应**: 因子管道由指纹指标自动决定, 无需人工干预
2. **前瞻偏差防护**: 指纹在扩展窗口上计算, 无未来信息泄露; 记忆衰退机制确保近期数据权重更高
3. **先验知识融合**: 构造规则语义分析 (语义先验) + 统计指纹 (数据后验), 贝叶斯融合
4. **中间状态追踪**: 指纹历史可追溯, 类型迁移可监测, 支持平滑过渡
5. **C¹ 连续性保持**: 类型迁移时平滑过渡 (指数衰减权重), 避免硬切换导致的管道不连续

### 10.2 鲁棒性设计

| 设计点 | 实现方式 |
|--------|---------|
| 三重筛选 | `_compute_ar1_median`: 有效样本数 + 变异系数 + 最少股票数 |
| NaN 安全 | 所有指标计算均处理 NaN, 不满足条件返回 NaN |
| 回退机制 | `_test_volatility_clustering`: statsmodels → 手动 Ljung-Box |
| 上限截断 | `_compute_level_diff_ic_ratio`: 上限 10; `_derive_complexity_need`: 上限 1.0 |
| 平滑处理 | JS 散度: 直方图 +1e-10 避免零值; 软分类: sigmoid 避免硬切换 |

### 10.3 可配置性

所有阈值均通过 `dataclass` 配置类暴露, 支持不同市场环境调整:

- `FingerprintConfig`: 窗口长度、半衰期、筛选阈值
- `ClassificationConfig`: AR(1) 阈值、sigmoid 陡峭度、混合区宽度
- `MonitorConfig`: 多时间尺度窗口、警报阈值、平滑过渡开关

---

## 11. 与外部系统的协同

### 11.1 与 Factor_Decoupler 的协同

```
Factor_Fingerprint                Factor_Decoupler
─────────────────                ─────────────────
提取指纹 → 分类因子                    差异化处理
    STATIC  ─────────────────────→  跳过解耦 (保留截面排序)
    DYNAMIC ─────────────────────→  三重中性化 (提取新息)
    MIXED   ─────────────────────→  条件性解耦
```

### 11.2 与 Factor_Imputer 的区别

| 维度 | Factor_Imputer_v2.0 | Factor_Fingerprint |
|------|---------------------|-------------------|
| **核心目标** | 选择最优插补策略 | 选择最优解耦策略 |
| **类型体系** | 业务语义 (财务/估值/技术/宏观) | 投资逻辑 (静态/动态/混合) |
| **检测依据** | 分布形态 + 缺失模式 | 时序持续性 + 截面稳定性 |
| **理论基础** | 启发式规则 | 时间序列分析 + 信息论 |
| **语义融合** | 关键词匹配 | 完整语义-统计贝叶斯融合 |
| **迁移监测** | 无 | 有 (持续监测类型变化) |

### 11.3 数据处理流水线位置

```
数据输入
    ↓
[Factor_Imputer]  ← 业务分类 (财务/估值/技术)
    ↓
缺失值插补 (截面中位数/ffill/滚动窗口)
    ↓
[Factor_Fingerprint]  ← 投资逻辑分类 (静态/动态/混合)
    ↓
解耦决策 (跳过/三重中性化/条件解耦)
    ↓
纯净信号输出
```

---

## 12. API 快速参考

### 12.1 包级导入

```python
from Factor_Fingerprint import (
    # 指纹提取
    FactorFingerprinter, FactorFingerprint, FingerprintConfig, FactorType,
    # 分类
    AdaptiveFactorClassifier, ClassificationConfig, ClassificationResult,
    # 语义理解
    FactorSemanticUnderstanding, FinancialTokenizer,
    FinancialKnowledgeGraph, SemanticMatcher,
    FactorMeta, ExtractedRule, extract_from_text, extract_to_meta,
    # 语义-统计融合
    SemanticPrior, BayesianFactorClassifier,
    ConflictArbitrator, ConflictDiagnosis,
    ArbitratedResult, SemanticStatisticalFusion,
    # 迁移监测
    FactorFingerprintMonitor, MonitorConfig, MigrationAlert,
    # 健康度监测
    FactorHealthMonitor, HealthConfig, HealthAlertLevel,
    FactorHealthReport, HealthAlert,
)
```

### 12.2 常用模式

**模式 1: 快速分类 (纯统计)**

```python
fp = FactorFingerprinter().extract_fingerprint(data)
result = AdaptiveFactorClassifier().classify(fp)
print(result.primary_type.value)  # 'static' | 'dynamic' | 'mixed'
```

**模式 2: 语义-统计融合 (推荐)**

```python
fusion = SemanticStatisticalFusion()
result = fusion.classify(
    description="市盈率因子，基于最新财报数据",
    fingerprint=fp,
    data_months=36
)
print(f"类型: {result.primary_type.value}, 置信度: {result.confidence:.2%}")
```

**模式 3: 持续监测**

```python
monitor = FactorFingerprintMonitor()
for t, fp in enumerate(fingerprints_over_time):
    monitor.add_fingerprint('my_factor', fp)
    alerts = monitor.check_type_migration('my_factor', fp)
    if alerts:
        for alert in alerts:
            print(f"警报: {alert.from_type.value} → {alert.to_type.value}")
```

**模式 4: 健康度监测**

```python
health_monitor = FactorHealthMonitor()
report = health_monitor.evaluate_health(
    factor_name='momentum',
    factor_data=factor_panel,
    returns_data=returns_panel,
    market_cap_data=mcap_panel,
)
print(f"健康分: {report.health_score:.1f}, 等级: {report.health_level.value}")
for alert in report.alerts:
    print(f"  [{alert.category}] {alert.metric_name}: {alert.recommendation}")
```

---

*本文档由 Code Wiki 生成器基于源代码分析自动生成。最后更新: 2026-07-01*