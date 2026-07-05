# Factor_Fingerprint Code Wiki

[中文](CODE_WIKI.md) | [English](CODE_WIKI_EN.md)

> Factor Fingerprint and Adaptive Classification System — Complete Code Documentation
> Version: v1.0.0 | Build date: 2026-05-17 | License: MIT

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Overall Architecture](#2-overall-architecture)
3. [File Structure](#3-file-structure)
4. [Core Module Details](#4-core-module-details)
   - [4.1 fingerprint.py — Factor Fingerprint Extractor](#41-fingerprintpy--factor-fingerprint-extractor)
   - [4.2 classifier.py — Adaptive Classifier](#42-classifierpy--adaptive-classifier)
   - [4.3 semantic.py — Semantic Understanding System](#43-semanticpy--semantic-understanding-system)
   - [4.4 semantic_fusion.py — Semantic-Statistical Fusion](#44-semantic_fusionpy--semantic-statistical-fusion)
   - [4.5 monitor.py — Migration Monitor](#45-monitorpy--migration-monitor)
5. [Key Data Flow](#5-key-data-flow)
6. [Module Dependencies](#6-module-dependencies)
7. [External Dependencies](#7-external-dependencies)
8. [Usage](#8-usage)
9. [Testing](#9-testing)
10. [Design Principles](#10-design-principles)
11. [Integration with External Systems](#11-integration-with-external-systems)
12. [API Quick Reference](#12-api-quick-reference)

---

## 1. Project Overview

**Factor_Fingerprint** is a quantitative factor fingerprint extraction and adaptive classification system. Based on time-series stability, cross-sectional stability, and semantic understanding, it generates a unique "fingerprint" vector for each quantitative factor, enabling intelligent factor type recognition and differentiated processing.

### Core Capabilities

```
Fingerprint Extraction → Adaptive Classification → Semantic-Statistical Fusion → Migration Monitoring → Health Monitoring
```

### Factor Classification Taxonomy

| Type | Meaning | Typical Examples | AR(1) Characteristic |
|------|---------|------------------|---------------------|
| `STATIC` | Static factor | PB, ROE, Market Cap | > 0.80 |
| `DYNAMIC` | Dynamic factor | Reversal, Turnover change | < 0.40 |
| `MIXED` | Mixed factor | Momentum, Growth | 0.40 ~ 0.80 |
| `UNKNOWN` | Unclassifiable | — | NaN |

### Design Philosophy

- **Data-driven Adaptation**: The factor pipeline is automatically determined by fingerprint metrics, requiring no manual intervention
- **Lookahead Bias Protection**: Fingerprints are computed on expanding windows, with no future information leakage
- **Prior Knowledge Fusion**: Combines construction-rule semantic analysis + statistical fingerprint via Bayesian fusion
- **Intermediate State Tracking**: Fingerprint history is traceable, type migration is monitorable

---

## 2. Overall Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  Factor_Fingerprint System Architecture           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  FactorFingerprinter                       │   │
│  │                  (Factor Fingerprint Extractor)            │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Time-series Stability      Cross-sectional Stability     │   │
│  │  ├─ AR(1) median            ├─ Skewness std               │   │
│  │  ├─ Rank autocorrelation    ├─ Kurtosis std               │   │
│  │  ├─ Vol clustering test     ├─ JS divergence mean         │   │
│  │  ├─ Half-life               ├─ Missing-rate CV            │   │
│  │  └─ Level/Diff IC ratio     └─ Factor coverage            │   │
│  │                                                              │   │
│  │  Composite Derivative Metrics: SD score · Complexity need · SNR estimate │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               AdaptiveFactorClassifier                     │   │
│  │               (Adaptive Classifier)                        │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Hard classification: AR(1) > 0.80 → STATIC | < 0.40 → DYNAMIC │
│  │  Soft classification: sigmoid probability weighting → static/mixed/dynamic probability │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   ┌──────────────────┐    ┌──────────────────────────┐   │   │
│  │   │  semantic.py      │    │  semantic_fusion.py       │   │   │
│  │   │  Semantic         │───▶│  Semantic-Statistical     │   │   │
│  │   │  Understanding    │    │  Fusion                   │   │   │
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
│  │               (Migration Monitor)                          │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Multi-time-scale: Fast (1 period) · Standard (3 periods) · Long-term (6 periods) │   │
│  │  Smooth transition: Exponential decay weights, avoid hard switching │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               FactorHealthMonitor                          │   │
│  │               (Health Monitor)                             │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Five-dimensional Assessment: Crowding (0.25) · Efficacy (0.35) · Capacity (0.15) │   │
│  │           · Decay (0.15) · Regime Sensitivity (0.10)     │   │
│  │  Four-level Alert: HEALTHY · WATCH · WARNING · CRITICAL  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. File Structure

```
Factor_Fingerprint/
├── __init__.py                          # Package entry, exports all core classes (v1.0.0)
├── demo.py                              # Demo script (5 scenarios)
├── README.md                            # Project documentation
├── core/
│   ├── fingerprint.py                   # Factor Fingerprint Extractor (523 lines)
│   ├── classifier.py                    # Adaptive Classifier (249 lines)
│   ├── semantic.py                      # Semantic Understanding System (423 lines)
│   ├── semantic_fusion.py               # Semantic-Statistical Fusion (536 lines)
│   ├── monitor.py                       # Migration Monitor (428 lines)
│   └── health.py                        # Health Monitor (~1300 lines)
├── tests/
│   ├── test_semantic_statistical_fusion.py  # Fusion system tests (281 lines)
│   ├── test_health_phase2.py               # Efficacy metrics tests (28 tests)
│   ├── test_health_phase3.py               # Decay metrics tests (19 tests)
│   ├── test_health_phase4.py               # Crowding metrics tests (16 tests)
│   ├── test_health_phase5.py               # Capacity metrics tests (11 tests)
│   ├── test_health_phase6.py               # Regime sensitivity tests (11 tests)
│   └── test_health_phase7.py               # Composite scoring + integration tests (25 tests)
└── docs/
    ├── factor_prior_semantic_analysis.md    # Prior semantic analysis value justification
    ├── factor_rule_based_fingerprint.md     # Rule-based fingerprint design document
    ├── nlp_rule_extraction_analysis.md      # NLP rule extraction approach analysis
    ├── nlp_semantic_improvement.md          # Semantic understanding improvement plan
    └── health_module_implementation_plan.md # Health module implementation plan
```

---

## 4. Core Module Details

### 4.1 fingerprint.py — Factor Fingerprint Extractor

**File path**: [core/fingerprint.py](file:///f:/Coding/Factor_Fingerprint/core/fingerprint.py)

#### 4.1.1 Enum Types

**`FactorType`** (Enum) — Factor type enumeration

| Member | Value | Meaning |
|--------|-------|---------|
| `STATIC` | `"static"` | Static factor: high autocorrelation, stable ranking |
| `DYNAMIC` | `"dynamic"` | Dynamic factor: low autocorrelation, innovation-driven |
| `MIXED` | `"mixed"` | Mixed factor: between static and dynamic |
| `UNKNOWN` | `"unknown"` | Unclassifiable |

#### 4.1.2 Data Classes

**`FactorFingerprint`** (NamedTuple) — Factor fingerprint vector, containing 13 metrics

```
┌─────────────────────────────────────────────────────────────────┐
│                    FactorFingerprint Fields                       │
├─────────────────┬───────────────────────────────────────────────┤
│  Time-series Stability │                                               │
│  ar1_median      │ float  AR(1) coefficient median (core metric)   │
│  rank_autocorr   │ float  Cross-sectional rank autocorrelation (Spearman) │
│  vol_clustering_pvalue │ float  Volatility clustering Ljung-Box p-value │
│  half_life       │ float  Autocorrelation coefficient half-life (periods to decay to 0.5) │
│  level_diff_ic_ratio │ float  Level vs difference IC ratio         │
├─────────────────┼───────────────────────────────────────────────┤
│  Cross-sectional Stability │                                          │
│  skewness_std    │ float  Std of cross-sectional skewness across periods │
│  kurtosis_std    │ float  Std of cross-sectional kurtosis across periods │
│  js_divergence_mean │ float  Mean JS divergence between adjacent periods │
│  missing_cv      │ float  Coefficient of variation of missing rate │
│  coverage_ratio  │ float  Factor coverage ratio                    │
├─────────────────┼───────────────────────────────────────────────┤
│  Composite Derivatives │                                              │
│  sd_score        │ float  Static-dynamic propensity score (0=dynamic, 1=static) │
│  complexity_need │ float  Processing complexity need (0=simple, 1=complex) │
│  snr_estimate    │ float  Signal-to-noise ratio estimate           │
└─────────────────┴───────────────────────────────────────────────┘
```

**`FingerprintConfig`** (dataclass) — Fingerprint extraction configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_window` | 24 | Minimum computation window (periods) |
| `decay_halflife` | 12 | Memory decay half-life |
| `min_obs_per_stock` | 12 | Minimum valid observations per stock |
| `min_stocks` | 10 | Minimum number of stocks to compute median |
| `min_cv_threshold` | 0.01 | Minimum coefficient of variation threshold |
| `js_bins` | 20 | Number of bins for JS divergence histogram |
| `vol_cluster_lags` | 12 | Lags for volatility clustering test |
| `ar1_max_lag` | 20 | Maximum lag for half-life computation |

#### 4.1.3 Main Class: `FactorFingerprinter`

**Constructor**: `__init__(config: Optional[FingerprintConfig] = None)`

**Core Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `extract_fingerprint` | `(factor_data: pd.DataFrame) -> FactorFingerprint` | Extract complete fingerprint for a single factor |
| `batch_extract` | `(factor_dict: Dict[str, pd.DataFrame]) -> Dict[str, FactorFingerprint]` | Batch extract fingerprints for multiple factors |

**Internal Methods (Time-series Stability)**:

| Method | Description |
|--------|-------------|
| `_compute_ar1_median` | Fit AR(1) model per stock, take coefficient median. Triple screening: valid sample count, coefficient of variation, minimum stock count |
| `_compute_rank_autocorr` | Mean Spearman correlation between current and next period cross-sectional rankings, exponentially weighted |
| `_test_volatility_clustering` | Ljung-Box test on cross-sectional standard deviation squared series, returns minimum p-value |
| `_estimate_half_life` | Lag at which cross-sectional mean series autocorrelation decays to 0.5, with linear interpolation |
| `_compute_level_diff_ic_ratio` | Level autocorrelation / difference autocorrelation, capped at 10 |

**Internal Methods (Cross-sectional Stability)**:

| Method | Description |
|--------|-------------|
| `_compute_skewness_std` | Std of skewness values across cross-sectional periods |
| `_compute_kurtosis_std` | Std of kurtosis values across cross-sectional periods |
| `_compute_js_divergence_mean` | Mean Jensen-Shannon divergence between adjacent period cross-sectional histograms |
| `_compute_missing_cv` | Std / mean of missing rates across periods |
| `_compute_coverage_ratio` | Mean of valid sample count / total sample count |

**Internal Methods (Composite Derivatives)**:

| Method | Formula | Description |
|--------|---------|-------------|
| `_derive_sd_score` | `0.35*AR(1) + 0.30*rank_autocorr + 0.20*half_life + 0.15*IC_ratio` | Weighted composite, components normalized to [0,1] |
| `_derive_complexity_need` | `0.4*skewness_vol + 0.4*kurtosis_vol + 0.2*JS_divergence` | More unstable distribution → higher complexity |
| `_estimate_snr` | `abs(mean) / std` | Signal-to-noise ratio of cross-sectional mean series |

**Utility Methods**:

| Method | Description |
|--------|-------------|
| `_exponential_weights` | Generate exponential decay weights, half-life controlled by `decay_halflife` |
| `_manual_ljungbox` | Manually compute Ljung-Box statistic (fallback when statsmodels unavailable) |

**Usage Example**:

```python
fingerprinter = FactorFingerprinter(
    config=FingerprintConfig(min_window=24, decay_halflife=12)
)
fp = fingerprinter.extract_fingerprint(factor_data)  # pd.DataFrame, shape (T, N)
print(f"AR(1)={fp.ar1_median:.4f}, SD_Score={fp.sd_score:.4f}")
```

---

### 4.2 classifier.py — Adaptive Classifier

**File path**: [core/classifier.py](file:///f:/Coding/Factor_Fingerprint/core/classifier.py)

#### 4.2.1 Data Classes

**`ClassificationConfig`** (dataclass) — Classifier configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `static_ar1_threshold` | 0.80 | AR(1) above this → STATIC |
| `dynamic_ar1_threshold` | 0.40 | AR(1) below this → DYNAMIC |
| `soft_boundary` | True | Enable soft boundary (sigmoid) |
| `sigmoid_steepness` | 10.0 | sigmoid steepness |
| `mixed_zone_width` | 0.10 | Mixed zone half-width |

**`ClassificationResult`** (dataclass) — Classification result

| Field | Type | Description |
|-------|------|-------------|
| `primary_type` | FactorType | Primary classification type |
| `primary_prob` | float | Primary type probability |
| `secondary_type` | Optional[FactorType] | Secondary type (may be None) |
| `secondary_prob` | float | Secondary type probability |
| `confidence` | float | Classification confidence [0, 1] |
| `is_hard` | bool | Whether hard classification (confidence > 0.8) |

#### 4.2.2 Main Class: `AdaptiveFactorClassifier`

**Constructor**: `__init__(config: Optional[ClassificationConfig] = None)`

**Core Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `classify` | `(fingerprint: FactorFingerprint) -> ClassificationResult` | Classify a single factor |
| `batch_classify` | `(fingerprints: Dict[str, FactorFingerprint]) -> Dict[str, ClassificationResult]` | Batch classify |
| `get_classification_summary` | `(classifications: Dict[str, ClassificationResult]) -> pd.DataFrame` | Get classification summary table |

**Classification Logic**:

```
classify(fingerprint)
  ├─ AR(1) is NaN → UNKNOWN
  ├─ _hard_classify(ar1)         # Hard classification: threshold judgment
  │   ├─ ar1 > 0.80 → STATIC
  │   ├─ ar1 < 0.40 → DYNAMIC
  │   └─ else → MIXED
  ├─ _soft_classify(ar1)         # Soft classification: sigmoid probability
  │   ├─ P(static) = sigmoid(k*(ar1 - 0.80))
  │   ├─ P(dynamic) = sigmoid(-k*(ar1 - 0.40))
  │   └─ P(mixed) = 1 - P(static) - P(dynamic)
  └─ _compute_confidence(ar1, hard_class)  # Confidence
      └─ Farther from threshold → higher confidence
```

**`_soft_classify` Detailed Logic**:

```
k = sigmoid_steepness (default 10.0)
s_threshold = 0.80 (static threshold)
d_threshold = 0.40 (dynamic threshold)

p_static  = 1 / (1 + exp(-k * (ar1 - s_threshold)))
p_dynamic = 1 / (1 + exp(k * (ar1 - d_threshold)))
p_mixed   = max(1 - p_static - p_dynamic, 0.0)

Normalize: p_static + p_mixed + p_dynamic = 1.0
```

**`_compute_confidence` Logic**:

- STATIC: `confidence = min(|ar1 - 0.80| / 0.2, 1.0)`
- DYNAMIC: `confidence = min(|0.40 - ar1| / 0.2, 1.0)`
- MIXED: `confidence = min(min_distance_to_boundary / 0.2, 1.0)`

---

### 4.3 semantic.py — Semantic Understanding System

**File path**: [core/semantic.py](file:///f:/Coding/Factor_Fingerprint/core/semantic.py)

#### 4.3.1 Layered Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              Semantic Understanding System Architecture       │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: FinancialTokenizer  (Finance-enhanced tokenizer)   │
│     ├─ jieba tokenization + finance custom dictionary (40+ financial terms) │
│     ├─ POS tagging (financial terms/nouns/verbs/quantifiers/time words) │
│     └─ Number+unit extraction (regex: "12 months", "20 days") │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: SemanticRoleLabeler  (Semantic role labeler)       │
│     ├─ 7 semantic relations: time range/operation type/data source/comparison object/ │
│     │   sorting method/modifier relation/negation relation   │
│     └─ Regex matching                                         │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: FinancialKnowledgeGraph  (Financial knowledge graph) │
│     ├─ 8 factor entities: momentum/reversal/value/quality/growth/size/liquidity/sentiment │
│     ├─ 9 indicator entities: PE/PB/PS/ROE/ROA/gross margin/net margin/turnover/market cap │
│     ├─ 8 operator entities: divide by/log/cumulate/rank/negate/difference/mean/std │
│     └─ Semantic disambiguation: scoring based on name and alias matching │
├──────────────────────────────────────────────────────────────┤
│  Layer 4: SemanticMatcher  (Semantic similarity matcher)     │
│     ├─ 6 factor templates: momentum/reversal/value/quality/growth/liquidity │
│     ├─ Keyword matching (fallback scheme)                    │
│     └─ Returns Top-K matches, including category and typical stability │
└──────────────────────────────────────────────────────────────┘
```

#### 4.3.2 Data Models

**`Token`** (dataclass) — Tokenization result

| Field | Type | Description |
|-------|------|-------------|
| `text` | str | Token text |
| `pos` | str | POS tag |
| `offset` | int | Offset |
| `entity_type` | str | Entity type (financial term/noun/verb...) |

**`FactorMeta`** (dataclass) — Factor construction metadata (prior knowledge)

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Factor name |
| `category` | Optional[str] | Factor category (momentum/value/quality/...) |
| `is_price_based` | bool | Whether based on price data |
| `is_financial_report` | bool | Whether based on financial report data |
| `is_analyst_data` | bool | Whether based on analyst data |
| `lookback_window` | Optional[float] | Lookback window |
| `rebalance_freq` | str | Rebalance frequency (monthly) |
| `known_issues` | List[str] | Known issues |
| `typical_stability` | Optional[str] | Expected stability (static/dynamic/mixed) |
| `literature_support` | bool | Whether supported by literature |

**`ExtractedRule`** (dataclass) — Rule extracted from text

| Field | Type | Description |
|-------|------|-------------|
| `category` | Optional[str] | Inferred factor category |
| `category_confidence` | float | Category confidence |
| `is_price_based` | bool | Whether based on price |
| `is_financial_report` | bool | Whether based on financial report |
| `is_analyst_data` | bool | Whether based on analyst data |
| `lookback_windows` | List[Tuple[float, str]] | List of time windows |
| `rebalance_freq` | str | Rebalance frequency |
| `operators` | List[str] | List of operators |
| `data_sources` | List[str] | List of data sources |
| `semantic_roles` | Dict[str, List[str]] | Semantic roles |
| `knowledge_graph_matches` | List[Dict] | Knowledge graph matches |
| `overall_confidence` | float | Overall confidence |
| `inferred_factor_type` | Optional[str] | Inferred factor type |

#### 4.3.3 Main Class: `FactorSemanticUnderstanding`

**Constructor**: `__init__()` — Initialize all 4 layer components

**Core Methods**:

| Method | Description |
|--------|-------------|
| `understand(text: str) -> ExtractedRule` | 4-layer pipeline: tokenization → semantic roles → knowledge graph disambiguation → semantic matching → composite decision |
| `_extract_candidates(tokens) -> List[str]` | Extract candidate entities from tokens (financial terms + nouns) |
| `_make_decision(rule) -> ExtractedRule` | Composite decision: knowledge graph > semantic matching > basic rules; compute overall confidence |

**Decision Priority**:

```
1. Knowledge graph exact match → category_confidence = 0.8
2. Semantic similarity match → category_confidence = similarity * 0.7
3. Basic rule → no category
```

**Overall Confidence Calculation**:

```
score = category_confidence * 0.4
      + (0.2 if lookback_windows else 0)
      + (0.2 if data_sources else 0)
      + (0.2 if semantic_roles else 0)
overall_confidence = min(score, 1.0)
```

#### 4.3.4 Convenience Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `extract_from_text` | `(text: str) -> ExtractedRule` | Extract rule from text, auto-creates system instance |
| `extract_to_meta` | `(text: str, name: str) -> FactorMeta` | Extract and convert to FactorMeta |

---

### 4.4 semantic_fusion.py — Semantic-Statistical Fusion

**File path**: [core/semantic_fusion.py](file:///f:/Coding/Factor_Fingerprint/core/semantic_fusion.py)

#### 4.4.1 Fusion Design Philosophy

```
Cold start (data < 30%)  → Trust semantics (semantic is the only reliable basis)
Observation period (30% ~ 70%) → Weighted fusion (Bayesian posterior = prior × likelihood)
Mature period (data > 70%) → Trust statistics (statistical fingerprint reflects factor essence)
On conflict              → Conservative downgrade (use most conservative mixed pipeline)
```

#### 4.4.2 Data Classes

**`SemanticPrior`** (dataclass) — Semantic prior

| Field | Type | Description |
|-------|------|-------------|
| `expected_type` | FactorType | Semantically inferred factor type |
| `prior_strength` | float | Prior strength [0, 1] |
| `confidence` | float | Semantic analysis confidence |

**Category to Type Mapping** (`CATEGORY_TO_TYPE`):

| Category | Factor Type | Category | Factor Type |
|----------|-------------|----------|-------------|
| `value` | STATIC | `quality` | STATIC |
| `size` | STATIC | `momentum` | MIXED |
| `growth` | MIXED | `reversal` | DYNAMIC |
| `liquidity` | DYNAMIC | `sentiment` | DYNAMIC |

**AR(1) Prior Distribution Parameters** (`AR1_PRIOR_MAP`):

| Type | Mean | Std |
|------|------|-----|
| STATIC | 0.85 | 0.10 |
| MIXED | 0.60 | 0.15 |
| DYNAMIC | 0.20 | 0.10 |
| UNKNOWN | 0.50 | 0.25 |

**Key Methods**:

| Method | Description |
|--------|-------------|
| `to_ar1_prior() -> Tuple[float, float]` | Convert semantic type to AR(1) Gaussian prior (mean, std) |
| `from_extracted_rule(rule) -> SemanticPrior` | Create prior from ExtractedRule, compute prior_strength |
| `from_description(description) -> SemanticPrior` | Create prior directly from natural language description (with semantic analysis) |

#### 4.4.3 Conflict Diagnosis Types

**`ConflictDiagnosis`** (Enum):

| Value | Meaning |
|-------|---------|
| `description_error` | Description error: semantic analysis inconsistent with actual construction |
| `regime_change` | Market regime change: factor nature has structurally changed |
| `impurity` | Construction impurity: factor construction mixed with other factor components |
| `noise` | Statistical noise: insufficient data leads to unreliable statistics |
| `human_review` | Human review required: persistent conflict cannot be auto-resolved |

#### 4.4.4 Main Class: `BayesianFactorClassifier`

Inherits from `AdaptiveFactorClassifier`, adds semantic prior support.

**Key Methods**:

| Method | Description |
|--------|-------------|
| `classify_with_prior(fingerprint, prior) -> ClassificationResult` | Bayesian classification with semantic prior |
| `_estimate_data_weight(fingerprint) -> float` | Estimate data sufficiency weight (based on AR(1) distance from boundary) |

**Classification Logic**:

```
classify_with_prior(fingerprint, prior)
  ├─ No prior → degrades to pure statistical classification
  ├─ AR(1) is NaN but prior exists → trust semantics
  ├─ Statistics and prior agree → boost confidence: confidence * (1 + prior_strength * 0.3)
  └─ Conflict:
      ├─ data_weight > 0.7 → statistics dominant, mark as soft classification
      └─ data_weight ≤ 0.7 → semantics dominant, mark as soft classification
```

#### 4.4.5 Main Class: `ConflictArbitrator`

**`ArbitratedResult`** — Extends `ClassificationResult`, adds conflict information:

| New Field | Type | Description |
|-----------|------|-------------|
| `conflict_reason` | Optional[str] | Conflict reason |
| `diagnosis` | Optional[ConflictDiagnosis] | Diagnosis type |

**Core Method**:

| Method | Description |
|--------|-------------|
| `arbitrate(semantic, statistical, data_sufficiency) -> ArbitratedResult` | Arbitrate conflict between semantic and statistical results |

**Arbitration Strategy**:

```
arbitrate(semantic, statistical, data_sufficiency)
  ├─ Agree → _high_confidence_lock: confidence * 1.1, is_hard=True
  ├─ Disagree:
  │   ├─ data_sufficiency < 0.3 → _semantic_override: semantic first, low confidence
  │   ├─ 0.3 ≤ data_sufficiency < 0.7 → _fallback_to_mixed: downgrade to MIXED
  │   └─ data_sufficiency ≥ 0.7 → _alert_human_review: statistics dominant, mark for human review
```

#### 4.4.6 Main Class: `SemanticStatisticalFusion`

End-to-end fusion interface, integrating semantic analysis, statistical fingerprint, Bayesian classification, and conflict arbitration.

**Constructor**: `__init__()` — Initialize `FactorSemanticUnderstanding`, `BayesianFactorClassifier`, `ConflictArbitrator`

**Core Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `classify` | `(description: str, fingerprint: FactorFingerprint, data_months: int) -> ArbitratedResult` | Single-factor fusion classification |
| `classify_batch` | `(descriptions, fingerprints, data_months) -> Dict[str, ArbitratedResult]` | Batch fusion classification |
| `_compute_data_sufficiency` | `(data_months, fingerprint) -> float` | Compute data sufficiency (length + quality) |

**Complete Pipeline**:

```
classify(description, fingerprint, data_months)
  Step 1: semantic_prior = SemanticPrior.from_description(description)
  Step 2: statistical_result = classifier.classify_with_prior(fingerprint, semantic_prior)
  Step 3: data_sufficiency = _compute_data_sufficiency(data_months, fingerprint)
  Step 4: Agree → high-confidence lock; Conflict → ConflictArbitrator.arbitrate()
```

**Data Sufficiency Calculation**:

```
length_score = min(data_months / 24, 1.0)           # Data length: 24 months for full score
quality_score = 0.5*AR(1) valid + 0.3*rank_autocorr valid + 0.2*SD_score valid
data_sufficiency = length_score * 0.6 + quality_score * 0.4
```

---

### 4.5 monitor.py — Migration Monitor

**File path**: [core/monitor.py](file:///f:/Coding/Factor_Fingerprint/core/monitor.py)

#### 4.5.1 Data Classes

**`MigrationAlert`** (NamedTuple) — Migration alert

| Field | Type | Description |
|-------|------|-------------|
| `factor_name` | str | Factor name |
| `from_type` | FactorType | Type before migration |
| `to_type` | FactorType | Type after migration |
| `level` | str | Alert level: WARNING / INFO / CRITICAL |
| `window` | int | Monitoring window length |
| `timestamp` | datetime | Timestamp |
| `recommendation` | str | Recommended action |
| `fingerprint_distance` | float | Fingerprint distance |

**`MonitorConfig`** (dataclass) — Monitor configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `short_window` | 1 | Fast monitoring window (periods) |
| `medium_window` | 3 | Standard monitoring window (periods) |
| `long_window` | 6 | Long-term monitoring window (periods) |
| `short_threshold` | 0.3 | Fast alert threshold (fingerprint distance) |
| `medium_threshold` | 0.2 | Standard alert threshold |
| `long_threshold` | 0.15 | Long-term alert threshold |
| `migration_consecutive` | 3 | Consecutive change periods to trigger migration |
| `enable_smooth_transition` | True | Whether to enable smooth transition |

#### 4.5.2 Main Class: `FactorFingerprintMonitor`

**Internal State**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `fingerprint_history` | `Dict[str, List[FactorFingerprint]]` | Fingerprint history: {factor name: [fingerprint series]} |
| `classification_history` | `Dict[str, List[ClassificationResult]]` | Classification history: {factor name: [classification result series]} |
| `alert_history` | `List[MigrationAlert]` | Alert history |

**Core Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_fingerprint` | `(factor_name, fingerprint)` | Add new fingerprint to history, sync classification |
| `check_type_migration` | `(factor_name, current_fingerprint) -> List[MigrationAlert]` | Check type migration, return triggered alerts |
| `get_transition_weights` | `(factor_name, current_fingerprint) -> Dict[FactorType, float]` | Get transition weights during type migration |
| `get_migration_summary` | `(factor_name=None) -> pd.DataFrame` | Get migration summary table |
| `get_factor_stability_score` | `(factor_name) -> float` | Get factor stability score [0, 1] |

**Multi-time-scale Monitoring**:

```
check_type_migration(factor_name, current_fingerprint)
  ├─ add_fingerprint: record current fingerprint + classification
  ├─ _check_short_migration:  1-period sharp change → WARNING (distance > 0.3)
  ├─ _check_medium_migration: 3-period trend change → INFO (distance > 0.2)
  └─ _check_long_migration:   6-period structural drift → INFO (distance > 0.15, dominant type change)
```

**Smooth Transition Weights** (`get_transition_weights`):

```
- Last 3 periods same type → no migration, return {current_type: 1.0}
- Migration → exponential decay: more recent types get higher weight (decay=0.7)
  Example: last 5 period types are [S, S, D, D, D] → normalized weights
```

**Fingerprint Distance** (`_compute_fingerprint_distance`):

```
Uses normalized Euclidean distance on key metrics: ar1_median, rank_autocorr, skewness_std, kurtosis_std
distance = ||vec1 - vec2|| / sqrt(len)
```

**Stability Score** (`get_factor_stability_score`):

```
Based on type consistency of last 6 classification periods
stability = max_count / 6
Less than 3 periods → 0.5 (neutral)
```

### 4.6 health.py — Factor Health Monitor

**File path**: [core/health.py](file:///f:/Coding/Factor_Fingerprint/core/health.py)

#### 4.6.1 Design Philosophy

Five-dimensional orthogonal assessment framework, computing each dimension score independently then weighted fusion, generating comprehensive health report:

```
Crowding     ── 0.25 ─┐
Efficacy     ── 0.35 ─┤
Capacity     ── 0.15 ─┼── Weighted composite → Health score [0-100] + Level determination
Decay        ── 0.15 ─┤
Regime       ── 0.10 ─┘
```

#### 4.6.2 Enum Types

**`HealthAlertLevel`** (Enum) — Health alert level

| Member | Value | Meaning |
|--------|-------|---------|
| `HEALTHY` | `"healthy"` | Healthy: all metrics normal (score ≥ 60, no alerts) |
| `WATCH` | `"watch"` | Watch: score < 60 but no above-threshold alerts |
| `WARNING` | `"warning"` | Warning: 1-2 dimensions exceed threshold |
| `CRITICAL` | `"critical"` | Critical: ≥3 dimensions exceed threshold or decay trend alert exists |

#### 4.6.3 Data Classes

**`HealthAlert`** (NamedTuple) — Health single alert

| Field | Type | Description |
|-------|------|-------------|
| `metric_name` | str | Metric name |
| `metric_value` | float | Current value |
| `threshold` | float | Warning threshold |
| `direction` | str | Threshold direction: `'above'` / `'below'` |
| `level` | HealthAlertLevel | Alert level |
| `category` | str | Dimension: `crowding`/`efficacy`/`capacity`/`decay`/`regime` |
| `recommendation` | str | Recommended action |

**`HealthConfig`** (dataclass) — Health monitoring configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_data_periods` | 12 | Minimum data periods |
| `min_stocks` | 10 | Minimum number of stocks |
| **Weights** | | |
| `health_score_weights` | `{efficacy:0.35, crowding:0.25, capacity:0.15, decay:0.15, regime:0.10}` | Five-dimensional weighted weights |
| **Crowding thresholds** | | |
| `crowding_corr_threshold` | 0.70 | Pairwise correlation threshold |
| `crowding_hhi_threshold` | 0.15 | Position concentration HHI threshold |
| `crowding_turnover_threshold` | 0.50 | Annualized turnover threshold |
| `crowding_reversal_threshold` | 0.30 | Return reversal threshold |
| **Efficacy thresholds** | | |
| `efficacy_icir_threshold` | 0.50 | Minimum IC IR threshold |
| `efficacy_ic_win_rate_threshold` | 0.55 | Minimum IC win rate threshold |
| **Decay thresholds** | | |
| `decay_mk_trend_pvalue` | 0.05 | Mann-Kendall trend significance |
| `decay_long_short_ratio_threshold` | 0.50 | Long-short return decay ratio threshold |
| **Capacity thresholds** | | |
| `capacity_effective_n_threshold` | 30 | Effective stock count threshold |
| `capacity_top5_threshold` | 0.40 | Top5 concentration threshold |

**`FactorHealthReport`** (dataclass) — Health comprehensive report

| Field | Type | Description |
|-------|------|-------------|
| `factor_name` | str | Factor name |
| `health_score` | float | Composite health score [0, 100] |
| `health_level` | HealthAlertLevel | Health level |
| `crowding_score` | float | Crowding score [0, 100] |
| `efficacy_score` | float | Efficacy score |
| `capacity_score` | float | Capacity score |
| `decay_score` | float | Decay score |
| `regime_score` | float | Regime sensitivity score |
| `crowding_metrics` | Dict[str, float] | Crowding sub-metrics |
| `efficacy_metrics` | Dict[str, float] | Efficacy sub-metrics |
| `capacity_metrics` | Dict[str, float] | Capacity sub-metrics |
| `decay_metrics` | Dict[str, float] | Decay sub-metrics |
| `regime_metrics` | Dict[str, float] | Regime sensitivity sub-metrics |
| `alerts` | List[HealthAlert] | Alert list |

#### 4.6.4 Main Class: `FactorHealthMonitor`

**Constructor**: `__init__(config: Optional[HealthConfig] = None)`

**Internal State**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `health_history` | `Dict[str, List[FactorHealthReport]]` | Health history: {factor name: [report series]} |
| `alert_history` | `List[HealthAlert]` | Alert history |

**Public Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `evaluate_health` | `(factor_name, factor_data, returns_data, market_cap_data, volume_data) -> FactorHealthReport` | Evaluate single factor health |
| `evaluate_health_batch` | `(factor_dict, returns_data, ...) -> Dict[str, FactorHealthReport]` | Batch evaluate multiple factors |
| `get_health_summary` | `(factor_name=None) -> pd.DataFrame` | Get health summary table |
| `get_health_trend` | `(factor_name, lookback=12) -> pd.DataFrame` | Get last N periods health trend |

**Private Methods — Composite Scoring**:

| Method | Formula | Description |
|--------|---------|-------------|
| `_compute_health_score` | `Σ(dim_score[k] × weight[k])` | Weighted composite, clip to [0, 100] |
| `_determine_health_level` | See level determination logic | Based on score + alert dimension count |
| `_normalize_score` | `(value - worst) / (best - worst) × 100` | Linear mapping to [0, 100], NaN → 50 |

**Private Methods — Efficacy (weight 0.35)**:

| Method | Formula | Description |
|--------|---------|-------------|
| `_compute_rank_ic` | Spearman(factor_t, return_t+1) | Cross-sectional rank correlation coefficient |
| `_compute_ic_series` | Compute RankIC per period | Generate IC time series |
| `_compute_ic_ir` | mean(IC) / std(IC, ddof=1) | IC information ratio |
| `_compute_rolling_ic_mean` | Exponentially weighted moving average IC | Recent IC mean |
| `_compute_ic_win_rate` | count(IC > 0) / N | IC positive ratio |
| `_compute_ic_autocorr` | lag-1 autocorrelation | IC persistence |

**Private Methods — Crowding (weight 0.25)**:

| Method | Formula | Description |
|--------|---------|-------------|
| `_compute_pairwise_corr_concentration` | mean(pairwise_corr) | Pairwise correlation of factor values across stocks |
| `_compute_position_hhi` | `Σ(w_i²) / (Σw_i)²` | Position weight Herfindahl-Hirschman Index |
| `_compute_turnover` | mean(abs(rank_t - rank_t-1)) / N | Ranking change rate |
| `_compute_return_reversal` | | Return reversal risk (requires returns_data) |

**Private Methods — Capacity (weight 0.15)**:

| Method | Formula | Description |
|--------|---------|-------------|
| `_compute_effective_n` | `1 / HHI` | Effective stock count |
| `_compute_top5_concentration` | top5 weight sum | Top5 holding concentration |
| `_compute_cap_weighted_concentration` | | Market-cap weighted concentration (requires market_cap_data) |

**Private Methods — Decay (weight 0.15)**:

| Method | Formula | Description |
|--------|---------|-------------|
| `_compute_mann_kendall` | S = Σ sign(x_j - x_i) | Non-parametric monotonic trend test, returns (tau, p_value) |
| `_compute_long_short_decay` | | Long-short return decay ratio (first 1/3 vs last 1/3) |
| `_compute_rolling_ic_slope` | polyfit(IC, deg=1) | IC series linear regression slope |

**Private Methods — Regime Sensitivity (weight 0.10)**:

| Method | Formula | Description |
|--------|---------|-------------|
| `_split_bull_bear` | Based on market return sign | Split bull/bear market periods |
| `_compute_bull_bear_ic_ratio` | IC_bull / IC_bear | Bull/bear IC ratio |
| `_compute_vol_conditional_ic_corr` | corr(IC, volatility) | Volatility-conditional IC correlation |

#### 4.6.5 Level Determination Logic

```
_determine_health_level(score, alerts)
  ├─ WARNING/CRITICAL alerts ≥ 3 different dimensions → CRITICAL
  ├─ WARNING/CRITICAL alerts ≥ 1 dimension → WARNING
  ├─ score < 60 → WATCH
  └─ score ≥ 60 → HEALTHY
```

#### 4.6.6 Usage Example

```python
from Factor_Fingerprint import FactorHealthMonitor, HealthConfig

monitor = FactorHealthMonitor(HealthConfig(min_data_periods=12))

# Single factor evaluation
report = monitor.evaluate_health(
    factor_name='momentum_12m1m',
    factor_data=factor_panel,      # pd.DataFrame, T×N
    returns_data=returns_panel,    # pd.DataFrame, T×N
    market_cap_data=mcap_panel,    # pd.DataFrame, T×N (optional)
)
print(f"Health score: {report.health_score:.1f}, Level: {report.health_level.value}")

# Batch evaluation
results = monitor.evaluate_health_batch(
    {'momentum': data1, 'value': data2},
    returns_data=returns_panel,
    market_cap_data=mcap_panel,
)

# Trend query
trend = monitor.get_health_trend('momentum_12m1m', lookback=12)
```

---

## 5. Key Data Flow

### 5.1 Complete Processing Pipeline

```
Input: Factor panel data (T × N) + Construction description text
  │
  ├─[1]─ FactorFingerprinter.extract_fingerprint(data)
  │      └─ Output: FactorFingerprint (13 metrics)
  │
  ├─[2]─ FactorSemanticUnderstanding.understand(description)
  │      └─ Output: ExtractedRule (semantic analysis result)
  │
  ├─[3]─ SemanticPrior.from_extracted_rule(rule)
  │      └─ Output: SemanticPrior (prior distribution)
  │
  ├─[4]─ BayesianFactorClassifier.classify_with_prior(fp, prior)
  │      └─ Output: ClassificationResult (fusion classification)
  │
  ├─[5]─ ConflictArbitrator.arbitrate(semantic, statistical, data_sufficiency)
  │      └─ Output: ArbitratedResult (arbitration result)
  │
  └─[6]─ FactorFingerprintMonitor.add_fingerprint(name, fp)
         └─ Output: MigrationAlert[] (migration alerts, if any)

  └─[7]─ FactorHealthMonitor.evaluate_health(name, data, returns, mcap)
         └─ Output: FactorHealthReport (five-dimensional health report)
```

### 5.2 Data Flow Diagram

```
                    ┌───────────────────┐
                    │  Factor panel data T×N │
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │ FactorFingerprinter│
                    │  extract_fingerprint│
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │ FactorFingerprint  │
                    │ (13-dim fingerprint │
                    │  vector)           │
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

## 6. Module Dependencies

```
                    ┌──────────────────┐
                    │   __init__.py    │  (Package entry)
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                  │
    ┌─────────▼────┐ ┌──────▼───────┐ ┌────────▼─────────┐
    │ fingerprint  │ │  classifier  │ │    semantic       │
    │   .py        │ │    .py       │ │     .py           │
    │              │◄┤              │ │                   │
    │ FactorType   │ │ imports      │ │ (independent, no  │
    │ FactorFinger │ │ FactorFinger │ │  internal deps)   │
    │ print        │ │ print,       │ │                   │
    │              │ │ FactorType   │ │ FinancialTokenizer│
    └──────┬───────┘ └──────┬───────┘ │ SemanticRoleLabel │
           │                │         │ FinancialKnowledge│
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

Dependency layers:
  Layer 0: fingerprint.py (base layer, no internal dependencies)
  Layer 1: classifier.py (depends on fingerprint), semantic.py (independent)
  Layer 2: semantic_fusion.py (depends on fingerprint + classifier + semantic)
  Layer 3: monitor.py (depends on fingerprint + classifier)
  Layer 4: health.py (depends on fingerprint + classifier)
```

---

## 7. External Dependencies

### 7.1 Required Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | — | Numerical computation, matrix operations |
| `pandas` | — | Panel data processing, DataFrame |
| `scipy` | — | Statistical tests (Spearman, chi2, Jensen-Shannon) |
| `jieba` | — | Chinese tokenization (semantic understanding module) |

### 7.2 Optional Dependencies

| Package | Purpose | Fallback |
|---------|---------|----------|
| `statsmodels` | Ljung-Box test (`acorr_ljungbox`) | Manual implementation (`_manual_ljungbox`) |

### 7.3 Installation Command

```bash
pip install numpy pandas scipy jieba
```

---

## 8. Usage

### 8.1 Running the Demo Script

```bash
cd Factor_Fingerprint
python demo.py
```

The demo script contains 5 scenarios:

1. **`demo_fingerprint()`** — Fingerprint extraction: generates 3 types of simulated data with different AR(1) strengths, extracts and displays fingerprint metrics
2. **`demo_classifier()`** — Factor classification: classifies 3 factor types, shows hard and soft classification results
3. **`demo_semantic()`** — Semantic understanding: performs semantic analysis on 5 factor construction descriptions
4. **`demo_integration()`** — Complete pipeline: semantic understanding → fingerprint extraction → classification → consistency check
5. **`demo_health()`** — Health monitoring: evaluates factor five-dimensional health, displays composite score and alerts

### 8.2 Running Tests

```bash
cd Factor_Fingerprint
python tests/test_semantic_statistical_fusion.py
```

Test coverage:
- `TestSemanticPrior` (5 tests): Semantic prior creation, AR(1) mapping, creation from ExtractedRule
- `TestBayesianClassifier` (4 tests): No-prior / consistent-prior / conflicting-prior classification
- `TestConflictArbitrator` (4 tests): Consistent / insufficient-data / observation-period / persistent-conflict arbitration
- `TestSemanticStatisticalIntegration` (3 tests): End-to-end pipeline, cold start, conflict scenarios

### 8.3 Programming Interface Usage

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

# 1. Fingerprint extraction
fingerprinter = FactorFingerprinter()
fp = fingerprinter.extract_fingerprint(factor_data)  # pd.DataFrame, T×N

# 2. Pure statistical classification
classifier = AdaptiveFactorClassifier()
result = classifier.classify(fp)

# 3. Semantic-statistical fusion
fusion = SemanticStatisticalFusion()
arbitrated = fusion.classify(
    description="Price-to-earnings ratio factor, based on latest financial report data",
    fingerprint=fp,
    data_months=36
)

# 4. Migration monitoring
monitor = FactorFingerprintMonitor()
monitor.add_fingerprint('PB', fp_t1)
alerts = monitor.check_type_migration('PB', fp_t2)
```

---

## 9. Testing

**Test file**: [tests/test_semantic_statistical_fusion.py](file:///f:/Coding/Factor_Fingerprint/tests/test_semantic_statistical_fusion.py)

**Health test files** (6 total):
- [tests/test_health_phase2.py](file:///f:/Coding/Factor_Fingerprint/tests/test_health_phase2.py) — Efficacy metrics (28 tests)
- [tests/test_health_phase3.py](file:///f:/Coding/Factor_Fingerprint/tests/test_health_phase3.py) — Decay metrics (19 tests)
- [tests/test_health_phase4.py](file:///f:/Coding/Factor_Fingerprint/tests/test_health_phase4.py) — Crowding metrics (16 tests)
- [tests/test_health_phase5.py](file:///f:/Coding/Factor_Fingerprint/tests/test_health_phase5.py) — Capacity metrics (11 tests)
- [tests/test_health_phase6.py](file:///f:/Coding/Factor_Fingerprint/tests/test_health_phase6.py) — Regime sensitivity (11 tests)
- [tests/test_health_phase7.py](file:///f:/Coding/Factor_Fingerprint/tests/test_health_phase7.py) — Composite scoring + integration (25 tests)

**Test framework**: `pytest` (health), `unittest` (fusion)

**Test Classes**:

| Test Class | Test Count | Coverage |
|------------|------------|----------|
| `TestSemanticPrior` | 5 | Prior creation, AR(1) mapping (STATIC/DYNAMIC/MIXED), creation from ExtractedRule |
| `TestBayesianClassifier` | 4 | No-prior degradation, consistent-prior confidence boost, insufficient-data trust semantics, soft classification on conflict |
| `TestConflictArbitrator` | 4 | Consistent high-confidence, insufficient-data semantic override, observation-period downgrade to mixed, persistent-conflict human review |
| `TestSemanticStatisticalIntegration` | 3 | Complete pipeline, cold start, conflict scenarios |

**How to Run**:

```bash
# Fusion system tests (unittest)
python tests/test_semantic_statistical_fusion.py

# Health module tests (pytest, all 110 tests)
python -m pytest tests/test_health_phase*.py -v

# Run only a specific phase
python -m pytest tests/test_health_phase7.py -v
```

---

## 10. Design Principles

### 10.1 Core Principles

1. **Data-driven Adaptation**: The factor pipeline is automatically determined by fingerprint metrics, requiring no manual intervention
2. **Lookahead Bias Protection**: Fingerprints are computed on expanding windows, with no future information leakage; memory decay mechanism ensures recent data gets higher weight
3. **Prior Knowledge Fusion**: Construction-rule semantic analysis (semantic prior) + statistical fingerprint (data posterior), Bayesian fusion
4. **Intermediate State Tracking**: Fingerprint history is traceable, type migration is monitorable, smooth transition supported
5. **C¹ Continuity Preservation**: Smooth transition during type migration (exponential decay weights), avoiding pipeline discontinuity caused by hard switching

### 10.2 Robustness Design

| Design Point | Implementation |
|--------------|----------------|
| Triple screening | `_compute_ar1_median`: valid sample count + coefficient of variation + minimum stock count |
| NaN safety | All metric computations handle NaN, return NaN when conditions not met |
| Fallback mechanism | `_test_volatility_clustering`: statsmodels → manual Ljung-Box |
| Upper-bound clamping | `_compute_level_diff_ic_ratio`: upper bound 10; `_derive_complexity_need`: upper bound 1.0 |
| Smoothing | JS divergence: histogram +1e-10 to avoid zero values; soft classification: sigmoid to avoid hard switching |

### 10.3 Configurability

All thresholds are exposed via `dataclass` configuration classes, supporting adjustment for different market environments:

- `FingerprintConfig`: window length, half-life, screening thresholds
- `ClassificationConfig`: AR(1) thresholds, sigmoid steepness, mixed zone width
- `MonitorConfig`: multi-time-scale windows, alert thresholds, smooth transition switch

---

## 11. Integration with External Systems

### 11.1 Integration with Factor_Decoupler

```
Factor_Fingerprint                Factor_Decoupler
─────────────────                ─────────────────
Extract fingerprint → Classify factor    Differentiated processing
    STATIC  ─────────────────────→  Skip decoupling (preserve cross-sectional ranking)
    DYNAMIC ─────────────────────→  Triple neutralization (extract innovation)
    MIXED   ─────────────────────→  Conditional decoupling
```

### 11.2 Comparison with Factor_Imputer

| Dimension | Factor_Imputer_v2.0 | Factor_Fingerprint |
|-----------|---------------------|--------------------|
| **Core goal** | Select optimal imputation strategy | Select optimal decoupling strategy |
| **Type system** | Business semantics (financial/valuation/technical/macro) | Investment logic (static/dynamic/mixed) |
| **Detection basis** | Distribution shape + missing pattern | Time-series persistence + cross-sectional stability |
| **Theoretical basis** | Heuristic rules | Time-series analysis + information theory |
| **Semantic fusion** | Keyword matching | Complete semantic-statistical Bayesian fusion |
| **Migration monitoring** | None | Yes (continuously monitors type changes) |

### 11.3 Position in Data Processing Pipeline

```
Data input
    ↓
[Factor_Imputer]  ← Business classification (financial/valuation/technical)
    ↓
Missing value imputation (cross-sectional median/ffill/rolling window)
    ↓
[Factor_Fingerprint]  ← Investment logic classification (static/dynamic/mixed)
    ↓
Decoupling decision (skip/triple neutralization/conditional decoupling)
    ↓
Pure signal output
```

---

## 12. API Quick Reference

### 12.1 Package-level Imports

```python
from Factor_Fingerprint import (
    # Fingerprint extraction
    FactorFingerprinter, FactorFingerprint, FingerprintConfig, FactorType,
    # Classification
    AdaptiveFactorClassifier, ClassificationConfig, ClassificationResult,
    # Semantic understanding
    FactorSemanticUnderstanding, FinancialTokenizer,
    FinancialKnowledgeGraph, SemanticMatcher,
    FactorMeta, ExtractedRule, extract_from_text, extract_to_meta,
    # Semantic-statistical fusion
    SemanticPrior, BayesianFactorClassifier,
    ConflictArbitrator, ConflictDiagnosis,
    ArbitratedResult, SemanticStatisticalFusion,
    # Migration monitoring
    FactorFingerprintMonitor, MonitorConfig, MigrationAlert,
    # Health monitoring
    FactorHealthMonitor, HealthConfig, HealthAlertLevel,
    FactorHealthReport, HealthAlert,
)
```

### 12.2 Common Patterns

**Pattern 1: Quick Classification (Pure Statistical)**

```python
fp = FactorFingerprinter().extract_fingerprint(data)
result = AdaptiveFactorClassifier().classify(fp)
print(result.primary_type.value)  # 'static' | 'dynamic' | 'mixed'
```

**Pattern 2: Semantic-Statistical Fusion (Recommended)**

```python
fusion = SemanticStatisticalFusion()
result = fusion.classify(
    description="Price-to-earnings ratio factor, based on latest financial report data",
    fingerprint=fp,
    data_months=36
)
print(f"Type: {result.primary_type.value}, Confidence: {result.confidence:.2%}")
```

**Pattern 3: Continuous Monitoring**

```python
monitor = FactorFingerprintMonitor()
for t, fp in enumerate(fingerprints_over_time):
    monitor.add_fingerprint('my_factor', fp)
    alerts = monitor.check_type_migration('my_factor', fp)
    if alerts:
        for alert in alerts:
            print(f"Alert: {alert.from_type.value} → {alert.to_type.value}")
```

**Pattern 4: Health Monitoring**

```python
health_monitor = FactorHealthMonitor()
report = health_monitor.evaluate_health(
    factor_name='momentum',
    factor_data=factor_panel,
    returns_data=returns_panel,
    market_cap_data=mcap_panel,
)
print(f"Health score: {report.health_score:.1f}, Level: {report.health_level.value}")
for alert in report.alerts:
    print(f"  [{alert.category}] {alert.metric_name}: {alert.recommendation}")
```

---

*This document was auto-generated by the Code Wiki generator based on source code analysis. Last updated: 2026-07-01*
