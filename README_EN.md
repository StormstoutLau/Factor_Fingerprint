# Factor Fingerprint

[中文](README.md) | [English](README_EN.md)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT">
  <img src="https://img.shields.io/badge/Status-Stable-brightgreen?style=for-the-badge" alt="Status: Stable">
</p>

**Factor Fingerprint & Adaptive Classification System** — Generates a unique "fingerprint" for quantitative factors based on temporal stability, cross-sectional stability, and semantic understanding, enabling intelligent factor-type recognition and differentiated processing.

> **Core Capabilities**: Fingerprint Extraction → Adaptive Classification → Semantic-Statistical Fusion → Migration Monitoring → Health Monitoring

---

## Table of Contents

- [Core Features](#core-features)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Module Details](#module-details)
- [Academic Foundations](#academic-foundations)
- [Synergy with Factor_Decoupler](#synergy-with-factor_decoupler)
- [API Reference](#api-reference)
- [File Structure](#file-structure)
- [Version Information](#version-information)

---

## Core Features

### 1. Multi-Dimensional Factor Fingerprint Extraction

| Dimension | Metric | Description |
|-----------|--------|-------------|
| **Temporal Stability** | AR(1) median, rank autocorrelation, half-life | Measures the persistence of factor ranking |
| **Cross-Sectional Stability** | Skewness/kurtosis std, JS divergence | Measures the stability of the cross-sectional distribution |
| **Composite Derivatives** | Static-dynamic score, complexity need | Comprehensive judgment of factor type |

### 2. Adaptive Classification System

```
AR(1) > 0.80  →  STATIC (static factors: value, profitability, quality)
AR(1) < 0.40  →  DYNAMIC (dynamic factors: reversal, volatility changes)
0.40 ~ 0.80   →  MIXED (mixed factors: momentum)
```

Supports both **hard classification** (threshold-based) and **soft classification** (sigmoid probability-weighted) modes.

### 3. Semantic-Statistical Fusion

- **Trust Semantics on Cold Start**: When data is insufficient, semantic analysis is the only reliable basis
- **Trust Statistics with Sufficient Data**: Statistical fingerprints reflect the factor's essence
- **Conservative Downgrade on Conflict**: When semantics and statistics disagree, use the most conservative mixed pipeline

### 4. Migration Monitoring

Continuously monitors changes in factor fingerprints and issues alerts when factor types migrate. Supports a soft-transition weight mechanism.

### 5. Health Monitoring

Comprehensively evaluates factor health across five dimensions (crowding, efficacy, capacity, decay, regime sensitivity), produces a composite health score, and emits tiered alerts (HEALTHY / WATCH / WARNING / CRITICAL).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FactorFingerprinter                       │
│                    (Factor Fingerprint Extractor)            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Temporal    │  │ Cross-      │  │ Composite           │ │
│  │ Stability   │  │ Sectional   │  │ Derivatives         │ │
│  │ AR(1)       │  │ Skew/Kurt   │  │ Static-Dynamic      │ │
│  │ Rank AC     │  │ JS Div      │  │ Complexity          │ │
│  │ Half-life   │  │ Coverage    │  │                     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         └─────────────────┼────────────────────┘            │
│                           ↓                                  │
│              ┌─────────────────────────┐                     │
│              │   FactorFingerprint      │                     │
│              │   (Fingerprint Vector)   │                     │
│              └────────────┬────────────┘                     │
└───────────────────────────┼─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 AdaptiveFactorClassifier                     │
│                 (Adaptive Classifier)                        │
├─────────────────────────────────────────────────────────────┤
│  Hard: threshold → STATIC / DYNAMIC / MIXED                 │
│  Soft: sigmoid probability → static / mixed / dynamic prob  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              SemanticStatisticalFusion                       │
│              (Semantic-Statistical Fusion)                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ SemanticPrior│  │ Statistical │  │ ConflictArbitrator │ │
│  │ (Prior)     │  │ Fingerprint │  │ (Arbitrator)        │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         └─────────────────┼────────────────────┘            │
│                           ↓                                  │
│              ┌─────────────────────────┐                     │
│              │   ArbitratedResult       │                     │
│              │   (Fused Classification) │                     │
│              └─────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  FactorHealthMonitor                         │
│                  (Factor Health Monitor)                     │
├─────────────────────────────────────────────────────────────┤
│  Crowding  │ Efficacy │ Capacity │ Decay │ Regime           │
│  0.25      │ 0.35     │ 0.15     │ 0.15  │ 0.10            │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/StormstoutLau/factor_pipeline.git
cd factor_pipeline/Factor_Fingerprint

# Install dependencies
pip install numpy pandas scipy jieba
```

### Basic Usage

```python
import pandas as pd
from Factor_Fingerprint import (
    FactorFingerprinter,
    AdaptiveFactorClassifier,
    FactorType,
)

# Prepare factor data (T x N): time x stocks
factor_data = pd.DataFrame(...)

# 1. Extract fingerprint
fingerprinter = FactorFingerprinter()
fingerprint = fingerprinter.extract_fingerprint(factor_data)

print(f"AR(1) median: {fingerprint.ar1_median:.4f}")
print(f"Static-dynamic score: {fingerprint.sd_score:.4f}")

# 2. Classify
classifier = AdaptiveFactorClassifier()
result = classifier.classify(fingerprint)

print(f"Hard class: {result['hard_class'].value}")
print(f"Soft probability: static={result['soft_prob']['static']:.2%}")
```

### Semantic-Statistical Fusion

```python
from Factor_Fingerprint import (
    SemanticStatisticalFusion,
    SemanticPrior,
)

# Build semantic prior
semantic = SemanticPrior.from_description(
    name="momentum factor",
    description="cumulative return over the past 12 months excluding the most recent month",
    lookback_windows=[12, 1],
)

# Fused classification
fusion = SemanticStatisticalFusion()
result = fusion.classify(
    fingerprint=fingerprint,
    semantic_prior=semantic,
    data_coverage=0.8,  # data coverage
)

print(f"Fused class: {result.final_type.value}")
print(f"Confidence: {result.confidence:.2%}")
```

### Running the Demo

```bash
python demo.py
```

The demo script covers 5 scenarios: fingerprint extraction, classification, semantic understanding, end-to-end integration, and health monitoring.

---

## Module Details

### core/fingerprint.py — Factor Fingerprint Extraction

| Class/Function | Description |
|----------------|-------------|
| `FactorFingerprinter` | Main fingerprint extractor class |
| `FactorFingerprint` | Fingerprint named tuple |
| `FactorType` | Factor type enum (STATIC/DYNAMIC/MIXED) |
| `FingerprintConfig` | Fingerprint extraction configuration |

**Fingerprint Metric Reference**:

| Metric | Type | Description | Static Typical | Dynamic Typical |
|--------|------|-------------|----------------|-----------------|
| `ar1_median` | Temporal | AR(1) coefficient median | > 0.80 | < 0.40 |
| `rank_autocorr` | Temporal | Cross-sectional rank autocorrelation | > 0.95 | < 0.80 |
| `half_life` | Temporal | Autocorrelation half-life | > 10 periods | < 3 periods |
| `skewness_std` | Cross-sectional | Skewness std | < 0.05 | > 0.15 |
| `js_divergence_mean` | Cross-sectional | JS divergence mean | < 0.05 | > 0.15 |
| `sd_score` | Composite | Static-dynamic score | > 0.70 | < 0.30 |

**Threshold Design Rationale**:

| Metric | Threshold | Role | Theoretical Basis |
|--------|-----------|------|-------------------|
| **AR(1)** | > 0.80 / < 0.40 | **Primary classifier** | Box-Jenkins time-series analysis: 0.80 marks "strong autocorrelation", 0.40 marks "weak autocorrelation" |
| **Rank autocorrelation** | > 0.95 / < 0.80 | Auxiliary validation | Spearman rank correlation: > 0.95 means cross-sectional ranking is nearly unchanged, < 0.80 means significant ranking changes |
| **Half-life** | > 10 / < 3 periods | Auxiliary metric | Information decay theory: lag at which autocorrelation drops to 0.5; > 10 periods indicates extreme persistence |
| **Skewness std** | < 0.05 / > 0.15 | Cross-sectional stability | Distribution shape stability: low variance in period skewness indicates a stable cross-section |
| **JS divergence** | < 0.05 / > 0.15 | Cross-sectional stability | Jensen-Shannon information theory: similarity between adjacent cross-sections; < 0.05 is nearly identical |
| **SD score** | > 0.70 / < 0.30 | Composite metric | Weighted blend: AR(1) 35% + rank AC 30% + half-life 20% + level/diff IC ratio 15% |

**Key Design Points**:

1. **AR(1) is the sole hard-classification criterion**: Hard classification uses only AR(1) thresholds (0.80 / 0.40); other metrics feed soft probabilities and composite assessment
2. **Thresholds are empirical**: Tuned for typical financial-factor behavior, not strict statistical test critical values
3. **Fully configurable**: All thresholds are adjustable via `ClassificationConfig` to suit different markets

**Financial Intuition**:

- **Static factors** (PB, ROE): Fundamentals change slowly; AR(1) typically > 0.80 with stable cross-sectional ranking (rank AC > 0.95)
- **Dynamic factors** (reversal, turnover changes): Signals decay quickly; AR(1) typically < 0.40 with large ranking changes (rank AC < 0.80)
- **Mixed factors** (momentum): In between, 0.40 ~ 0.80

### core/classifier.py — Adaptive Classification

| Class | Description |
|-------|-------------|
| `AdaptiveFactorClassifier` | Adaptive classifier |
| `ClassificationConfig` | Classification threshold config |
| `ClassificationResult` | Classification result |

**Classification thresholds** (configurable):

```python
config = ClassificationConfig(
    static_ar1_threshold=0.80,   # AR(1) > 0.80 → static
    dynamic_ar1_threshold=0.40,  # AR(1) < 0.40 → dynamic
    soft_boundary=True,          # enable soft boundary
    sigmoid_steepness=10.0,      # sigmoid steepness
)
```

### core/semantic.py — Semantic Understanding

| Class/Function | Description |
|----------------|-------------|
| `FactorSemanticUnderstanding` | Main semantic understanding system |
| `FinancialTokenizer` | Finance-enhanced tokenizer |
| `FinancialKnowledgeGraph` | Financial knowledge graph |
| `SemanticMatcher` | Semantic similarity matcher |
| `extract_from_text` | Extract factor metadata from text |

**Supported semantic analyses**:
- Factor name recognition (momentum, value, quality, etc.)
- Time-window extraction ("past 12 months", "20 days")
- Data source recognition (financials, market data, alternative data)
- Construction-method classification (cumulative return, ratio, ranking, etc.)

### core/semantic_fusion.py — Semantic-Statistical Fusion

| Class | Description |
|-------|-------------|
| `SemanticStatisticalFusion` | Main fusion classifier |
| `SemanticPrior` | Semantic prior distribution |
| `BayesianFactorClassifier` | Bayesian classifier |
| `ConflictArbitrator` | Conflict arbitration engine |
| `ArbitratedResult` | Arbitrated result |

**Conflict-handling strategy**:

| Data Coverage | Trust Bias | Conflict Handling |
|---------------|------------|-------------------|
| < 30% | Semantics first | Use semantic classification, flag low confidence |
| 30% ~ 70% | Weighted fusion | Bayesian posterior = prior × likelihood |
| > 70% | Statistics first | Use statistical classification, semantics as auxiliary validation |

### core/monitor.py — Migration Monitoring

| Class | Description |
|-------|-------------|
| `FactorFingerprintMonitor` | Fingerprint monitor |
| `MonitorConfig` | Monitoring config |
| `MigrationAlert` | Migration alert |

### core/health.py — Factor Health Monitoring

| Class | Description |
|-------|-------------|
| `FactorHealthMonitor` | Factor health monitor — main entry |
| `HealthConfig` | Health monitoring configuration |
| `FactorHealthReport` | Composite health report |
| `HealthAlert` | Single health alert |
| `HealthAlertLevel` | Alert level enum (HEALTHY / WATCH / WARNING / CRITICAL) |

**Five Evaluation Dimensions**:

| Dimension | Weight | Description | Key Sub-metrics |
|-----------|--------|-------------|-----------------|
| **Crowding** | 0.25 | Higher score = less crowded | Pairwise correlation concentration, HHI, turnover, return reversal |
| **Efficacy** | 0.35 | Highest weight — "efficacy is the precondition for a factor to exist" | IC IR, IC win rate, rolling IC |
| **Capacity** | 0.15 | Higher score = larger capacity | Effective holdings, Top5 concentration |
| **Decay** | 0.15 | Higher score = slower decay | Mann-Kendall trend, long/short return ratio |
| **Regime Sensitivity** | 0.10 | Higher score = more regime-robust | Bull/bear IC ratio, volatility-IC correlation |

**Four Alert Levels**:

| Level | Trigger Condition |
|-------|-------------------|
| `HEALTHY` | All metrics normal |
| `WATCH` | 1–2 metrics approaching threshold |
| `WARNING` | 1–2 metrics exceeding threshold |
| `CRITICAL` | Multiple metrics exceeding threshold + decay trend |

**Design Philosophy** (consistent with the rest of the project):
- Data-driven adaptation: health scores are computed automatically from statistical metrics
- Look-ahead-bias protection: all rolling metrics use only historical data
- Intermediate-state tracking: health history is fully traceable
- Multi-dimensional orthogonal evaluation: five dimensions computed independently, then weighted-fused

**Typical usage**:

```python
from Factor_Fingerprint import FactorHealthMonitor

monitor = FactorHealthMonitor()
report = monitor.evaluate_health(
    factor_name='momentum_with_decay',
    factor_data=factor_df,        # T×N factor panel
    returns_data=returns_df,     # T×N returns panel (optional)
    market_cap_data=mcap_df,     # T×N market-cap panel (optional)
)

print(f"Health score: {report.health_score:.1f}/100")
print(f"Level: {report.health_level.value}")
print(f"Crowding: {report.crowding_score:.1f}")
print(f"Efficacy: {report.efficacy_score:.1f}")
```

---

## Academic Foundations

### Classical Foundations

- **Barra Multi-Factor Model** — Standard framework for factor classification and risk decomposition
- **Carhart (1997)** — Momentum factor definition in the four-factor model
- **Fama-French (1993, 2015)** — Classic definitions of value, size, profitability, and investment factors

### Frontier Method Connections

| Method | Reference | Connection to This Implementation |
|--------|-----------|-----------------------------------|
| **Factor models** | Bai & Ng (2002), *Econometrica* | Fingerprint metrics extendable to factor-loading analysis |
| **Interactive fixed effects** | Bai (2009), *Econometrica* | Theoretical basis for cross-sectional stability analysis |
| **Text analysis** | Loughran & McDonald (2011), *JFE* | Lexicon-based approach for financial text analysis |
| **Knowledge graph** | Financial semantic networks | Graph representation of factor-construction rules |

---

## Synergy with Factor_Decoupler

Factor_Fingerprint and Factor_Decoupler form a complete factor-processing pipeline:

```
Factor_Fingerprint                Factor_Decoupler
─────────────────                ─────────────────
Extract fingerprint → Classify       Differentiated processing
    STATIC  ─────────────────────→  Skip decoupling (preserve cross-sectional ranking)
    DYNAMIC ─────────────────────→  Triple neutralization (extract innovation)
    MIXED   ─────────────────────→  Conditional decoupling
```

See [factor_pipeline/pipelines_v2.py](../factor_pipeline/pipelines_v2.py) for the detailed synergy design.

---

## Difference from Factor_Imputer

### Positioning

| Dimension | Factor_Imputer_v2.0 | Factor_Fingerprint |
|-----------|---------------------|--------------------|
| **Core goal** | Choose optimal **imputation strategy** | Choose optimal **decoupling strategy** |
| **Type system** | Business semantics (financial/valuation/technical/macro) | Investment logic (static/dynamic/mixed) |
| **Detection basis** | Distribution shape + missing pattern | Temporal persistence + cross-sectional stability |
| **Theoretical basis** | Heuristic rules | Time-series analysis + information theory |
| **Output use** | Imputation-method parameters (grouping, weights) | Decoupling/neutralization decisions |
| **Semantic fusion** | Keyword matching (planned) | Full Bayesian semantic-statistical fusion |
| **Migration monitoring** | None | Yes (continuous type-change monitoring) |

### Type-System Comparison

**Factor_Imputer business classification** (for imputation):
- `financial` — Financial factors (ROE, revenue, etc.) → industry-group median imputation
- `valuation` — Valuation factors (PE, PB, etc.) → industry-group median imputation
- `growth` — Growth factors (growth rates) → KNN/rolling imputation
- `technical` — Technical factors (MACD, RSI) → forward fill
- `quality` — Quality factors → industry-group imputation
- `risk` — Risk factors → market-cap-group imputation
- `macro` — Macro factors → forward fill / rolling window

**Factor_Fingerprint investment-logic classification** (for decoupling):
- `STATIC` — Static factors (PB, ROE) → **skip decoupling**, preserve cross-sectional ranking
- `DYNAMIC` — Dynamic factors (reversal, turnover changes) → **triple neutralization**, extract temporal innovation
- `MIXED` — Mixed factors (momentum) → **conditional decoupling**

### Complementary Relationship

```
Data input
    ↓
[Factor_Imputer]  ← business classification (financial/valuation/technical)
    ↓
Missing-value imputation (cross-sectional median / ffill / rolling window)
    ↓
[Factor_Fingerprint]  ← investment-logic classification (static/dynamic/mixed)
    ↓
Decoupling decision (skip / triple neutralization / conditional)
    ↓
Clean signal output
```

**Analogy**:
- Factor_Imputer's factor types = **Hospital department classification** (internal medicine / surgery / pediatrics) → determines treatment (imputation method)
- Factor_Fingerprint's factor types = **Disease nature classification** (acute / chronic / mixed) → determines medication strategy (decoupling strategy)

The two are **complementary, not substitutes**, jointly forming a complete factor preprocessing pipeline.

---

## API Reference

### FactorFingerprinter

| Method | Signature | Description |
|--------|-----------|-------------|
| `extract_fingerprint` | `(data: pd.DataFrame) -> FactorFingerprint` | Extract a single factor's fingerprint |
| `batch_extract` | `(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, FactorFingerprint]` | Batch extraction |

### AdaptiveFactorClassifier

| Method | Signature | Description |
|--------|-----------|-------------|
| `classify` | `(fp: FactorFingerprint) -> Dict` | Classify a single factor |
| `batch_classify` | `(fps: Dict[str, FactorFingerprint]) -> Dict[str, Dict]` | Batch classification |

### SemanticStatisticalFusion

| Method | Signature | Description |
|--------|-----------|-------------|
| `classify` | `(fingerprint, semantic_prior, data_coverage) -> ArbitratedResult` | Fused classification |

### FactorHealthMonitor

| Method | Signature | Description |
|--------|-----------|-------------|
| `evaluate_health` | `(factor_name, factor_data, returns_data=None, market_cap_data=None, volume_data=None) -> FactorHealthReport` | Evaluate a single factor's health across 5 dimensions |
| `evaluate_health_batch` | `(factor_dict, returns_data=None, market_cap_data=None, volume_data=None) -> Dict[str, FactorHealthReport]` | Batch health evaluation |
| `get_health_summary` | `(factor_name=None) -> pd.DataFrame` | Health summary table (latest report per factor, or single factor's history) |

---

## File Structure

```
Factor_Fingerprint/
├── __init__.py                 # Package entry, exports core classes
├── core/
│   ├── __init__.py             # Core module entry
│   ├── fingerprint.py          # Factor fingerprint extractor
│   ├── classifier.py           # Adaptive classifier
│   ├── semantic.py             # Semantic understanding system
│   ├── semantic_fusion.py      # Semantic-statistical fusion
│   ├── monitor.py              # Migration monitoring
│   └── health.py               # Factor health monitor
├── tests/
│   ├── test_semantic_statistical_fusion.py  # Fusion tests
│   ├── test_health_phase2.py                # Health module phase-2 tests
│   ├── test_health_phase3.py                # Health module phase-3 tests
│   ├── test_health_phase4.py                # Health module phase-4 tests
│   ├── test_health_phase5.py                # Health module phase-5 tests
│   ├── test_health_phase6.py                # Health module phase-6 tests
│   └── test_health_phase7.py                # Health module phase-7 tests
├── docs/
│   ├── factor_prior_semantic_analysis.md    # Semantic analysis doc
│   ├── factor_rule_based_fingerprint.md     # Rule-based fingerprint doc
│   ├── health_module_implementation_plan.md # Health module implementation plan
│   ├── nlp_rule_extraction_analysis.md      # NLP rule extraction
│   └── nlp_semantic_improvement.md          # Semantic improvement
├── demo.py                     # Demo script (5 scenarios)
└── README.md                   # This document
```

---

## Version Information

- **Version**: v1.0.0
- **Python**: 3.10+
- **Dependencies**: numpy, pandas, scipy, jieba
- **Build date**: 2026.05.17
- **Status**: STABLE
- **License**: MIT
