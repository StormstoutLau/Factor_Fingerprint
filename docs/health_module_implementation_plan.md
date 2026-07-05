# health.py 执行方案

> 因子健康度监测模块 — 架构一致性实施计划
> 版本: v1.0 | 日期: 2026-06-30

---

## 0. 架构约定提取 (从现有模块逆向)

对 `fingerprint.py` / `classifier.py` / `monitor.py` / `semantic_fusion.py` 四个模块做结构对比，提取出以下统一约定，新模块必须严格遵守：

### 0.1 文件级约定

```
# -*- coding: utf-8 -*-
"""
模块中文名 (English Name)

一句话描述 + 设计哲学（与项目保持一致）。
"""
```

### 0.2 导入顺序

```python
# 1. 标准库 typing
from typing import Dict, Any, List, Optional, Tuple, NamedTuple

# 2. 标准库 dataclasses / enum
from dataclasses import dataclass, field
from enum import Enum

# 3. 标准库 datetime
from datetime import datetime

# 4. 数值计算
import numpy as np
import pandas as pd
from scipy import stats

# 5. 日志
import logging

# 6. 项目内部相对导入
from .fingerprint import FactorFingerprint, FactorType
from .classifier import ClassificationResult, AdaptiveFactorClassifier

logger = logging.getLogger(__name__)
```

### 0.3 数据类层次约定

| 层级 | 类型 | 用途 | 示例 |
|------|------|------|------|
| **枚举** | `Enum` | 分类标签 | `FactorType`, `ConflictDiagnosis` |
| **不可变记录** | `NamedTuple` | 指标向量/警报 | `FactorFingerprint`, `MigrationAlert` |
| **可配置参数** | `@dataclass` | 阈值/窗口 | `FingerprintConfig`, `MonitorConfig` |
| **可变结果** | `@dataclass` | 含状态的结果 | `ClassificationResult`, `ArbitratedResult` |

**关键规则**:
- NamedTuple 必须有 `to_dict()` 方法
- dataclass 的 Config 类使用 `field(default_factory=...)` 处理可变默认值
- 所有字段带类型注解和 `# 行内注释`

### 0.4 主类约定

```
class XxxMonitor:
    """
    中文描述

    Usage:
        monitor = XxxMonitor()
        result = monitor.main_method(...)
    """

    def __init__(self, config: Optional[XxxConfig] = None):
        self.config = config or XxxConfig()
        logger.info(f"XxxMonitor initialized with config: {self.config}")

    # ==================== 公共 API ====================

    def main_method(self, ...) -> XxxResult:
        """完整文档: Parameters / Returns"""

    def batch_method(self, ...) -> Dict[str, XxxResult]:
        """批量处理"""

    def get_summary(self, ...) -> pd.DataFrame:
        """汇总表格"""

    # ==================== 私有实现 ====================

    def _compute_xxx(self, ...) -> float:
        """单指标计算"""

    def _derive_xxx(self, ...) -> float:
        """综合衍生指标"""
```

### 0.5 方法文档约定

所有公共方法遵循:
```python
def method_name(self, param1: Type1, param2: Type2) -> ReturnType:
    """
    一句话功能描述

    详细说明（可选，复杂逻辑时添加）

    Parameters
    ----------
    param1 : Type1
        参数说明
    param2 : Type2
        参数说明

    Returns
    -------
    ReturnType
        返回值说明
    """
```

### 0.6 鲁棒性约定

- 所有指标计算前检查数据长度: `if len(data) < min_required: return np.nan`
- 所有除法前检查分母: `if abs(denom) < 1e-10: return np.nan`
- 所有结果截断: `return float(min(value, upper_bound))`
- 所有统计检验前检查变异: `if series.nunique() <= 1: return np.nan`
- 日志级别: `logger.info` 用于关键操作, `logger.warning` 用于异常但可处理, `logger.debug` 用于中间状态

### 0.7 配置的可变默认值约定

```python
@dataclass
class XxxConfig:
    # 简单类型直接赋值
    threshold: float = 0.40

    # 可变类型用 field(default_factory=...)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'a': 0.5, 'b': 0.5
    })
```

---

## 1. 模块设计

### 1.1 文件: `core/health.py`

### 1.2 数据类体系

#### 枚举: `HealthAlertLevel`

```python
class HealthAlertLevel(Enum):
    """健康度警报等级"""
    HEALTHY = "healthy"       # 健康: 所有指标正常
    WATCH = "watch"           # 关注: 1-2个指标接近阈值
    WARNING = "warning"       # 警告: 1-2个指标超阈值
    CRITICAL = "critical"     # 危急: 多个指标超阈值 + 衰减趋势
```

#### 不可变记录: `HealthAlert` (NamedTuple)

```python
class HealthAlert(NamedTuple):
    """健康度单项警报"""
    metric_name: str              # 指标名称
    metric_value: float           # 当前值
    threshold: float              # 预警阈值
    direction: str                # 'above' 或 'below' (超阈值方向)
    level: HealthAlertLevel       # 警报等级
    category: str                 # 'crowding'/'efficacy'/'capacity'/'decay'/'regime'
    timestamp: datetime
    recommendation: str           # 建议措施

    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'direction': self.direction,
            'level': self.level.value,
            'category': self.category,
            'timestamp': self.timestamp.isoformat(),
            'recommendation': self.recommendation,
        }
```

#### 可配置参数: `HealthConfig` (dataclass)

```python
@dataclass
class HealthConfig:
    """健康度监测配置"""
    # ---- 拥挤度 ----
    crowding_corr_threshold: float = 0.40        # 配对相关性集中度阈值
    crowding_hhi_threshold: float = 0.10         # 持仓集中度HHI阈值
    crowding_turnover_threshold: float = 0.80    # 年化换手率阈值
    crowding_reversal_threshold: float = -0.10   # 收益反转阈值

    # ---- 效能 ----
    efficacy_icir_threshold: float = 0.30        # IC IR阈值
    efficacy_ic_win_rate_threshold: float = 0.55 # IC胜率阈值
    efficacy_rolling_ic_window: int = 12         # 滚动IC窗口

    # ---- 容量 ----
    capacity_effective_n_threshold: int = 20     # 有效持仓数阈值
    capacity_top5_concentration_threshold: float = 0.40  # Top5集中度阈值

    # ---- 衰减 ----
    decay_mk_trend_pvalue: float = 0.05          # Mann-Kendall趋势p值
    decay_long_short_ratio_threshold: float = 0.50  # 多空收益衰减比
    decay_lookback_long: int = 36                # 长期回看窗口
    decay_lookback_short: int = 12               # 短期回看窗口

    # ---- 体制敏感性 ----
    regime_bull_bear_ratio_threshold: float = 2.0    # 牛熊IC比上限
    regime_bull_bear_ratio_lower: float = 0.5        # 牛熊IC比下限
    regime_vol_corr_threshold: float = 0.30          # 波动率IC相关性阈值

    # ---- 综合 ----
    health_score_weights: Dict[str, float] = field(default_factory=lambda: {
        'efficacy': 0.35,     # 效能权重最高: "有效"是因子存在的前提
        'crowding': 0.25,
        'capacity': 0.15,
        'decay': 0.15,
        'regime': 0.10,
    })
    rolling_window: int = 12               # 默认滚动窗口
    min_data_months: int = 12              # 最少数据月数
```

#### 可变结果: `FactorHealthReport` (dataclass)

```python
@dataclass
class FactorHealthReport:
    """因子健康度综合报告"""
    factor_name: str
    timestamp: datetime

    # 综合评分
    health_score: float                  # 0-100 综合健康分
    health_level: HealthAlertLevel

    # 分维度得分 (0-100, 越高越好)
    crowding_score: float                # 拥挤度得分 (高=不拥挤)
    efficacy_score: float                # 效能得分
    capacity_score: float                # 容量得分
    decay_score: float                   # 衰减得分 (高=衰减慢)
    regime_score: float                  # 体制稳健性得分

    # 详细指标字典
    crowding_metrics: Dict[str, float] = field(default_factory=dict)
    efficacy_metrics: Dict[str, float] = field(default_factory=dict)
    capacity_metrics: Dict[str, float] = field(default_factory=dict)
    decay_metrics: Dict[str, float] = field(default_factory=dict)
    regime_metrics: Dict[str, float] = field(default_factory=dict)

    # 警报
    alerts: List[HealthAlert] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'factor_name': self.factor_name,
            'timestamp': self.timestamp.isoformat(),
            'health_score': self.health_score,
            'health_level': self.health_level.value,
            'crowding_score': self.crowding_score,
            'efficacy_score': self.efficacy_score,
            'capacity_score': self.capacity_score,
            'decay_score': self.decay_score,
            'regime_score': self.regime_score,
            'crowding_metrics': self.crowding_metrics,
            'efficacy_metrics': self.efficacy_metrics,
            'capacity_metrics': self.capacity_metrics,
            'decay_metrics': self.decay_metrics,
            'regime_metrics': self.regime_metrics,
            'alerts': [a.to_dict() for a in self.alerts],
        }
```

### 1.3 主类: `FactorHealthMonitor`

```python
class FactorHealthMonitor:
    """
    因子健康度监测器

    综合评估因子的拥挤度、效能、容量、衰减趋势和体制敏感性，
    生成健康度综合报告和分级警报。

    设计哲学（与项目保持一致）：
    - 数据驱动自适应：健康评分由统计指标自动计算
    - 前瞻偏差防护：所有滚动指标仅使用历史数据
    - 中间状态追踪：健康度历史可追溯
    - 多维度正交评估：五维指标独立计算，加权融合

    Usage:
        monitor = FactorHealthMonitor()
        report = monitor.evaluate_health(
            factor_name='PB',
            factor_data=factor_df,       # T×N 因子面板
            returns_data=returns_df,     # T×N 收益面板 (可选)
            market_cap_data=mcap_df,     # T×N 市值面板 (可选)
        )
        print(f"健康分: {report.health_score:.0f}/100")
        print(f"等级: {report.health_level.value}")
    """

    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        self.fingerprinter = FactorFingerprinter()
        self.classifier = AdaptiveFactorClassifier()
        # 健康度历史
        self.health_history: Dict[str, List[FactorHealthReport]] = {}
        # 警报历史
        self.alert_history: List[HealthAlert] = []
        logger.info(f"FactorHealthMonitor initialized with config: {self.config}")

    # ==================== 公共 API ====================

    def evaluate_health(self,
                        factor_name: str,
                        factor_data: pd.DataFrame,
                        returns_data: Optional[pd.DataFrame] = None,
                        market_cap_data: Optional[pd.DataFrame] = None,
                        volume_data: Optional[pd.DataFrame] = None,
                        ) -> FactorHealthReport:
        """
        评估因子健康度

        五维评估：拥挤度 → 效能 → 容量 → 衰减 → 体制敏感性

        Parameters
        ----------
        factor_name : str
            因子名称
        factor_data : pd.DataFrame, shape (T, N)
            因子面板数据
        returns_data : pd.DataFrame, optional
            收益面板数据，用于IC计算和收益反转
        market_cap_data : pd.DataFrame, optional
            市值面板数据，用于容量计算
        volume_data : pd.DataFrame, optional
            成交量面板数据，用于流动性计算

        Returns
        -------
        FactorHealthReport
            健康度综合报告
        """

    def evaluate_health_batch(self,
                              factor_dict: Dict[str, pd.DataFrame],
                              returns_data: Optional[pd.DataFrame] = None,
                              market_cap_data: Optional[pd.DataFrame] = None,
                              volume_data: Optional[pd.DataFrame] = None,
                              ) -> Dict[str, FactorHealthReport]:
        """批量评估多个因子的健康度"""

    def get_health_summary(self,
                           factor_name: Optional[str] = None
                           ) -> pd.DataFrame:
        """获取健康度摘要表"""

    def get_health_trend(self,
                         factor_name: str,
                         lookback: int = 12
                         ) -> pd.DataFrame:
        """获取健康度趋势（最近N期）"""

    # ==================== 综合评分 ====================

    def _compute_health_score(self,
                              dim_scores: Dict[str, float],
                              ) -> Tuple[float, HealthAlertLevel]:
        """加权合成健康分 + 等级判定"""

    def _determine_health_level(self,
                                score: float,
                                alerts: List[HealthAlert],
                                ) -> HealthAlertLevel:
        """
        判定健康等级

        CRITICAL: 多个维度超阈值 + 衰减趋势
        WARNING:  1-2个维度超阈值
        WATCH:    接近阈值但未超标
        HEALTHY:  所有指标正常
        """

    def _normalize_score(self,
                         value: float,
                         best: float,
                         worst: float,
                         ) -> float:
        """将指标值映射到 [0, 100] 得分"""

    # ==================== 拥挤度 (Crowding) ====================

    def _evaluate_crowding(self,
                           factor_data: pd.DataFrame,
                           returns_data: Optional[pd.DataFrame] = None,
                           ) -> Tuple[float, Dict[str, float], List[HealthAlert]]:
        """
        评估因子拥挤度

        指标:
        1. 配对相关性集中度 (pairwise_corr_concentration)
        2. 持仓集中度HHI (position_hhi)
        3. 因子换手率 (turnover)
        4. 收益反转风险 (return_reversal)
        """

    def _compute_pairwise_corr_concentration(self,
                                             factor_data: pd.DataFrame,
                                             top_quantile: float = 0.2
                                             ) -> float:
        """配对相关性集中度: 多头组合内股票的平均两两相关性"""

    def _compute_position_hhi(self,
                              factor_data: pd.DataFrame,
                              ) -> float:
        """持仓集中度: 等权多头的 HHI = Σ(1/N)²"""

    def _compute_turnover(self,
                          factor_data: pd.DataFrame,
                          ) -> float:
        """因子换手率: mean(|rank_t - rank_{t-1}|) / N"""

    def _compute_return_reversal(self,
                                 factor_data: pd.DataFrame,
                                 returns_data: pd.DataFrame,
                                 ) -> float:
        """收益反转: 因子多头组合收益的1阶自相关"""

    # ==================== 效能 (Efficacy) ====================

    def _evaluate_efficacy(self,
                           factor_data: pd.DataFrame,
                           returns_data: pd.DataFrame,
                           ) -> Tuple[float, Dict[str, float], List[HealthAlert]]:
        """
        评估因子效能

        指标:
        1. IC信息比 (ic_ir)
        2. 滚动IC均值 (rolling_ic_mean)
        3. IC胜率 (ic_win_rate)
        4. IC自相关 (ic_autocorr)
        """

    def _compute_ic_series(self,
                           factor_data: pd.DataFrame,
                           returns_data: pd.DataFrame,
                           ) -> np.ndarray:
        """计算 Rank IC 序列: 每期截面因子值与下期收益的Spearman相关"""

    def _compute_ic_ir(self, ic_series: np.ndarray) -> float:
        """IC IR = mean(IC) / std(IC)"""

    def _compute_ic_win_rate(self, ic_series: np.ndarray) -> float:
        """IC胜率 = P(IC > 0)"""

    def _compute_ic_autocorr(self, ic_series: np.ndarray) -> float:
        """IC自相关: corr(IC_t, IC_{t-1})"""

    # ==================== 容量 (Capacity) ====================

    def _evaluate_capacity(self,
                           factor_data: pd.DataFrame,
                           market_cap_data: Optional[pd.DataFrame] = None,
                           ) -> Tuple[float, Dict[str, float], List[HealthAlert]]:
        """
        评估因子容量

        指标:
        1. 有效持仓数 (effective_n)
        2. Top5集中度 (top5_concentration)
        3. 市值加权集中度 (cap_weighted_concentration)
        """

    def _compute_effective_n(self,
                             factor_data: pd.DataFrame,
                             ) -> float:
        """有效持仓数: 1 / Σ(weight_i²) — 等权下的简化"""

    def _compute_top5_concentration(self,
                                    factor_data: pd.DataFrame,
                                    ) -> float:
        """Top5集中度: 前5大持仓占比"""

    def _compute_cap_weighted_concentration(self,
                                            factor_data: pd.DataFrame,
                                            market_cap_data: pd.DataFrame,
                                            ) -> float:
        """市值加权集中度"""

    # ==================== 衰减 (Decay) ====================

    def _evaluate_decay(self,
                        factor_data: pd.DataFrame,
                        returns_data: pd.DataFrame,
                        ) -> Tuple[float, Dict[str, float], List[HealthAlert]]:
        """
        评估因子衰减

        指标:
        1. Mann-Kendall IC趋势检验 (mk_trend_pvalue)
        2. 多空收益衰减比 (long_short_decay_ratio)
        3. 滚动IC斜率 (rolling_ic_slope)
        """

    def _compute_mann_kendall(self, series: np.ndarray) -> Tuple[float, float]:
        """Mann-Kendall 趋势检验: 返回 (tau, p_value)"""

    def _compute_long_short_decay(self,
                                  factor_data: pd.DataFrame,
                                  returns_data: pd.DataFrame,
                                  ) -> float:
        """多空收益衰减比: ret_short / ret_long"""

    def _compute_rolling_ic_slope(self,
                                  ic_series: np.ndarray,
                                  ) -> float:
        """滚动IC斜率: 对最近N期IC做线性回归"""

    # ==================== 体制敏感性 (Regime Sensitivity) ====================

    def _evaluate_regime(self,
                         factor_data: pd.DataFrame,
                         returns_data: pd.DataFrame,
                         ) -> Tuple[float, Dict[str, float], List[HealthAlert]]:
        """
        评估体制敏感性

        指标:
        1. 牛熊IC比 (bull_bear_ic_ratio)
        2. 波动率条件IC (vol_conditional_ic_corr)
        """

    def _split_bull_bear(self,
                         returns_data: pd.DataFrame,
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """基于市场收益正负划分牛熊期"""

    def _compute_bull_bear_ic_ratio(self,
                                    ic_series: np.ndarray,
                                    market_returns: np.ndarray,
                                    ) -> float:
        """牛熊IC比 = mean(IC|bull) / mean(IC|bear)"""

    def _compute_vol_conditional_ic_corr(self,
                                         ic_series: np.ndarray,
                                         returns_data: pd.DataFrame,
                                         ) -> float:
        """波动率条件IC: corr(IC_t, VIX_proxy_t) — VIX_proxy = 市场收益的滚动标准差"""

    # ==================== 工具方法 ====================

    def _check_data_sufficiency(self,
                                factor_data: pd.DataFrame,
                                returns_data: Optional[pd.DataFrame],
                                ) -> Tuple[bool, str]:
        """检查数据是否足够进行评估"""

    def _create_alert(self,
                      metric_name: str,
                      value: float,
                      threshold: float,
                      direction: str,
                      level: HealthAlertLevel,
                      category: str,
                      recommendation: str,
                      ) -> HealthAlert:
        """创建警报的工厂方法"""

    def _compute_rank_ic(self,
                         factor_t: pd.Series,
                         returns_t1: pd.Series,
                         ) -> float:
        """单期 Rank IC 计算"""
```

---

## 2. 分步实施计划

### Phase 1: 核心骨架 (1 session)

**目标**: 创建文件、数据类、空方法体，确保能 import 无报错

| 步骤 | 产出 | 验证 |
|------|------|------|
| 1.1 | 创建 `core/health.py` 文件框架 | `from Factor_Fingerprint.core.health import FactorHealthMonitor` 通过 |
| 1.2 | 实现 `HealthAlertLevel` 枚举 | — |
| 1.3 | 实现 `HealthAlert` (NamedTuple + to_dict) | — |
| 1.4 | 实现 `HealthConfig` (dataclass) | — |
| 1.5 | 实现 `FactorHealthReport` (dataclass + to_dict) | — |
| 1.6 | 实现 `FactorHealthMonitor.__init__` + 空方法签名 | 实例化无报错 |
| 1.7 | 更新 `core/__init__.py` 导出 | 包级导入通过 |

**验证命令**:
```bash
python -c "from Factor_Fingerprint.core.health import FactorHealthMonitor, HealthConfig, HealthAlertLevel; print('PASS')"
```

### Phase 2: 效能指标 (P0, 1 session)

**目标**: 实现效能维度的 4 个指标 + 评分逻辑

| 步骤 | 方法 | 验证 |
|------|------|------|
| 2.1 | `_compute_ic_series` | 用模拟数据验证 IC 序列形状 |
| 2.2 | `_compute_ic_ir` | 验证 ICIR 公式 |
| 2.3 | `_compute_ic_win_rate` | 验证正IC占比 |
| 2.4 | `_compute_ic_autocorr` | 验证自相关计算 |
| 2.5 | `_evaluate_efficacy` | 组合评分 + 警报生成 |
| 2.6 | `_normalize_score` 工具方法 | 验证 [0,100] 映射 |

**验证命令**:
```bash
python tests/test_health.py  # 测试 TestEfficacyMetrics
```

### Phase 3: 衰减指标 (P0, 1 session)

| 步骤 | 方法 | 验证 |
|------|------|------|
| 3.1 | `_compute_mann_kendall` | 对上升/下降/随机序列验证 |
| 3.2 | `_compute_long_short_decay` | 验证衰减比公式 |
| 3.3 | `_compute_rolling_ic_slope` | 验证斜率符号 |
| 3.4 | `_evaluate_decay` | 组合评分 + 警报生成 |

### Phase 4: 拥挤度指标 (P1, 1 session)

| 步骤 | 方法 | 验证 |
|------|------|------|
| 4.1 | `_compute_pairwise_corr_concentration` | 验证高相关面板 vs 低相关面板 |
| 4.2 | `_compute_position_hhi` | 验证等权 vs 集中持仓 |
| 4.3 | `_compute_turnover` | 验证稳定 vs 高换手 |
| 4.4 | `_compute_return_reversal` | 验证正/负自相关 |
| 4.5 | `_evaluate_crowding` | 组合评分 + 警报生成 |

### Phase 5: 容量指标 (P1, 1 session)

| 步骤 | 方法 | 验证 |
|------|------|------|
| 5.1 | `_compute_effective_n` | 验证 N=100 等权 → EN≈100 |
| 5.2 | `_compute_top5_concentration` | 验证集中度 |
| 5.3 | `_compute_cap_weighted_concentration` | 验证市值加权 |
| 5.4 | `_evaluate_capacity` | 组合评分 + 警报生成 |

### Phase 6: 体制敏感性 (P2, 1 session)

| 步骤 | 方法 | 验证 |
|------|------|------|
| 6.1 | `_split_bull_bear` | 验证牛熊划分 |
| 6.2 | `_compute_bull_bear_ic_ratio` | 验证牛熊IC差异 |
| 6.3 | `_compute_vol_conditional_ic_corr` | 验证波动率相关性 |
| 6.4 | `_evaluate_regime` | 组合评分 + 警报生成 |

### Phase 7: 综合评分 + 集成 (1 session)

| 步骤 | 产出 | 验证 |
|------|------|------|
| 7.1 | `_compute_health_score` 加权合成 | 验证五维加权公式 |
| 7.2 | `_determine_health_level` 等级判定 | 验证 CRITICAL/WARNING/WATCH/HEALTHY |
| 7.3 | `evaluate_health` 主方法整合 | 端到端测试 |
| 7.4 | `evaluate_health_batch` 批量方法 | 批量测试 |
| 7.5 | `get_health_summary` 摘要表 | DataFrame 输出 |
| 7.6 | `get_health_trend` 趋势 | 历史追踪 |
| 7.7 | `_check_data_sufficiency` 数据检查 | 边界条件 |

### Phase 8: 测试 + 文档 (1 session)

| 步骤 | 产出 | 验证 |
|------|------|------|
| 8.1 | `tests/test_health.py` 完整测试套件 | `python -m pytest tests/test_health.py` |
| 8.2 | `demo_health()` 添加到 demo.py | `python demo.py` 包含健康度场景 |
| 8.3 | 更新 CODE_WIKI.md | 新模块文档 |

---

## 3. 测试计划

### 3.1 测试文件: `tests/test_health.py`

```python
# -*- coding: utf-8 -*-
"""
因子健康度监测器测试
"""
import unittest
import numpy as np
import pandas as pd
from Factor_Fingerprint.core.health import (
    FactorHealthMonitor, HealthConfig, HealthAlertLevel,
    FactorHealthReport, HealthAlert,
)

class TestHealthConfig(unittest.TestCase):
    """配置类测试"""

    def test_default_config(self):
        config = HealthConfig()
        self.assertEqual(config.efficacy_icir_threshold, 0.30)

    def test_custom_config(self):
        config = HealthConfig(efficacy_icir_threshold=0.50)
        self.assertEqual(config.efficacy_icir_threshold, 0.50)

    def test_weights_sum_to_one(self):
        config = HealthConfig()
        w = config.health_score_weights
        self.assertAlmostEqual(sum(w.values()), 1.0, delta=0.01)


class TestEfficacyMetrics(unittest.TestCase):
    """效能指标测试"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()
        # 生成模拟数据: 强有效因子 (IC ~ 0.05)
        np.random.seed(42)
        self.T, self.N = 60, 100
        self.factor_data = pd.DataFrame(
            np.random.randn(self.T, self.N),
            index=pd.date_range('2020-01-01', periods=self.T, freq='M'),
            columns=[f'S{i}' for i in range(self.N)]
        )
        self.returns_data = pd.DataFrame(
            0.05 * self.factor_data.values + 0.02 * np.random.randn(self.T, self.N),
            index=self.factor_data.index,
            columns=self.factor_data.columns
        )

    def test_compute_ic_series(self):
        ic = self.monitor._compute_ic_series(self.factor_data, self.returns_data)
        self.assertGreater(len(ic), 0)
        self.assertTrue(np.all(np.isfinite(ic)))

    def test_ic_ir_positive_for_effective_factor(self):
        ic = self.monitor._compute_ic_series(self.factor_data, self.returns_data)
        ir = self.monitor._compute_ic_ir(ic)
        self.assertGreater(ir, 0.0)

    def test_ic_win_rate(self):
        ic = self.monitor._compute_ic_series(self.factor_data, self.returns_data)
        wr = self.monitor._compute_ic_win_rate(ic)
        self.assertGreater(wr, 0.0)
        self.assertLessEqual(wr, 1.0)

    def test_ic_autocorr(self):
        ic = self.monitor._compute_ic_series(self.factor_data, self.returns_data)
        ac = self.monitor._compute_ic_autocorr(ic)
        self.assertGreaterEqual(ac, -1.0)
        self.assertLessEqual(ac, 1.0)

    def test_evaluate_efficacy_returns_score(self):
        score, metrics, alerts = self.monitor._evaluate_efficacy(
            self.factor_data, self.returns_data
        )
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIn('ic_ir', metrics)


class TestDecayMetrics(unittest.TestCase):
    """衰减指标测试"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_mann_kendall_increasing_trend(self):
        series = np.arange(1, 51, dtype=float)
        tau, p_value = self.monitor._compute_mann_kendall(series)
        self.assertGreater(tau, 0.5)
        self.assertLess(p_value, 0.05)

    def test_mann_kendall_decreasing_trend(self):
        series = np.arange(50, 0, -1, dtype=float)
        tau, p_value = self.monitor._compute_mann_kendall(series)
        self.assertLess(tau, -0.5)
        self.assertLess(p_value, 0.05)

    def test_mann_kendall_random_no_trend(self):
        np.random.seed(42)
        series = np.random.randn(100)
        tau, p_value = self.monitor._compute_mann_kendall(series)
        self.assertGreater(p_value, 0.05)


class TestCrowdingMetrics(unittest.TestCase):
    """拥挤度指标测试"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()
        np.random.seed(42)
        self.T, self.N = 60, 50
        # 高相关因子: 所有股票高度同步
        self.high_corr_data = pd.DataFrame(
            np.random.randn(self.T, 1) + 0.1 * np.random.randn(self.T, self.N),
            columns=[f'S{i}' for i in range(self.N)]
        )
        # 低相关因子: 各股票独立
        self.low_corr_data = pd.DataFrame(
            np.random.randn(self.T, self.N),
            columns=[f'S{i}' for i in range(self.N)]
        )

    def test_pairwise_corr_higher_for_high_corr(self):
        high = self.monitor._compute_pairwise_corr_concentration(self.high_corr_data)
        low = self.monitor._compute_pairwise_corr_concentration(self.low_corr_data)
        self.assertGreater(high, low)

    def test_hhi_for_equal_weight(self):
        hhi = self.monitor._compute_position_hhi(self.low_corr_data)
        expected = 1.0 / self.N  # 等权 HHI
        self.assertAlmostEqual(hhi, expected, delta=0.01)

    def test_turnover_between_zero_and_one(self):
        to = self.monitor._compute_turnover(self.low_corr_data)
        self.assertGreaterEqual(to, 0.0)
        self.assertLessEqual(to, 1.0)


class TestCapacityMetrics(unittest.TestCase):
    """容量指标测试"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()
        np.random.seed(42)
        self.T, self.N = 60, 100
        self.data = pd.DataFrame(
            np.random.randn(self.T, self.N),
            columns=[f'S{i}' for i in range(self.N)]
        )

    def test_effective_n_equals_n_for_equal_weight(self):
        en = self.monitor._compute_effective_n(self.data)
        self.assertAlmostEqual(en, self.N, delta=5)

    def test_top5_concentration(self):
        conc = self.monitor._compute_top5_concentration(self.data)
        self.assertAlmostEqual(conc, 5.0 / self.N, delta=0.02)


class TestRegimeMetrics(unittest.TestCase):
    """体制敏感性测试"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()
        np.random.seed(42)
        self.T, self.N = 60, 100
        self.returns_data = pd.DataFrame(
            np.random.randn(self.T, self.N),
            columns=[f'S{i}' for i in range(self.N)]
        )

    def test_bull_bear_split(self):
        bull_idx, bear_idx = self.monitor._split_bull_bear(self.returns_data)
        self.assertGreater(len(bull_idx), 0)
        self.assertGreater(len(bear_idx), 0)
        self.assertEqual(len(bull_idx) + len(bear_idx), self.T)


class TestHealthScoreComputation(unittest.TestCase):
    """综合评分测试"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_normalize_score(self):
        self.assertEqual(self.monitor._normalize_score(0.5, 0.0, 1.0), 50.0)
        self.assertEqual(self.monitor._normalize_score(1.0, 0.0, 1.0), 100.0)
        self.assertEqual(self.monitor._normalize_score(0.0, 0.0, 1.0), 0.0)

    def test_health_level_healthy(self):
        alerts = []
        level = self.monitor._determine_health_level(85.0, alerts)
        self.assertEqual(level, HealthAlertLevel.HEALTHY)

    def test_health_level_warning(self):
        alerts = [HealthAlert(
            metric_name='ic_ir', metric_value=0.20, threshold=0.30,
            direction='below', level=HealthAlertLevel.WARNING,
            category='efficacy', timestamp=pd.Timestamp.now(),
            recommendation='关注IC IR下降'
        )]
        level = self.monitor._determine_health_level(55.0, alerts)
        self.assertEqual(level, HealthAlertLevel.WARNING)

    def test_health_level_critical(self):
        alerts = [
            HealthAlert('ic_ir', 0.15, 0.30, 'below', HealthAlertLevel.WARNING,
                       'efficacy', pd.Timestamp.now(), ''),
            HealthAlert('mk_trend', 0.01, 0.05, 'below', HealthAlertLevel.WARNING,
                       'decay', pd.Timestamp.now(), ''),
            HealthAlert('pairwise_corr', 0.50, 0.40, 'above', HealthAlertLevel.WARNING,
                       'crowding', pd.Timestamp.now(), ''),
        ]
        level = self.monitor._determine_health_level(35.0, alerts)
        self.assertEqual(level, HealthAlertLevel.CRITICAL)


class TestEndToEnd(unittest.TestCase):
    """端到端测试"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()
        np.random.seed(42)
        self.T, self.N = 60, 100
        self.factor_data = pd.DataFrame(
            np.random.randn(self.T, self.N),
            index=pd.date_range('2020-01-01', periods=self.T, freq='M'),
            columns=[f'S{i}' for i in range(self.N)]
        )
        self.returns_data = pd.DataFrame(
            0.05 * self.factor_data.values + 0.02 * np.random.randn(self.T, self.N),
            index=self.factor_data.index,
            columns=self.factor_data.columns
        )
        self.mcap_data = pd.DataFrame(
            np.abs(np.random.randn(self.T, self.N)) * 1e10,
            index=self.factor_data.index,
            columns=self.factor_data.columns
        )

    def test_full_evaluate_health(self):
        report = self.monitor.evaluate_health(
            'test_factor', self.factor_data, self.returns_data, self.mcap_data
        )
        self.assertIsInstance(report, FactorHealthReport)
        self.assertGreaterEqual(report.health_score, 0)
        self.assertLessEqual(report.health_score, 100)
        self.assertIsInstance(report.health_level, HealthAlertLevel)

    def test_evaluate_without_optional_data(self):
        """缺少可选数据时不应崩溃"""
        report = self.monitor.evaluate_health('test_factor', self.factor_data)
        self.assertIsInstance(report, FactorHealthReport)

    def test_batch_evaluate(self):
        factor_dict = {
            'f1': self.factor_data,
            'f2': self.factor_data * 0.5,
        }
        reports = self.monitor.evaluate_health_batch(
            factor_dict, self.returns_data, self.mcap_data
        )
        self.assertEqual(len(reports), 2)

    def test_health_summary(self):
        self.monitor.evaluate_health('f1', self.factor_data, self.returns_data)
        summary = self.monitor.get_health_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)

    def test_data_insufficient(self):
        """数据不足时应返回低置信度报告"""
        small_data = self.factor_data.iloc[:6]
        report = self.monitor.evaluate_health('small', small_data, self.returns_data.iloc[:6])
        self.assertIsInstance(report, FactorHealthReport)


if __name__ == '__main__':
    unittest.main()
```

---

## 4. 集成点

### 4.1 `core/__init__.py` 更新

```python
from .health import (
    FactorHealthMonitor, HealthConfig, HealthAlertLevel,
    FactorHealthReport, HealthAlert,
)
```

### 4.2 `__init__.py` 更新

```python
from .core.health import (
    FactorHealthMonitor, HealthConfig, HealthAlertLevel,
    FactorHealthReport, HealthAlert,
)
```

### 4.3 `demo.py` 新增场景

```python
def demo_health():
    """演示因子健康度监测"""
    print("\n" + "=" * 60)
    print("Demo 5: 因子健康度监测")
    print("=" * 60)

    monitor = FactorHealthMonitor(
        config=HealthConfig(efficacy_icir_threshold=0.30)
    )

    # 生成模拟数据
    ...
    report = monitor.evaluate_health(
        'test_factor', factor_data, returns_data, mcap_data
    )
    print(f"综合健康分: {report.health_score:.0f}/100")
    print(f"健康等级: {report.health_level.value}")
    print(f"  拥挤度: {report.crowding_score:.0f}")
    print(f"  效能:   {report.efficacy_score:.0f}")
    print(f"  容量:   {report.capacity_score:.0f}")
    print(f"  衰减:   {report.decay_score:.0f}")
    print(f"  体制:   {report.regime_score:.0f}")
    if report.alerts:
        print(f"\n  ⚠ 警报 ({len(report.alerts)}条):")
        for a in report.alerts:
            print(f"    [{a.level.value}] {a.category}/{a.metric_name}: {a.recommendation}")
```

---

## 5. 实施总览

| Phase | 内容 | 预估工作量 | 依赖 |
|-------|------|-----------|------|
| 1 | 核心骨架 | 1 session | — |
| 2 | 效能指标 (P0) | 1 session | Phase 1 |
| 3 | 衰减指标 (P0) | 1 session | Phase 1 |
| 4 | 拥挤度指标 (P1) | 1 session | Phase 1 |
| 5 | 容量指标 (P1) | 1 session | Phase 1 |
| 6 | 体制敏感性 (P2) | 1 session | Phase 1 |
| 7 | 综合评分 + 集成 | 1 session | Phase 2-6 |
| 8 | 测试 + 文档 | 1 session | Phase 7 |

**总计**: 8 sessions, 8 个独立可验证的交付物。