# -*- coding: utf-8 -*-
"""
Phase 2: 效能指标专项测试 (TDD)

严格验证每个效能指标的计算准确性。
"""
import unittest
import sys
sys.path.insert(0, 'f:/Coding')
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from Factor_Fingerprint.core.health import FactorHealthMonitor, HealthConfig


class TestRankIC(unittest.TestCase):
    """_compute_rank_ic — 单期Rank IC计算"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_perfect_positive_rank_ic(self):
        """完全正相关: x=[1..20], y=[1..20] → Spearman = 1.0"""
        stocks = [f'S{i}' for i in range(20)]
        x = pd.Series(range(1, 21), index=stocks, dtype=float)
        y = pd.Series(range(1, 21), index=stocks, dtype=float)
        ic = self.monitor._compute_rank_ic(x, y)
        self.assertAlmostEqual(ic, 1.0, delta=0.001)

    def test_perfect_negative_rank_ic(self):
        """完全负相关: x=[1..20], y=[20..1] → Spearman = -1.0"""
        stocks = [f'S{i}' for i in range(20)]
        x = pd.Series(range(1, 21), index=stocks, dtype=float)
        y = pd.Series(range(20, 0, -1), index=stocks, dtype=float)
        ic = self.monitor._compute_rank_ic(x, y)
        self.assertAlmostEqual(ic, -1.0, delta=0.001)

    def test_zero_rank_ic(self):
        """无关: 随机数据 → Spearman 不接近 ±1.0"""
        np.random.seed(42)
        stocks = [f'S{i}' for i in range(50)]
        x = pd.Series(np.random.randn(50), index=stocks)
        y = pd.Series(np.random.randn(50), index=stocks)
        ic = self.monitor._compute_rank_ic(x, y)
        # 随机数据不应接近 ±1.0 (50个样本，|IC| < 0.5 的概率 > 99.9%)
        self.assertLess(abs(ic), 0.5)

    def test_insufficient_stocks_returns_nan(self):
        """少于10只股票应返回NaN"""
        stocks = [f'S{i}' for i in range(5)]
        x = pd.Series(range(1, 6), index=stocks, dtype=float)
        y = pd.Series(range(1, 6), index=stocks, dtype=float)
        ic = self.monitor._compute_rank_ic(x, y)
        self.assertTrue(np.isnan(ic))

    def test_mismatched_indices(self):
        """指数不匹配时只使用交集"""
        stocks_x = [f'X{i}' for i in range(15)]
        stocks_y = [f'Y{i}' for i in range(10)] + [f'X{i}' for i in range(5)]
        x = pd.Series(range(15), index=stocks_x, dtype=float)
        y = pd.Series(range(15), index=stocks_y, dtype=float)
        ic = self.monitor._compute_rank_ic(x, y)
        # 只有5个交集，但阈值是10，应返回NaN
        self.assertTrue(np.isnan(ic))

    def test_constant_factor_returns_nan(self):
        """常数值因子应返回NaN（无变异）"""
        stocks = [f'S{i}' for i in range(20)]
        x = pd.Series(np.ones(20), index=stocks)
        y = pd.Series(range(20), index=stocks, dtype=float)
        ic = self.monitor._compute_rank_ic(x, y)
        self.assertTrue(np.isnan(ic))


class TestICSeries(unittest.TestCase):
    """_compute_ic_series — IC序列计算"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_ic_series_length(self):
        """IC序列长度 = T-1"""
        np.random.seed(42)
        T, N = 30, 50
        factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        returns_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        ic = self.monitor._compute_ic_series(factor_data, returns_data)
        self.assertLessEqual(len(ic), T - 1)

    def test_ic_series_all_finite(self):
        """IC序列应全部为有限值"""
        np.random.seed(42)
        T, N = 30, 50
        factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        returns_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        ic = self.monitor._compute_ic_series(factor_data, returns_data)
        self.assertTrue(np.all(np.isfinite(ic)))

    def test_ic_series_positive_for_effective_factor(self):
        """有效因子（IC=0.10，强信号）应有正IC均值"""
        np.random.seed(123)
        T, N = 60, 100
        factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        # 强信号: 0.10 * factor + 小噪声
        returns_data = pd.DataFrame(
            0.10 * factor_data.values + 0.01 * np.random.randn(T, N),
            columns=[f'S{i}' for i in range(N)]
        )
        ic = self.monitor._compute_ic_series(factor_data, returns_data)
        mean_ic = np.mean(ic)
        print(f"IC mean: {mean_ic:.4f}")
        self.assertGreater(mean_ic, 0.0)


class TestICIR(unittest.TestCase):
    """_compute_ic_ir — IC信息比"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_ic_ir_manual_calculation(self):
        """手算验证: ICIR = mean(IC) / std(IC)"""
        ic = np.array([0.05, 0.03, 0.04, 0.06, 0.02])
        icir = self.monitor._compute_ic_ir(ic)
        manual = np.mean(ic) / np.std(ic, ddof=1)
        self.assertAlmostEqual(icir, manual, delta=0.0001)

    def test_ic_ir_high_for_consistent_ic(self):
        """高一致性IC → 高ICIR（微小波动，非零标准差）"""
        ic = np.array([0.049, 0.050, 0.051, 0.050, 0.049, 0.051, 0.050, 0.049])
        icir = self.monitor._compute_ic_ir(ic)
        self.assertGreater(icir, 20.0)  # 均值0.05, std很小 → 高ICIR

    def test_ic_ir_negative_for_negative_ic(self):
        """负IC均值 → 负ICIR"""
        ic = np.array([-0.05, -0.03, -0.04])
        icir = self.monitor._compute_ic_ir(ic)
        self.assertLess(icir, 0.0)

    def test_ic_ir_nan_for_insufficient_data(self):
        """数据不足 → NaN"""
        ic = np.array([0.05, 0.03])
        icir = self.monitor._compute_ic_ir(ic)
        self.assertTrue(np.isnan(icir))

    def test_ic_ir_nan_for_zero_std(self):
        """零标准差 → NaN"""
        ic = np.array([0.05, 0.05, 0.05])
        icir = self.monitor._compute_ic_ir(ic)
        self.assertTrue(np.isnan(icir))


class TestICWinRate(unittest.TestCase):
    """_compute_ic_win_rate — IC胜率"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_win_rate_manual(self):
        """手算验证: 3/5 = 0.6"""
        ic = np.array([0.05, -0.02, 0.03, 0.01, -0.01])
        wr = self.monitor._compute_ic_win_rate(ic)
        self.assertAlmostEqual(wr, 0.6, delta=0.001)

    def test_win_rate_all_positive(self):
        """全部正IC → 1.0"""
        ic = np.array([0.05, 0.03, 0.04])
        wr = self.monitor._compute_ic_win_rate(ic)
        self.assertAlmostEqual(wr, 1.0, delta=0.001)

    def test_win_rate_all_negative(self):
        """全部负IC → 0.0"""
        ic = np.array([-0.05, -0.03, -0.04])
        wr = self.monitor._compute_ic_win_rate(ic)
        self.assertAlmostEqual(wr, 0.0, delta=0.001)

    def test_win_rate_nan_for_insufficient(self):
        """数据不足 → NaN"""
        ic = np.array([0.05, 0.03])
        wr = self.monitor._compute_ic_win_rate(ic)
        self.assertTrue(np.isnan(wr))


class TestICAutocorr(unittest.TestCase):
    """_compute_ic_autocorr — IC自相关"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_autocorr_positive_persistence(self):
        """正持续性IC → 正自相关"""
        # 缓慢上升的序列: [0.01, 0.02, 0.03, ..., 0.20] → 正自相关
        ic = np.linspace(0.01, 0.20, 30)
        ac = self.monitor._compute_ic_autocorr(ic)
        self.assertGreater(ac, 0.0)

    def test_autocorr_alternating(self):
        """交替IC → 负自相关"""
        ic = np.array([0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05] * 3)
        ac = self.monitor._compute_ic_autocorr(ic)
        self.assertLess(ac, 0.0)

    def test_autocorr_nan_for_insufficient(self):
        """数据不足 → NaN"""
        ic = np.array([0.05, 0.03, 0.04, 0.02])
        ac = self.monitor._compute_ic_autocorr(ic)
        self.assertTrue(np.isnan(ac))

    def test_autocorr_manual_verification(self):
        """手算验证: 手动计算corr(IC_t, IC_{t-1})"""
        ic = np.array([0.05, 0.03, 0.04, 0.06, 0.02, 0.05, 0.03])
        ac = self.monitor._compute_ic_autocorr(ic)
        manual = pd.Series(ic).autocorr(1)
        self.assertAlmostEqual(ac, manual, delta=0.0001)


class TestEfficacyEvaluation(unittest.TestCase):
    """_evaluate_efficacy — 效能综合评估"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()
        np.random.seed(42)
        T, N = 60, 100
        self.factor_data = pd.DataFrame(
            np.random.randn(T, N), columns=[f'S{i}' for i in range(N)]
        )
        # 有效因子
        self.returns_data = pd.DataFrame(
            0.05 * self.factor_data.values + 0.02 * np.random.randn(T, N),
            columns=[f'S{i}' for i in range(N)]
        )

    def test_efficacy_score_range(self):
        """效能得分应在 [0, 100]"""
        score, metrics, alerts = self.monitor._evaluate_efficacy(
            self.factor_data, self.returns_data
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_efficacy_metrics_keys(self):
        """效能指标应包含所有必需key"""
        _, metrics, _ = self.monitor._evaluate_efficacy(
            self.factor_data, self.returns_data
        )
        self.assertIn('ic_ir', metrics)
        self.assertIn('rolling_ic_mean', metrics)
        self.assertIn('ic_win_rate', metrics)
        self.assertIn('ic_autocorr', metrics)

    def test_efficacy_alerts_for_weak_factor(self):
        """弱因子应触发警报"""
        # 完全随机IC → 低效能
        np.random.seed(42)
        returns_random = pd.DataFrame(
            np.random.randn(60, 100), columns=[f'S{i}' for i in range(100)]
        )
        _, _, alerts = self.monitor._evaluate_efficacy(
            self.factor_data, returns_random
        )
        self.assertGreater(len(alerts), 0)

    def test_efficacy_without_returns(self):
        """无收益数据时返回中性得分"""
        score, metrics, alerts = self.monitor._evaluate_efficacy(
            self.factor_data, None
        )
        self.assertEqual(score, 50.0)
        self.assertEqual(len(metrics), 0)
        self.assertEqual(len(alerts), 0)


class TestRollingICMean(unittest.TestCase):
    """_compute_rolling_ic_mean — 滚动IC均值"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_rolling_mean_uses_window(self):
        """滚动均值应使用配置的窗口长度"""
        ic = np.arange(1, 21, dtype=float)
        mean = self.monitor._compute_rolling_ic_mean(ic)
        # 默认window=12, 应使用最后12个值
        manual = np.mean(ic[-12:])
        self.assertAlmostEqual(mean, manual, delta=0.0001)

    def test_rolling_mean_short_series(self):
        """短序列应使用全部数据"""
        ic = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = self.monitor._compute_rolling_ic_mean(ic)
        self.assertAlmostEqual(mean, 3.0, delta=0.0001)


if __name__ == '__main__':
    unittest.main(verbosity=2)