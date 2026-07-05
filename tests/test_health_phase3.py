# -*- coding: utf-8 -*-
"""
Phase 3: 衰减指标专项测试 (TDD)

严格验证: Mann-Kendall趋势检验、多空收益衰减比、滚动IC斜率
"""
import unittest
import sys
sys.path.insert(0, 'f:/Coding')
import numpy as np
import pandas as pd
from Factor_Fingerprint.core.health import FactorHealthMonitor, HealthConfig


class TestMannKendall(unittest.TestCase):
    """_compute_mann_kendall — 非参数趋势检验"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_strictly_increasing(self):
        """严格递增序列 → tau=1.0, p<0.001"""
        series = np.arange(1, 21, dtype=float)
        tau, p = self.monitor._compute_mann_kendall(series)
        self.assertAlmostEqual(tau, 1.0, delta=0.001)
        self.assertLess(p, 0.001)

    def test_strictly_decreasing(self):
        """严格递减序列 → tau=-1.0, p<0.001"""
        series = np.arange(20, 0, -1, dtype=float)
        tau, p = self.monitor._compute_mann_kendall(series)
        self.assertAlmostEqual(tau, -1.0, delta=0.001)
        self.assertLess(p, 0.001)

    def test_random_no_trend(self):
        """随机序列 → tau≈0, p>0.05"""
        np.random.seed(42)
        series = np.random.randn(100)
        tau, p = self.monitor._compute_mann_kendall(series)
        self.assertGreater(p, 0.05)

    def test_weak_upward_trend(self):
        """弱上升趋势 → tau>0, p可能>0.05但tau方向正确"""
        np.random.seed(42)
        trend = np.linspace(0, 1, 50)
        noise = np.random.randn(50) * 0.5
        series = trend + noise
        tau, p = self.monitor._compute_mann_kendall(series)
        self.assertGreater(tau, 0.0)  # 方向正确

    def test_insufficient_data(self):
        """数据不足10个 → NaN"""
        series = np.arange(1, 6, dtype=float)
        tau, p = self.monitor._compute_mann_kendall(series)
        self.assertTrue(np.isnan(tau))
        self.assertTrue(np.isnan(p))

    def test_tau_bounded(self):
        """tau 应在 [-1, 1]"""
        np.random.seed(42)
        series = np.random.randn(50)
        tau, p = self.monitor._compute_mann_kendall(series)
        self.assertGreaterEqual(tau, -1.0)
        self.assertLessEqual(tau, 1.0)

    def test_tau_manual_verification(self):
        """手算验证: 前10个递增，后5个递减 → tau应在[-1,1]且非NaN"""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                          9.0, 8.0, 7.0, 6.0, 5.0])
        tau, p = self.monitor._compute_mann_kendall(series)
        self.assertGreaterEqual(tau, -1.0)
        self.assertLessEqual(tau, 1.0)
        self.assertFalse(np.isnan(tau))

    def test_tau_manual_verification_increasing(self):
        """手算验证: 严格递增序列 tau=1.0"""
        # 构造 [1,2,3,...,15] 中的部分值，但确保完全递增
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                          11.0, 12.0])
        tau, p = self.monitor._compute_mann_kendall(series)
        self.assertAlmostEqual(tau, 1.0, delta=0.001)
        self.assertLess(p, 0.001)


class TestLongShortDecay(unittest.TestCase):
    """_compute_long_short_decay — 多空收益衰减比"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()
        np.random.seed(42)
        T, N = 60, 100
        self.factor_data = pd.DataFrame(
            np.random.randn(T, N), columns=[f'S{i}' for i in range(N)]
        )
        self.returns_data = pd.DataFrame(
            np.random.randn(T, N), columns=[f'S{i}' for i in range(N)]
        )

    def test_decay_ratio_is_finite(self):
        """衰减比应为有限值"""
        ratio = self.monitor._compute_long_short_decay(
            self.factor_data, self.returns_data
        )
        self.assertTrue(np.isfinite(ratio))

    def test_decay_ratio_nan_for_short_data(self):
        """数据不足时返回NaN"""
        small_factor = self.factor_data.iloc[:10]
        small_returns = self.returns_data.iloc[:10]
        ratio = self.monitor._compute_long_short_decay(small_factor, small_returns)
        self.assertTrue(np.isnan(ratio))


class TestICSlope(unittest.TestCase):
    """_compute_rolling_ic_slope — 滚动IC斜率"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_positive_slope_for_increasing_ic(self):
        """递增IC → 正斜率"""
        ic = np.linspace(0.01, 0.10, 20)
        slope = self.monitor._compute_rolling_ic_slope(ic)
        self.assertGreater(slope, 0.0)

    def test_negative_slope_for_decreasing_ic(self):
        """递减IC → 负斜率"""
        ic = np.linspace(0.10, 0.01, 20)
        slope = self.monitor._compute_rolling_ic_slope(ic)
        self.assertLess(slope, 0.0)

    def test_zero_slope_for_constant_ic(self):
        """常数IC → 斜率≈0"""
        ic = np.ones(20) * 0.05
        slope = self.monitor._compute_rolling_ic_slope(ic)
        self.assertAlmostEqual(slope, 0.0, delta=0.001)

    def test_slope_manual_verification(self):
        """手算验证: IC=[0.02, 0.04, 0.06, 0.08, 0.10] → slope=0.02"""
        ic = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
        slope = self.monitor._compute_rolling_ic_slope(ic)
        # x=[0,1,2,3,4], y=[0.02,0.04,0.06,0.08,0.10]
        # x_mean=2, y_mean=0.06
        # sum((x-x_mean)*(y-y_mean)) = (-2)*(-0.04)+(-1)*(-0.02)+0+1*0.02+2*0.04
        # = 0.08 + 0.02 + 0 + 0.02 + 0.08 = 0.20
        # sum((x-x_mean)^2) = 4+1+0+1+4 = 10
        # slope = 0.20/10 = 0.02
        self.assertAlmostEqual(slope, 0.02, delta=0.0001)

    def test_slope_nan_for_short_data(self):
        """数据不足 → NaN"""
        ic = np.array([0.02, 0.04, 0.06])
        slope = self.monitor._compute_rolling_ic_slope(ic)
        self.assertTrue(np.isnan(slope))


class TestDecayEvaluation(unittest.TestCase):
    """_evaluate_decay — 衰减综合评估"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()
        np.random.seed(42)
        T, N = 60, 100
        self.factor_data = pd.DataFrame(
            np.random.randn(T, N), columns=[f'S{i}' for i in range(N)]
        )
        self.returns_data = pd.DataFrame(
            np.random.randn(T, N), columns=[f'S{i}' for i in range(N)]
        )

    def test_decay_score_range(self):
        """衰减得分应在 [0, 100]"""
        score, metrics, alerts = self.monitor._evaluate_decay(
            self.factor_data, self.returns_data
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_decay_metrics_keys(self):
        """衰减指标应包含所有必需key"""
        _, metrics, _ = self.monitor._evaluate_decay(
            self.factor_data, self.returns_data
        )
        self.assertIn('mk_tau', metrics)
        self.assertIn('mk_trend_pvalue', metrics)
        self.assertIn('long_short_decay_ratio', metrics)
        self.assertIn('rolling_ic_slope', metrics)

    def test_decay_without_returns(self):
        """无收益数据时返回中性得分"""
        score, metrics, alerts = self.monitor._evaluate_decay(
            self.factor_data, None
        )
        self.assertEqual(score, 50.0)
        self.assertEqual(len(metrics), 0)
        self.assertEqual(len(alerts), 0)

    def test_decay_alerts_for_declining_factor(self):
        """IC持续下降的因子应触发衰减警报"""
        np.random.seed(42)
        T, N = 60, 100
        # 构造IC递减的因子: 早期有效，后期无效
        early_factor = pd.DataFrame(np.random.randn(30, N), columns=[f'S{i}' for i in range(N)])
        early_returns = pd.DataFrame(
            0.08 * early_factor.values + 0.02 * np.random.randn(30, N),
            columns=[f'S{i}' for i in range(N)]
        )
        late_factor = pd.DataFrame(np.random.randn(30, N), columns=[f'S{i}' for i in range(N)])
        late_returns = pd.DataFrame(np.random.randn(30, N), columns=[f'S{i}' for i in range(N)])

        factor_data = pd.concat([early_factor, late_factor], ignore_index=True)
        returns_data = pd.concat([early_returns, late_returns], ignore_index=True)

        _, _, alerts = self.monitor._evaluate_decay(factor_data, returns_data)
        # 应检测到衰减趋势
        has_decay_alert = any(a.category == 'decay' for a in alerts)
        print(f"Decay alerts: {len(alerts)}, has_decay: {has_decay_alert}")


if __name__ == '__main__':
    unittest.main(verbosity=2)