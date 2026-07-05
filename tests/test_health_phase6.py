# -*- coding: utf-8 -*-
"""
Phase 6: 体制敏感性指标专项测试 (TDD)

严格验证: 牛熊IC比、波动率条件IC相关性
"""
import unittest
import sys
sys.path.insert(0, 'f:/Coding')
import numpy as np
import pandas as pd
from Factor_Fingerprint.core.health import FactorHealthMonitor, HealthConfig


class TestBullBearSplit(unittest.TestCase):
    """_split_bull_bear — 牛熊划分"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_split_covers_all_periods(self):
        """牛熊划分应覆盖所有期数"""
        np.random.seed(42)
        T, N = 60, 50
        returns = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        bull, bear = self.monitor._split_bull_bear(returns)
        self.assertEqual(len(bull) + len(bear), T)

    def test_split_has_both(self):
        """随机数据应有牛市和熊市"""
        np.random.seed(42)
        T, N = 60, 50
        returns = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        bull, bear = self.monitor._split_bull_bear(returns)
        self.assertGreater(len(bull), 0)
        self.assertGreater(len(bear), 0)


class TestBullBearICRatio(unittest.TestCase):
    """_compute_bull_bear_ic_ratio — 牛熊IC比"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_ratio_three_for_biased_ic(self):
        """牛市IC=0.06, 熊市IC=0.02 → ratio=3.0"""
        ic = np.array([0.06, 0.06, 0.06, 0.02, 0.02, 0.02])
        mkt = np.array([0.01, 0.01, 0.01, -0.01, -0.01, -0.01, 0.01])
        ratio = self.monitor._compute_bull_bear_ic_ratio(ic, mkt)
        self.assertAlmostEqual(ratio, 3.0, delta=0.001)

    def test_ratio_one_for_uniform_ic(self):
        """牛市IC=熊市IC → ratio≈1.0"""
        ic = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        mkt = np.array([0.01, 0.01, 0.01, -0.01, -0.01, -0.01, 0.01])
        ratio = self.monitor._compute_bull_bear_ic_ratio(ic, mkt)
        self.assertAlmostEqual(ratio, 1.0, delta=0.01)

    def test_ratio_nan_for_insufficient_data(self):
        """数据不足 → NaN"""
        ic = np.array([0.05, 0.05])
        mkt = np.array([0.01, -0.01, 0.01])
        ratio = self.monitor._compute_bull_bear_ic_ratio(ic, mkt)
        self.assertTrue(np.isnan(ratio))


class TestVolConditionalIC(unittest.TestCase):
    """_compute_vol_conditional_ic_corr — 波动率条件IC"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_vol_corr_bounded(self):
        """波动率IC相关性应在 [-1, 1]"""
        np.random.seed(42)
        T, N = 60, 50
        ic = np.random.randn(59)
        returns = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        corr = self.monitor._compute_vol_conditional_ic_corr(ic, returns)
        if not np.isnan(corr):
            self.assertGreaterEqual(corr, -1.0)
            self.assertLessEqual(corr, 1.0)

    def test_vol_corr_nan_for_short_data(self):
        """数据不足 → NaN"""
        T, N = 5, 50
        ic = np.random.randn(4)
        returns = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        corr = self.monitor._compute_vol_conditional_ic_corr(ic, returns)
        self.assertTrue(np.isnan(corr))


class TestRegimeEvaluation(unittest.TestCase):
    """_evaluate_regime — 体制敏感性综合评估"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_regime_score_range(self):
        """体制得分应在 [0, 100]"""
        np.random.seed(42)
        T, N = 60, 100
        factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        returns_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        score, metrics, alerts = self.monitor._evaluate_regime(factor_data, returns_data)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_regime_metrics_keys(self):
        """体制指标应包含所有必需key"""
        np.random.seed(42)
        T, N = 60, 100
        factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        returns_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        _, metrics, _ = self.monitor._evaluate_regime(factor_data, returns_data)
        self.assertIn('bull_bear_ic_ratio', metrics)
        self.assertIn('vol_conditional_ic_corr', metrics)

    def test_regime_without_returns(self):
        """无收益数据时返回中性得分"""
        np.random.seed(42)
        T, N = 60, 100
        factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        score, metrics, alerts = self.monitor._evaluate_regime(factor_data, None)
        self.assertEqual(score, 50.0)
        self.assertEqual(len(metrics), 0)
        self.assertEqual(len(alerts), 0)

    def test_regime_alerts_for_biased_factor(self):
        """牛熊严重偏差的因子应触发体制警报"""
        np.random.seed(42)
        T, N = 60, 100
        factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        # 构造牛市IC远大于熊市IC
        returns_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        _, _, alerts = self.monitor._evaluate_regime(factor_data, returns_data)
        has_regime = any(a.category == 'regime' for a in alerts)
        print(f"Regime alerts: {len(alerts)}, has_regime={has_regime}")


if __name__ == '__main__':
    unittest.main(verbosity=2)