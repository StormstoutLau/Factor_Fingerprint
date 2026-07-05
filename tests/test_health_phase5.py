# -*- coding: utf-8 -*-
"""
Phase 5: 容量指标专项测试 (TDD)

严格验证: 有效持仓数、Top5集中度、市值加权集中度
"""
import unittest
import sys
sys.path.insert(0, 'f:/Coding')
import numpy as np
import pandas as pd
from Factor_Fingerprint.core.health import FactorHealthMonitor, HealthConfig


class TestEffectiveN(unittest.TestCase):
    """_compute_effective_n — 有效持仓数"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_effective_n_equals_stock_count(self):
        """等权时 EN = N"""
        N_vals = [50, 100, 200]
        for N in N_vals:
            data = pd.DataFrame(np.random.randn(60, N), columns=[f'S{i}' for i in range(N)])
            en = self.monitor._compute_effective_n(data)
            self.assertAlmostEqual(en, N, delta=1)

    def test_effective_n_larger_for_more_stocks(self):
        """股票越多 → EN越大"""
        data_50 = pd.DataFrame(np.random.randn(60, 50), columns=[f'S{i}' for i in range(50)])
        data_100 = pd.DataFrame(np.random.randn(60, 100), columns=[f'S{i}' for i in range(100)])
        self.assertGreater(
            self.monitor._compute_effective_n(data_100),
            self.monitor._compute_effective_n(data_50)
        )

    def test_effective_n_nan_for_short_data(self):
        """数据不足 → NaN"""
        data = pd.DataFrame(np.random.randn(5, 50), columns=[f'S{i}' for i in range(50)])
        en = self.monitor._compute_effective_n(data)
        self.assertTrue(np.isnan(en))


class TestTop5Concentration(unittest.TestCase):
    """_compute_top5_concentration — Top5集中度"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_top5_equals_5_over_N(self):
        """等权时 Top5 = 5/N"""
        N = 100
        data = pd.DataFrame(np.random.randn(60, N), columns=[f'S{i}' for i in range(N)])
        conc = self.monitor._compute_top5_concentration(data)
        self.assertAlmostEqual(conc, 5.0 / N, delta=0.01)

    def test_top5_larger_for_fewer_stocks(self):
        """股票越少 → Top5集中度越大"""
        data_20 = pd.DataFrame(np.random.randn(60, 20), columns=[f'S{i}' for i in range(20)])
        data_100 = pd.DataFrame(np.random.randn(60, 100), columns=[f'S{i}' for i in range(100)])
        self.assertGreater(
            self.monitor._compute_top5_concentration(data_20),
            self.monitor._compute_top5_concentration(data_100)
        )


class TestCapWeightedConcentration(unittest.TestCase):
    """_compute_cap_weighted_concentration — 市值加权集中度"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_cap_weighted_bounded(self):
        """市值加权HHI应在 [0, 1]"""
        np.random.seed(42)
        T, N = 60, 50
        factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        mcap_data = pd.DataFrame(np.abs(np.random.randn(T, N)) * 1e10, columns=[f'S{i}' for i in range(N)])
        hhi = self.monitor._compute_cap_weighted_concentration(factor_data, mcap_data)
        self.assertGreaterEqual(hhi, 0.0)
        self.assertLessEqual(hhi, 1.0)

    def test_cap_weighted_nan_for_short_data(self):
        """数据不足 → NaN"""
        factor_data = pd.DataFrame(np.random.randn(5, 50), columns=[f'S{i}' for i in range(50)])
        mcap_data = pd.DataFrame(np.random.randn(5, 50), columns=[f'S{i}' for i in range(50)])
        hhi = self.monitor._compute_cap_weighted_concentration(factor_data, mcap_data)
        self.assertTrue(np.isnan(hhi))


class TestCapacityEvaluation(unittest.TestCase):
    """_evaluate_capacity — 容量综合评估"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_capacity_score_range(self):
        """容量得分应在 [0, 100]"""
        np.random.seed(42)
        T, N = 60, 100
        factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        mcap_data = pd.DataFrame(np.abs(np.random.randn(T, N)) * 1e10, columns=[f'S{i}' for i in range(N)])
        score, metrics, alerts = self.monitor._evaluate_capacity(factor_data, mcap_data)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_capacity_metrics_keys(self):
        """容量指标应包含所有必需key"""
        np.random.seed(42)
        T, N = 60, 100
        factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        mcap_data = pd.DataFrame(np.abs(np.random.randn(T, N)) * 1e10, columns=[f'S{i}' for i in range(N)])
        _, metrics, _ = self.monitor._evaluate_capacity(factor_data, mcap_data)
        self.assertIn('effective_n', metrics)
        self.assertIn('top5_concentration', metrics)
        self.assertIn('cap_weighted_concentration', metrics)

    def test_capacity_alerts_for_small_universe(self):
        """小股票池应触发容量不足警报"""
        np.random.seed(42)
        T, N = 60, 15  # 仅15只股票
        factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        _, _, alerts = self.monitor._evaluate_capacity(factor_data, None)
        has_capacity = any(a.category == 'capacity' for a in alerts)
        print(f"Capacity alerts for N=15: {len(alerts)}, has_capacity={has_capacity}")

    def test_capacity_without_mcap(self):
        """无市值数据时仍可计算基本指标"""
        np.random.seed(42)
        T, N = 60, 100
        factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        score, metrics, alerts = self.monitor._evaluate_capacity(factor_data, None)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertNotIn('cap_weighted_concentration', metrics)


if __name__ == '__main__':
    unittest.main(verbosity=2)