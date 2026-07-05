# -*- coding: utf-8 -*-
"""
Phase 7: 综合评分+集成测试 (TDD)

严格验证: 加权合成、等级判定、端到端流水线、批量、摘要、趋势
"""
import unittest
import sys
sys.path.insert(0, 'f:/Coding')
import numpy as np
import pandas as pd
from Factor_Fingerprint.core.health import (
    FactorHealthMonitor, HealthConfig, HealthAlertLevel,
    FactorHealthReport, HealthAlert,
)


class TestNormalizeScore(unittest.TestCase):
    """_normalize_score — 指标归一化"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_midpoint(self):
        """中点值 → 50"""
        s = self.monitor._normalize_score(0.5, 0.0, 1.0)
        self.assertAlmostEqual(s, 50.0, delta=0.01)

    def test_best_value(self):
        """最优值 → 100"""
        s = self.monitor._normalize_score(0.0, 0.0, 1.0)  # best=0.0, worst=1.0
        self.assertAlmostEqual(s, 100.0, delta=0.01)

    def test_worst_value(self):
        """最差值 → 0"""
        s = self.monitor._normalize_score(1.0, 0.0, 1.0)  # best=0.0, worst=1.0
        self.assertAlmostEqual(s, 0.0, delta=0.01)

    def test_nan_returns_neutral(self):
        """NaN → 50 (中性)"""
        s = self.monitor._normalize_score(np.nan, 0.0, 1.0)
        self.assertAlmostEqual(s, 50.0, delta=0.01)

    def test_best_greater_than_worst(self):
        """best > worst: 高分=好"""
        s = self.monitor._normalize_score(100.0, 200.0, 50.0)
        self.assertAlmostEqual(s, 33.33, delta=0.1)

    def test_clipped_to_range(self):
        """超出范围应被截断到 [0, 100]"""
        s = self.monitor._normalize_score(2.0, 0.0, 1.0)
        self.assertAlmostEqual(s, 0.0, delta=0.01)
        s = self.monitor._normalize_score(-1.0, 0.0, 1.0)
        self.assertAlmostEqual(s, 100.0, delta=0.01)


class TestHealthScore(unittest.TestCase):
    """_compute_health_score — 加权合成"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_weighted_composite(self):
        """加权合成: 80*0.35 + 70*0.25 + 90*0.15 + 60*0.15 + 75*0.10 = 75.5"""
        dim_scores = {
            'efficacy': 80, 'crowding': 70, 'capacity': 90,
            'decay': 60, 'regime': 75
        }
        score, level = self.monitor._compute_health_score(dim_scores)
        expected = 80*0.35 + 70*0.25 + 90*0.15 + 60*0.15 + 75*0.10
        self.assertAlmostEqual(score, expected, delta=0.1)

    def test_all_perfect(self):
        """全满分 → 100"""
        dim_scores = {'efficacy': 100, 'crowding': 100, 'capacity': 100,
                      'decay': 100, 'regime': 100}
        score, level = self.monitor._compute_health_score(dim_scores)
        self.assertAlmostEqual(score, 100.0, delta=0.1)
        self.assertEqual(level, HealthAlertLevel.HEALTHY)

    def test_all_zero(self):
        """全零分 → 0"""
        dim_scores = {'efficacy': 0, 'crowding': 0, 'capacity': 0,
                      'decay': 0, 'regime': 0}
        score, level = self.monitor._compute_health_score(dim_scores)
        self.assertAlmostEqual(score, 0.0, delta=0.1)


class TestHealthLevel(unittest.TestCase):
    """_determine_health_level — 等级判定"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()

    def test_healthy_no_alerts(self):
        """无警报+高分 → HEALTHY"""
        level = self.monitor._determine_health_level(85.0, [])
        self.assertEqual(level, HealthAlertLevel.HEALTHY)

    def test_watch_low_score(self):
        """低分无警报 → WATCH"""
        level = self.monitor._determine_health_level(50.0, [])
        self.assertEqual(level, HealthAlertLevel.WATCH)

    def test_warning_one_dimension(self):
        """1个维度警报 → WARNING"""
        alerts = [HealthAlert('test', 0.1, 0.3, 'below', HealthAlertLevel.WARNING,
                             'efficacy', pd.Timestamp.now(), 'test')]
        level = self.monitor._determine_health_level(55.0, alerts)
        self.assertEqual(level, HealthAlertLevel.WARNING)

    def test_critical_three_dimensions(self):
        """3个维度警报 → CRITICAL"""
        alerts = [
            HealthAlert('a', 0.1, 0.3, 'below', HealthAlertLevel.WARNING,
                       'efficacy', pd.Timestamp.now(), ''),
            HealthAlert('b', 0.1, 0.3, 'below', HealthAlertLevel.WARNING,
                       'decay', pd.Timestamp.now(), ''),
            HealthAlert('c', 0.1, 0.3, 'below', HealthAlertLevel.WARNING,
                       'crowding', pd.Timestamp.now(), ''),
        ]
        level = self.monitor._determine_health_level(35.0, alerts)
        self.assertEqual(level, HealthAlertLevel.CRITICAL)


class TestEndToEnd(unittest.TestCase):
    """evaluate_health — 端到端流水线"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()
        np.random.seed(42)
        T, N = 60, 100
        dates = pd.date_range('2020-01-01', periods=T, freq='ME')
        stocks = [f'S{i}' for i in range(N)]
        self.factor_data = pd.DataFrame(np.random.randn(T, N), index=dates, columns=stocks)
        self.returns_data = pd.DataFrame(
            0.05 * self.factor_data.values + 0.02 * np.random.randn(T, N),
            index=dates, columns=stocks
        )
        self.mcap_data = pd.DataFrame(
            np.abs(np.random.randn(T, N)) * 1e10, index=dates, columns=stocks
        )

    def test_full_pipeline_returns_report(self):
        """完整流水线返回FactorHealthReport"""
        report = self.monitor.evaluate_health(
            'test_factor', self.factor_data, self.returns_data, self.mcap_data
        )
        self.assertIsInstance(report, FactorHealthReport)

    def test_report_has_all_dimensions(self):
        """报告包含所有五维得分"""
        report = self.monitor.evaluate_health(
            'test_factor', self.factor_data, self.returns_data, self.mcap_data
        )
        self.assertIsInstance(report.crowding_score, float)
        self.assertIsInstance(report.efficacy_score, float)
        self.assertIsInstance(report.capacity_score, float)
        self.assertIsInstance(report.decay_score, float)
        self.assertIsInstance(report.regime_score, float)

    def test_health_score_bounded(self):
        """综合健康分在 [0, 100]"""
        report = self.monitor.evaluate_health(
            'test_factor', self.factor_data, self.returns_data, self.mcap_data
        )
        self.assertGreaterEqual(report.health_score, 0.0)
        self.assertLessEqual(report.health_score, 100.0)

    def test_pipeline_without_optional_data(self):
        """缺少可选数据时不应崩溃"""
        report = self.monitor.evaluate_health('test_factor', self.factor_data)
        self.assertIsInstance(report, FactorHealthReport)

    def test_pipeline_records_history(self):
        """流水线应记录健康度历史"""
        self.monitor.evaluate_health('hist_factor', self.factor_data, self.returns_data)
        self.assertIn('hist_factor', self.monitor.health_history)
        self.assertEqual(len(self.monitor.health_history['hist_factor']), 1)

    def test_data_insufficient_returns_watch(self):
        """数据不足时返回WATCH级别"""
        small_data = self.factor_data.iloc[:6]
        small_ret = self.returns_data.iloc[:6]
        report = self.monitor.evaluate_health('small', small_data, small_ret)
        self.assertEqual(report.health_level, HealthAlertLevel.WATCH)
        self.assertEqual(report.health_score, 0.0)

    def test_to_dict_serializable(self):
        """to_dict 应可序列化"""
        report = self.monitor.evaluate_health(
            'test_factor', self.factor_data, self.returns_data, self.mcap_data
        )
        d = report.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn('health_score', d)
        self.assertIn('health_level', d)


class TestBatchAndSummary(unittest.TestCase):
    """批量评估和摘要"""

    def setUp(self):
        self.monitor = FactorHealthMonitor()
        np.random.seed(42)
        T, N = 60, 100
        self.factor_data = pd.DataFrame(np.random.randn(T, N), columns=[f'S{i}' for i in range(N)])
        self.returns_data = pd.DataFrame(
            0.05 * self.factor_data.values + 0.02 * np.random.randn(T, N),
            columns=[f'S{i}' for i in range(N)]
        )

    def test_batch_returns_all_factors(self):
        """批量应返回所有因子"""
        factor_dict = {'f1': self.factor_data, 'f2': self.factor_data * 0.5}
        reports = self.monitor.evaluate_health_batch(
            factor_dict, self.returns_data
        )
        self.assertEqual(len(reports), 2)
        self.assertIn('f1', reports)
        self.assertIn('f2', reports)

    def test_summary_returns_dataframe(self):
        """摘要应返回DataFrame"""
        self.monitor.evaluate_health('f1', self.factor_data, self.returns_data)
        summary = self.monitor.get_health_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)

    def test_summary_specific_factor(self):
        """指定因子摘要"""
        self.monitor.evaluate_health('f1', self.factor_data, self.returns_data)
        summary = self.monitor.get_health_summary('f1')
        self.assertEqual(len(summary), 1)

    def test_trend_records_correctly(self):
        """趋势应累积记录"""
        for _ in range(3):
            self.monitor.evaluate_health('trend', self.factor_data, self.returns_data)
        trend = self.monitor.get_health_trend('trend')
        self.assertEqual(len(trend), 3)

    def test_trend_empty_for_unknown(self):
        """未知因子趋势为空"""
        trend = self.monitor.get_health_trend('unknown')
        self.assertEqual(len(trend), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)