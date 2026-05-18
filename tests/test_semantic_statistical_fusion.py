# -*- coding: utf-8 -*-
"""
语义-统计融合测试套件

测试语义规则与统计规则的协调机制。
采用 TDD 原则：先写测试，再实现功能。
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.fingerprint import FactorFingerprint, FactorType
from core.classifier import ClassificationResult


class TestSemanticPrior(unittest.TestCase):
    """测试语义先验：将语义输出转换为统计分类器可理解的先验分布"""

    def test_semantic_prior_creation(self):
        """测试 SemanticPrior 能正确创建"""
        from core.semantic_fusion import SemanticPrior
        
        prior = SemanticPrior(
            expected_type=FactorType.STATIC,
            prior_strength=0.8,
            confidence=0.9
        )
        
        self.assertEqual(prior.expected_type, FactorType.STATIC)
        self.assertEqual(prior.prior_strength, 0.8)
        self.assertEqual(prior.confidence, 0.9)
    
    def test_ar1_prior_mapping_static(self):
        """测试静态因子的 AR(1) 先验分布"""
        from core.semantic_fusion import SemanticPrior
        
        prior = SemanticPrior(expected_type=FactorType.STATIC)
        mean, std = prior.to_ar1_prior()
        
        self.assertGreater(mean, 0.8)
        self.assertLess(std, 0.15)
    
    def test_ar1_prior_mapping_dynamic(self):
        """测试动态因子的 AR(1) 先验分布"""
        from core.semantic_fusion import SemanticPrior
        
        prior = SemanticPrior(expected_type=FactorType.DYNAMIC)
        mean, std = prior.to_ar1_prior()
        
        self.assertLess(mean, 0.4)
        self.assertLess(std, 0.15)
    
    def test_ar1_prior_mapping_mixed(self):
        """测试混合因子的 AR(1) 先验分布"""
        from core.semantic_fusion import SemanticPrior
        
        prior = SemanticPrior(expected_type=FactorType.MIXED)
        mean, std = prior.to_ar1_prior()
        
        self.assertGreaterEqual(mean, 0.4)
        self.assertLessEqual(mean, 0.8)
    
    def test_from_extracted_rule(self):
        """测试从 ExtractedRule 创建 SemanticPrior"""
        from core.semantic import ExtractedRule
        from core.semantic_fusion import SemanticPrior
        
        rule = ExtractedRule()
        rule.category = 'value'
        rule.category_confidence = 0.85
        rule.overall_confidence = 0.75
        
        prior = SemanticPrior.from_extracted_rule(rule)
        
        self.assertEqual(prior.expected_type, FactorType.STATIC)
        self.assertGreater(prior.prior_strength, 0)
        self.assertGreater(prior.confidence, 0)


class TestBayesianClassifier(unittest.TestCase):
    """测试贝叶斯分类器：支持语义先验的分类"""

    def test_classify_without_prior(self):
        """测试无先验时退化为纯统计分类"""
        from core.semantic_fusion import BayesianFactorClassifier
        
        classifier = BayesianFactorClassifier()
        fp = FactorFingerprint(ar1_median=0.85)
        
        result = classifier.classify_with_prior(fp)
        
        self.assertEqual(result.primary_type, FactorType.STATIC)
        self.assertGreater(result.confidence, 0)
    
    def test_classify_with_consistent_prior(self):
        """测试先验与统计一致时提升置信度"""
        from core.semantic_fusion import BayesianFactorClassifier, SemanticPrior
        
        classifier = BayesianFactorClassifier()
        fp = FactorFingerprint(ar1_median=0.85)
        prior = SemanticPrior(expected_type=FactorType.STATIC, prior_strength=0.8)
        
        result_with_prior = classifier.classify_with_prior(fp, prior)
        result_without = classifier.classify_with_prior(fp)
        
        self.assertEqual(result_with_prior.primary_type, FactorType.STATIC)
        self.assertGreaterEqual(
            result_with_prior.confidence,
            result_without.confidence
        )
    
    def test_classify_with_conflicting_prior_insufficient_data(self):
        """测试数据不足时优先信任语义先验"""
        from core.semantic_fusion import BayesianFactorClassifier, SemanticPrior
        
        classifier = BayesianFactorClassifier()
        fp = FactorFingerprint(ar1_median=np.nan)
        prior = SemanticPrior(expected_type=FactorType.STATIC, prior_strength=0.8)
        
        result = classifier.classify_with_prior(fp, prior)
        
        self.assertEqual(result.primary_type, FactorType.STATIC)
        self.assertFalse(result.is_hard)
    
    def test_classify_with_conflicting_prior_sufficient_data(self):
        """测试数据充足但冲突时标记为软分类"""
        from core.semantic_fusion import BayesianFactorClassifier, SemanticPrior
        
        classifier = BayesianFactorClassifier()
        fp = FactorFingerprint(ar1_median=0.85)
        prior = SemanticPrior(expected_type=FactorType.DYNAMIC, prior_strength=0.6)
        
        result = classifier.classify_with_prior(fp, prior)
        
        self.assertEqual(result.primary_type, FactorType.STATIC)
        self.assertFalse(result.is_hard)


class TestConflictArbitrator(unittest.TestCase):
    """测试冲突仲裁引擎"""

    def test_consistent_results(self):
        """测试语义与统计一致时高置信度锁定"""
        from core.semantic_fusion import ConflictArbitrator, SemanticPrior
        from core.classifier import ClassificationResult
        
        arbitrator = ConflictArbitrator()
        semantic = SemanticPrior(expected_type=FactorType.STATIC)
        statistical = ClassificationResult(
            primary_type=FactorType.STATIC,
            primary_prob=0.9,
            confidence=0.85
        )
        
        result = arbitrator.arbitrate(semantic, statistical, data_sufficiency=0.8)
        
        self.assertEqual(result.primary_type, FactorType.STATIC)
        self.assertGreater(result.confidence, 0.85)
    
    def test_insufficient_data_trusts_semantic(self):
        """测试数据不足时信任语义"""
        from core.semantic_fusion import ConflictArbitrator, SemanticPrior
        from core.classifier import ClassificationResult
        
        arbitrator = ConflictArbitrator()
        semantic = SemanticPrior(expected_type=FactorType.STATIC, prior_strength=0.8)
        statistical = ClassificationResult(
            primary_type=FactorType.DYNAMIC,
            primary_prob=0.6,
            confidence=0.5
        )
        
        result = arbitrator.arbitrate(semantic, statistical, data_sufficiency=0.2)
        
        self.assertEqual(result.primary_type, FactorType.STATIC)
    
    def test_observation_period_fallback(self):
        """测试观察期降级到混合"""
        from core.semantic_fusion import ConflictArbitrator, SemanticPrior
        from core.classifier import ClassificationResult
        
        arbitrator = ConflictArbitrator()
        semantic = SemanticPrior(expected_type=FactorType.STATIC)
        statistical = ClassificationResult(
            primary_type=FactorType.DYNAMIC,
            primary_prob=0.7,
            confidence=0.6
        )
        
        result = arbitrator.arbitrate(semantic, statistical, data_sufficiency=0.5)
        
        self.assertEqual(result.primary_type, FactorType.MIXED)
        self.assertFalse(result.is_hard)
    
    def test_persistent_conflict_alert(self):
        """测试持续冲突触发人工审查标记"""
        from core.semantic_fusion import ConflictArbitrator, SemanticPrior
        from core.classifier import ClassificationResult
        
        arbitrator = ConflictArbitrator()
        semantic = SemanticPrior(expected_type=FactorType.STATIC)
        statistical = ClassificationResult(
            primary_type=FactorType.DYNAMIC,
            primary_prob=0.8,
            confidence=0.9
        )
        
        result = arbitrator.arbitrate(semantic, statistical, data_sufficiency=0.9)
        
        self.assertIsNotNone(result.conflict_reason)
        self.assertIn('human_review', result.conflict_reason)


class TestSemanticStatisticalIntegration(unittest.TestCase):
    """测试端到端集成"""

    def test_full_pipeline_with_description(self):
        """测试带描述的完整流水线"""
        from core.semantic_fusion import SemanticStatisticalFusion
        from core.fingerprint import FactorFingerprinter
        
        fusion = SemanticStatisticalFusion()
        
        description = "市盈率因子，基于最新财报数据"
        fp = FactorFingerprint(ar1_median=0.82)
        
        result = fusion.classify(description, fp, data_months=36)
        
        self.assertEqual(result.primary_type, FactorType.STATIC)
        self.assertGreater(result.confidence, 0.7)
    
    def test_cold_start_with_description(self):
        """测试冷启动：只有描述没有数据"""
        from core.semantic_fusion import SemanticStatisticalFusion
        
        fusion = SemanticStatisticalFusion()
        
        description = "短期反转因子，基于过去1个月收益率"
        fp = FactorFingerprint(ar1_median=np.nan)
        
        result = fusion.classify(description, fp, data_months=6)
        
        self.assertEqual(result.primary_type, FactorType.DYNAMIC)
        self.assertFalse(result.is_hard)
    
    def test_conflict_scenario(self):
        """测试冲突场景：语义说静态，统计说动态"""
        from core.semantic_fusion import SemanticStatisticalFusion
        
        fusion = SemanticStatisticalFusion()
        
        description = "市净率因子"
        fp = FactorFingerprint(ar1_median=0.25)
        
        result = fusion.classify(description, fp, data_months=36)
        
        self.assertIsNotNone(result.conflict_reason)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticPrior))
    suite.addTests(loader.loadTestsFromTestCase(TestBayesianClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestConflictArbitrator))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticStatisticalIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
