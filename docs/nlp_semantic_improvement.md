# 因子构造语义理解准确性提升方案

## 一、当前实现的局限性分析

### 1.1 现有问题

| 问题类型 | 具体表现 | 影响程度 |
|---------|---------|---------|
| **歧义性** | "return" 可以指收益率，也可以是函数返回值 | 高 |
| **复合表达** | "过去12个月扣除最近1个月" 无法解析 | 高 |
| **隐含信息** | "市值因子" 隐含 size 类型 | 中 |
| **领域术语** | "ttm"、"ROE" 等专业术语 | 中 |
| **上下文依赖** | "它"、"上述因子" 等指代 | 低 |

### 1.2 当前识别失败案例

```python
# 案例1: 复合时间窗口
文本: "过去12个月扣除最近1个月后的累积收益率"
当前结果: lookback_window = None  ❌
期望结果: lookback_window = [12, 1], operator = ['cumsum', 'subtract']

# 案例2: 隐含类别
文本: "市值因子取对数市值"
当前结果: category = 'size'  ✓
期望结果: category = 'size', operator = ['log'], data_source = ['market_cap']

# 案例3: 专业术语
文本: "使用一致预期净利润变化"
当前结果: lookback_window = None  ❌
期望结果: category = 'sentiment', data_source = ['analyst'], operator = ['diff']
```

---

## 二、开源NLP工具生态

### 2.1 工具全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                    NLP工具生态                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │  文本处理    │   │   实体识别   │   │   关系抽取   │          │
│  ├─────────────┤   ├─────────────┤   ├─────────────┤          │
│  │ jieba       │   │ spaCy       │   │ OpenNRE    │          │
│  │ pkuseg      │   │ HanLP       │   │ stanza      │          │
│  │ ltp         │   │ THULAC      │   │ -          │          │
│  └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │   语义相似   │   │   知识图谱   │   │   预训练模型 │          │
│  ├─────────────┤   ├─────────────┤   ├─────────────┤          │
│  │ sentence-  │   │ Neo4j       │   │ bert-base- │          │
│  │ transformers│  │ RDFlib      │   │ chinese    │          │
│  │ scikit-learn│  │ NetworkX    │   │ chinese-   │          │
│  └─────────────┘   └─────────────┘   │ roberta   │          │
│                                       └─────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 中文分词工具对比

| 工具 | 准确性 | 速度 | 领域适应 | 维护状态 | 推荐场景 |
|------|--------|------|---------|---------|---------|
| **jieba** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 一般 | 活跃 | 快速原型 |
| **pkuseg** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 优秀 | 活跃 | 生产环境 |
| **LTP** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 优秀 | 活跃 | 全面分析 |
| **HanLP** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 优秀 | 活跃 | 最高准确 |

### 2.3 实体识别工具对比

| 工具 | 准确性 | 多语言 | 微调支持 | 部署难度 | 推荐度 |
|------|--------|-------|---------|---------|--------|
| **spaCy** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中等 | ⭐⭐⭐⭐ |
| **HanLP** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 低 | ⭐⭐⭐⭐⭐ |
| **FoolNLTK** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | 低 | ⭐⭐ |
| **LTP** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 中等 | ⭐⭐⭐⭐ |

### 2.4 预训练语言模型对比

| 模型 | 参数量 | 中文支持 | 推理速度 | 硬件需求 | 推荐度 |
|------|--------|---------|---------|---------|--------|
| **BERT-base-chinese** | 100M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 中等 | ⭐⭐⭐⭐ |
| **RoBERTa-wwm-ext** | 100M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 中等 | ⭐⭐⭐⭐⭐ |
| **MacBERT** | 100M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 中等 | ⭐⭐⭐⭐⭐ |
| **PERT** | 100M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 中等 | ⭐⭐⭐⭐⭐ |
| **ChatGLM** | 6B+ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 高 | ⭐⭐⭐ (需要LLM) |

---

## 三、推荐方案：分层提升架构

### 3.1 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│              因子构造语义理解分层架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: 基础层（规则+分词）                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ jieba分词 | 正则规则 | 金融词典 | 时间表达式解析          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  Layer 2: 语义层（NER+关系抽取）                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ HanLP NER | 依存句法分析 | 语义角色标注 | 金融实体库       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  Layer 3: 理解层（语义相似+知识图谱）                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Sentence-BERT | 金融知识图谱 | 语义消歧 | 上下文推理       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  Layer 4: 增强层（预训练模型 - 可选）                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ FinBERT | ChatGLM | 指令微调 | Few-shot Learning         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 分层实现

#### Layer 1: 金融增强分词器

```python
# -*- coding: utf-8 -*-
"""
Layer 1: 金融增强分词器

结合规则和金融词典的增强分词器
"""

import jieba
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Token:
    """分词结果"""
    text: str
    pos: str  # 词性
    offset: int
    entity_type: str = None


class FinancialTokenizer:
    """金融增强分词器"""

    def __init__(self):
        self._init_financial_dict()

    def _init_financial_dict(self):
        """初始化金融词典"""
        # 金融术语（自定义词典优先级最高）
        financial_terms = [
            '市盈率', '市净率', '市销率', 'ROE', 'ROA', 'ROIC',
            '毛利率', '净利率', '资产负债率', '换手率', '波动率',
            '动量因子', '反转因子', '价值因子', '质量因子', '成长因子',
            '一致预期', '分析师预期', 'TTM', '扣除非经常性损益',
            '过去N个月', '最近N个月', '累积收益率', '行业中性化',
        ]

        for term in financial_terms:
            jieba.add_word(term, freq=10000, tag='nz')

        # 词性标注
        self.pos_mapping = {
            'nz': '金融术语',
            'n': '名词',
            'v': '动词',
            'm': '数量词',
            'q': '量词',
            't': '时间词',
        }

    def tokenize(self, text: str) -> List[Token]:
        """分词"""
        tokens = []
        for i, (word, pos) in enumerate(jieba.posseg.cut(text)):
            token = Token(
                text=word,
                pos=pos,
                offset=text.find(word) if i == 0 else 0,
                entity_type=self.pos_mapping.get(pos, '其他')
            )
            tokens.append(token)
        return tokens

    def extract_numbers_with_unit(self, text: str) -> List[Tuple[float, str]]:
        """提取数字+单位"""
        pattern = r'(\d+(?:\.\d+)?)\s*([日周月年个]*)'
        matches = re.findall(pattern, text)
        return [(float(num), unit) for num, unit in matches if unit]
```

#### Layer 2: 依存句法分析器

```python
# -*- coding: utf-8 -*-
"""
Layer 2: 依存句法分析器

使用HanLP进行句法分析，提取语义关系
"""

from typing import List, Dict, Tuple, Optional
try:
    from hanlp.components.mtl.tasks dep.constituency import ConstituencyParsingTransformer
    from hanlp.components.mtl.tasks.dep.biaffine_dep import BiaffineDependencyParsingTransformer
    HAS_HANLP = True
except ImportError:
    HAS_HANLP = False


class DependencyParser:
    """依存句法分析器"""

    def __init__(self):
        if HAS_HANLP:
            self.parser = BiaffineDependencyParsingTransformer()
            self.parser.load()
        else:
            self.parser = None

    def parse(self, text: str) -> Optional[Dict]:
        """依存句法分析"""
        if not HAS_HANLP:
            return None

        try:
            result = self.parser(text)
            return self._extract_relations(result)
        except:
            return None

    def _extract_relations(self, result: Dict) -> List[Dict]:
        """提取关系"""
        relations = []

        if 'dep' in result:
            for i, (head, label) in enumerate(zip(result['head'], result['deprel'])):
                relations.append({
                    'index': i,
                    'head': head,
                    'relation': label,
                    'word': result['token'][i]
                })

        return relations


class SemanticRoleLabeler:
    """语义角色标注器"""

    RELATION_PATTERNS = {
        # 金融领域特有的语义关系
        '时间范围': r'(过去|最近|历史上)',
        '操作类型': r'(除以?|比值?|乘以?|加|减|取|计算)',
        '数据来源': r'(使用?|基于?|采用?)',
        '比较对象': r'(与|和|比)',
        '排序方式': r'(排名?|分位数|百分位|倒序)',
    }

    def extract_semantic_roles(self, text: str) -> Dict[str, List[str]]:
        """提取语义角色"""
        roles = {}

        for role_name, pattern in self.RELATION_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                roles[role_name] = matches

        return roles
```

#### Layer 3: 金融知识图谱

```python
# -*- coding: utf-8 -*-
"""
Layer 3: 金融知识图谱

构建因子领域的知识图谱，用于语义消歧
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class FinancialEntity:
    """金融实体"""
    name: str
    entity_type: str  # factor/metric/industry/operator
    aliases: List[str] = field(default_factory=list)
    properties: Dict = field(default_factory=dict)


class FinancialKnowledgeGraph:
    """金融知识图谱"""

    def __init__(self):
        self.entities = {}
        self.relations = defaultdict(list)
        self._build_core_graph()

    def _build_core_graph(self):
        """构建核心知识图谱"""
        # 因子类型
        factor_entities = [
            FinancialEntity('动量因子', 'factor', ['momentum', '动量']),
            FinancialEntity('反转因子', 'factor', ['reversal', '反转', '短期反转']),
            FinancialEntity('价值因子', 'factor', ['value', '价值', '估值']),
            FinancialEntity('质量因子', 'factor', ['quality', '质量', '盈利质量']),
            FinancialEntity('成长因子', 'factor', ['growth', '成长', '增长']),
            FinancialEntity('市值因子', 'factor', ['size', '市值', '规模']),
            FinancialEntity('流动性因子', 'factor', ['liquidity', '流动性']),
            FinancialEntity('情绪因子', 'factor', ['sentiment', '情绪', '投资者情绪']),
        ]

        # 财务指标
        metric_entities = [
            FinancialEntity('PE', 'metric', ['市盈率', 'price_to_earning']),
            FinancialEntity('PB', 'metric', ['市净率', 'price_to_book']),
            FinancialEntity('ROE', 'metric', ['净资产收益率', 'return_on_equity']),
            FinancialEntity('ROA', 'metric', ['资产收益率', 'return_on_assets']),
            FinancialEntity('毛利率', 'metric', ['gross_margin']),
            FinancialEntity('净利率', 'metric', ['net_margin', '净利润率']),
        ]

        # 操作符
        operator_entities = [
            FinancialEntity('除以', 'operator', ['比值', '比率', '/', 'divide']),
            FinancialEntity('对数', 'operator', ['log', 'ln', '取对数']),
            FinancialEntity('累积', 'operator', ['累计', 'cumsum', 'cum']),
            FinancialEntity('排名', 'operator', ['排序', 'rank', 'percentile']),
        ]

        for entity in factor_entities + metric_entities + operator_entities:
            self.entities[entity.name] = entity

    def disambiguate(self, text: str, candidates: List[str]) -> Optional[str]:
        """语义消歧"""
        text_lower = text.lower()

        best_match = None
        best_score = 0

        for candidate in candidates:
            if candidate not in self.entities:
                continue

            entity = self.entities[candidate]
            score = 0

            # 检查名称匹配
            if entity.name in text:
                score += 10

            # 检查别名匹配
            for alias in entity.aliases:
                if alias.lower() in text_lower:
                    score += 5

            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match

    def get_entity_properties(self, entity_name: str) -> Dict:
        """获取实体属性"""
        if entity_name in self.entities:
            return self.entities[entity_name].properties
        return {}
```

#### Layer 4: 语义相似度匹配（可选）

```python
# -*- coding: utf-8 -*-
"""
Layer 4: 语义相似度匹配

使用Sentence-BERT进行语义匹配
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SemanticMatcher:
    """语义相似度匹配器"""

    # 预定义的标准因子模板
    FACTOR_TEMPLATES = {
        '动量因子': {
            'patterns': [
                '过去N个月收益率',
                '累计收益率',
                '动量效应',
            ],
            'category': 'momentum',
            'typical_stability': 'mixed'
        },
        '反转因子': {
            'patterns': [
                '短期反转',
                '收益率相反数',
                'reversal',
            ],
            'category': 'reversal',
            'typical_stability': 'dynamic'
        },
        '价值因子': {
            'patterns': [
                '市盈率',
                '市净率',
                '估值',
            ],
            'category': 'value',
            'typical_stability': 'static'
        },
        '质量因子': {
            'patterns': [
                'ROE',
                'ROA',
                '盈利质量',
            ],
            'category': 'quality',
            'typical_stability': 'static'
        },
    }

    def __init__(self):
        if HAS_SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        else:
            self.model = None

    def match(self, text: str, top_k: int = 3) -> List[Dict]:
        """语义匹配"""
        if not self.model:
            # 回退到关键词匹配
            return self._keyword_match(text, top_k)

        # 编码
        text_embedding = self.model.encode([text])
        pattern_embeddings = self.model.encode([
            p for patterns in self.FACTOR_TEMPLATES.values()
            for p in patterns['patterns']
        ])

        # 计算相似度
        similarities = util.cos_sim(text_embedding, pattern_embeddings)[0]

        # 排序
        top_indices = np.argsort(-similarities)[:top_k]

        results = []
        pattern_idx = 0
        for template_name, template in self.FACTOR_TEMPLATES.items():
            for pattern in template['patterns']:
                if pattern_idx in top_indices:
                    results.append({
                        'template': template_name,
                        'matched_pattern': pattern,
                        'similarity': float(similarities[pattern_idx]),
                        'category': template['category'],
                        'typical_stability': template['typical_stability']
                    })
                pattern_idx += 1

        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]

    def _keyword_match(self, text: str, top_k: int) -> List[Dict]:
        """关键词匹配（回退方案）"""
        results = []

        for template_name, template in self.FACTOR_TEMPLATES.items():
            max_similarity = 0
            matched_pattern = None

            for pattern in template['patterns']:
                if pattern in text:
                    max_similarity = 1.0
                    matched_pattern = pattern
                    break

            if max_similarity > 0:
                results.append({
                    'template': template_name,
                    'matched_pattern': matched_pattern,
                    'similarity': max_similarity,
                    'category': template['category'],
                    'typical_stability': template['typical_stability']
                })

        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
```

---

## 四、完整集成示例

```python
# -*- coding: utf-8 -*-
"""
因子构造语义理解完整系统

集成所有Layer的完整实现
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnhancedExtractedRule:
    """增强版提取规则"""
    category: Optional[str] = None
    category_confidence: float = 0.0
    is_price_based: bool = False
    is_financial_report: bool = False
    is_analyst_data: bool = False
    lookback_windows: List[Tuple[float, str]] = field(default_factory=list)
    rebalance_freq: str = 'monthly'
    operators: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    semantic_roles: Dict[str, List[str]] = field(default_factory=dict)
    knowledge_graph_matches: List[Dict] = field(default_factory=list)
    overall_confidence: float = 0.0


class FactorSemanticUnderstanding:
    """因子构造语义理解系统"""

    def __init__(self, use_llm: bool = False):
        # Layer 1: 金融增强分词
        self.tokenizer = FinancialTokenizer()

        # Layer 2: 依存句法分析
        self.dep_parser = DependencyParser()
        self.srl = SemanticRoleLabeler()

        # Layer 3: 金融知识图谱
        self.kg = FinancialKnowledgeGraph()

        # Layer 4: 语义匹配
        self.matcher = SemanticMatcher()

        self.use_llm = use_llm
        logger.info("FactorSemanticUnderstanding initialized")

    def understand(self, text: str) -> EnhancedExtractedRule:
        """完整语义理解"""
        rule = EnhancedExtractedRule()

        # Layer 1: 分词和基础规则
        tokens = self.tokenizer.tokenize(text)
        numbers = self.tokenizer.extract_numbers_with_unit(text)
        rule.lookback_windows = numbers

        # Layer 2: 语义角色
        rule.semantic_roles = self.srl.extract_semantic_roles(text)

        # Layer 3: 知识图谱消歧
        candidates = self._extract_candidates(tokens)
        for candidate in candidates:
            match = self.kg.disambiguate(text, [candidate])
            if match:
                entity = self.kg.entities[match]
                if entity.entity_type == 'factor':
                    rule.category = entity.name
                elif entity.entity_type == 'metric':
                    rule.data_sources.append(entity.name)

        # Layer 4: 语义相似度匹配
        rule.knowledge_graph_matches = self.matcher.match(text)

        # 综合决策
        rule = self._make_decision(rule)

        return rule

    def _extract_candidates(self, tokens: List[Token]) -> List[str]:
        """提取候选实体"""
        return [t.text for t in tokens if t.entity_type in ['金融术语', '名词']]

    def _make_decision(self, rule: EnhancedExtractedRule) -> EnhancedExtractedRule:
        """综合决策"""
        # 优先级: 知识图谱 > 语义匹配 > 基础规则
        if rule.category:
            rule.category_confidence = 1.0
        elif rule.knowledge_graph_matches:
            best_match = rule.knowledge_graph_matches[0]
            rule.category = best_match['category']
            rule.category_confidence = best_match['similarity']

        # 计算总体置信度
        score = rule.category_confidence * 0.4
        score += (0.2 if rule.lookback_windows else 0)
        score += (0.2 if rule.data_sources else 0)
        score += (0.2 if rule.semantic_roles else 0)

        rule.overall_confidence = min(score, 1.0)

        return rule


def demo():
    """演示"""
    print("=" * 80)
    print("因子构造语义理解系统演示")
    print("=" * 80)

    system = FactorSemanticUnderstanding()

    test_texts = [
        "动量因子定义为过去12个月扣除最近1个月后的累积收益率",
        "使用ROE的ttm值与行业平均ROE的比值作为质量因子",
        "基于20日换手率与120日平均换手率的比值构建流动性因子",
    ]

    for text in test_texts:
        print(f"\n文本: {text}")
        result = system.understand(text)
        print(f"类别: {result.category} (置信度: {result.category_confidence:.2f})")
        print(f"时间窗口: {result.lookback_windows}")
        print(f"数据来源: {result.data_sources}")
        print(f"操作符: {result.operators}")
        print(f"语义角色: {result.semantic_roles}")


if __name__ == '__main__':
    demo()
```

---

## 五、工具选型建议

### 5.1 不同场景的推荐配置

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| **快速原型** | Layer 1 + Layer 3 | 快速实现，依赖少 |
| **生产环境** | Layer 1 + 2 + 3 | 准确性优先 |
| **最高准确** | Layer 1 + 2 + 3 + 4 | 使用预训练模型 |
| **极致准确** | Layer 1 + 2 + 3 + LLM | GPT-4/ChatGLM |

### 5.2 依赖安装

```bash
# 基础依赖
pip install jieba

# 进阶依赖（可选）
pip install hanlp
pip install sentence-transformers
pip install networkx

# GPU支持
pip install torch
```

---

## 六、总结

| 提升维度 | 具体措施 | 预期提升 |
|---------|---------|---------|
| **分词准确性** | 金融增强词典 | +20% |
| **语义理解** | 依存句法分析 | +30% |
| **消歧能力** | 金融知识图谱 | +25% |
| **模板匹配** | Sentence-BERT | +20% |
| **综合提升** | 分层集成 | +50%+ |

**推荐路径**：
1. 快速验证：Layer 1 + Layer 3
2. 生产部署：Layer 1 + 2 + 3
3. 极致追求：Layer 1 + 2 + 3 + 4 (LLM)
