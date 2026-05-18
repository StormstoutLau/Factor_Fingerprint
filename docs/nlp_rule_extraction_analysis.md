# 自然语言构造规则提取分析

## 一、当前实现机制分析

### 1.1 当前规则识别方式

当前实现使用**简单的字符串匹配**来识别因子类别：

```python
# FactorConstructionRuleEngine.infer_category() 中的代码
name_lower = factor_meta.name.lower()

if any(k in name_lower for k in ['pe', 'pb', 'ps', 'ev/ebit', 'price_to', 'value']):
    return 'value'
elif any(k in name_lower for k in ['momentum', 'mom', 'return']):
    return 'momentum'
elif any(k in name_lower for k in ['reversal', 'rever', 'reverse']):
    return 'reversal'
# ...
```

### 1.2 当前实现的局限性

| 维度 | 当前实现 | 局限性 |
|------|---------|-------|
| **输入形式** | 因子名称（字符串） | 只能处理简单名称 |
| **识别方式** | 关键词匹配 | 无法处理语义 |
| **上下文理解** | 无 | 无法理解复杂描述 |
| **专业术语** | 有限 | 无法识别新术语 |
| **模糊表述** | 无法处理 | "可能基于价格"等 |

### 1.3 实际场景中的自然语言描述

研究者通常用自然语言描述因子构造规则，例如：

```
文本1: "动量因子定义为过去12个月扣除最近1个月后的累积收益率"
文本2: "我们使用ROE的ttm值与行业平均ROE的比值作为质量因子"
文本3: "基于20日换手率与120日平均换手率的比值构建流动性因子"
文本4: "反转因子是过去1周的日收益率的相反数"
文本5: "市值因子取对数市值，PB使用最新财报的账面价值除以总市值"
```

---

## 二、自然语言处理方案

### 2.1 方案对比

| 方案 | 准确性 | 实现复杂度 | 适用场景 | 推荐度 |
|------|--------|----------|---------|--------|
| **规则表达式** | ⭐⭐ | ⭐ | 简单名称 | ⭐⭐ |
| **正则表达式** | ⭐⭐⭐ | ⭐⭐ | 固定模式 | ⭐⭐⭐ |
| **关键词+TF-IDF** | ⭐⭐⭐ | ⭐⭐ | 半结构化文本 | ⭐⭐⭐ |
| **命名实体识别(NER)** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 专业文本 | ⭐⭐⭐⭐ |
| **LLM/GPT** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 复杂文本 | ⭐⭐⭐⭐⭐ |
| **混合方案** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 生产环境 | ⭐⭐⭐⭐⭐ |

### 2.2 推荐方案：混合架构

```
┌─────────────────────────────────────────────────────────┐
│                  混合规则提取架构                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   自然语言文本输入                                        │
│         ↓                                               │
│   ┌─────────────┐                                        │
│   │  预处理模块  │  清洗、分句、标准化                       │
│   └──────┬──────┘                                        │
│          ↓                                               │
│   ┌─────────────┐                                        │
│   │ 快速规则匹配 │  正则/关键词（覆盖80%常见模式）           │
│   └──────┬──────┘                                        │
│          ↓                                               │
│   ┌─────────────┐                                        │
│   │   NER模块   │  提取关键实体（时间窗口、数值操作符）     │
│   └──────┬──────┘                                        │
│          ↓                                               │
│   ┌─────────────┐                                        │
│   │  LLM增强模块│  处理复杂/模糊表述（可选）               │
│   └──────┬──────┘                                        │
│          ↓                                               │
│   ┌─────────────┐                                        │
│   │  规则融合器 │  多源规则合并、去冲突                     │
│   └──────┬──────┘                                        │
│          ↓                                               │
│   FactorMeta 输出                                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 三、详细实现方案

### 3.1 快速规则匹配（覆盖80%场景）

使用正则表达式匹配常见模式：

```python
import re
from typing import Dict, List, Tuple, Optional

class QuickRuleMatcher:
    """快速规则匹配器"""
    
    # 时间窗口正则
    TIME_WINDOW_PATTERNS = [
        (r'(\d+)[日周月年]', 'window_days', int),      # "12个月"、"20日"
        (r'过去(\d+)[日周月年]', 'window_start', int),  # "过去12个月"
        (r'最近(\d+)[日周月年]', 'window_recent', int), # "最近1个月"
        (r'(ttm|TTM|近\d+期)', 'period_type', 'ttm'),  # "ttm值"
    ]
    
    # 数据来源正则
    DATA_SOURCE_PATTERNS = [
        (r'(收盘价?|price|close)', 'data_source', 'price'),
        (r'(换手率?|turnover|volume)', 'data_source', 'turnover'),
        (r'(市值|cap|market_value)', 'data_source', 'market_cap'),
        (r'(ROE|ROA|毛利率|净利率)', 'data_source', 'financial'),
        (r'(财报|公告|年报|季报)', 'data_source', 'financial_report'),
    ]
    
    # 操作符正则
    OPERATOR_PATTERNS = [
        (r'[×xX*]?(除以?|/|(?<!指)标?比)', 'operator', 'divide'),
        (r'[加加上+]+', 'operator', 'add'),
        (r'[减下去\-]+', 'operator', 'subtract'),
        (r'乘以?[×xX*]', 'operator', 'multiply'),
        (r'取?(相反数?|倒数|对数|log)', 'operator', 'transform'),
        (r'(排名|分位数|百分位)', 'operator', 'rank'),
    ]
    
    def extract(self, text: str) -> Dict[str, any]:
        """从文本中提取规则"""
        result = {}
        
        # 提取时间窗口
        for pattern, key, converter in self.TIME_WINDOW_PATTERNS:
            match = re.search(pattern, text)
            if match:
                if converter == int:
                    result[key] = converter(match.group(1))
                else:
                    result[key] = converter
                break
        
        # 提取数据来源
        for pattern, key, value in self.DATA_SOURCE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                result[key] = value
                break
        
        # 提取操作符
        operators = []
        for pattern, key, value in self.OPERATOR_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                operators.append(value)
        if operators:
            result['operators'] = operators
        
        return result
```

### 3.2 命名实体识别（NER）模块

使用 spaCy 或 similar NLP 库提取关键实体：

```python
import spacy
from typing import List, Dict, Any

class FactorEntityExtractor:
    """因子实体提取器"""
    
    def __init__(self):
        # 加载中文模型（需要下载：python -m spacy download zh_core_web_sm）
        try:
            self.nlp = spacy.load('zh_core_web_sm')
        except:
            self.nlp = None
            print("Warning: spaCy Chinese model not loaded")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取因子相关实体"""
        if self.nlp is None:
            return {}
        
        doc = self.nlp(text)
        
        entities = {
            'TIME_EXPRESSION': [],  # 时间表达：过去12个月、最近1周
            'FINANCIAL_METRIC': [],  # 财务指标：ROE、PB、PE
            'OPERATOR': [],          # 操作符：除以、比值、累积
            'DATA_SOURCE': [],       # 数据来源：收盘价、换手率
        }
        
        # 自定义规则增强NER
        time_patterns = [
            r'\d+[日周月年]',
            r'过去\d+[日周月年]',
            r'最近\d+[日周月年]',
            r'ttm|TTM',
        ]
        
        for pattern in time_patterns:
            for match in re.finditer(pattern, text):
                entities['TIME_EXPRESSION'].append(match.group())
        
        # 财务指标关键词
        financial_keywords = [
            'ROE', 'ROA', 'ROIC', 'PB', 'PE', 'PS', 'PCF',
            '毛利率', '净利率', '负债率', '营收增长率',
        ]
        
        for kw in financial_keywords:
            if kw.lower() in text.lower():
                entities['FINANCIAL_METRIC'].append(kw)
        
        return entities
    
    def extract_relationships(self, text: str) -> List[Dict[str, str]]:
        """提取关系三元组"""
        # 简化版：提取"A与B的比值"等模式
        relationships = []
        
        patterns = [
            (r'([^和与]+)与([^和与]+)的比值', 'ratio'),
            (r'([^和与]+)除以([^和与]+)', 'divide'),
            (r'([^和与]+)与([^和与]+)的和', 'add'),
        ]
        
        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                relationships.append({
                    'type': rel_type,
                    'source': match.group(1).strip(),
                    'target': match.group(2).strip(),
                })
        
        return relationships
```

### 3.3 LLM增强模块（处理复杂文本）

使用大语言模型处理复杂或模糊的文本：

```python
from typing import Dict, Any, Optional
import json

class LLMEnhancedExtractor:
    """LLM增强的规则提取器"""
    
    SYSTEM_PROMPT = """你是一个量化投资因子构造规则提取专家。
给定一段描述因子构造规则的文本，你需要提取以下信息：

1. **category**: 因子类别（value/momentum/reversal/quality/liquidity/sentiment/other）
2. **is_price_based**: 是否基于价格数据（true/false）
3. **is_financial_report**: 是否基于财报数据（true/false）
4. **lookback_window**: 回溯窗口天数（数字或null）
5. **rebalance_freq**: 调仓频率（daily/weekly/monthly/quarterly）
6. **known_issues**: 已知问题列表
7. **typical_stability**: 预期稳定性（static/dynamic/mixed）
8. **literature_support**: 是否有文献支撑（true/false）

请以JSON格式输出。"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'gpt-4'):
        self.api_key = api_key
        self.model = model
    
    def extract(self, text: str) -> Dict[str, Any]:
        """使用LLM提取规则"""
        
        # 如果没有API Key，返回空
        if not self.api_key:
            return {}
        
        try:
            response = self._call_llm(text)
            return json.loads(response)
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return {}
    
    def _call_llm(self, text: str) -> str:
        """调用LLM API"""
        # 这里是伪代码，实际需要调用OpenAI API
        # response = openai.ChatCompletion.create(
        #     model=self.model,
        #     messages=[
        #         {"role": "system", "content": self.SYSTEM_PROMPT},
        #         {"role": "user", "content": text}
        #     ]
        # )
        # return response.choices[0].message.content
        pass
```

### 3.4 规则融合器

```python
class RuleFusionEngine:
    """规则融合引擎"""
    
    def __init__(self):
        self.quick_matcher = QuickRuleMatcher()
        self.entity_extractor = FactorEntityExtractor()
        self.llm_extractor = LLMEnhancedExtractor()
    
    def fuse(self, text: str, use_llm: bool = False) -> FactorMeta:
        """融合多源规则"""
        
        # 1. 快速规则匹配（最高优先级）
        quick_rules = self.quick_matcher.extract(text)
        
        # 2. NER实体提取
        entities = self.entity_extractor.extract_entities(text)
        ner_rules = self._entities_to_rules(entities)
        
        # 3. 规则合并（快速匹配优先）
        fused_rules = {**ner_rules, **quick_rules}
        
        # 4. LLM增强（可选，用于复杂文本）
        if use_llm and self.llm_extractor.api_key:
            llm_rules = self.llm_extractor.extract(text)
            fused_rules = {**fused_rules, **llm_rules}
        
        # 5. 转换为FactorMeta
        return self._rules_to_meta(fused_rules)
    
    def _entities_to_rules(self, entities: Dict) -> Dict:
        """将实体转换为规则"""
        rules = {}
        
        # 从时间表达推断回溯窗口
        if entities.get('TIME_EXPRESSION'):
            time_expr = entities['TIME_EXPRESSION'][0]
            # 解析时间表达
            match = re.search(r'(\d+)', time_expr)
            if match:
                window = int(match.group(1))
                # 转换为月数
                if '日' in time_expr:
                    window = window / 21  # 约21个交易日
                elif '周' in time_expr:
                    window = window / 4
                elif '年' in time_expr:
                    window = window * 12
                rules['lookback_window'] = window
        
        # 从财务指标推断是否基于财报
        if entities.get('FINANCIAL_METRIC'):
            rules['is_financial_report'] = True
        
        return rules
    
    def _rules_to_meta(self, rules: Dict) -> FactorMeta:
        """将规则转换为FactorMeta"""
        return FactorMeta(
            name=rules.get('name', 'Unknown'),
            category=rules.get('category'),
            is_price_based=rules.get('is_price_based', False),
            is_financial_report=rules.get('is_financial_report', False),
            lookback_window=rules.get('lookback_window'),
            rebalance_freq=rules.get('rebalance_freq', 'monthly'),
        )
```

---

## 四、完整使用示例

```python
class NaturalLanguageRuleExtractor:
    """自然语言规则提取器（完整封装）"""
    
    def __init__(self, use_llm: bool = False, api_key: Optional[str] = None):
        self.fusion_engine = RuleFusionEngine()
        self.use_llm = use_llm
        self.api_key = api_key
    
    def extract(self, text: str) -> FactorMeta:
        """从自然语言文本提取因子元数据"""
        return self.fusion_engine.fuse(text, use_llm=self.use_llm)
    
    def extract_batch(self, texts: List[str]) -> List[FactorMeta]:
        """批量提取"""
        return [self.extract(text) for text in texts]


# 使用示例
if __name__ == '__main__':
    extractor = NaturalLanguageRuleExtractor()
    
    test_texts = [
        "动量因子定义为过去12个月扣除最近1个月后的累积收益率",
        "我们使用ROE的ttm值与行业平均ROE的比值作为质量因子",
        "基于20日换手率与120日平均换手率的比值构建流动性因子",
    ]
    
    for text in test_texts:
        print(f"\n文本: {text}")
        meta = extractor.extract(text)
        print(f"提取结果: category={meta.category}, "
              f"lookback={meta.lookback_window}, "
              f"financial={meta.is_financial_report}")
```

---

## 五、方案选择建议

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| **简单因子名称** | 关键词匹配 | 快速、低成本 |
| **因子库标准化** | 正则+NER | 覆盖80%场景 |
| **研究报告解析** | 正则+NER+LLM | 复杂文本需要LLM |
| **生产环境** | 混合架构 | 平衡准确性和成本 |

### 5.1 分阶段实施方案

```
Phase 1: 快速部署
├── 关键词匹配（已实现）
├── 正则表达式（1周）
└── 基础NER（2周）

Phase 2: 能力增强
├── 完整NER训练（2周）
├── 规则库扩充（持续）
└── 评估指标建立

Phase 3: 智能化升级
├── LLM集成（1周）
├── 反馈学习机制
└── 持续优化
```

---

## 六、总结

### 当前实现
- ✅ 简单的关键词匹配
- ✅ 可处理简单因子名称
- ⚠️ 无法处理自然语言文本

### 推荐方案
| 组件 | 技术选型 | 优先级 |
|------|---------|--------|
| 快速匹配 | 正则表达式 | P0 |
| 实体提取 | spaCy NER | P1 |
| 复杂文本 | LLM (GPT-4) | P2 |
| 规则融合 | 自研融合器 | P0 |
