# KYC 项目：如何选取最重要的 Feature

**Author**：Yanda Cheng  
**Project**：KYC (Know Your Customer)  
**Purpose**：系统设计中如何选择最重要的功能特性  
**Level**：Senior 级别

---

## 📋 目录

1. [核心问题](#核心问题)
2. [Feature 优先级评估框架](#feature-优先级评估框架)
3. [KYC 项目实际应用](#kyc-项目实际应用)
4. [面试回答模板](#面试回答模板)

---

## 核心问题

### 问题场景

**面试问题**：如何选取最重要的 feature？

**挑战**：
- 功能特性很多，资源有限
- 需要平衡业务价值和技术复杂度
- 需要量化评估，避免主观判断

---

## Feature 优先级评估框架

### 1. RICE 评分模型（推荐）

**RICE = Reach × Impact × Confidence / Effort**

- **Reach（覆盖范围）**：这个功能会影响多少用户/请求？
- **Impact（影响程度）**：对每个用户的影响有多大？
- **Confidence（信心度）**：我们对这个功能的信心有多高？
- **Effort（工作量）**：实现这个功能需要多少工作量？

**评分标准**：

```python
class RICEScoring:
    """
    RICE 评分模型
    """
    def __init__(self):
        # Reach: 覆盖范围（0-100）
        self.reach_scale = {
            "all_users": 100,      # 所有用户
            "most_users": 75,      # 大部分用户
            "some_users": 50,      # 部分用户
            "few_users": 25,       # 少数用户
            "rare_users": 10       # 极少数用户
        }
        
        # Impact: 影响程度（0.25-3.0）
        self.impact_scale = {
            "massive": 3.0,        # 巨大影响
            "high": 2.0,           # 高影响
            "medium": 1.0,         # 中等影响
            "low": 0.5,            # 低影响
            "minimal": 0.25        # 极小影响
        }
        
        # Confidence: 信心度（50%-100%）
        self.confidence_scale = {
            "high": 100,           # 高信心（有数据支持）
            "medium": 80,          # 中等信心（有一些数据）
            "low": 50              # 低信心（猜测）
        }
        
        # Effort: 工作量（人-月）
        self.effort_scale = {
            "small": 1,            # 1 人-月
            "medium": 2,           # 2 人-月
            "large": 4,            # 4 人-月
            "xlarge": 8            # 8 人-月
        }
    
    def calculate_rice_score(self, feature):
        """
        计算 RICE 分数
        """
        reach = self.reach_scale[feature["reach"]]
        impact = self.impact_scale[feature["impact"]]
        confidence = self.confidence_scale[feature["confidence"]] / 100.0
        effort = self.effort_scale[feature["effort"]]
        
        rice_score = (reach * impact * confidence) / effort
        
        return {
            "rice_score": rice_score,
            "reach": reach,
            "impact": impact,
            "confidence": confidence,
            "effort": effort
        }
```

### 2. 业务价值 vs 技术复杂度矩阵

**四象限分析**：

```
高业务价值
    ↑
    │  [快速实现]    [战略投资]
    │  高价值+低复杂度  高价值+高复杂度
    │
    │  [避免]        [评估]
    │  低价值+低复杂度  低价值+高复杂度
    │
    └──────────────────────────→ 高复杂度
```

**决策原则**：
- **快速实现**（高价值+低复杂度）：优先实现
- **战略投资**（高价值+高复杂度）：长期规划
- **避免**（低价值+低复杂度）：不实现
- **评估**（低价值+高复杂度）：重新评估

### 3. Kano 模型（用户满意度）

**三类需求**：

1. **基本需求（Must-have）**：必须有，否则用户不满意
2. **期望需求（Performance）**：越多越好，线性提升满意度
3. **兴奋需求（Delight）**：没有也可以，但有会大幅提升满意度

**优先级**：
- **基本需求** > **期望需求** > **兴奋需求**

---

## KYC 项目实际应用

### KYC 项目 Feature 列表

```python
kyc_features = [
    {
        "name": "身份证字段提取",
        "description": "从身份证图片中提取姓名、身份证号、地址等字段",
        "reach": "all_users",          # 所有用户都需要
        "impact": "massive",           # 核心功能
        "confidence": "high",          # 技术成熟
        "effort": "medium",            # 2 人-月
        "category": "must_have"        # 基本需求
    },
    {
        "name": "字段验证",
        "description": "验证提取的字段格式是否正确（如身份证号格式）",
        "reach": "all_users",
        "impact": "high",              # 重要但不核心
        "confidence": "high",
        "effort": "small",             # 1 人-月
        "category": "must_have"
    },
    {
        "name": "多模型支持",
        "description": "支持切换不同的 OCR 模型（Qwen2.5-VL-32B vs 7B）",
        "reach": "some_users",         # 部分用户需要
        "impact": "medium",            # 中等影响
        "confidence": "medium",        # 需要验证
        "effort": "medium",            # 2 人-月
        "category": "performance"
    },
    {
        "name": "实时监控 Dashboard",
        "description": "实时监控系统指标（Schema Pass Rate、Latency 等）",
        "reach": "few_users",          # 只有运维人员需要
        "impact": "high",              # 对运维很重要
        "confidence": "high",
        "effort": "medium",            # 2 人-月
        "category": "performance"
    },
    {
        "name": "自动回滚",
        "description": "检测到异常自动回滚到稳定版本",
        "reach": "all_users",          # 保护所有用户
        "impact": "high",              # 提高系统稳定性
        "confidence": "high",
        "effort": "large",             # 4 人-月
        "category": "must_have"
    },
    {
        "name": "A/B 测试框架",
        "description": "支持 A/B 测试，对比不同模型/配置的效果",
        "reach": "some_users",         # 只有部分场景需要
        "impact": "medium",            # 中等影响
        "confidence": "low",           # 不确定效果
        "effort": "large",             # 4 人-月
        "category": "delight"
    }
]
```

### RICE 评分计算

```python
# 计算每个 Feature 的 RICE 分数
rice_scorer = RICEScoring()

feature_scores = []
for feature in kyc_features:
    score = rice_scorer.calculate_rice_score(feature)
    feature_scores.append({
        "name": feature["name"],
        "rice_score": score["rice_score"],
        "category": feature["category"],
        "details": score
    })

# 按 RICE 分数排序
feature_scores.sort(key=lambda x: x["rice_score"], reverse=True)

# 输出结果
for feature in feature_scores:
    print(f"{feature['name']}: RICE = {feature['rice_score']:.2f}")
```

**预期结果**：

```
身份证字段提取: RICE = 300.00  (最高优先级)
字段验证: RICE = 200.00
自动回滚: RICE = 75.00
实时监控 Dashboard: RICE = 37.50
多模型支持: RICE = 25.00
A/B 测试框架: RICE = 6.25  (最低优先级)
```

### 业务价值 vs 技术复杂度分析

```python
def analyze_feature_matrix(features):
    """
    分析 Feature 的业务价值和技术复杂度
    """
    matrix = {
        "quick_wins": [],      # 快速实现
        "strategic": [],       # 战略投资
        "avoid": [],           # 避免
        "evaluate": []         # 重新评估
    }
    
    for feature in features:
        business_value = calculate_business_value(feature)
        technical_complexity = calculate_technical_complexity(feature)
        
        if business_value >= 0.7 and technical_complexity <= 0.5:
            matrix["quick_wins"].append(feature)
        elif business_value >= 0.7 and technical_complexity > 0.5:
            matrix["strategic"].append(feature)
        elif business_value < 0.7 and technical_complexity <= 0.5:
            matrix["avoid"].append(feature)
        else:
            matrix["evaluate"].append(feature)
    
    return matrix
```

**KYC 项目分析结果**：

```
快速实现（优先）:
  - 字段验证（高价值+低复杂度）
  - 实时监控 Dashboard（高价值+低复杂度）

战略投资（长期规划）:
  - 身份证字段提取（高价值+高复杂度，但必须做）
  - 自动回滚（高价值+高复杂度）

避免:
  - （KYC 项目中没有低价值功能）

重新评估:
  - A/B 测试框架（低价值+高复杂度，需要重新评估）
```

---

## 综合评估方法

### 1. 多维度评分

```python
class FeaturePrioritization:
    """
    Feature 优先级评估（多维度）
    """
    def __init__(self):
        self.weights = {
            "rice_score": 0.4,          # RICE 分数权重 40%
            "business_value": 0.3,      # 业务价值权重 30%
            "technical_risk": 0.2,      # 技术风险权重 20%
            "strategic_alignment": 0.1   # 战略对齐权重 10%
        }
    
    def calculate_priority_score(self, feature):
        """
        计算综合优先级分数
        """
        # 1. RICE 分数（归一化到 0-1）
        rice_score = self.calculate_rice_score(feature)
        normalized_rice = rice_score / 300.0  # 假设最高分 300
        
        # 2. 业务价值（0-1）
        business_value = self.calculate_business_value(feature)
        
        # 3. 技术风险（0-1，风险越低分数越高）
        technical_risk = self.calculate_technical_risk(feature)
        technical_risk_score = 1.0 - technical_risk
        
        # 4. 战略对齐（0-1）
        strategic_alignment = self.calculate_strategic_alignment(feature)
        
        # 综合分数
        priority_score = (
            self.weights["rice_score"] * normalized_rice +
            self.weights["business_value"] * business_value +
            self.weights["technical_risk"] * technical_risk_score +
            self.weights["strategic_alignment"] * strategic_alignment
        )
        
        return {
            "priority_score": priority_score,
            "rice_score": normalized_rice,
            "business_value": business_value,
            "technical_risk_score": technical_risk_score,
            "strategic_alignment": strategic_alignment
        }
    
    def calculate_business_value(self, feature):
        """
        计算业务价值（0-1）
        """
        # 基于 Kano 模型
        if feature["category"] == "must_have":
            return 1.0
        elif feature["category"] == "performance":
            return 0.7
        elif feature["category"] == "delight":
            return 0.4
        else:
            return 0.5
    
    def calculate_technical_risk(self, feature):
        """
        计算技术风险（0-1，越高风险越大）
        """
        # 基于技术复杂度、团队经验、依赖关系等
        risk_factors = {
            "new_technology": 0.3,      # 新技术
            "complex_integration": 0.2,  # 复杂集成
            "external_dependency": 0.2,  # 外部依赖
            "team_experience": 0.3      # 团队经验
        }
        
        total_risk = sum(risk_factors.values())
        return total_risk
    
    def calculate_strategic_alignment(self, feature):
        """
        计算战略对齐度（0-1）
        """
        # 基于公司战略、产品路线图等
        strategic_features = [
            "身份证字段提取",
            "自动回滚",
            "实时监控 Dashboard"
        ]
        
        if feature["name"] in strategic_features:
            return 1.0
        else:
            return 0.5
```

### 2. 依赖关系分析

```python
def analyze_dependencies(features):
    """
    分析 Feature 之间的依赖关系
    """
    dependencies = {
        "身份证字段提取": [],
        "字段验证": ["身份证字段提取"],
        "多模型支持": ["身份证字段提取"],
        "实时监控 Dashboard": ["身份证字段提取", "字段验证"],
        "自动回滚": ["实时监控 Dashboard"],
        "A/B 测试框架": ["多模型支持"]
    }
    
    # 拓扑排序，确定实现顺序
    execution_order = topological_sort(dependencies)
    
    return execution_order
```

**实现顺序**：

```
1. 身份证字段提取（无依赖）
2. 字段验证（依赖：身份证字段提取）
3. 多模型支持（依赖：身份证字段提取）
4. 实时监控 Dashboard（依赖：身份证字段提取、字段验证）
5. 自动回滚（依赖：实时监控 Dashboard）
6. A/B 测试框架（依赖：多模型支持）
```

---

## 面试回答模板

### 30 秒版本

**问题**：如何选取最重要的 feature？

**回答**：
> "我们使用 RICE 评分模型（Reach × Impact × Confidence / Effort）来量化评估每个 feature。同时结合业务价值 vs 技术复杂度矩阵，优先实现高价值+低复杂度的快速实现功能。对于 KYC 项目，身份证字段提取是最高优先级，因为它是核心功能，影响所有用户，而且技术相对成熟。"

---

### 2 分钟版本

**问题**：详细说明如何选取最重要的 feature。

**回答**：
> "我们使用多维度评估框架来选择最重要的 feature：
> 
> **1. RICE 评分模型**：
> - Reach（覆盖范围）：影响多少用户
> - Impact（影响程度）：对每个用户的影响
> - Confidence（信心度）：我们对这个功能的信心
> - Effort（工作量）：实现需要多少工作量
> 
> RICE = (Reach × Impact × Confidence) / Effort
> 
> **2. 业务价值 vs 技术复杂度矩阵**：
> - 快速实现（高价值+低复杂度）：优先实现
> - 战略投资（高价值+高复杂度）：长期规划
> - 避免（低价值+低复杂度）：不实现
> - 重新评估（低价值+高复杂度）：重新评估
> 
> **3. Kano 模型**：
> - 基本需求（Must-have）：必须有
> - 期望需求（Performance）：越多越好
> - 兴奋需求（Delight）：锦上添花
> 
> **KYC 项目实际应用**：
> - 身份证字段提取：RICE = 300（最高），基本需求，快速实现
> - 字段验证：RICE = 200，基本需求，快速实现
> - 自动回滚：RICE = 75，基本需求，战略投资
> - A/B 测试框架：RICE = 6.25，兴奋需求，重新评估"

---

### 5 分钟版本

**问题**：详细说明如何选取最重要的 feature，包括评估框架、实际应用和决策流程。

**回答**：
> "我们使用多维度评估框架来选择最重要的 feature，包括 RICE 评分、业务价值矩阵、Kano 模型和依赖关系分析。
> 
> **1. RICE 评分模型**：
> 
> RICE = (Reach × Impact × Confidence) / Effort
> 
> - **Reach**：覆盖范围（0-100）
>   - 所有用户：100
>   - 大部分用户：75
>   - 部分用户：50
> 
> - **Impact**：影响程度（0.25-3.0）
>   - 巨大影响：3.0
>   - 高影响：2.0
>   - 中等影响：1.0
> 
> - **Confidence**：信心度（50%-100%）
>   - 高信心（有数据支持）：100%
>   - 中等信心（有一些数据）：80%
>   - 低信心（猜测）：50%
> 
> - **Effort**：工作量（人-月）
>   - 小：1 人-月
>   - 中：2 人-月
>   - 大：4 人-月
> 
> **2. 业务价值 vs 技术复杂度矩阵**：
> 
> 我们使用四象限分析：
> - **快速实现**（高价值+低复杂度）：优先实现，快速获得价值
> - **战略投资**（高价值+高复杂度）：长期规划，分阶段实现
> - **避免**（低价值+低复杂度）：不实现，节省资源
> - **重新评估**（低价值+高复杂度）：重新评估，可能需要调整
> 
> **3. Kano 模型**：
> 
> 根据用户满意度分类：
> - **基本需求**（Must-have）：必须有，否则用户不满意
> - **期望需求**（Performance）：越多越好，线性提升满意度
> - **兴奋需求**（Delight）：没有也可以，但有会大幅提升满意度
> 
> 优先级：基本需求 > 期望需求 > 兴奋需求
> 
> **4. 依赖关系分析**：
> 
> 我们分析 feature 之间的依赖关系，使用拓扑排序确定实现顺序。例如：
> - 身份证字段提取（无依赖）→ 优先实现
> - 字段验证（依赖：身份证字段提取）→ 其次实现
> - 自动回滚（依赖：实时监控 Dashboard）→ 最后实现
> 
> **KYC 项目实际应用**：
> 
> 我们评估了 6 个 feature：
> 
> 1. **身份证字段提取**：
>    - RICE = 300（最高）
>    - 基本需求，快速实现
>    - 优先级：P0
> 
> 2. **字段验证**：
>    - RICE = 200
>    - 基本需求，快速实现
>    - 优先级：P0
> 
> 3. **自动回滚**：
>    - RICE = 75
>    - 基本需求，战略投资
>    - 优先级：P1
> 
> 4. **实时监控 Dashboard**：
>    - RICE = 37.5
>    - 期望需求，快速实现
>    - 优先级：P1
> 
> 5. **多模型支持**：
>    - RICE = 25
>    - 期望需求，战略投资
>    - 优先级：P2
> 
> 6. **A/B 测试框架**：
>    - RICE = 6.25（最低）
>    - 兴奋需求，重新评估
>    - 优先级：P3
> 
> **决策流程**：
> 
> 1. **收集 Feature 列表**：列出所有候选 feature
> 2. **RICE 评分**：计算每个 feature 的 RICE 分数
> 3. **矩阵分析**：分析业务价值和技术复杂度
> 4. **Kano 分类**：根据用户满意度分类
> 5. **依赖分析**：分析 feature 之间的依赖关系
> 6. **综合排序**：综合所有因素，确定优先级
> 7. **资源分配**：根据优先级分配资源"

---

## 最佳实践

### 1. 量化评估

- ✅ 使用 RICE 等量化模型，避免主观判断
- ✅ 定期重新评估，根据数据调整优先级
- ✅ 记录评估过程，便于追溯和复盘

### 2. 多维度考虑

- ✅ 不仅考虑业务价值，还要考虑技术复杂度
- ✅ 考虑依赖关系，确定实现顺序
- ✅ 考虑战略对齐，确保符合公司方向

### 3. 灵活调整

- ✅ 根据实际数据调整优先级
- ✅ 根据用户反馈调整功能
- ✅ 根据技术进展调整计划

---

## 总结

### 核心方法

1. **RICE 评分模型**：量化评估每个 feature
2. **业务价值矩阵**：四象限分析
3. **Kano 模型**：用户满意度分类
4. **依赖关系分析**：确定实现顺序

### 关键要点

- ✅ **量化评估**：避免主观判断
- ✅ **多维度考虑**：业务价值、技术复杂度、依赖关系
- ✅ **灵活调整**：根据数据反馈调整优先级

---

**Remember**: The most important feature is not always the most complex one, but the one that delivers the most value with the least effort.
