# Day 3｜回归测试与门禁：把 AI 系统变成"可回测工程"

---
doc_type: tutorial
layer: L2
scope_in:  回归测试（Golden Set、Before/After对比）、Release Gate（门禁指标、阻断机制）、评估报告模板
scope_out: 具体测试实现代码（见 howto）；测试数据准备（见 reference）；CI/CD 集成（见 reference）
inputs:   (读者) 需求：理解回归测试设计，知道如何建立 Golden Set 和 Release Gate，确保改动不会把系统搞坏
outputs:  回归测试完整设计 + Golden Set 构建策略 + Release Gate 设计 + Before/After 对比流程 + KYC 项目实际案例
entrypoints: [ Golden Set 构建, Release Gate 设计, 回归测试流程 ]
children: [ KYC_Day03_A1_B1_Golden_Set存储和使用详解.md, KYC_Day03_A1_B2_Datadog数据发送详解.md, KYC_Day03_A1_B3_大厂级回归测试实践详解.md, KYC_Day03_A1_B4_评估报告模板和生成详解.md, KYC_Day03_A1_B5_测试用例优先级管理详解.md ]
related: [ 回归测试, Golden Set, Release Gate, Before/After 对比, 评估报告, KYC_Day01_A1_详细讲解_指标与测试.md, KYC_Day02_A1_可观测性详解.md ]
---

## Definition（定义）

**核心问题**：**如何确保每次 prompt/模型/validator 改动不会把系统搞坏？**

**核心答案**：
- ✅ **Golden Set（黄金测试集）**：50-200 条测试用例，覆盖正常、边界、异常、长尾场景
- ✅ **回归测试流程**：Before/After 对比，识别退化（Regression）
- ✅ **Release Gate（门禁）**：通过阈值、阻断机制，不达标禁止发布

**类比**：
- **Golden Set** = **体检套餐**（覆盖所有关键检查项）
- **回归测试** = **体检报告对比**（这次体检 vs 上次体检）
- **Release Gate** = **体检合格标准**（不达标不能入职）

---

## 🎯 为什么要练回归测试？

### Senior 的价值定位

**不是"能交付"**：
- ❌ 只关注功能实现（能跑就行）
- ❌ 不关心改动的影响（改了再说）

**而是"能交付且不会搞坏系统"**：
- ✅ 每次改动都有证据（Before/After 对比）
- ✅ 建立门禁机制（不达标禁止发布）
- ✅ 快速识别退化（Regression Detection）

**面试中的价值**：
- ✅ 能讲出"Golden Set 构建策略"：如何选择测试用例，如何分类
- ✅ 能设计"Release Gate"：门禁指标、阈值设定、阻断机制
- ✅ 能说明"回归测试流程"：Before/After 对比、退化分析、发布决策

---

## 📊 Golden Set（黄金测试集）详解

### 1. Golden Set 是什么？

**定义**：**一组精心挑选的测试用例，代表系统的关键场景和边界情况**。

**核心价值**：
- ✅ **代表性强**：覆盖系统的主要使用场景
- ✅ **稳定性高**：测试用例本身不会频繁变化
- ✅ **可重复**：每次回归测试都用同一组用例
- ✅ **可对比**：Before/After 对比才有意义

**类比**：
- **Golden Set** = **体检套餐**（覆盖所有关键检查项）
- **不是所有数据**：不是把所有生产数据都拿来测试（太多、太杂）
- **而是精选数据**：选择最能代表系统能力的测试用例

---

### 2. Golden Set 构建策略

**目标规模**：**50-200 条测试用例**（平衡覆盖度和执行成本）

**构建方式**：**通常是手动和自动结合**（业界常见做法）

| 构建方式 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| **手动挑选** | ✅ 质量高、代表性强<br>✅ 覆盖关键场景<br>✅ 人工判断边界情况 | ⚠️ 耗时耗力<br>⚠️ 可能遗漏 | 初始构建、关键场景 |
| **自动生成** | ✅ 快速、省时<br>✅ 覆盖全面<br>✅ 可重复 | ⚠️ 质量可能不高<br>⚠️ 可能包含噪音 | 大规模数据、初步筛选 |
| **混合方式**（推荐） | ✅ 结合两者优势<br>✅ 人工筛选 + 自动补全 | ⚠️ 需要两套流程 | 生产环境（业界常见） |

**业界常见做法（混合方式）**：

1. **第一步：自动生成候选集**
   - 从生产数据中采样（覆盖不同场景）
   - 从历史错误中提取（覆盖失败场景）
   - 从用户反馈中收集（覆盖真实场景）

2. **第二步：手动筛选**
   - 人工审核候选集
   - 选择代表性的测试用例
   - 确保覆盖关键场景

3. **第三步：定期补充**
   - 从生产环境的错误中提取新的测试用例
   - 定期审查和更新 Golden Set

**代码示例**：

```python
# 混合方式构建 Golden Set
def build_golden_set_hybrid():
    """混合方式构建 Golden Set"""
    
    # 第一步：自动生成候选集
    candidates = []
    
    # 1. 从生产数据中采样
    production_data = load_production_data(last_30_days=True)
    sampled_cases = random.sample(production_data, min(100, len(production_data)))
    candidates.extend(sampled_cases)
    
    # 2. 从历史错误中提取
    error_cases = load_error_cases(last_90_days=True)
    candidates.extend(error_cases)  # 错误案例全部包含
    
    # 3. 从用户反馈中收集
    user_feedback = load_user_feedback(high_priority=True)
    candidates.extend(user_feedback)
    
    # 第二步：手动筛选（需要人工审核）
    golden_set = manual_review_and_select(candidates, target_count=100)
    
    # 第三步：定期补充（从新错误中提取）
    new_errors = get_new_errors_since_last_review()
    for error in new_errors:
        if should_add_to_golden_set(error):
            golden_set.append(create_test_case_from_error(error))
    
    return golden_set

def manual_review_and_select(candidates: list, target_count: int) -> list:
    """人工审核和筛选候选集"""
    # 这里需要人工判断：
    # - 这个用例是否代表性强？
    # - 这个用例是否覆盖关键场景？
    # - 这个用例是否会频繁变化？（如果是，不适合加入 Golden Set）
    
    reviewed_cases = []
    for case in candidates:
        # 人工判断：是否应该加入 Golden Set
        if should_include_in_golden_set(case):
            reviewed_cases.append(case)
    
    # 确保覆盖关键场景
    ensure_coverage(reviewed_cases, target_count)
    
    return reviewed_cases[:target_count]
```

**实际工作流程**：

```
生产数据（大量）
    ↓ [自动采样]
候选集（1000+ 条）
    ↓ [人工筛选]
Golden Set（100 条）
    ↓ [定期审查]
补充新用例
    ↓
更新 Golden Set
```

**为什么不能完全自动？**

- ❌ **自动无法判断"代表性"**：不知道哪些用例最能代表系统能力
- ❌ **自动无法判断"稳定性"**：不知道哪些用例会频繁变化
- ❌ **自动无法覆盖"边界情况"**：需要人工设计边界测试用例

**为什么不能完全手动？**

- ❌ **手动耗时耗力**：从大量数据中手动选择效率低
- ❌ **手动可能遗漏**：可能遗漏重要的场景
- ❌ **手动难以扩展**：随着数据量增加，手动成本越来越高

**业界最佳实践（混合方式）**：

1. **自动生成候选集**（快速、全面）
2. **手动筛选和审核**（质量、代表性）
3. **定期自动补充**（从错误中提取）
4. **人工定期审查**（确保质量）

---

**📦 Golden Set 存储和使用**：
- 详见：[Golden Set 存储和使用详解](./KYC_Day03_A1_B1_Golden_Set存储和使用详解.md)
- **核心要点**：
  - ✅ **存储方案**：Git 仓库（代码）+ 对象存储 S3（测试数据文件）
  - ✅ **使用方式**：CI/CD 自动触发 + 定期回归测试
  - ✅ **结果监控**：发送到 Datadog 进行监控和告警
  - ❌ **不要只用本地存储**：无法团队协作
  - ❌ **不要用 Datadog 存储数据**：Datadog 是监控平台，不是存储平台

---

**覆盖场景**：

| 场景类型 | 占比 | 说明 | 示例 |
|---------|------|------|------|
| **正常场景** | 20% | 清晰、标准格式 | 标准身份证、标准护照 |
| **边界场景** | 30% | 模糊、遮挡、低质量 | 模糊照片、部分遮挡、低分辨率 |
| **异常场景** | 30% | 版式变化、多页、复杂布局 | 非常规布局、多栏布局、表格密集 |
| **长尾场景** | 20% | 罕见格式、特殊字符、极端情况 | 特殊字符、多语言混排、极端长度 |

**KYC 项目示例**：

```python
# Golden Set 构建示例
golden_set = {
    "normal_cases": [
        {
            "case_id": "normal_001",
            "file_path": "test_data/normal/id_card_standard.jpg",
            "expected_fields": {
                "name": "张三",
                "id_number": "110101199001011234",
                "date_of_birth": "1990-01-01"
            },
            "category": "normal",
            "description": "标准身份证，清晰、标准格式"
        },
        # ... 更多正常场景
    ],
    "edge_cases": [
        {
            "case_id": "edge_001",
            "file_path": "test_data/edge/id_card_blurry.jpg",
            "expected_fields": {
                "name": "李四",
                "id_number": "110101199002021234"
            },
            "category": "edge",
            "description": "模糊照片，测试 OCR 能力"
        },
        # ... 更多边界场景
    ],
    "anomaly_cases": [
        {
            "case_id": "anomaly_001",
            "file_path": "test_data/anomaly/passport_multipage.pdf",
            "expected_fields": {
                "name": "王五",
                "passport_number": "E12345678"
            },
            "category": "anomaly",
            "description": "多页护照，测试多页处理能力"
        },
        # ... 更多异常场景
    ],
    "longtail_cases": [
        {
            "case_id": "longtail_001",
            "file_path": "test_data/longtail/id_card_special_chars.jpg",
            "expected_fields": {
                "name": "赵六·史密斯",
                "id_number": "110101199003031234"
            },
            "category": "longtail",
            "description": "特殊字符，测试字符处理能力"
        },
        # ... 更多长尾场景
    ]
}

# 统计
total_cases = (
    len(golden_set["normal_cases"]) +
    len(golden_set["edge_cases"]) +
    len(golden_set["anomaly_cases"]) +
    len(golden_set["longtail_cases"])
)
# 目标：50-200 条
```

---

### 3. Golden Set 分类

#### a) Hard Cases（困难案例）

**定义**：**容易出错的场景，用于测试系统的鲁棒性**。

**分类**：

1. **模糊/遮挡**：
   - 图片模糊（测试 OCR 能力）
   - 部分遮挡（测试部分信息提取）
   - 低分辨率（测试图像处理能力）

2. **版式变化**：
   - 非常规布局（测试布局理解能力）
   - 多栏布局（测试多栏解析）
   - 表格密集（测试表格提取能力）

3. **长尾情况**：
   - 特殊字符/编码（测试字符处理）
   - 多语言混排（测试多语言支持）
   - 极端长度（测试长文本处理）

**KYC 项目示例**：

```python
hard_cases = [
    {
        "case_id": "hard_001",
        "file_path": "test_data/hard/id_card_blurry.jpg",
        "difficulty": "blur",
        "expected_challenge": "OCR 可能无法识别模糊文字",
        "test_purpose": "测试系统对模糊图片的处理能力"
    },
    {
        "case_id": "hard_002",
        "file_path": "test_data/hard/passport_partial_occlusion.jpg",
        "difficulty": "occlusion",
        "expected_challenge": "部分信息被遮挡",
        "test_purpose": "测试系统对部分遮挡的处理能力"
    },
    # ... 更多困难案例
]
```

---

#### b) Critical Cases（关键案例）

**定义**：**业务关键字段，必须提取成功的场景**。

**分类**：

1. **必须提取成功的字段**：
   - 身份证号（KYC 核心字段）
   - 姓名（身份验证必需）
   - 出生日期（年龄验证）

2. **对业务影响大的字段**：
   - 有效期（判断证件是否过期）
   - 签发机关（验证证件真实性）

**KYC 项目示例**：

```python
critical_cases = [
    {
        "case_id": "critical_001",
        "file_path": "test_data/critical/id_card_standard.jpg",
        "critical_fields": ["id_number", "name", "date_of_birth"],
        "business_impact": "high",
        "test_purpose": "确保核心字段提取准确"
    },
    {
        "case_id": "critical_002",
        "file_path": "test_data/critical/passport_expiry.jpg",
        "critical_fields": ["expiry_date", "passport_number"],
        "business_impact": "high",
        "test_purpose": "确保有效期提取准确（影响业务决策）"
    },
    # ... 更多关键案例
]
```

---

### 4. Golden Set 维护

**版本管理**：
- ✅ **Golden Set 版本化**：与模型/prompt 版本绑定
- ✅ **变更记录**：记录每次 Golden Set 的变更（新增、删除、修改用例）
- ✅ **版本对比**：不同版本的 Golden Set 可以对比

**定期审查**：
- ✅ **每月回顾**：补充新发现的 edge cases
- ✅ **生产数据反馈**：从生产环境的错误中提取新的测试用例
- ✅ **覆盖率检查**：确保 Golden Set 覆盖所有关键场景

**自动化**：
- ✅ **CI/CD 集成**：每次代码变更自动运行回归测试
- ✅ **自动化报告**：自动生成 Before/After 对比报告
- ✅ **门禁阻断**：不达标自动阻止发布

**代码示例**：

```python
# Golden Set 版本管理
class GoldenSetManager:
    def __init__(self):
        self.version = "v1.0.0"
        self.cases = []
        self.version_history = []
    
    def add_case(self, case: dict):
        """添加测试用例"""
        case["added_in_version"] = self.version
        self.cases.append(case)
    
    def remove_case(self, case_id: str):
        """删除测试用例"""
        self.cases = [c for c in self.cases if c["case_id"] != case_id]
        self.version_history.append({
            "version": self.version,
            "action": "remove",
            "case_id": case_id
        })
    
    def update_version(self, new_version: str):
        """更新版本"""
        self.version_history.append({
            "version": self.version,
            "cases_count": len(self.cases)
        })
        self.version = new_version
    
    def export(self) -> dict:
        """导出 Golden Set"""
        return {
            "version": self.version,
            "total_cases": len(self.cases),
            "cases": self.cases,
            "version_history": self.version_history
        }
```

---

## 🚪 Release Gate（门禁）详解

### 1. Release Gate 是什么？

**定义**：**发布前的质量门禁，通过阈值判断是否允许发布**。

**核心价值**：
- ✅ **防止退化**：不达标的改动不能发布
- ✅ **证据化**：每次发布都有数据支撑
- ✅ **自动化**：CI/CD 自动判断，减少人为错误

**类比**：
- **Release Gate** = **体检合格标准**（不达标不能入职）
- **不是主观判断**：不是"我觉得可以发布"
- **而是客观标准**：所有指标必须达到阈值

---

### 2. 核心门禁指标

#### a) Schema Pass Rate（Schema 通过率）

**定义**：**Schema 验证通过的请求数 / 总请求数**

**计算公式**：
```python
schema_pass_rate = (
    sum(1 for r in results if r["status"] == "success" and r["schema_valid"] == True) /
    len(results)
) * 100
```

**KYC 项目阈值**：
- ✅ **门禁阈值**：`> 95%`（95% 以上通过才能发布）
- ✅ **当前值**：需要从 Golden Set 测试结果计算
- ✅ **门禁规则**：如果 < 95%，**禁止发布**

**示例**：
```
Golden Set: 100 个测试用例
Schema Pass: 96 个
Schema Pass Rate: 96% ✅ 通过（> 95%）

如果 Schema Pass Rate = 94% ❌ 不通过（< 95%）
→ 禁止发布，需要修复后再测
```

---

#### b) 字段级准确率（Field-level Accuracy）

**定义**：**正确提取的字段数 / 总字段数**

**计算公式**：
```python
def calculate_field_accuracy(results: list, golden_set: list) -> float:
    """计算字段级准确率"""
    total_fields = 0
    correct_fields = 0
    
    for result, expected in zip(results, golden_set):
        for field_name, expected_value in expected["expected_fields"].items():
            total_fields += 1
            actual_value = result.get("extracted_fields", {}).get(field_name)
            if actual_value == expected_value:
                correct_fields += 1
    
    return (correct_fields / total_fields) * 100 if total_fields > 0 else 0
```

**KYC 项目阈值**：
- ✅ **门禁阈值**：`> 90%`（90% 以上字段准确才能发布）
- ✅ **门禁规则**：如果 < 90%，**禁止发布**

**示例**：
```
Golden Set: 100 个测试用例，每个用例 5 个字段
总字段数: 500
正确字段数: 460
字段级准确率: 92% ✅ 通过（> 90%）

如果字段级准确率 = 88% ❌ 不通过（< 90%）
→ 禁止发布，需要修复后再测
```

---

#### c) 字段级一致性（Field-level Consistency）

**定义**：**相同输入多次提取结果一致的字段数 / 总字段数**

**计算公式**：
```python
def calculate_field_consistency(case_id: str, num_runs: int = 3) -> float:
    """计算字段级一致性"""
    # 对同一个用例运行多次
    results = []
    for _ in range(num_runs):
        result = run_test_case(case_id)
        results.append(result)
    
    # 计算每个字段的一致性
    consistent_fields = 0
    total_fields = 0
    
    for field_name in expected_fields:
        total_fields += 1
        values = [r["extracted_fields"][field_name] for r in results]
        if len(set(values)) == 1:  # 所有值都相同
            consistent_fields += 1
    
    return (consistent_fields / total_fields) * 100 if total_fields > 0 else 0
```

**KYC 项目阈值**：
- ✅ **门禁阈值**：`> 85%`（85% 以上字段一致才能发布）
- ✅ **门禁规则**：如果 < 85%，**禁止发布**

**为什么需要一致性？**
- ✅ **稳定性**：相同输入应该产生相同输出
- ✅ **可预测性**：用户期望一致的结果
- ✅ **质量保证**：不一致可能意味着系统不稳定

---

#### d) Fallback 比例（Fallback Rate）

**定义**：**触发降级的请求数 / 总请求数**

**计算公式**：
```python
fallback_rate = (
    sum(1 for r in results if r.get("fallback_triggered") == True) /
    len(results)
) * 100
```

**KYC 项目阈值**：
- ✅ **门禁阈值**：`< 5%`（Fallback 率低于 5% 才能发布）
- ✅ **门禁规则**：如果 > 5%，**禁止发布**

**为什么限制 Fallback？**
- ✅ **质量保证**：Fallback 率高可能意味着主流程有问题
- ✅ **成本控制**：Fallback 通常使用更昂贵的模型
- ✅ **用户体验**：Fallback 可能导致延迟增加

---

#### e) 成本上限（Cost Cap）

**定义**：**tokens / request 或 $ / request**

**计算公式**：
```python
avg_cost_per_request = (
    sum(r.get("cost_usd", 0) for r in results) /
    len(results)
)

avg_tokens_per_request = (
    sum(r.get("tokens_used", 0) for r in results) /
    len(results)
)
```

**KYC 项目阈值**：
- ✅ **门禁阈值**：`< $0.002 / request` 或 `< 2000 tokens/request`
- ✅ **门禁规则**：如果 > 阈值，**禁止发布**

**为什么限制成本？**
- ✅ **成本控制**：防止成本意外增加
- ✅ **预算管理**：确保在预算范围内
- ✅ **可扩展性**：成本过高会影响系统扩展

---

### 3. Release Gate 决策流程

**流程**：

```
代码变更（Prompt/Model/Validator）
    ↓
运行 Golden Set 回归测试
    ↓
计算所有门禁指标
    ↓
对比 Before/After
    ↓
检查所有指标是否满足阈值
    ↓
┌─────────────────┐
│ 所有指标通过？   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
   Yes       No
    │         │
    ▼         ▼
允许发布    禁止发布
            ↓
        修复后再测
```

**代码示例**：

```python
def check_release_gate(before_results: dict, after_results: dict) -> dict:
    """检查 Release Gate"""
    gates = {
        "schema_pass_rate": {
            "threshold": 95,
            "before": before_results["schema_pass_rate"],
            "after": after_results["schema_pass_rate"],
            "passed": after_results["schema_pass_rate"] >= 95
        },
        "field_accuracy": {
            "threshold": 90,
            "before": before_results["field_accuracy"],
            "after": after_results["field_accuracy"],
            "passed": after_results["field_accuracy"] >= 90
        },
        "field_consistency": {
            "threshold": 85,
            "before": before_results["field_consistency"],
            "after": after_results["field_consistency"],
            "passed": after_results["field_consistency"] >= 85
        },
        "fallback_rate": {
            "threshold": 5,
            "before": before_results["fallback_rate"],
            "after": after_results["fallback_rate"],
            "passed": after_results["fallback_rate"] < 5
        },
        "cost_per_request": {
            "threshold": 0.002,
            "before": before_results["avg_cost_per_request"],
            "after": after_results["avg_cost_per_request"],
            "passed": after_results["avg_cost_per_request"] < 0.002
        }
    }
    
    # 检查是否所有门禁都通过
    all_passed = all(gate["passed"] for gate in gates.values())
    
    return {
        "all_passed": all_passed,
        "gates": gates,
        "decision": "允许发布" if all_passed else "禁止发布",
        "failed_gates": [name for name, gate in gates.items() if not gate["passed"]]
    }
```

---

## 📊 回归测试流程详解

### 1. 回归测试是什么？

**定义**：**在代码变更后，运行 Golden Set，对比 Before/After 结果，识别退化（Regression）**。

**核心价值**：
- ✅ **证据化**：每次改动都有数据支撑
- ✅ **快速发现**：快速识别性能退化
- ✅ **自动化**：CI/CD 自动运行，减少人工成本

**类比**：
- **回归测试** = **体检报告对比**（这次体检 vs 上次体检）
- **不是单次测试**：不是只看这次的结果
- **而是对比测试**：对比 Before/After，看是否有退化

---

### 2. 回归测试执行步骤

#### 步骤 1：准备环境

```python
# 1. 部署新版本到测试环境
def deploy_to_test_env(version: str):
    """部署新版本到测试环境"""
    # 部署代码
    deploy_code(version)
    
    # 等待服务就绪
    wait_for_service_ready()
    
    # 健康检查
    health_check()

# 2. 准备 Golden Set 数据
def prepare_golden_set(golden_set_version: str):
    """准备 Golden Set 数据"""
    golden_set = load_golden_set(golden_set_version)
    return golden_set
```

---

#### 步骤 2：运行测试

```python
# 批量执行 Golden Set
def run_regression_test(golden_set: list, version: str) -> list:
    """运行回归测试"""
    results = []
    
    for case in golden_set:
        result = run_test_case(
            case_id=case["case_id"],
            file_path=case["file_path"],
            version=version
        )
        results.append(result)
    
    return results

# 收集结果（输出、延迟、成本）
def collect_results(results: list) -> dict:
    """收集测试结果"""
    return {
        "total_cases": len(results),
        "schema_pass_count": sum(1 for r in results if r["schema_valid"]),
        "field_accuracy": calculate_field_accuracy(results),
        "field_consistency": calculate_field_consistency(results),
        "fallback_count": sum(1 for r in results if r.get("fallback_triggered")),
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results),
        "avg_cost_usd": sum(r.get("cost_usd", 0) for r in results) / len(results),
        "results": results
    }
```

---

#### 步骤 3：计算指标

```python
# 计算所有门禁指标
def calculate_all_metrics(results: list) -> dict:
    """计算所有门禁指标"""
    return {
        "schema_pass_rate": calculate_schema_pass_rate(results),
        "field_accuracy": calculate_field_accuracy(results),
        "field_consistency": calculate_field_consistency(results),
        "fallback_rate": calculate_fallback_rate(results),
        "avg_cost_per_request": calculate_avg_cost(results),
        "p95_latency_ms": calculate_p95_latency(results),
        "p99_latency_ms": calculate_p99_latency(results)
    }
```

---

#### 步骤 4：对比基准

```python
# 与上一版本对比（Before/After）
def compare_before_after(before_results: dict, after_results: dict) -> dict:
    """对比 Before/After"""
    comparison = {}
    
    for metric_name in before_results.keys():
        before_value = before_results[metric_name]
        after_value = after_results[metric_name]
        delta = after_value - before_value
        delta_percent = (delta / before_value * 100) if before_value != 0 else 0
        
        comparison[metric_name] = {
            "before": before_value,
            "after": after_value,
            "delta": delta,
            "delta_percent": delta_percent,
            "regression": delta < 0  # 负值表示退化
        }
    
    return comparison
```

---

#### 步骤 5：门禁判断

```python
# 所有指标是否满足阈值？
def check_all_gates(metrics: dict, gates: dict) -> bool:
    """检查所有门禁是否通过"""
    for gate_name, gate_config in gates.items():
        metric_value = metrics.get(gate_name)
        threshold = gate_config["threshold"]
        operator = gate_config.get("operator", ">=")  # >=, <=, <, >
        
        if operator == ">=":
            if metric_value < threshold:
                return False
        elif operator == "<=":
            if metric_value > threshold:
                return False
        elif operator == "<":
            if metric_value >= threshold:
                return False
        elif operator == ">":
            if metric_value <= threshold:
                return False
    
    return True
```

---

#### 步骤 6：发布决策

```python
# 通过 → 允许发布
# 不通过 → 修复后再测
def make_release_decision(gate_result: dict) -> dict:
    """发布决策"""
    if gate_result["all_passed"]:
        return {
            "decision": "允许发布",
            "reason": "所有门禁指标通过",
            "action": "proceed"
        }
    else:
        return {
            "decision": "禁止发布",
            "reason": f"以下门禁未通过: {', '.join(gate_result['failed_gates'])}",
            "action": "block",
            "failed_gates": gate_result["failed_gates"]
        }
```

---

## 📊 Before/After 对比详解

### 1. 对比维度

#### a) 准确率对比

```python
# Before/After 准确率对比
comparison = {
    "schema_pass_rate": {
        "before": 96.0,  # %
        "after": 94.0,   # %
        "delta": -2.0,   # 退化 2%
        "regression": True
    },
    "field_accuracy": {
        "before": 92.0,  # %
        "after": 90.0,   # %
        "delta": -2.0,   # 退化 2%
        "regression": True
    }
}
```

---

#### b) 延迟对比

```python
# Before/After 延迟对比
comparison = {
    "p95_latency_ms": {
        "before": 2000,  # ms
        "after": 2500,   # ms
        "delta": 500,    # 增加 500ms
        "regression": True
    },
    "p99_latency_ms": {
        "before": 3000,  # ms
        "after": 4000,   # ms
        "delta": 1000,   # 增加 1000ms
        "regression": True
    }
}
```

---

#### c) 成本对比

```python
# Before/After 成本对比
comparison = {
    "avg_cost_per_request": {
        "before": 0.0015,  # $
        "after": 0.0025,   # $
        "delta": 0.001,    # 增加 $0.001
        "delta_percent": 66.7,  # 增加 66.7%
        "regression": True
    },
    "avg_tokens_per_request": {
        "before": 1500,   # tokens
        "after": 2500,    # tokens
        "delta": 1000,    # 增加 1000 tokens
        "regression": True
    }
}
```

---

#### d) Fallback 比例对比

```python
# Before/After Fallback 比例对比
comparison = {
    "fallback_rate": {
        "before": 3.0,   # %
        "after": 7.0,    # %
        "delta": 4.0,    # 增加 4%
        "regression": True
    }
}
```

---

### 2. 退化分析（Regression Analysis）

**定义**：**识别性能退化的原因和影响**。

**分析步骤**：

1. **识别退化指标**：
   - 哪些指标退化了？
   - 退化程度如何？

2. **分析失败用例**：
   - 哪些用例失败了？
   - 失败原因是什么？

3. **定位根因**：
   - 是 Prompt 的问题？
   - 是模型的问题？
   - 是 Validator 的问题？

4. **评估影响**：
   - 退化的影响范围？
   - 是否需要立即修复？

**代码示例**：

```python
def analyze_regression(before_results: dict, after_results: dict) -> dict:
    """分析退化"""
    comparison = compare_before_after(before_results, after_results)
    
    # 识别退化指标
    regressed_metrics = [
        name for name, comp in comparison.items()
        if comp.get("regression", False)
    ]
    
    # 分析失败用例
    failed_cases = [
        r for r in after_results["results"]
        if r["status"] == "fail" or r.get("schema_valid") == False
    ]
    
    # 失败原因分类
    failure_reasons = {}
    for case in failed_cases:
        reason = case.get("error_code", "unknown")
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    return {
        "regressed_metrics": regressed_metrics,
        "failed_cases_count": len(failed_cases),
        "failure_reasons": failure_reasons,
        "top_failed_cases": failed_cases[:10]  # Top 10
    }
```

---

## 💡 KYC 项目回归测试实现检查清单

### Golden Set 建立

- [ ] **建立 Golden Set**（50-200 条测试用例）
  - [ ] 正常场景：20%（10-40 条）
  - [ ] 边界场景：30%（15-60 条）
  - [ ] 异常场景：30%（15-60 条）
  - [ ] 长尾场景：20%（10-40 条）

- [ ] **Golden Set 分类**
  - [ ] Hard Cases（困难案例）
  - [ ] Critical Cases（关键案例）

- [ ] **Golden Set 版本管理**
  - [ ] 版本化 Golden Set
  - [ ] 变更记录
  - [ ] 定期审查和更新

---

### Release Gate 配置

- [ ] **定义门禁指标和阈值**
  - [ ] Schema Pass Rate: `> 95%`
  - [ ] 字段级准确率: `> 90%`
  - [ ] 字段级一致性: `> 85%`
  - [ ] Fallback 比例: `< 5%`
  - [ ] 成本上限: `< $0.002 / request`

- [ ] **实现门禁判断逻辑**
  - [ ] 自动计算所有指标
  - [ ] 自动判断是否通过
  - [ ] 自动生成报告

---

### 回归测试自动化

- [ ] **实现回归测试脚本**
  - [ ] 批量执行 Golden Set
  - [ ] 收集结果（输出、延迟、成本）
  - [ ] 计算所有指标

- [ ] **集成到 CI/CD 流程**
  - [ ] 代码变更自动触发回归测试
  - [ ] 门禁不通过自动阻止发布
  - [ ] 自动生成 Before/After 对比报告

---

### 评估报告模板

- [ ] **建立回归报告模板**
  - [ ] 基本信息（版本、变更内容）
  - [ ] Golden Set 信息
  - [ ] 指标对比（Before/After）
  - [ ] 退化分析
  - [ ] 门禁结论

---

## 💡 总结

### 核心答案

**如何确保每次 prompt/模型/validator 改动不会把系统搞坏？**

**答案**：
1. ✅ **Golden Set（黄金测试集）**：50-200 条测试用例，覆盖正常、边界、异常、长尾场景
2. ✅ **回归测试流程**：Before/After 对比，识别退化（Regression）
3. ✅ **Release Gate（门禁）**：通过阈值、阻断机制，不达标禁止发布

**效果**：
- ✅ **证据化**：每次改动都有数据支撑
- ✅ **防止退化**：不达标的改动不能发布
- ✅ **自动化**：CI/CD 自动运行，减少人工成本

### 关键要点

1. **Golden Set 是基础**：选择代表性的测试用例，覆盖关键场景
2. **Release Gate 是保障**：通过阈值判断，防止退化
3. **Before/After 对比是关键**：对比才能发现退化

### 面试话术

- ✅ "我们建立 Golden Set（50-200 条测试用例），覆盖正常、边界、异常、长尾场景。每次代码变更（Prompt/Model/Validator）都会运行回归测试，对比 Before/After 结果，识别退化。我们设置 Release Gate：Schema Pass Rate > 95%，字段级准确率 > 90%，Fallback 比例 < 5%，成本 < $0.002/request。如果任何指标不达标，自动阻止发布，确保改动不会把系统搞坏。"

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | Day 3 回归测试与门禁（本文档） |
| **Related** | 回归测试、Golden Set、Release Gate、Before/After 对比、评估报告、[KYC_Day01_A1_详细讲解_指标与测试.md](../day01/KYC_Day01_A1_详细讲解_指标与测试.md)、[KYC_Day02_A1_可观测性详解.md](../day02/KYC_Day02_A1_可观测性详解.md) |
