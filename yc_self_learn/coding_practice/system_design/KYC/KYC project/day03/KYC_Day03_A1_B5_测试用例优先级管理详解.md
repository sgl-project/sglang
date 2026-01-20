# 测试用例优先级管理详解

---
doc_type: tutorial
layer: L2
scope_in:  测试用例优先级管理（Level 0-2）、优先级定义（P0/P1/P2）、优先级执行策略、优先级维护
scope_out: 具体优先级分配工具（见 reference）；优先级自动分配算法（见 reference）
inputs:  (读者) 需求：理解测试用例优先级管理，从 Level 0 到 Level 2 的完整实践
outputs:  优先级管理完整设计 + 执行策略 + 维护流程 + 可视化架构 + KYC 项目实际案例
entrypoints: [ 优先级定义, 优先级分配, 优先级执行, 优先级维护 ]
children: []
related: [ Golden Set, 回归测试, 测试用例管理, KYC_Day03_A1_回归测试与门禁详解.md ]
---

## Definition（定义）

**核心问题**：**如何管理测试用例的优先级？如何根据优先级优化测试执行？**

**核心答案**：
- ✅ **Level 0（基础）**：理解什么是优先级（P0/P1/P2）
- ✅ **Level 1（中级）**：如何分配优先级（手动/自动）
- ✅ **Level 2（高级）**：优先级执行策略（快速失败、并行执行、优先级维护）

---

## 📊 Level 0：优先级基础概念

### 1. 什么是测试用例优先级？

**定义**：**根据测试用例的重要性和影响范围，给用例分配不同的优先级**

**优先级分类**：

| 优先级 | 名称 | 定义 | 失败影响 | 执行策略 |
|--------|------|------|---------|---------|
| **P0** | Critical（关键） | 必须通过的用例，失败会阻断发布 | 🔴 严重 | 优先执行，失败立即停止 |
| **P1** | High（重要） | 应该通过的用例，失败需要审核 | 🟡 中等 | 重要执行，失败需要审核 |
| **P2** | Medium（一般） | 可选用例，失败不影响发布 | 🟢 轻微 | 可选执行，失败不影响发布 |

### 2. 优先级决策树

```
测试用例
    ↓
是核心功能吗？
    ├─ 是 → P0（关键用例）
    │       └─ 失败会阻断发布
    │
    └─ 否 → 继续判断
            ↓
        是重要功能吗？
            ├─ 是 → P1（重要用例）
            │       └─ 失败需要审核
            │
            └─ 否 → P2（一般用例）
                    └─ 失败不影响发布
```

### 3. KYC 项目优先级示例

```python
# KYC 项目优先级分配示例
priority_examples = {
    "P0": [
        {
            "case_id": "critical_001",
            "category": "身份证号提取",
            "priority": "P0",
            "reason": "核心功能，失败会导致业务无法进行",
            "failure_impact": "阻断发布"
        },
        {
            "case_id": "critical_002",
            "category": "姓名提取",
            "priority": "P0",
            "reason": "核心功能，失败会导致业务无法进行",
            "failure_impact": "阻断发布"
        }
    ],
    "P1": [
        {
            "case_id": "important_001",
            "category": "地址提取",
            "priority": "P1",
            "reason": "重要功能，失败会影响用户体验",
            "failure_impact": "需要审核"
        }
    ],
    "P2": [
        {
            "case_id": "normal_001",
            "category": "次要字段提取",
            "priority": "P2",
            "reason": "辅助功能，失败不影响核心业务",
            "failure_impact": "不影响发布"
        }
    ]
}
```

---

## 📊 Level 1：优先级分配策略

### 1. 优先级分配流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   测试用例优先级分配流程（Level 1）                        │
└─────────────────────────────────────────────────────────────────────────┘

【步骤 1：用例分类】
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  所有测试用例（100 条）                                                  │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  ① 核心功能用例？                                                  │ │
│  │     ├─ 是 → P0（10-20 条，10-20%）                                │ │
│  │     │   - 身份证号提取                                              │ │
│  │     │   - 姓名提取                                                  │ │
│  │     │   - 出生日期提取                                              │ │
│  │     │                                                              │ │
│  │     └─ 否 → 继续判断                                               │ │
│  │                                                                    │ │
│  │  ② 重要功能用例？                                                  │ │
│  │     ├─ 是 → P1（30-40 条，30-40%）                                │ │
│  │     │   - 地址提取                                                  │ │
│  │     │   - 有效期提取                                                │ │
│  │     │   - 签发机关提取                                              │ │
│  │     │                                                              │ │
│  │     └─ 否 → P2（40-50 条，40-50%）                                │ │
│  │         - 次要字段提取                                              │ │
│  │         - 边界场景测试                                              │ │
│  │                                                                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

【步骤 2：优先级分配方法】

方法 A：手动分配（小项目）
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  开发者/QA 手动分配                                                      │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  1. 人工审核每个用例                                                │ │
│  │  2. 根据业务重要性分配优先级                                        │ │
│  │  3. 记录分配原因                                                    │ │
│  │                                                                    │ │
│  │  优点：                                                             │ │
│  │  ✅ 质量高（人工判断）                                              │ │
│  │  ✅ 考虑业务场景                                                    │ │
│  │                                                                    │ │
│  │  缺点：                                                             │ │
│  │  ⚠️ 耗时耗力                                                        │ │
│  │  ⚠️ 主观性强                                                        │ │
│  │                                                                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

方法 B：自动分配（大厂）
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  自动分配算法                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  1. 基于规则自动分配                                                │ │
│  │     - 核心字段 → P0                                                │ │
│  │     - 重要字段 → P1                                                │ │
│  │     - 其他字段 → P2                                                │ │
│  │                                                                    │ │
│  │  2. 基于历史数据自动分配                                            │ │
│  │     - 历史失败次数多 → P0                                          │ │
│  │     - 历史失败次数少 → P1/P2                                       │ │
│  │                                                                    │ │
│  │  3. 基于业务规则自动分配                                            │ │
│  │     - 用户量大的场景 → P0                                          │ │
│  │     - 用户量小的场景 → P1/P2                                       │ │
│  │                                                                    │ │
│  │  优点：                                                             │ │
│  │  ✅ 快速、可重复                                                    │ │
│  │  ✅ 客观、一致                                                      │ │
│  │                                                                    │ │
│  │  缺点：                                                             │ │
│  │  ⚠️ 可能不够准确                                                    │ │
│  │  ⚠️ 需要人工审核                                                    │ │
│  │                                                                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

方法 C：混合方式（推荐）
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  自动分配 + 人工审核                                                     │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  ① 自动分配（快速、初步）                                           │ │
│  │     ↓                                                              │ │
│  │  ② 人工审核（质量、准确性）                                         │ │
│  │     ↓                                                              │ │
│  │  ③ 最终确定优先级                                                   │ │
│  │                                                                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. 优先级分配规则

#### 规则 1：基于业务重要性

```python
def assign_priority_by_business_importance(case: Dict) -> str:
    """基于业务重要性分配优先级"""
    
    # 核心字段 → P0
    core_fields = ["id_number", "name", "date_of_birth"]
    if any(field in case["fields"] for field in core_fields):
        return "P0"
    
    # 重要字段 → P1
    important_fields = ["address", "expiry_date", "issuing_authority"]
    if any(field in case["fields"] for field in important_fields):
        return "P1"
    
    # 其他字段 → P2
    return "P2"
```

#### 规则 2：基于历史失败率

```python
def assign_priority_by_failure_rate(case: Dict, historical_results: List[Dict]) -> str:
    """基于历史失败率分配优先级"""
    
    case_id = case["case_id"]
    
    # 统计历史失败次数
    failure_count = sum(
        1 for result in historical_results
        if result["case_id"] == case_id and result["status"] == "fail"
    )
    
    total_runs = len([r for r in historical_results if r["case_id"] == case_id])
    failure_rate = failure_count / total_runs if total_runs > 0 else 0
    
    # 失败率 > 20% → P0（经常失败的用例，需要重点测试）
    if failure_rate > 0.2:
        return "P0"
    
    # 失败率 5-20% → P1
    elif failure_rate > 0.05:
        return "P1"
    
    # 失败率 < 5% → P2
    else:
        return "P2"
```

#### 规则 3：基于用户影响

```python
def assign_priority_by_user_impact(case: Dict, user_stats: Dict) -> str:
    """基于用户影响分配优先级"""
    
    case_category = case["category"]
    
    # 用户量大的场景 → P0
    if user_stats.get(case_category, {}).get("user_count", 0) > 10000:
        return "P0"
    
    # 用户量中等的场景 → P1
    elif user_stats.get(case_category, {}).get("user_count", 0) > 1000:
        return "P1"
    
    # 用户量小的场景 → P2
    else:
        return "P2"
```

### 3. 优先级分配代码示例

```python
class TestCasePriorityManager:
    """测试用例优先级管理器"""
    
    def __init__(self):
        self.priority_rules = {
            "business_importance": assign_priority_by_business_importance,
            "failure_rate": assign_priority_by_failure_rate,
            "user_impact": assign_priority_by_user_impact
        }
    
    def assign_priority(self, case: Dict, historical_results: List[Dict] = None, user_stats: Dict = None) -> str:
        """分配优先级（混合策略）"""
        
        priorities = []
        
        # 策略 1：基于业务重要性
        priority1 = self.priority_rules["business_importance"](case)
        priorities.append(priority1)
        
        # 策略 2：基于历史失败率（如果有历史数据）
        if historical_results:
            priority2 = self.priority_rules["failure_rate"](case, historical_results)
            priorities.append(priority2)
        
        # 策略 3：基于用户影响（如果有用户数据）
        if user_stats:
            priority3 = self.priority_rules["user_impact"](case, user_stats)
            priorities.append(priority3)
        
        # 取最高优先级（P0 > P1 > P2）
        priority_map = {"P0": 0, "P1": 1, "P2": 2}
        return min(priorities, key=lambda p: priority_map[p])
    
    def batch_assign_priority(self, cases: List[Dict], historical_results: List[Dict] = None, user_stats: Dict = None) -> List[Dict]:
        """批量分配优先级"""
        
        for case in cases:
            case["priority"] = self.assign_priority(case, historical_results, user_stats)
        
        return cases

# 使用示例
manager = TestCasePriorityManager()

cases = [
    {"case_id": "case_001", "fields": ["id_number", "name"], "category": "id_card"},
    {"case_id": "case_002", "fields": ["address"], "category": "id_card"},
    {"case_id": "case_003", "fields": ["minor_field"], "category": "id_card"}
]

cases_with_priority = manager.batch_assign_priority(cases)
# 结果：
# case_001: P0（包含核心字段 id_number, name）
# case_002: P1（包含重要字段 address）
# case_003: P2（包含次要字段）
```

---

## 📊 Level 2：优先级执行策略

### 1. 优先级执行架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  优先级执行架构（Level 2）                                │
└─────────────────────────────────────────────────────────────────────────┘

测试调度器（Test Scheduler）
  ├─ 用例分发（按优先级）
  │   ├─ P0 用例（10-20 条）→ 优先执行
  │   ├─ P1 用例（30-40 条）→ 次要执行
  │   └─ P2 用例（40-50 条）→ 可选执行
  │
  ├─ 执行策略
  │   ├─ 快速失败（Fast Fail）
  │   │   └─ P0 用例失败 → 立即停止，不执行后续用例
  │   │
  │   ├─ 并行执行（Parallel）
  │   │   ├─ P0 用例：单线程执行（确保顺序）
  │   │   ├─ P1 用例：多线程并行执行
  │   │   └─ P2 用例：多线程并行执行
  │   │
  │   └─ 资源分配（Resource Allocation）
  │       ├─ P0 用例：分配最多资源（确保快速执行）
  │       ├─ P1 用例：分配中等资源
  │       └─ P2 用例：分配最少资源（可延迟执行）
  │
  └─ 结果收集
      ├─ 实时监控 P0 用例结果
      ├─ 如果 P0 失败，立即告警
      └─ 汇总所有优先级的结果

执行流程：
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  开始执行                                                               │
│    ↓                                                                    │
│  【阶段 1：执行 P0 用例】（10-20 条，单线程，优先执行）                 │
│    │                                                                    │
│    ├─ 执行用例 1（P0）                                                  │
│    │   ├─ 通过 ✅ → 继续执行下一个 P0 用例                             │
│    │   └─ 失败 ❌ → 立即停止，报告失败，不执行后续用例                  │
│    │                                                                    │
│    ├─ 执行用例 2（P0）                                                  │
│    │   ├─ 通过 ✅ → 继续执行下一个 P0 用例                             │
│    │   └─ 失败 ❌ → 立即停止，报告失败                                  │
│    │                                                                    │
│    └─ ... 执行所有 P0 用例                                              │
│        ↓                                                                │
│  所有 P0 用例通过？                                                     │
│    ├─ 是 ✅ → 继续执行 P1/P2 用例                                      │
│    └─ 否 ❌ → 停止执行，报告失败，阻断发布                              │
│        ↓                                                                │
│  【阶段 2：执行 P1 用例】（30-40 条，多线程并行执行）                  │
│    │                                                                    │
│    ├─ 并行执行 P1 用例（5-10 个线程）                                  │
│    │   ├─ 收集所有 P1 用例结果                                         │
│    │   └─ 统计通过/失败数量                                            │
│    │                                                                    │
│    └─ P1 用例失败率 > 10%？                                            │
│        ├─ 是 ⚠️ → 告警，需要审核                                       │
│        └─ 否 ✅ → 继续执行 P2 用例                                     │
│            ↓                                                            │
│  【阶段 3：执行 P2 用例】（40-50 条，多线程并行执行，可选）            │
│    │                                                                    │
│    ├─ 并行执行 P2 用例（10-20 个线程）                                │
│    │   └─ 收集结果（失败不影响发布）                                   │
│    │                                                                    │
│    └─ 生成完整报告                                                      │
│        ↓                                                                │
│  结束执行                                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. 快速失败（Fast Fail）策略

**定义**：**P0 用例失败时，立即停止执行，不执行后续用例**

**流程图**：

```
执行 P0 用例
    ↓
用例 1（P0）
    ↓
┌──────────────┐
│ 通过？        │
└──────┬───────┘
       │
    ┌──┴──┐
    │     │
   Yes   No
    │     │
    │     └─→ ❌ 立即停止
    │         ├─ 报告失败
    │         ├─ 不执行后续用例
    │         └─ 阻断发布
    │
    ↓
用例 2（P0）
    ↓
┌──────────────┐
│ 通过？        │
└──────┬───────┘
       │
    ┌──┴──┐
    │     │
   Yes   No
    │     │
    │     └─→ ❌ 立即停止
    │
    ↓
... 所有 P0 用例通过
    ↓
✅ 继续执行 P1/P2 用例
```

**代码示例**：

```python
def execute_golden_set_with_priority(golden_set: List[Dict], fast_fail: bool = True):
    """按优先级执行 Golden Set（支持快速失败）"""
    
    # 按优先级分组
    p0_cases = [case for case in golden_set if case.get("priority") == "P0"]
    p1_cases = [case for case in golden_set if case.get("priority") == "P1"]
    p2_cases = [case for case in golden_set if case.get("priority") == "P2"]
    
    results = {
        "p0": {"passed": 0, "failed": 0, "cases": []},
        "p1": {"passed": 0, "failed": 0, "cases": []},
        "p2": {"passed": 0, "failed": 0, "cases": []}
    }
    
    # 阶段 1：执行 P0 用例（单线程，快速失败）
    print("执行 P0 用例（关键用例）...")
    for case in p0_cases:
        result = run_test_case(case)
        results["p0"]["cases"].append(result)
        
        if result["status"] == "pass":
            results["p0"]["passed"] += 1
        else:
            results["p0"]["failed"] += 1
            
            # 快速失败：P0 用例失败，立即停止
            if fast_fail:
                print(f"❌ P0 用例失败: {case['case_id']}")
                print("🚫 快速失败：停止执行后续用例")
                return {
                    "status": "blocked",
                    "reason": f"P0 用例失败: {case['case_id']}",
                    "results": results
                }
    
    print(f"✅ P0 用例全部通过: {results['p0']['passed']}/{len(p0_cases)}")
    
    # 阶段 2：执行 P1 用例（多线程并行）
    if p1_cases:
        print("执行 P1 用例（重要用例）...")
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            p1_results = list(executor.map(run_test_case, p1_cases))
        
        for result in p1_results:
            results["p1"]["cases"].append(result)
            if result["status"] == "pass":
                results["p1"]["passed"] += 1
            else:
                results["p1"]["failed"] += 1
        
        print(f"P1 用例结果: {results['p1']['passed']}/{len(p1_cases)} 通过")
    
    # 阶段 3：执行 P2 用例（多线程并行，可选）
    if p2_cases:
        print("执行 P2 用例（一般用例）...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            p2_results = list(executor.map(run_test_case, p2_cases))
        
        for result in p2_results:
            results["p2"]["cases"].append(result)
            if result["status"] == "pass":
                results["p2"]["passed"] += 1
            else:
                results["p2"]["failed"] += 1
        
        print(f"P2 用例结果: {results['p2']['passed']}/{len(p2_cases)} 通过")
    
    return {
        "status": "completed",
        "results": results
    }
```

### 3. 并行执行策略

**不同优先级的并行度**：

```
优先级        执行方式          并行度          资源分配
─────────────────────────────────────────────────────
P0（关键）    单线程执行        1 个线程        最多资源（确保快速）
P1（重要）    多线程并行        5-10 个线程     中等资源
P2（一般）    多线程并行        10-20 个线程    最少资源（可延迟）
```

**并行执行架构**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        并行执行架构                                       │
└─────────────────────────────────────────────────────────────────────────┘

测试调度器
    │
    ├─ P0 执行器（单线程）
    │   ├─ 用例 1（P0）→ 执行
    │   ├─ 用例 2（P0）→ 等待用例 1 完成
    │   └─ 用例 3（P0）→ 等待用例 2 完成
    │
    ├─ P1 执行器池（5-10 个线程）
    │   ├─ 线程 1 → 用例 1（P1）
    │   ├─ 线程 2 → 用例 2（P1）
    │   ├─ 线程 3 → 用例 3（P1）
    │   ├─ ...
    │   └─ 线程 10 → 用例 10（P1）
    │
    └─ P2 执行器池（10-20 个线程）
        ├─ 线程 1 → 用例 1（P2）
        ├─ 线程 2 → 用例 2（P2）
        ├─ ...
        └─ 线程 20 → 用例 20（P2）

执行时间对比：
─────────────────────────────────────────────────────
方式              执行时间（100 条用例）
─────────────────────────────────────────────────────
串行执行          100 × 5s = 500s（8.3 分钟）
按优先级并行      10s（P0）+ 50s（P1）+ 25s（P2）= 85s（1.4 分钟）
                  提升 5.9x
```

### 4. 资源分配策略

**不同优先级的资源分配**：

```python
def allocate_resources_by_priority(priority: str, available_resources: Dict) -> Dict:
    """根据优先级分配资源"""
    
    if priority == "P0":
        # P0 用例：分配最多资源
        return {
            "cpu_cores": available_resources["cpu_cores"] * 0.5,  # 50% CPU
            "memory_mb": available_resources["memory_mb"] * 0.5,  # 50% 内存
            "gpu_utilization": available_resources["gpu_utilization"] * 0.6,  # 60% GPU
            "timeout_seconds": 30  # 30 秒超时
        }
    
    elif priority == "P1":
        # P1 用例：分配中等资源
        return {
            "cpu_cores": available_resources["cpu_cores"] * 0.3,  # 30% CPU
            "memory_mb": available_resources["memory_mb"] * 0.3,  # 30% 内存
            "gpu_utilization": available_resources["gpu_utilization"] * 0.3,  # 30% GPU
            "timeout_seconds": 20  # 20 秒超时
        }
    
    else:  # P2
        # P2 用例：分配最少资源
        return {
            "cpu_cores": available_resources["cpu_cores"] * 0.2,  # 20% CPU
            "memory_mb": available_resources["memory_mb"] * 0.2,  # 20% 内存
            "gpu_utilization": available_resources["gpu_utilization"] * 0.1,  # 10% GPU
            "timeout_seconds": 10  # 10 秒超时
        }
```

---

## 📊 Level 2：优先级维护策略

### 1. 优先级维护流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    优先级维护流程（Level 2）                              │
└─────────────────────────────────────────────────────────────────────────┘

【维护触发条件】
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  定期维护（每周/每月）                                                   │
│    ↓                                                                    │
│  分析测试结果                                                            │
│    ├─ P0 用例连续 10 次通过 → 考虑降级为 P1                            │
│    ├─ P1 用例连续 10 次失败 → 考虑升级为 P0                            │
│    └─ P2 用例从未失败 → 考虑移除或降级                                  │
│                                                                         │
│  业务变更触发                                                            │
│    ├─ 新功能上线 → 相关用例升级为 P0/P1                                │
│    ├─ 旧功能下线 → 相关用例降级为 P2 或移除                            │
│    └─ 业务优先级变化 → 重新分配优先级                                  │
│                                                                         │
│  性能优化触发                                                            │
│    ├─ 测试执行时间过长 → 优化用例优先级                                │
│    └─ 资源使用不均 → 重新分配资源                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

【维护流程】
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ① 收集数据                                                             │
│     ├─ 历史测试结果（最近 30 天）                                       │
│     ├─ 用例失败率                                                       │
│     ├─ 用例执行频率                                                     │
│     └─ 业务重要性变化                                                   │
│                                                                         │
│  ② 分析优先级                                                           │
│     ├─ 计算每个用例的"优先级分数"                                       │
│     ├─ 识别需要调整优先级的用例                                         │
│     └─ 生成优先级调整建议                                               │
│                                                                         │
│  ③ 人工审核                                                             │
│     ├─ 审核优先级调整建议                                               │
│     ├─ 确认或拒绝调整                                                   │
│     └─ 记录调整原因                                                     │
│                                                                         │
│  ④ 执行调整                                                             │
│     ├─ 更新用例优先级                                                   │
│     ├─ 更新执行策略                                                     │
│     └─ 通知相关人员                                                     │
│                                                                         │
│  ⑤ 验证调整                                                             │
│     ├─ 运行回归测试                                                     │
│     ├─ 验证执行时间和资源使用                                           │
│     └─ 确认调整效果                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. 优先级分数计算

```python
def calculate_priority_score(case: Dict, historical_results: List[Dict], business_importance: float) -> float:
    """计算用例的优先级分数"""
    
    case_id = case["case_id"]
    
    # 1. 历史失败率（0-1）
    failure_count = sum(1 for r in historical_results if r["case_id"] == case_id and r["status"] == "fail")
    total_runs = len([r for r in historical_results if r["case_id"] == case_id])
    failure_rate = failure_count / total_runs if total_runs > 0 else 0
    
    # 2. 执行频率（0-1）
    execution_frequency = total_runs / 30 if total_runs > 0 else 0  # 假设 30 天
    execution_frequency = min(execution_frequency, 1.0)  # 归一化
    
    # 3. 业务重要性（0-1，外部输入）
    business_importance_score = business_importance
    
    # 4. 计算综合分数（加权平均）
    priority_score = (
        failure_rate * 0.4 +           # 失败率权重 40%
        execution_frequency * 0.3 +    # 执行频率权重 30%
        business_importance_score * 0.3  # 业务重要性权重 30%
    )
    
    return priority_score

def suggest_priority_adjustment(case: Dict, priority_score: float, current_priority: str) -> Dict:
    """建议优先级调整"""
    
    # 优先级分数阈值
    p0_threshold = 0.7
    p1_threshold = 0.4
    
    suggested_priority = None
    reason = ""
    
    if priority_score >= p0_threshold:
        suggested_priority = "P0"
        reason = "优先级分数高，建议升级为 P0"
    elif priority_score >= p1_threshold:
        suggested_priority = "P1"
        reason = "优先级分数中等，建议保持或调整为 P1"
    else:
        suggested_priority = "P2"
        reason = "优先级分数低，建议降级为 P2"
    
    # 如果建议与当前优先级不同，需要调整
    if suggested_priority != current_priority:
        return {
            "case_id": case["case_id"],
            "current_priority": current_priority,
            "suggested_priority": suggested_priority,
            "priority_score": priority_score,
            "reason": reason,
            "action": "adjust"
        }
    else:
        return {
            "case_id": case["case_id"],
            "current_priority": current_priority,
            "suggested_priority": suggested_priority,
            "priority_score": priority_score,
            "reason": "优先级合理，无需调整",
            "action": "keep"
        }
```

---

## 📊 优先级管理可视化

### 1. 优先级分布图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    优先级分布可视化                                       │
└─────────────────────────────────────────────────────────────────────────┘

Golden Set 用例分布（100 条）
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  P0（关键用例）：15 条（15%）                                           │
│  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                                         │
│  P1（重要用例）：35 条（35%）                                           │
│  ███████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                                         │
│  P2（一般用例）：50 条（50%）                                           │
│  ███████████████████████████████████████████████████████░░░░░░░░░░░░  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

执行时间分布（按优先级）
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  P0（单线程）：15 × 5s = 75s（12.5 分钟）                              │
│  ████████████████████████████████████████████████████████████████████  │
│                                                                         │
│  P1（5 线程并行）：35 / 5 × 5s = 35s（5.8 分钟）                       │
│  ██████████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                                         │
│  P2（10 线程并行）：50 / 10 × 5s = 25s（4.2 分钟）                     │
│  ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                                         │
│  总执行时间：75s + 35s + 25s = 135s（2.25 分钟）                       │
│                                                                         │
│  对比：串行执行需要 100 × 5s = 500s（8.3 分钟）                        │
│  提升：500s / 135s ≈ 3.7x                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. 优先级执行流程图（详细）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  优先级执行流程详细图                                     │
└─────────────────────────────────────────────────────────────────────────┘

开始执行回归测试
    │
    ▼
┌─────────────────────────┐
│ 阶段 1：执行 P0 用例     │
│ （15 条，单线程）        │
└───────────┬─────────────┘
            │
            ▼
    ┌───────────────┐
    │ 用例 1（P0）  │
    └───────┬───────┘
            │
        ┌───┴───┐
        │ 通过？ │
        └───┬───┘
            │
        ┌───┴───┐
       Yes      No
        │       │
        │       └─────→ ❌ 快速失败
        │                 ├─ 报告失败
        │                 ├─ 停止执行
        │                 └─ 阻断发布
        │
        ▼
    ┌───────────────┐
    │ 用例 2（P0）  │
    └───────┬───────┘
            │
        ┌───┴───┐
        │ 通过？ │
        └───┬───┘
            │
        ┌───┴───┐
       Yes      No
        │       │
        │       └─────→ ❌ 快速失败
        │
        ▼
    ...（执行所有 P0 用例）
        │
        ▼
┌─────────────────────────┐
│ ✅ 所有 P0 用例通过      │
│ （15/15）               │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 阶段 2：执行 P1 用例     │
│ （35 条，5 线程并行）    │
└───────────┬─────────────┘
            │
            ▼
    ┌──────────────────────┐
    │ 线程池（5 个线程）   │
    ├──────────────────────┤
    │ 线程 1 → 用例 1（P1）│
    │ 线程 2 → 用例 2（P1）│
    │ 线程 3 → 用例 3（P1）│
    │ 线程 4 → 用例 4（P1）│
    │ 线程 5 → 用例 5（P1）│
    └──────────────────────┘
            │
            ▼
    （继续并行执行剩余 P1 用例）
            │
            ▼
┌─────────────────────────┐
│ P1 用例结果汇总          │
│ （通过/失败统计）        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ P1 失败率 > 10%？       │
└───────────┬─────────────┘
            │
        ┌───┴───┐
       No       Yes
        │       │
        │       └─────→ ⚠️ 告警
        │                 ├─ 需要审核
        │                 └─ 可能影响发布
        │
        ▼
┌─────────────────────────┐
│ 阶段 3：执行 P2 用例     │
│ （50 条，10 线程并行）   │
└───────────┬─────────────┘
            │
            ▼
    ┌──────────────────────┐
    │ 线程池（10 个线程）  │
    ├──────────────────────┤
    │ 线程 1-10 → 并行执行 │
    └──────────────────────┘
            │
            ▼
    （P2 用例失败不影响发布）
            │
            ▼
┌─────────────────────────┐
│ 生成完整报告             │
│ ├─ P0 结果              │
│ ├─ P1 结果              │
│ └─ P2 结果              │
└───────────┬─────────────┘
            │
            ▼
    结束执行
```

---

## 💡 总结

### Level 0-2 总结

| 级别 | 内容 | 关键概念 |
|------|------|---------|
| **Level 0** | 优先级基础 | P0/P1/P2 定义、优先级分类 |
| **Level 1** | 优先级分配 | 手动/自动/混合分配、分配规则 |
| **Level 2** | 优先级执行和维护 | 快速失败、并行执行、资源分配、优先级维护 |

### 关键要点

1. **优先级是动态的**：不是固定不变的，需要定期维护和调整
2. **快速失败策略**：P0 用例失败立即停止，节省时间和资源
3. **并行执行优化**：不同优先级使用不同的并行度，优化执行时间
4. **资源分配策略**：P0 用例分配最多资源，确保快速执行

---

**下一步**：
- 查看 [回归测试与门禁详解](./KYC_Day03_A1_回归测试与门禁详解.md)
- 查看 [测试用例覆盖度分析详解](./KYC_Day03_A1_B6_测试用例覆盖度分析详解.md)
