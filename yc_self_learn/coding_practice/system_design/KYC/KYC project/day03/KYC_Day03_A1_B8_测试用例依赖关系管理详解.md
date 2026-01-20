# Day 3_A1_B8：测试用例依赖关系管理详解

**优先级**：🟡 P1 - 重要但不紧急  
**目的**：管理测试用例之间的依赖关系，优化测试执行顺序，支持并行执行

---

## 💡 测试级别说明

### 测试级别 vs 指标级别（L0/L1/L2）

**重要区分**：

| 概念 | 含义 | 示例 |
|------|------|------|
| **指标级别（L0/L1/L2）** | 指标的分类（稳定性/业务收益/长期健康） | L0 稳定性指标：成功率、延迟 |
| **测试级别** | 测试的类型（单元测试/集成测试/回归测试） | 回归测试、集成测试、单元测试 |

**Golden Set 回归测试的级别**：
- ✅ **测试类型**：回归测试（Regression Test）
- ✅ **指标级别**：主要用于验证 **L0 稳定性指标**（成功率、延迟、错误率）
- ✅ **测试级别**：**Level 0 级别测试**（验证系统基本功能是否正常）

**测试级别分类**：
- **Level 0 测试**：基础功能测试（Golden Set 回归测试）→ 验证 L0 稳定性指标
- **Level 1 测试**：业务场景测试（端到端业务流程测试）→ 验证 L1 业务收益指标
- **Level 2 测试**：长期健康测试（可维护性、可扩展性测试）→ 验证 L2 长期健康指标

**测试用例依赖关系**：
- ✅ **适用于所有测试级别**：单元测试、集成测试、回归测试都可以有依赖关系
- ✅ **Golden Set 回归测试（Level 0）**：大多数用例独立（没有依赖），可以并行执行
- ✅ **集成测试（Level 1）**：可能有依赖关系（需要按顺序执行）

---

## 🎯 什么是测试用例依赖关系？

### 💡 测试级别说明

**重要区分**：**测试级别 vs 指标级别（L0/L1/L2）**

| 概念 | 含义 | 示例 |
|------|------|------|
| **指标级别（L0/L1/L2）** | 指标的分类（稳定性/业务收益/长期健康） | L0 稳定性指标：成功率、延迟 |
| **测试级别** | 测试的类型（单元测试/集成测试/回归测试） | 回归测试、集成测试、单元测试 |

**Golden Set 回归测试与 L0 指标的关系**：
- ✅ **回归测试**：验证系统功能是否正常（测试类型）
- ✅ **L0 指标**：衡量系统稳定性（指标类型）
- ✅ **关系**：回归测试通过后，会计算 L0 指标（成功率、延迟等）

**测试用例依赖关系**：
- ✅ **适用于所有测试级别**：单元测试、集成测试、回归测试都可以有依赖关系
- ✅ **Golden Set 回归测试**：大多数用例独立（没有依赖），可以并行执行
- ✅ **集成测试**：可能有依赖关系（需要按顺序执行）

---

### 核心概念

**测试用例依赖关系**：**某些测试用例必须在其他测试用例执行完成后才能执行**

### 类比理解

**类比 1：做菜的顺序**
```
❌ 没有依赖关系（错误）：
1. 炒菜（但还没切菜）
2. 切菜（但还没买菜）
3. 买菜

✅ 有依赖关系（正确）：
1. 买菜（没有依赖）
2. 切菜（依赖：买菜完成）
3. 炒菜（依赖：切菜完成）
```

**类比 2：软件安装**
```
❌ 没有依赖关系（错误）：
1. 安装应用（但依赖库还没装）
2. 安装依赖库（但系统库还没装）
3. 安装系统库

✅ 有依赖关系（正确）：
1. 安装系统库（没有依赖）
2. 安装依赖库（依赖：系统库已安装）
3. 安装应用（依赖：依赖库已安装）
```

---

## 💡 为什么需要依赖关系？

### 问题场景

**场景 1：用例之间有数据依赖**

```
用例 A：测试"创建用户"功能
用例 B：测试"查询用户"功能（需要用户已存在）
用例 C：测试"删除用户"功能（需要用户已存在）

❌ 没有依赖关系：
- 用例 B 和 C 可能在用例 A 之前执行
- 结果：用例 B 和 C 失败（用户不存在）

✅ 有依赖关系：
- 用例 A → 用例 B → 用例 C（顺序执行）
- 结果：所有用例都能成功执行
```

**场景 2：用例之间有状态依赖**

```
用例 A：测试"登录"功能
用例 B：测试"查看个人信息"功能（需要已登录）
用例 C：测试"修改个人信息"功能（需要已登录）

❌ 没有依赖关系：
- 用例 B 和 C 可能在用例 A 之前执行
- 结果：用例 B 和 C 失败（未登录）

✅ 有依赖关系：
- 用例 A → 用例 B → 用例 C（顺序执行）
- 结果：所有用例都能成功执行
```

**场景 3：用例之间有资源依赖**

```
用例 A：测试"创建数据库连接"功能
用例 B：测试"查询数据"功能（需要数据库连接）
用例 C：测试"写入数据"功能（需要数据库连接）

❌ 没有依赖关系：
- 用例 B 和 C 可能在用例 A 之前执行
- 结果：用例 B 和 C 失败（数据库连接不存在）

✅ 有依赖关系：
- 用例 A → 用例 B → 用例 C（顺序执行）
- 结果：所有用例都能成功执行
```

---

## 📊 KYC 项目中的依赖关系示例

### 示例 1：数据准备依赖

```
用例 A：测试"上传身份证图片"功能
用例 B：测试"OCR 识别"功能（依赖：图片已上传）
用例 C：测试"字段提取"功能（依赖：OCR 已完成）
用例 D：测试"验证字段"功能（依赖：字段已提取）

依赖关系：
A → B → C → D

执行顺序：
1. 执行用例 A（上传图片）
2. 执行用例 B（OCR 识别，使用用例 A 的结果）
3. 执行用例 C（字段提取，使用用例 B 的结果）
4. 执行用例 D（验证字段，使用用例 C 的结果）
```

### 示例 2：环境设置依赖

```
用例 A：测试"初始化模型"功能
用例 B：测试"加载配置"功能（依赖：模型已初始化）
用例 C：测试"处理请求"功能（依赖：模型和配置都已准备好）

依赖关系：
A → B → C

执行顺序：
1. 执行用例 A（初始化模型）
2. 执行用例 B（加载配置，使用用例 A 的模型）
3. 执行用例 C（处理请求，使用用例 A 和 B 的结果）
```

---

## 🔧 依赖关系定义

### 💡 依赖关系的传递机制

**核心问题**：**依赖关系是如何传递的？系统如何知道用例 A 依赖用例 B，用例 B 依赖用例 C，所以用例 A 也依赖用例 C？**

#### 传递机制说明

**依赖关系传递**：**通过依赖图（Dependency Graph）和拓扑排序算法实现**

```
依赖关系传递示例：

直接依赖：
case_A → case_B  （case_B 直接依赖 case_A）
case_B → case_C  （case_C 直接依赖 case_B）

传递依赖：
case_A → case_B → case_C  （case_C 间接依赖 case_A）

系统如何知道？
通过依赖图（Graph）和拓扑排序（Topological Sort）：
1. 构建依赖图：记录所有直接依赖关系
2. 拓扑排序：根据依赖图，自动计算传递依赖
3. 执行顺序：自动确定执行顺序（case_A → case_B → case_C）
```

---

### 💻 简单代码流程示例

**完整流程演示**：

```python
from collections import deque, defaultdict

# ============================================
# 步骤 1：定义测试用例（只定义直接依赖）
# ============================================
test_cases = [
    {"case_id": "case_A", "dependencies": []},           # 没有依赖
    {"case_id": "case_B", "dependencies": ["case_A"]},  # 直接依赖 case_A
    {"case_id": "case_C", "dependencies": ["case_B"]},  # 直接依赖 case_B（间接依赖 case_A）
]

print("=" * 60)
print("步骤 1：定义测试用例（只定义直接依赖）")
print("=" * 60)
for case in test_cases:
    print(f"  {case['case_id']}: dependencies = {case['dependencies']}")

# ============================================
# 步骤 2：构建依赖图（存储直接依赖关系）
# ============================================
graph = defaultdict(list)  # 依赖图：dep -> [依赖 dep 的用例列表]
in_degree = defaultdict(int)  # 入度：每个用例有多少个直接依赖

# 初始化入度
for case in test_cases:
    case_id = case["case_id"]
    in_degree[case_id] = 0

# 构建依赖图和入度
for case in test_cases:
    case_id = case["case_id"]
    dependencies = case.get("dependencies", [])
    
    for dep in dependencies:
        graph[dep].append(case_id)  # dep → case_id
        in_degree[case_id] += 1

print("\n" + "=" * 60)
print("步骤 2：构建依赖图（存储直接依赖关系）")
print("=" * 60)
print("依赖图（Graph）：")
for dep, dependents in graph.items():
    print(f"  {dep} → {dependents}")
print("\n入度（In-Degree）：")
for case_id, degree in in_degree.items():
    print(f"  {case_id}: 入度 = {degree}")

# ============================================
# 步骤 3：拓扑排序（自动处理传递依赖）
# ============================================
queue = deque()
result = []
in_degree_copy = in_degree.copy()  # 使用副本，不修改原始数据

# 找到所有入度为 0 的用例（没有依赖的用例）
for case_id in in_degree_copy:
    if in_degree_copy[case_id] == 0:
        queue.append(case_id)

print("\n" + "=" * 60)
print("步骤 3：拓扑排序（自动处理传递依赖）")
print("=" * 60)

iteration = 1
while queue:
    print(f"\n--- 迭代 {iteration} ---")
    
    # 执行当前用例
    current = queue.popleft()
    result.append(current)
    print(f"✅ 执行用例: {current} (入度 = {in_degree_copy[current]})")
    
    # 更新依赖关系（传递依赖在这里自动处理）
    for neighbor in graph[current]:
        old_degree = in_degree_copy[neighbor]
        in_degree_copy[neighbor] -= 1
        print(f"   更新 {neighbor}: 入度 {old_degree} → {in_degree_copy[neighbor]}")
        
        # 如果入度变为 0，说明所有依赖都已执行完成
        if in_degree_copy[neighbor] == 0:
            queue.append(neighbor)
            print(f"   → {neighbor} 可以执行了（所有依赖已完成）")
    
    iteration += 1

# ============================================
# 步骤 4：输出最终执行顺序
# ============================================
print("\n" + "=" * 60)
print("步骤 4：最终执行顺序（传递依赖自动处理）")
print("=" * 60)
print(f"执行顺序: {' → '.join(result)}")
print("\n传递依赖说明：")
print("  case_C 直接依赖 case_B")
print("  case_B 直接依赖 case_A")
print("  → case_C 间接依赖 case_A（传递依赖，自动处理）")
```

**输出结果**：

```
============================================================
步骤 1：定义测试用例（只定义直接依赖）
============================================================
  case_A: dependencies = []
  case_B: dependencies = ['case_A']
  case_C: dependencies = ['case_B']

============================================================
步骤 2：构建依赖图（存储直接依赖关系）
============================================================
依赖图（Graph）：
  case_A → ['case_B']
  case_B → ['case_C']

入度（In-Degree）：
  case_A: 入度 = 0
  case_B: 入度 = 1
  case_C: 入度 = 1

============================================================
步骤 3：拓扑排序（自动处理传递依赖）
============================================================

--- 迭代 1 ---
✅ 执行用例: case_A (入度 = 0)
   更新 case_B: 入度 1 → 0
   → case_B 可以执行了（所有依赖已完成）

--- 迭代 2 ---
✅ 执行用例: case_B (入度 = 0)
   更新 case_C: 入度 1 → 0
   → case_C 可以执行了（所有依赖已完成）

--- 迭代 3 ---
✅ 执行用例: case_C (入度 = 0)

============================================================
步骤 4：最终执行顺序（传递依赖自动处理）
============================================================
执行顺序: case_A → case_B → case_C

传递依赖说明：
  case_C 直接依赖 case_B
  case_B 直接依赖 case_A
  → case_C 间接依赖 case_A（传递依赖，自动处理）
```

---

### 🔍 传递依赖处理过程详解

**关键步骤**：

```
初始状态：
  case_A: 入度 = 0（没有依赖）
  case_B: 入度 = 1（依赖 case_A）
  case_C: 入度 = 1（依赖 case_B）

迭代 1：执行 case_A
  ✅ case_A 执行完成
  → case_B 的入度从 1 变为 0（因为 case_A 已完成）
  → case_B 可以执行了

迭代 2：执行 case_B
  ✅ case_B 执行完成
  → case_C 的入度从 1 变为 0（因为 case_B 已完成）
  → case_C 可以执行了

迭代 3：执行 case_C
  ✅ case_C 执行完成

结果：case_A → case_B → case_C
```

**传递依赖自动处理**：
- ✅ case_C 只定义了直接依赖 case_B
- ✅ 算法自动推导 case_C 也依赖 case_A（通过 case_B）
- ✅ 通过入度更新机制，确保 case_A 在 case_B 之前执行，case_B 在 case_C 之前执行

#### 依赖图（Dependency Graph）

**依赖图**：**用图结构表示用例之间的依赖关系**

```
依赖图结构：

节点（Node）：测试用例
边（Edge）：依赖关系（A → B 表示 B 依赖 A）

示例：
case_A (没有依赖)
    ↓
case_B (依赖 case_A)
    ↓
case_C (依赖 case_B)

依赖图表示：
graph = {
    "case_A": [],           # case_A 没有依赖
    "case_B": ["case_A"],   # case_B 依赖 case_A
    "case_C": ["case_B"]    # case_C 依赖 case_B
}

传递依赖计算：
- case_C 直接依赖 case_B
- case_B 直接依赖 case_A
- 因此：case_C 间接依赖 case_A（传递依赖）
```

#### 入度（In-Degree）

**入度**：**每个用例有多少个直接依赖**

```
入度计算：

case_A: 入度 = 0（没有依赖）
case_B: 入度 = 1（依赖 case_A）
case_C: 入度 = 1（依赖 case_B）

拓扑排序算法：
1. 找到所有入度为 0 的用例（case_A）
2. 执行这些用例
3. 移除这些用例，更新依赖关系
   - case_B 的入度从 1 变为 0（因为 case_A 已完成）
4. 重复步骤 1-3
```

---

### 1. 依赖关系表示方法

#### 方法 1：在用例中定义依赖（推荐）

```python
# 测试用例定义（带依赖关系）
test_cases = [
    {
        "case_id": "case_001",
        "description": "上传身份证图片",
        "dependencies": [],  # 没有依赖
        "file_path": "test_data/id_card_001.jpg"
    },
    {
        "case_id": "case_002",
        "description": "OCR 识别",
        "dependencies": ["case_001"],  # 依赖 case_001
        "file_path": "test_data/id_card_001.jpg"
    },
    {
        "case_id": "case_003",
        "description": "字段提取",
        "dependencies": ["case_002"],  # 依赖 case_002
        "file_path": "test_data/id_card_001.jpg"
    },
    {
        "case_id": "case_004",
        "description": "验证字段",
        "dependencies": ["case_003"],  # 依赖 case_003
        "file_path": "test_data/id_card_001.jpg"
    }
]
```

#### 方法 2：独立的依赖配置文件

```python
# dependencies.yaml
dependencies:
  case_001: []  # 没有依赖
  case_002: ["case_001"]  # 依赖 case_001
  case_003: ["case_002"]  # 依赖 case_002
  case_004: ["case_003"]  # 依赖 case_003
```

---

### 2. 依赖图构建

**依赖图（Dependency Graph）**：用图表示用例之间的依赖关系

```
依赖图示例：

case_001 (没有依赖)
    ↓
case_002 (依赖 case_001)
    ↓
case_003 (依赖 case_002)
    ↓
case_004 (依赖 case_003)

或者：

case_A (没有依赖)
    ↓
case_B (依赖 case_A)
    ↓
case_C (依赖 case_B)
case_D (依赖 case_B)  ← 两个用例都依赖 case_B，可以并行执行
```

---

## 🔄 依赖执行策略

### 1. 拓扑排序（Topological Sort）

**拓扑排序**：**根据依赖关系，确定用例的执行顺序**

**算法原理**：
1. 找到所有没有依赖的用例（入度为 0）
2. 执行这些用例
3. 移除这些用例，更新依赖关系
4. 重复步骤 1-3，直到所有用例执行完成

**实现示例**：

```python
from collections import deque, defaultdict

def topological_sort(test_cases: list) -> list:
    """拓扑排序：确定用例执行顺序"""
    # 1. 构建依赖图
    graph = defaultdict(list)  # 依赖图：case_id -> [依赖的用例列表]
    in_degree = defaultdict(int)  # 入度：每个用例有多少个依赖
    
    # 初始化
    for case in test_cases:
        case_id = case["case_id"]
        in_degree[case_id] = 0
    
    # 构建图和入度
    for case in test_cases:
        case_id = case["case_id"]
        dependencies = case.get("dependencies", [])
        
        for dep in dependencies:
            graph[dep].append(case_id)  # dep → case_id（dep 完成后才能执行 case_id）
            in_degree[case_id] += 1
    
    # 2. 拓扑排序（自动处理传递依赖）
    queue = deque()
    result = []
    
    # 找到所有入度为 0 的用例（没有依赖的用例）
    for case_id in in_degree:
        if in_degree[case_id] == 0:
            queue.append(case_id)
    
    # 执行拓扑排序
    while queue:
        current = queue.popleft()
        result.append(current)
        
        # 🔑 关键：更新依赖关系（传递依赖在这里自动处理）
        # 当 current 执行完成后，所有依赖 current 的用例的入度都会减 1
        # 如果某个用例的入度变为 0，说明它的所有依赖都已执行完成
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # 🔑 传递依赖示例：
    # case_A → case_B → case_C
    # 
    # 初始状态：
    #   in_degree[case_A] = 0
    #   in_degree[case_B] = 1（依赖 case_A）
    #   in_degree[case_C] = 1（依赖 case_B）
    #
    # 执行过程：
    #   1. 执行 case_A（入度为 0）
    #   2. case_B 的入度从 1 变为 0（因为 case_A 已完成）
    #   3. 执行 case_B（入度变为 0）
    #   4. case_C 的入度从 1 变为 0（因为 case_B 已完成）
    #   5. 执行 case_C（入度变为 0）
    #
    # 结果：case_A → case_B → case_C（传递依赖自动处理）
    
    # 检查是否有循环依赖
    if len(result) != len(test_cases):
        raise ValueError("Circular dependency detected!")
    
    return result

# 使用示例
test_cases = [
    {"case_id": "case_001", "dependencies": []},
    {"case_id": "case_002", "dependencies": ["case_001"]},
    {"case_id": "case_003", "dependencies": ["case_002"]},
    {"case_id": "case_004", "dependencies": ["case_003"]}
]

execution_order = topological_sort(test_cases)
print(execution_order)
# 输出：['case_001', 'case_002', 'case_003', 'case_004']
```

---

### 2. 并行执行优化

**核心思想**：**没有依赖关系的用例可以并行执行**

**并行执行策略**：

```
依赖图：
case_A (没有依赖)
    ↓
case_B (依赖 case_A)
    ↓
case_C (依赖 case_B)
case_D (依赖 case_B)  ← case_C 和 case_D 可以并行执行

执行计划：
批次 1：执行 case_A（没有依赖）
批次 2：执行 case_B（依赖 case_A）
批次 3：并行执行 case_C 和 case_D（都依赖 case_B）
```

**实现示例**：

```python
def get_execution_batches(test_cases: list) -> list:
    """获取执行批次（每批次内的用例可以并行执行）"""
    # 1. 构建依赖图
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    for case in test_cases:
        case_id = case["case_id"]
        in_degree[case_id] = 0
    
    for case in test_cases:
        case_id = case["case_id"]
        dependencies = case.get("dependencies", [])
        
        for dep in dependencies:
            graph[dep].append(case_id)
            in_degree[case_id] += 1
    
    # 2. 按批次执行
    batches = []
    remaining = set(case["case_id"] for case in test_cases)
    
    while remaining:
        # 找到当前批次：所有入度为 0 的用例
        current_batch = []
        for case_id in list(remaining):
            if in_degree[case_id] == 0:
                current_batch.append(case_id)
                remaining.remove(case_id)
        
        if not current_batch:
            raise ValueError("Circular dependency detected!")
        
        batches.append(current_batch)
        
        # 更新依赖关系
        for case_id in current_batch:
            for neighbor in graph[case_id]:
                in_degree[neighbor] -= 1
    
    return batches

# 使用示例
test_cases = [
    {"case_id": "case_A", "dependencies": []},
    {"case_id": "case_B", "dependencies": ["case_A"]},
    {"case_id": "case_C", "dependencies": ["case_B"]},
    {"case_id": "case_D", "dependencies": ["case_B"]}
]

batches = get_execution_batches(test_cases)
print(batches)
# 输出：
# [
#   ['case_A'],           # 批次 1：case_A 单独执行
#   ['case_B'],           # 批次 2：case_B 单独执行
#   ['case_C', 'case_D']  # 批次 3：case_C 和 case_D 并行执行
# ]
```

---

## 🚀 并行执行实现

### 1. 单线程顺序执行（无依赖优化）

```python
# 当前实现（所有用例顺序执行）
def run_regression_test_sequential(golden_set: list) -> list:
    """顺序执行所有用例"""
    results = []
    
    for case in golden_set:
        result = run_test_case(case)
        results.append(result)
    
    return results

# 问题：如果 100 个用例，每个用例 1 秒，总耗时 100 秒
```

---

### 2. 多线程并行执行（有依赖优化）

```python
import concurrent.futures
from threading import Lock

def run_regression_test_parallel(golden_set: list, max_workers: int = 4) -> list:
    """并行执行用例（考虑依赖关系）"""
    # 1. 获取执行批次
    batches = get_execution_batches(golden_set)
    
    # 2. 按批次执行
    results = []
    case_dict = {case["case_id"]: case for case in golden_set}
    
    for batch in batches:
        # 当前批次内的用例可以并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for case_id in batch:
                case = case_dict[case_id]
                future = executor.submit(run_test_case, case)
                futures.append((case_id, future))
            
            # 等待当前批次完成
            for case_id, future in futures:
                result = future.result()
                results.append(result)
    
    return results

# 优势：如果 100 个用例，分 10 个批次，每批次 10 个用例并行执行
# 总耗时：10 秒（假设每个用例 1 秒）
```

---

### 3. 异步执行（更高效）

```python
import asyncio

async def run_test_case_async(case: dict) -> dict:
    """异步执行单个用例"""
    # 模拟异步执行
    result = await asyncio.to_thread(run_test_case, case)
    return result

async def run_regression_test_async(golden_set: list) -> list:
    """异步执行用例（考虑依赖关系）"""
    batches = get_execution_batches(golden_set)
    case_dict = {case["case_id"]: case for case in golden_set}
    results = []
    
    for batch in batches:
        # 当前批次内的用例并行执行
        tasks = [run_test_case_async(case_dict[case_id]) for case_id in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    return results

# 使用示例
results = asyncio.run(run_regression_test_async(golden_set))
```

---

## 📊 KYC 项目实际应用场景

### 场景 1：Golden Set 回归测试（大多数用例独立）

**特点**：
- ✅ **大多数用例独立**：每个用例测试不同的文档
- ✅ **可以完全并行执行**：没有依赖关系

**实现**：

```python
# KYC 项目的 Golden Set 用例通常是独立的
golden_set = [
    {"case_id": "normal_001", "file_path": "test_data/normal/id_card_001.jpg", "dependencies": []},
    {"case_id": "normal_002", "file_path": "test_data/normal/id_card_002.jpg", "dependencies": []},
    {"case_id": "edge_001", "file_path": "test_data/edge/id_card_blurry.jpg", "dependencies": []},
    # ... 所有用例都没有依赖关系
]

# 可以完全并行执行
def run_golden_set_parallel(golden_set: list, max_workers: int = 10) -> list:
    """并行执行 Golden Set（所有用例独立）"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_test_case, case) for case in golden_set]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results
```

---

### 场景 2：集成测试（有依赖关系）

**特点**：
- ⚠️ **用例之间有依赖**：需要按顺序执行
- ⚠️ **需要依赖关系管理**

**实现**：

```python
# 集成测试用例（有依赖关系）
integration_tests = [
    {
        "case_id": "setup_001",
        "description": "初始化模型和配置",
        "dependencies": [],
        "action": "setup"
    },
    {
        "case_id": "test_001",
        "description": "测试处理请求",
        "dependencies": ["setup_001"],  # 依赖 setup_001
        "action": "test"
    },
    {
        "case_id": "test_002",
        "description": "测试批量处理",
        "dependencies": ["setup_001"],  # 依赖 setup_001（可以并行执行）
        "action": "test"
    },
    {
        "case_id": "cleanup_001",
        "description": "清理资源",
        "dependencies": ["test_001", "test_002"],  # 依赖 test_001 和 test_002
        "action": "cleanup"
    }
]

# 按依赖关系执行
batches = get_execution_batches(integration_tests)
# 输出：
# [
#   ['setup_001'],                    # 批次 1
#   ['test_001', 'test_002'],         # 批次 2（并行执行）
#   ['cleanup_001']                   # 批次 3
# ]
```

---

## 🔍 循环依赖检测

### 问题场景

**循环依赖**：**用例 A 依赖用例 B，用例 B 依赖用例 A**

```
❌ 循环依赖示例：
case_A 依赖 case_B
case_B 依赖 case_A

问题：无法确定执行顺序
```

### 检测方法

```python
def detect_circular_dependency(test_cases: list) -> bool:
    """检测循环依赖"""
    # 使用 DFS 检测环
    graph = defaultdict(list)
    for case in test_cases:
        case_id = case["case_id"]
        dependencies = case.get("dependencies", [])
        for dep in dependencies:
            graph[case_id].append(dep)
    
    visited = set()
    rec_stack = set()
    
    def has_cycle(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    for case in test_cases:
        case_id = case["case_id"]
        if case_id not in visited:
            if has_cycle(case_id):
                return True
    
    return False

# 使用示例
test_cases = [
    {"case_id": "case_A", "dependencies": ["case_B"]},
    {"case_id": "case_B", "dependencies": ["case_A"]}  # 循环依赖
]

if detect_circular_dependency(test_cases):
    print("⚠️ 检测到循环依赖！")
```

---

## 🛠️ KYC 项目完整实现

### 依赖关系管理器

```python
from collections import defaultdict, deque
from typing import List, Dict, Set
import concurrent.futures

class TestCaseDependencyManager:
    """测试用例依赖关系管理器"""
    
    def __init__(self):
        self.graph = defaultdict(list)  # 依赖图
        self.in_degree = defaultdict(int)  # 入度
    
    def add_test_case(self, case: Dict):
        """添加测试用例"""
        case_id = case["case_id"]
        dependencies = case.get("dependencies", [])
        
        # 初始化入度
        if case_id not in self.in_degree:
            self.in_degree[case_id] = 0
        
        # 构建依赖图
        for dep in dependencies:
            self.graph[dep].append(case_id)
            self.in_degree[case_id] += 1
    
    def get_execution_order(self) -> List[str]:
        """获取执行顺序（拓扑排序）"""
        queue = deque()
        result = []
        in_degree_copy = self.in_degree.copy()
        
        # 找到所有入度为 0 的用例
        for case_id in in_degree_copy:
            if in_degree_copy[case_id] == 0:
                queue.append(case_id)
        
        # 拓扑排序
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in self.graph[current]:
                in_degree_copy[neighbor] -= 1
                if in_degree_copy[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查循环依赖
        if len(result) != len(self.in_degree):
            raise ValueError("Circular dependency detected!")
        
        return result
    
    def get_execution_batches(self) -> List[List[str]]:
        """获取执行批次（每批次可以并行执行）"""
        batches = []
        remaining = set(self.in_degree.keys())
        in_degree_copy = self.in_degree.copy()
        
        while remaining:
            # 找到当前批次
            current_batch = []
            for case_id in list(remaining):
                if in_degree_copy[case_id] == 0:
                    current_batch.append(case_id)
                    remaining.remove(case_id)
            
            if not current_batch:
                raise ValueError("Circular dependency detected!")
            
            batches.append(current_batch)
            
            # 更新依赖关系
            for case_id in current_batch:
                for neighbor in self.graph[case_id]:
                    in_degree_copy[neighbor] -= 1
        
        return batches
    
    def detect_circular_dependency(self) -> bool:
        """检测循环依赖"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for case_id in self.in_degree:
            if case_id not in visited:
                if has_cycle(case_id):
                    return True
        
        return False

# 使用示例
manager = TestCaseDependencyManager()

# 添加测试用例
test_cases = [
    {"case_id": "case_A", "dependencies": []},
    {"case_id": "case_B", "dependencies": ["case_A"]},
    {"case_id": "case_C", "dependencies": ["case_B"]},
    {"case_id": "case_D", "dependencies": ["case_B"]}
]

for case in test_cases:
    manager.add_test_case(case)

# 获取执行顺序
execution_order = manager.get_execution_order()
print(f"执行顺序: {execution_order}")
# 输出：['case_A', 'case_B', 'case_C', 'case_D']

# 获取执行批次
batches = manager.get_execution_batches()
print(f"执行批次: {batches}")
# 输出：[['case_A'], ['case_B'], ['case_C', 'case_D']]

# 并行执行
def run_tests_with_dependencies(test_cases: List[Dict], max_workers: int = 4) -> List[Dict]:
    """考虑依赖关系的并行执行"""
    manager = TestCaseDependencyManager()
    for case in test_cases:
        manager.add_test_case(case)
    
    batches = manager.get_execution_batches()
    case_dict = {case["case_id"]: case for case in test_cases}
    results = []
    
    for batch in batches:
        # 当前批次并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_test_case, case_dict[case_id]) for case_id in batch]
            batch_results = [future.result() for future in concurrent.futures.as_completed(futures)]
            results.extend(batch_results)
    
    return results
```

---

## 💡 面试话术

### 核心话术

1. ✅ **什么是依赖关系**：
   - "测试用例依赖关系是指某些用例必须在其他用例执行完成后才能执行。比如用例 B 依赖用例 A，意味着用例 A 必须先执行，用例 B 才能执行。"

2. ✅ **为什么需要依赖关系**：
   - "用例之间可能有数据依赖、状态依赖或资源依赖。比如 KYC 项目中，'OCR 识别'用例依赖'上传图片'用例，因为需要图片文件存在才能进行 OCR。"

3. ✅ **如何管理依赖关系**：
   - "我们使用**拓扑排序**算法确定用例执行顺序。首先构建依赖图，然后找到所有没有依赖的用例（入度为 0），执行这些用例，再更新依赖关系，重复这个过程直到所有用例执行完成。"

4. ✅ **如何并行执行**：
   - "我们使用**执行批次**策略：将用例分成多个批次，每个批次内的用例没有依赖关系，可以并行执行。不同批次之间按顺序执行，确保依赖关系得到满足。这样可以大大提高测试执行效率。"

---

## 📝 实施检查清单

- [ ] **定义依赖关系**：在用例中定义 dependencies 字段
- [ ] **构建依赖图**：实现依赖图构建算法
- [ ] **拓扑排序**：实现拓扑排序算法
- [ ] **循环依赖检测**：实现循环依赖检测
- [ ] **并行执行**：实现批次并行执行
- [ ] **性能优化**：优化执行效率

---

## 🔗 相关文档

- **Parent**: [KYC_Day03_A1_回归测试与门禁详解.md](./KYC_Day03_A1_回归测试与门禁详解.md)
- **Related**: Golden Set、测试用例管理、回归测试

---

**最后更新**：2025-01-19
