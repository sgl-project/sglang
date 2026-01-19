# Day 2_A1_B2_C3：如何测试论证修复有效性详解

---
doc_type: glossary
layer: L3
scope_in:  如何测试论证修复有效性、Prompt 修复测试、反光图片处理测试、A/B 测试、回放测试、评估指标
scope_out: 具体测试实现（见 howto）；深入的测试策略（见 L4）
inputs:   (读者) 疑问：假设我们改了 Prompt，如何论证对于反光部分的修复有效？
outputs:  如何测试论证修复有效性 + Prompt 修复测试方法 + 反光图片处理测试 + 评估指标 + 实际例子
entrypoints: [ 核心问题 ]
children: []
related: [ 测试方法, A/B 测试, 回放测试, 评估指标, Prompt 修复, 反光图片处理, KYC_Day02_A1_B2_C2_定位问题后的下一步行动详解.md ]
---

## Definition（定义）

**核心问题**：**假设我们改了 Prompt，如何论证对于反光部分的修复有效？**

**核心答案**（简化版，三步流程）：
- ✅ **步骤 1**：**Unit Test（单元测试）**，用小的测试数据集，快速验证新 Prompt 是否能工作
- ✅ **步骤 2**：**Compare Test（对比测试）**，用历史数据回放，对比旧 Prompt 和新 Prompt 的效果
- ✅ **步骤 3**：**A/B Test（灰度发布）**，在生产环境小流量测试，验证真实效果

**关键理解**：
- ✅ **Unit Test**：快速验证新 Prompt 是否能工作（小数据集，几分钟）
- ✅ **Compare Test**：对比旧 Prompt 和新 Prompt 的效果（历史数据，几小时）
- ✅ **A/B Test**：在生产环境验证真实效果（小流量，几天）

**完整流程**：
```
1. Unit Test（快速验证）
   ↓ 通过
2. Compare Test（对比评估）
   ↓ 通过
3. A/B Test（灰度发布）
   ↓ 通过
4. 全面上线
```

---

## 🎯 核心问题

### 如何测试论证 Prompt 修复有效？（三步流程）

**场景**：我们修改了 Prompt，要求 LLM 输出 `user_name` 字段，如何论证修复有效？

**简化流程（三步）**：
```
1. Unit Test（单元测试）
   - 用小的测试数据集（10-20 张反光图片）
   - 快速验证新 Prompt 是否能工作
   - 时间：几分钟
   ↓ 通过
   
2. Compare Test（对比测试）
   - 用历史数据回放（100+ 张反光图片）
   - 对比旧 Prompt 和新 Prompt 的效果
   - 时间：几小时
   ↓ 通过
   
3. A/B Test（灰度发布）
   - 在生产环境小流量测试（1% → 10% → 50% → 100%）
   - 验证真实效果
   - 时间：几天到几周
   ↓ 通过
   
4. 全面上线
```

**关键区别**：
- ✅ **Unit Test**：快速验证（小数据集，几分钟）
- ✅ **Compare Test**：对比评估（历史数据，几小时）
- ✅ **A/B Test**：真实验证（生产环境，几天）

---

## 📊 详细步骤（三步流程）

### 步骤 1：Unit Test（单元测试）

**目的**：快速验证新 Prompt 是否能工作。

**操作步骤**：

**1.1 构建小的测试数据集**（10-20 张反光图片）：
```
来源：
1. 历史数据：从生产环境中收集有反光问题的图片
   - 从 Logs 中筛选 error_code: SCHEMA_VALIDATION_FAILED
   - 从失败请求中提取 file_id 和 file_url
   - 下载原始图片

2. 人工收集：人工收集各种反光场景的图片
   - 强光反光（窗户、灯光）
   - 弱光反光（屏幕、金属）
   - 模糊反光（对焦不准）

3. 模拟生成：使用图像处理工具模拟反光效果
   - 在正常图片上添加反光效果
   - 生成不同强度的反光图片
```

**1.2 使用新 Prompt 处理测试数据**：
```
操作：
1. 使用新 Prompt（v2.2.0）处理 10-20 张测试图片
2. 检查结果：是否都能识别出 user_name 字段

结果：
- 如果 10-20 张图片都能识别出 user_name → ✅ 通过 Unit Test
- 如果有图片无法识别 → ❌ 不通过，需要修改 Prompt

时间：几分钟
```

**例子**：
```python
# Unit Test：快速验证新 Prompt
def unit_test_new_prompt(test_images: list, prompt_version: str = "v2.2.0"):
    """Unit Test：快速验证新 Prompt"""
    results = []
    
    for image_file in test_images[:20]:  # 只用前 20 张
        result = process_image_with_prompt(
            image_path=image_file,
            prompt_version=prompt_version
        )
        
        # 检查是否包含 user_name 字段
        if "user_name" in result.get("llm_output", {}):
            results.append({"file_id": image_file, "status": "pass"})
        else:
            results.append({"file_id": image_file, "status": "fail"})
    
    # 判断是否通过
    pass_count = sum(1 for r in results if r["status"] == "pass")
    pass_rate = pass_count / len(results)
    
    if pass_rate >= 0.8:  # 80% 通过率
        return {"status": "pass", "pass_rate": pass_rate}
    else:
        return {"status": "fail", "pass_rate": pass_rate}
```

**关键点**：
- ✅ **快速验证**：只用 10-20 张图片，几分钟就能完成
- ✅ **简单判断**：只要能识别出 user_name 字段就算通过
- ✅ **快速反馈**：如果失败，立即知道需要修改 Prompt

---

### 步骤 2：Compare Test（对比测试）

**目的**：用历史数据回放，对比旧 Prompt 和新 Prompt 的效果。

**操作步骤**：

**2.1 构建大的测试数据集**（100+ 张反光图片）：
```
来源：
1. 从生产环境收集 100+ 张有反光问题的图片
   - 从 Logs 中筛选 error_code: SCHEMA_VALIDATION_FAILED
   - 提取 file_id 和 file_url
   - 下载原始图片

2. 标注正确答案：
   - 人工查看图片，标注正确答案
   - 标注反光程度（轻度、中度、重度）

目的：对比旧 Prompt 和新 Prompt 的效果
时间：几小时
```

**2.2 使用旧 Prompt 处理测试数据**：
```
操作：
1. 使用旧 Prompt（Prompt v2.1.0）处理测试数据集
2. 记录处理结果（LLM 输出、识别率、错误率等）

结果：
{
  "prompt_version": "v2.1.0",
  "total_images": 100,
  "success_count": 60,  # 成功识别
  "failure_count": 40,  # 识别失败
  "success_rate": 0.60,  # 成功率 60%
  "error_rate": 0.40,  # 错误率 40%
  "results": [
    {
      "file_id": "doc_001.jpg",
      "status": "failure",
      "error_code": "SCHEMA_VALIDATION_FAILED",
      "missing_fields": ["user_name"]
    },
    ...
  ]
}
```

**例子**：
```python
# 使用旧 Prompt 处理测试数据
def replay_with_old_prompt(test_data_path: str, prompt_version: str = "v2.1.0"):
    """使用旧 Prompt 回放测试"""
    results = []
    
    for image_file in os.listdir(f"{test_data_path}/images"):
        # 使用旧 Prompt 处理图片
        result = process_image_with_prompt(
            image_path=f"{test_data_path}/images/{image_file}",
            prompt_version=prompt_version
        )
        
        results.append({
            "file_id": image_file,
            "status": result["status"],
            "llm_output": result["llm_output"],
            "error_code": result.get("error_code"),
            "missing_fields": result.get("missing_fields", [])
        })
    
    # 计算指标
    success_count = sum(1 for r in results if r["status"] == "success")
    failure_count = len(results) - success_count
    
    return {
        "prompt_version": prompt_version,
        "total_images": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / len(results),
        "error_rate": failure_count / len(results),
        "results": results
    }
```

**2.2 使用新 Prompt 处理测试数据**：
```
操作：
1. 使用新 Prompt（Prompt v2.2.0）处理测试数据集
2. 记录处理结果（LLM 输出、识别率、错误率等）

结果：
{
  "prompt_version": "v2.2.0",
  "total_images": 100,
  "success_count": 85,  # 成功识别
  "failure_count": 15,  # 识别失败
  "success_rate": 0.85,  # 成功率 85%
  "error_rate": 0.15,  # 错误率 15%
  "results": [
    {
      "file_id": "doc_001.jpg",
      "status": "success",
      "llm_output": {
        "user_name": "张三",
        "user_id": "u123",
        "birth_date": "1990-01-01"
      }
    },
    ...
  ]
}
```

**例子**：
```python
# 使用新 Prompt 处理测试数据
def replay_with_new_prompt(test_data_path: str, prompt_version: str = "v2.2.0"):
    """使用新 Prompt 回放测试"""
    results = []
    
    for image_file in os.listdir(f"{test_data_path}/images"):
        # 使用新 Prompt 处理图片
        result = process_image_with_prompt(
            image_path=f"{test_data_path}/images/{image_file}",
            prompt_version=prompt_version
        )
        
        results.append({
            "file_id": image_file,
            "status": result["status"],
            "llm_output": result["llm_output"],
            "error_code": result.get("error_code"),
            "missing_fields": result.get("missing_fields", [])
        })
    
    # 计算指标
    success_count = sum(1 for r in results if r["status"] == "success")
    failure_count = len(results) - success_count
    
    return {
        "prompt_version": prompt_version,
        "total_images": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / len(results),
        "error_rate": failure_count / len(results),
        "results": results
    }
```

**关键点**：
- ✅ **使用相同的数据集**：旧 Prompt 和新 Prompt 使用相同的测试数据集
- ✅ **记录详细结果**：记录每个图片的处理结果（成功/失败、错误代码等）
- ✅ **计算指标**：计算识别率、错误率等指标

---

**2.3 使用新 Prompt 处理测试数据**：
```
操作：
1. 使用新 Prompt（v2.2.0）处理 100+ 张测试图片
2. 记录处理结果（识别率、错误率等）
```

**2.4 对比评估**：
```
对比结果：
- 旧 Prompt：成功率 60%，错误率 40%
- 新 Prompt：成功率 85%，错误率 15%
- 提升：成功率 +25%，错误率 -25%

结论：✅ 新 Prompt 明显优于旧 Prompt
```

**关键点**：
- ✅ **对比相同的数据集**：旧 Prompt 和新 Prompt 使用相同的测试数据集
- ✅ **定量评估**：用数字说话（识别率、错误率等）
- ✅ **时间较长**：100+ 张图片需要几小时处理
```
旧 Prompt（v2.1.0）：
- 成功率：60%（60/100）
- 错误率：40%（40/100）

新 Prompt（v2.2.0）：
- 成功率：85%（85/100）
- 错误率：15%（15/100）

对比：
- 成功率提升：+25%（从 60% 提升到 85%）
- 错误率下降：-25%（从 40% 下降到 15%）

结论：✅ 新 Prompt 明显优于旧 Prompt
```

**3.2 对比反光程度**：
```
按反光程度对比：

轻度反光（30 张）：
- 旧 Prompt：成功率 80%（24/30）
- 新 Prompt：成功率 95%（28/30）
- 提升：+15%

中度反光（50 张）：
- 旧 Prompt：成功率 50%（25/50）
- 新 Prompt：成功率 85%（42/50）
- 提升：+35%

重度反光（20 张）：
- 旧 Prompt：成功率 55%（11/20）
- 新 Prompt：成功率 75%（15/20）
- 提升：+20%

结论：✅ 新 Prompt 对各种反光程度都有提升，尤其是中度反光
```

**3.3 对比具体案例**：
```
案例 1：doc_001.jpg（中度反光）
- 旧 Prompt：失败（缺少 user_name 字段）
- 新 Prompt：成功（包含 user_name 字段："张三"）
- 结论：✅ 修复有效

案例 2：doc_002.jpg（重度反光）
- 旧 Prompt：失败（缺少 user_name 字段）
- 新 Prompt：成功（包含 user_name 字段："李四"）
- 结论：✅ 修复有效

案例 3：doc_003.jpg（轻度反光）
- 旧 Prompt：成功
- 新 Prompt：成功
- 结论：✅ 不受影响（轻度反光本来就能识别）
```

**3.4 对比评估报告**：
```
评估报告：
{
  "test_date": "2024-01-15",
  "test_dataset": "100 张反光图片",
  "old_prompt": {
    "version": "v2.1.0",
    "success_rate": 0.60,
    "error_rate": 0.40
  },
  "new_prompt": {
    "version": "v2.2.0",
    "success_rate": 0.85,
    "error_rate": 0.15
  },
  "improvement": {
    "success_rate_delta": 0.25,  # 提升 25%
    "error_rate_delta": -0.25,  # 下降 25%
    "improvement_rate": 41.67  # 改善率 41.67%（25%/60%）
  },
  "conclusion": "✅ 新 Prompt 明显优于旧 Prompt，可以上线"
}
```

**关键点**：
- ✅ **对比相同的数据集**：确保对比的公平性
- ✅ **多维度对比**：识别率、错误率、反光程度等
- ✅ **定量评估**：用数字说话，而不是主观判断

---

### 步骤 3：A/B Test（灰度发布）

**目的**：在生产环境小流量测试，验证真实效果（新数据，不是历史数据）。

**操作步骤**：

**4.1 配置 A/B 测试**：
```
配置：
1. 流量分配：1% 流量使用新 Prompt（v2.2.0），99% 使用旧 Prompt（v2.1.0）
2. 测试时间：3-7 天
3. 监控指标：识别率、错误率、人工审核率等

操作：
- 使用 Feature Flag 或配置中心控制流量分配
- 根据 request_id 或 user_id 的 hash 值分配流量
```

**例子**：
```python
# A/B 测试配置
def should_use_new_prompt(request_id: str) -> bool:
    """决定是否使用新 Prompt"""
    # 1% 流量使用新 Prompt
    hash_value = hash(request_id) % 100
    return hash_value < 1  # 只有 1% 的请求使用新 Prompt

def process_with_ab_test(request_id: str, image_path: str):
    """A/B 测试处理"""
    if should_use_new_prompt(request_id):
        prompt_version = "v2.2.0"  # 新 Prompt
        variant = "B"
    else:
        prompt_version = "v2.1.0"  # 旧 Prompt
        variant = "A"
    
    # 处理图片
    result = process_image_with_prompt(image_path, prompt_version)
    
    # 记录 A/B 测试标识
    result["ab_test_variant"] = variant
    result["prompt_version"] = prompt_version
    
    return result
```

**3.2 监控指标**（新数据，不是历史数据）：
```
监控指标：
1. 识别率（Success Rate）
   - 旧 Prompt（A 组）：60%
   - 新 Prompt（B 组）：85%

2. 错误率（Error Rate）
   - 旧 Prompt（A 组）：40%
   - 新 Prompt（B 组）：15%

3. 人工审核率（Manual Review Rate）
   - 旧 Prompt（A 组）：35%
   - 新 Prompt（B 组）：10%

4. 平均处理时间（Average Latency）
   - 旧 Prompt（A 组）：5 秒
   - 新 Prompt（B 组）：5.2 秒（稍微慢一点，但可以接受）
```

**例子**（Dashboard 显示）：
```
A/B 测试结果（过去 7 天）：
┌─────────────────────────────────────────────────────────┐
│ 指标               │ A 组（旧 Prompt）│ B 组（新 Prompt）│ 改善 │
├─────────────────────────────────────────────────────────┤
│ 识别率             │ 60%            │ 85%            │ +25% │
│ 错误率             │ 40%            │ 15%            │ -25% │
│ 人工审核率         │ 35%            │ 10%            │ -25% │
│ 平均处理时间       │ 5.0 秒         │ 5.2 秒         │ +0.2s│
│ 样本数量           │ 10,000         │ 100            │      │
└─────────────────────────────────────────────────────────┘

结论：✅ B 组（新 Prompt）明显优于 A 组（旧 Prompt）
```

**3.3 逐步扩大流量**：
```
流量扩大流程：
1. 1% 流量测试（3 天）
   - 监控指标：识别率、错误率
   - 如果效果好，进入下一步

2. 10% 流量测试（3 天）
   - 监控指标：识别率、错误率、人工审核率
   - 如果效果好，进入下一步

3. 50% 流量测试（3 天）
   - 监控指标：所有指标
   - 如果效果好，进入下一步

4. 100% 流量上线
   - 全面切换到新 Prompt
   - 持续监控指标
```

**关键点**：
- ✅ **小流量开始**：从 1% 流量开始，逐步扩大
- ✅ **监控指标**：持续监控识别率、错误率等指标
- ✅ **逐步扩大**：如果效果好，逐步扩大到 10%、50%、100%

---

## 💡 实际例子（KYC 项目）

### 完整测试流程示例

**场景**：我们修改了 Prompt，要求 LLM 输出 `user_name` 字段，需要论证修复有效。

**步骤 1：构建测试数据集**

```
操作：
1. 从生产环境收集 100 张反光图片
   - 从 Logs 中筛选 SCHEMA_VALIDATION_FAILED 错误
   - 提取 file_id 和 file_url
   - 下载原始图片

2. 人工标注正确答案
   - 人工查看图片，标注 user_name、user_id、birth_date
   - 标注反光程度（轻度、中度、重度）

结果：
- 测试数据集：100 张反光图片
- 反光程度分布：
  - 轻度：30 张
  - 中度：50 张
  - 重度：20 张
```

---

**步骤 2：回放测试**

```
操作：
1. 使用旧 Prompt（v2.1.0）处理测试数据集
   - 成功：60 张
   - 失败：40 张
   - 成功率：60%

2. 使用新 Prompt（v2.2.0）处理测试数据集
   - 成功：85 张
   - 失败：15 张
   - 成功率：85%

结果：
- 成功率提升：+25%（从 60% 提升到 85%）
- 错误率下降：-25%（从 40% 下降到 15%）
```

---

**步骤 3：对比评估**

```
评估结果：
1. 整体对比：
   - 成功率提升：+25%
   - 错误率下降：-25%
   - 结论：✅ 新 Prompt 明显优于旧 Prompt

2. 按反光程度对比：
   - 轻度反光：+15%（从 80% 提升到 95%）
   - 中度反光：+35%（从 50% 提升到 85%）
   - 重度反光：+20%（从 55% 提升到 75%）
   - 结论：✅ 新 Prompt 对各种反光程度都有提升

3. 具体案例对比：
   - 案例 1（doc_001.jpg）：失败 → 成功
   - 案例 2（doc_002.jpg）：失败 → 成功
   - 案例 3（doc_003.jpg）：成功 → 成功
   - 结论：✅ 修复有效
```

---

**步骤 4：A/B 测试（灰度发布）**

```
配置：
- 1% 流量使用新 Prompt（v2.2.0）
- 99% 流量使用旧 Prompt（v2.1.0）
- 测试时间：7 天

结果（过去 7 天）：
- A 组（旧 Prompt）：
  - 识别率：60%
  - 错误率：40%
  - 人工审核率：35%
  - 样本数量：10,000

- B 组（新 Prompt）：
  - 识别率：85%
  - 错误率：15%
  - 人工审核率：10%
  - 样本数量：100

结论：✅ B 组（新 Prompt）明显优于 A 组（旧 Prompt）
- 识别率提升：+25%
- 错误率下降：-25%
- 人工审核率下降：-25%
```

---

**步骤 5：逐步扩大流量**

```
流程：
1. 1% 流量测试（3 天）→ 效果好 → 继续
2. 10% 流量测试（3 天）→ 效果好 → 继续
3. 50% 流量测试（3 天）→ 效果好 → 继续
4. 100% 流量上线 → 全面切换到新 Prompt

结果：
- 识别率从 60% 提升到 85%
- 错误率从 40% 下降到 15%
- 人工审核率从 35% 下降到 10%
- 问题解决！✅
```

---

## 📊 评估指标

### 核心指标

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|---------|------|
| **识别率（Success Rate）** | 成功识别的图片占比 | 成功数量 / 总数量 | > 90% |
| **错误率（Error Rate）** | 识别失败的图片占比 | 失败数量 / 总数量 | < 10% |
| **人工审核率（Manual Review Rate）** | 转人工审核的图片占比 | 人工审核数量 / 总数量 | < 20% |
| **平均处理时间（Average Latency）** | 平均处理时间 | 总处理时间 / 总数量 | < 10 秒 |

---

### 细分指标

**按反光程度**：
- 轻度反光：成功率 > 95%
- 中度反光：成功率 > 80%
- 重度反光：成功率 > 60%

**按识别难度**：
- 容易识别：成功率 > 95%
- 中等难度：成功率 > 80%
- 困难识别：成功率 > 60%

---

## 💡 业界标准测试流程

### 业界通用的测试流程

**业界标准流程（ML/LLM 模型测试）**：

```
1. Offline Evaluation（离线评估）
   - Golden Set Test（黄金数据集测试）
   - Replay Test（回放测试）
   ↓
2. Shadow Test（影子测试）
   - 生产环境 Shadow 模式（不直接影响用户）
   ↓
3. A/B Test（灰度发布）
   - 生产环境小流量测试
   - 逐步扩大流量
```

**关键理解**：
- ✅ **Offline Evaluation**：离线评估（不进入生产环境）
- ✅ **Shadow Test**：影子测试（生产环境运行，但不影响用户）
- ✅ **A/B Test**：灰度发布（生产环境小流量测试，影响用户）

---

### 步骤 1：Offline Evaluation（离线评估）

**目的**：在不上线的情况下，评估新 Prompt 的效果。

**业界标准做法**：

**1.1 Golden Set Test（黄金数据集测试）**
```
做法：
1. 构建 Golden Set（黄金数据集）
   - 收集代表性样本（100-1000 张）
   - 人工标注正确答案（ground truth）
   - 涵盖各种场景（正常、反光、模糊等）

2. 使用新 Prompt 处理 Golden Set
   - 记录识别率、错误率等指标

3. 评估指标
   - 准确率（Accuracy）
   - 精确率（Precision）
   - 召回率（Recall）
   - F1 Score

判断标准：
- 如果指标达标（如准确率 > 90%）→ ✅ 通过
- 如果指标不达标 → ❌ 不通过，需要修改 Prompt
```

**1.2 Replay Test（回放测试）**
```
做法：
1. 收集历史数据（失败的请求）
   - 从生产环境收集有问题的图片
   - 包含失败原因和上下文

2. 使用新 Prompt 回放处理
   - 使用旧 Prompt 处理 → 记录结果
   - 使用新 Prompt 处理 → 记录结果

3. 对比评估
   - 对比识别率、错误率等指标
   - 对比具体的失败案例（之前失败，现在是否成功）

判断标准：
- 如果新 Prompt 明显优于旧 Prompt（如成功率提升 > 20%）→ ✅ 通过
- 如果新 Prompt 没有明显改善 → ❌ 不通过，需要修改 Prompt
```

**业界标准**：
- ✅ **Google/Meta/OpenAI**：使用 Golden Set + Replay Test 评估模型效果
- ✅ **Banking/Finance**：使用 Golden Set + Replay Test 评估欺诈检测模型
- ✅ **E-commerce**：使用 Golden Set + Replay Test 评估推荐系统

**Trade-off**：
- ✅ **优点**：不需要上线，风险低，可以快速迭代
- ⚠️ **缺点**：历史数据可能不代表未来的数据分布（Data Drift）

---

### 步骤 2：Shadow Test（影子测试）

**目的**：在生产环境运行新 Prompt，但不影响用户（不返回结果给用户）。

**业界标准做法**：

**做法**：
```
1. Shadow 模式运行新 Prompt
   - 生产环境请求同时使用旧 Prompt 和新 Prompt 处理
   - 旧 Prompt 的结果返回给用户（正常流程）
   - 新 Prompt 的结果只用于监控和评估（Shadow）

2. 监控和对比
   - 对比旧 Prompt 和新 Prompt 的处理结果
   - 监控新 Prompt 的性能指标（识别率、错误率、延迟等）
   - 观察是否有异常情况

3. 评估时间
   - 通常运行 1-3 天，收集足够的数据
   - 如果效果好，进入 A/B Test
```

**例子**：
```python
# Shadow Test 实现
def process_with_shadow_test(request_id: str, image_path: str):
    """Shadow Test 处理"""
    # 1. 旧 Prompt 处理（返回给用户）
    old_result = process_image_with_prompt(image_path, prompt_version="v2.1.0")
    
    # 2. 新 Prompt 处理（Shadow，不返回给用户）
    new_result = process_image_with_prompt(image_path, prompt_version="v2.2.0")
    
    # 3. 对比和监控
    compare_results(old_result, new_result)
    
    # 4. 返回旧 Prompt 的结果（用户无感知）
    return old_result
```

**业界标准**：
- ✅ **Netflix/Spotify**：使用 Shadow Test 评估推荐系统
- ✅ **Uber/Lyft**：使用 Shadow Test 评估定价模型
- ✅ **Banking/Finance**：使用 Shadow Test 评估风控模型

**Trade-off**：
- ✅ **优点**：真实环境数据，不影响用户，风险低
- ⚠️ **缺点**：需要运行两套系统，资源消耗增加（2x）

---

### 步骤 3：A/B Test（灰度发布）

**目的**：在生产环境小流量测试，影响部分用户。

**业界标准做法**：

**做法**：
```
1. 配置 A/B Test
   - 小流量使用新 Prompt（1-10%）
   - 大流量使用旧 Prompt（90-99%）

2. 监控指标
   - 识别率、错误率、人工审核率
   - 用户体验指标（满意度、投诉率等）

3. 逐步扩大流量
   - 1% → 10% → 50% → 100%
   - 每一步监控指标，如果效果好，继续扩大
```

**业界标准**：
- ✅ **Google/Meta**：使用 A/B Test 评估所有产品功能
- ✅ **Amazon**：使用 A/B Test 评估推荐算法和 UI 设计
- ✅ **Netflix**：使用 A/B Test 评估内容推荐和播放体验

**Trade-off**：
- ✅ **优点**：真实用户体验，可以验证真实效果
- ⚠️ **缺点**：影响部分用户，如果新 Prompt 有问题，会影响用户

---

## 📊 业界标准流程对比

### 完整流程对比

| 步骤 | 目的 | 数据来源 | 影响用户 | 时间 | 业界使用 |
|------|------|---------|---------|------|---------|
| **Offline Evaluation** | 离线评估新 Prompt 效果 | 历史数据 / Golden Set | ❌ 不影响 | 几小时 | ✅ 通用 |
| **Shadow Test** | 生产环境 Shadow 运行 | 生产环境实时数据 | ❌ 不影响 | 1-3 天 | ✅ 大公司常用 |
| **A/B Test** | 生产环境小流量测试 | 生产环境实时数据 | ✅ 影响部分用户 | 几天到几周 | ✅ 通用 |

---

### 业界常见流程（根据公司规模）

**小公司（成本敏感）**：
```
流程：
1. Offline Evaluation（Golden Set + Replay Test）
2. A/B Test（直接跳过 Shadow Test）

原因：Shadow Test 需要运行两套系统，成本高
```

**中公司（平衡成本和风险）**：
```
流程：
1. Offline Evaluation（Golden Set + Replay Test）
2. Shadow Test（可选，如果有足够资源）
3. A/B Test

原因：Shadow Test 可以降低风险，但成本较高
```

**大公司（重视风险控制）**：
```
流程：
1. Offline Evaluation（Golden Set + Replay Test）
2. Shadow Test（必须，降低风险）
3. A/B Test（逐步扩大流量）

原因：大公司用户量大，影响面广，必须严格控制风险
```

---

## 💡 总结

### 核心答案（业界标准流程）

**业界标准测试流程**：

**完整流程（三步）**：
1. **Offline Evaluation（离线评估）**：
   - Golden Set Test（黄金数据集测试）
   - Replay Test（回放测试）
   - 目的：在不上线的情况下，评估新 Prompt 的效果
   - 时间：几小时
   - 影响：❌ 不影响用户

2. **Shadow Test（影子测试）**（可选，大公司常用）：
   - 生产环境 Shadow 模式运行
   - 目的：真实环境数据，但不影响用户
   - 时间：1-3 天
   - 影响：❌ 不影响用户

3. **A/B Test（灰度发布）**：
   - 生产环境小流量测试
   - 目的：真实验证，影响部分用户
   - 时间：几天到几周
   - 影响：✅ 影响部分用户

### 关键区别

1. **Offline Evaluation vs Shadow Test vs A/B Test**：
   - ✅ **Offline Evaluation**：历史数据，不影响用户
   - ✅ **Shadow Test**：实时数据，不影响用户
   - ✅ **A/B Test**：实时数据，影响部分用户

2. **业界标准**：
   - ✅ **所有公司**：Offline Evaluation + A/B Test
   - ✅ **大公司**：Offline Evaluation + Shadow Test + A/B Test

3. **为什么需要 Shadow Test**：
   - ✅ **降低风险**：可以在不影响用户的情况下，验证真实效果
   - ✅ **真实数据**：使用生产环境的实时数据，比历史数据更真实
   - ⚠️ **成本较高**：需要运行两套系统（2x 资源消耗）

### 评估指标

- ✅ **识别率（Success Rate）**：> 90%
- ✅ **错误率（Error Rate）**：< 10%
- ✅ **人工审核率（Manual Review Rate）**：< 20%

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1_B2_C2 定位问题后的下一步行动详解（[KYC_Day02_A1_B2_C2_定位问题后的下一步行动详解.md](./KYC_Day02_A1_B2_C2_定位问题后的下一步行动详解.md)） |
| **Related** | 测试方法、A/B 测试、回放测试、评估指标、Prompt 修复、反光图片处理 |
