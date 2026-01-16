# Day 1｜指标体系详细讲解：L0/L1/L2、错误预算、测试设计

**基于项目**：KYC Identity Verification System (PoV)  
**目标**：深入理解三层指标体系、错误预算，以及如何设计测试来测量 p95/p99

---

## 📊 第一部分：什么是 L0/L1/L2 三层指标？

### 核心理念

**三层指标不是随意的分类，而是按照"谁关心、关心什么"来划分的**：

- **L0 稳定性**：运维/On-Call 关心 → "系统现在健康吗？"
- **L1 业务收益**：业务/PM 关心 → "系统创造价值了吗？"
- **L2 长期健康**：工程/Arch 关心 → "系统能持续进化吗？"

---

### L0 稳定性（监控/可用/延迟）- "系统现在健康吗？"

**定义**：实时监控系统是否正常工作的指标。

**谁关心**：
- On-Call 工程师（半夜告警时最关心）
- 运维团队（日常监控）
- 用户（直接感知到的问题）

**核心指标**：

#### 1. 成功率（Success Rate）

**定义**：正常响应的请求数 / 总请求数

**KYC 项目示例**：
- **计算方式**：`_summary.json` 中 `status: "success"` 的文档数 / 总文档数
- **目标**：99%（Production）
- **当前值**：95%（PoV 阶段）
- **告警阈值**：< 98% 触发告警

**为什么重要**：
- 直接反映系统可用性
- 用户最直观的感受
- On-Call 最关心的指标

#### 2. 延迟指标（Latency）- p50/p95/p99

**定义**：请求从发起到响应完成的时间。

**分位数解释**：

```
p50 (中位数) = 50% 的请求都在这个时间以内
p95 = 95% 的请求都在这个时间以内
p99 = 99% 的请求都在这个时间以内

例如：
- p50 = 3秒：50% 的请求在 3 秒内完成
- p95 = 8秒：95% 的请求在 8 秒内完成（5% 的请求超过 8 秒）
- p99 = 15秒：99% 的请求在 15 秒内完成（1% 的请求超过 15 秒）
```

**为什么用 p95/p99 而不是平均值？**

```
假设 100 个请求的延迟：
[1s, 2s, 3s, ..., 99s, 100s]

平均值 = 50.5 秒（被极端值拉高）
p95 = 95 秒（更真实反映大部分用户的体验）
p99 = 99 秒（捕获最坏情况的用户体验）
```

**平均值的问题**：
- 被极端值（outliers）拉高
- 无法反映"大部分用户的真实体验"
- 无法捕获"长尾问题"（那 5% 或 1% 的用户体验）

**KYC 项目示例**：
- **p50**：3-5 秒（单文档处理时间的中位数）
- **p95**：8-10 秒（SLO 目标：< 15 秒）
- **p99**：15-20 秒（SLO 目标：< 30 秒）

**组成分析**（单文档处理时间）：
- Preprocess（图片预处理）：100-200ms
- Rate Limiter Acquire：0-1000ms（如果有排队）
- Fireworks API Call：2000-8000ms（模型推理时间）
- Schema Validation：50-100ms
- Deterministic Rules：10-50ms
- Save Result：20-50ms

**告警阈值**：
- p95 > 15 秒 → Warning（触发告警）
- p99 > 30 秒 → Critical（立即回滚）

#### 3. 错误率（Error Rate）

**定义**：错误响应的请求数 / 总请求数

**KYC 项目示例**：
- **目标**：< 1%（Production）
- **当前值**：5%（PoV 阶段）
- **错误分类**（基于 `src/errors.py`）：
  - `IMAGE_FORMAT_UNSUPPORTED`：1%
  - `SCHEMA_VALIDATION_FAILED`：2%
  - `API_TIMEOUT`：1%
  - `RATE_LIMIT_EXCEEDED`：1%

**告警阈值**：
- Error Rate > 2% → Warning
- Error Rate > 5% → Critical（立即回滚）

#### 4. 回退率（Fallback Rate）

**定义**：触发降级/回退的请求数 / 总请求数

**KYC 项目示例**：
- **当前值**：0%（PoV 阶段无降级）
- **未来规划**：低质量图片 → OCR-only fallback

#### 5. 可用性（Availability）

**定义**：系统可用的时间 / 总时间

**计算方式**：
- 99% = 每月 < 7.3 小时不可用
- 99.9% = 每月 < 43 分钟不可用
- 99.99% = 每月 < 4.3 分钟不可用

**KYC 项目示例**：
- **目标**：99.9%（Production = 每月 < 43 分钟不可用）
- **当前值**：99%（PoV 阶段）

**SLA vs SLO vs SLI**：
- **SLA**（Service Level Agreement）：对用户的承诺（如 99.9% 可用性）
- **SLO**（Service Level Objective）：内部目标（如 99.95% 可用性，比 SLA 更严格）
- **SLI**（Service Level Indicator）：实际测量的指标（如当前的 99% 可用性）

---

### L1 业务收益（ROI）- "系统创造价值了吗？"

**定义**：系统为业务创造的实际价值（省钱、省时、降低风险）。

**谁关心**：
- 业务团队/PM（证明系统的价值）
- 财务部门（成本分析）
- 决策层（是否继续投资）

**核心指标**：

#### 1. 每单节省的人审分钟数

**KYC 项目示例**：
- **基线**：5-10 分钟/单（人工审核一张 ID 文档）
- **当前值**：3-5 秒/单（AI 处理时间）
- **节省**：5 分钟/单
- **ROI**：`5 分钟/单 × 1000 单/月 = 5000 分钟/月 = 83 小时/月`
- **节省效率**：`> 99%` 的时间节省

**为什么重要**：
- 直接证明系统的价值
- 可以计算人力成本节省
- 业务团队最关心的指标

#### 2. 错误拦截率带来的风险降低

**KYC 项目示例**：
- **关键拦截场景**（基于 `src/rules.py` 的 fraud markers）：
  - `expiry:expired`：过期文档拦截
  - `missing_critical:full_name,date_of_birth`：关键字段缺失拦截
  - `low_confidence_critical:document_number`：低置信度关键字段拦截
- **基线**：0%（无自动化拦截）
- **当前值**：15-20%（PoV 阶段）
- **风险降低估算**：200 件/月（避免的潜在合规风险）

**为什么重要**：
- 降低合规风险（KYC 是强监管领域）
- 减少人工审核的漏检
- 降低潜在的法律风险

#### 3. 吞吐提升带来的成本节省

**KYC 项目示例**：
- **成本分解**：
  - **Fireworks API 调用**：$0.001-0.002 / request（Qwen2.5-VL-32B）
  - **人工审核成本**：$5-10 / request（按小时工资计算）
- **基线成本**：$7.5 / request（平均人工成本）
- **当前成本**：$0.0015 / request（AI 成本）
- **节省**：`$7.5 / request × 1000 requests/月 = $7500 / 月`
- **ROI 倍数**：`5000x` 成本降低

**为什么重要**：
- 直接的成本节省
- 可以量化 ROI
- 财务部门最关心的指标

#### 4. 自动化率

**KYC 项目示例**：
- **定义**：无需人工介入的请求数 / 总请求数
- **自动化判断**（基于 `src/rules.py`）：
  - 所有关键字段提取成功（`full_name, date_of_birth, document_number, expiry_date, issuing_country`）
  - 置信度 > 阈值（如 `> 0.85`）
  - 通过确定性规则检查（expiry valid, quality good）
- **目标**：80%（20% 需要人工 review）
- **当前值**：60-70%（PoV 阶段）

**为什么重要**：
- 反映系统的成熟度
- 直接关联人力成本
- 业务团队关心的效率指标

---

### L2 长期健康（可维护/可扩展）- "系统能持续进化吗？"

**定义**：系统能否持续改进、降低维护成本、提高可扩展性的指标。

**谁关心**：
- 工程团队/架构师（系统的可持续性）
- Tech Lead（技术债务管理）
- CTO（技术战略）

**核心指标**：

#### 1. 变更失败率（Change Failure Rate）

**定义**：导致回滚/问题的发布数 / 总发布数

**KYC 项目示例**：
- **目标**：< 5%
- **当前值**：0%（PoV 阶段，尚未有 production releases）
- **基于**：Schema-First 设计（`src/schemas.py`）和确定性规则（`src/rules.py`）降低变更风险

**为什么重要**：
- 反映系统设计的稳定性
- 降低变更的风险成本
- 工程团队关心的稳定性指标

#### 2. 回滚频率（Rollback Frequency）

**定义**：回滚次数 / 总发布次数

**KYC 项目示例**：
- **目标**：< 2%
- **当前值**：0%（PoV 阶段）
- **近30天**：0 次回滚

**为什么重要**：
- 反映发布流程的成熟度
- 降低生产事故风险
- 工程团队关心的发布稳定性

#### 3. 回归门禁通过率（Regression Gate Pass Rate）

**定义**：通过回归测试的发布数 / 总发布数

**KYC 项目示例**：
- **回归测试覆盖**：
  - Unit tests (`tests/test_rules.py`, `tests/test_validators.py`)
  - Schema validation tests (`tests/test_validators.py`)
  - Error handling tests (`tests/test_errors.py`)
- **目标**：> 95%
- **当前值**：100%（PoV 阶段，所有单元测试通过）

**为什么重要**：
- 反映测试覆盖的完整性
- 降低回归问题的风险
- 工程团队关心的质量指标

#### 4. 告警噪音（Alert Noise - Precision）

**定义**：有效告警数 / 总告警数

**KYC 项目示例**：
- **目标**：> 80%（减少误报）
- **当前值**：N/A（PoV 阶段，无生产告警系统）
- **平均告警数/周**：0（PoV 阶段）

**为什么重要**：
- 降低 On-Call 的疲劳（减少误报）
- 提高告警的响应效率
- 运维团队关心的告警质量

#### 5. Toil（重复劳动）趋势

**定义**：每周花在重复性任务上的时间

**KYC 项目示例**：
- **Toil 来源**：
  - 手动运行 batch processing
  - 手动检查 `_summary.json`
  - 手动处理错误文档
- **目标**：< 5 小时/周
- **当前值**：2-3 小时/周（PoV 阶段）
- **趋势**：→（稳定）
- **自动化机会**：CI/CD 集成、自动化 batch scheduling、错误自动重试

**为什么重要**：
- 降低维护成本（重复劳动 = 浪费工程师时间）
- 提高工程效率
- 工程团队关心的效率指标

#### 6. Schema 兼容性

**定义**：Schema 变更导致的 breaking changes 数 / 总 schema 变更数

**KYC 项目示例**：
- **目标**：0 breaking changes（通过 versioning：`schema_version = "v1"`）
- **当前值**：0（PoV 阶段）
- **基于**：`src/schemas.py` 的版本化设计

**为什么重要**：
- 反映 API 设计的稳定性
- 降低下游系统的影响
- 架构师关心的 API 稳定性

#### 7. Auditability 覆盖率

**定义**：包含完整 trace_id 的文档数 / 总文档数

**KYC 项目示例**：
- **目标**：100%
- **当前值**：100%（PoV 阶段）
- **基于**：每个请求都有 `trace_id`（Privacy & Logging section）

**为什么重要**：
- 支持合规审计（KYC 是强监管领域）
- 支持问题定位
- 合规团队关心的可审计性

#### 8. PII 泄漏事件

**定义**：PII 泄漏事件数

**KYC 项目示例**：
- **目标**：0
- **当前值**：0
- **基于**：Logging rules（Never log: base64 image, prompt content, extracted PII fields）

**为什么重要**：
- 降低合规风险（PII 泄漏 = 严重合规问题）
- 降低法律风险
- 合规/安全团队最关心的指标

---

## 🎯 第二部分：什么是错误预算（Error Budget）？

### 核心理念

**错误预算不是"允许失败"，而是"平衡发布速度 vs 稳定性"的控制机制**。

### 错误预算的定义

**Error Budget = 100% - SLO**

**示例**：
- SLO = 99%（成功率）
- Error Budget = 1% = 1000 文档中允许 10 个失败（月度）

### 错误预算的作用

#### 1. 平衡"发布速度 vs 稳定性"

```
如果没有错误预算：
- 产品团队："我们要快速发布新功能！"
- 工程团队："不行，新功能可能引入 bug，影响稳定性！"
- 结果：双方争论不休，决策困难

有了错误预算：
- 错误预算充足（> 50%）→ 可以快速发布（产品团队开心）
- 错误预算不足（< 25%）→ 冻结发布，专注稳定性（工程团队开心）
- 结果：客观的决策机制，双方都能接受
```

#### 2. 量化稳定性成本

```
错误预算 = 稳定性成本

如果错误预算消耗太快：
- 说明系统稳定性有问题
- 需要投入更多资源在稳定性上
- 暂停新功能开发，专注稳定性修复
```

### 错误预算的计算

**L0 稳定性 Error Budget**：

**月度计算**：
- **总请求数**：1000 文档/月
- **SLO**：99%（成功率）
- **Error Budget**：1% = 10 个文档/月 可以失败

**当前状态**（PoV 阶段）：
- **成功率**：95%
- **实际失败数**：50 个文档/月（5% × 1000）
- **Error Budget 消耗**：50 / 10 = 500%（远超预算）

**决策**：
- 当前状态：`冻结`（Error Budget 严重超标）
- 需要：立即修复稳定性问题

### 错误预算的状态

#### 1. 正常状态（Error Budget > 50%）

**状态**：✅ 健康

**决策**：
- ✅ 可以继续发布新功能
- ✅ 可以承担一定风险

**示例**：
- Error Budget 剩余：80%
- 可以快速发布新功能

#### 2. 警告状态（Error Budget 25% - 50%）

**状态**：⚠️ 警告

**决策**：
- ⚠️ 限制高风险发布
- ⚠️ 增加审查流程

**示例**：
- Error Budget 剩余：30%
- 只能发布低风险功能，高风险功能需要更严格的测试

#### 3. 冻结状态（Error Budget < 25%）

**状态**：🛑 冻结

**决策**：
- 🛑 **冻结所有新功能发布**
- 🛑 **只允许稳定性修复和优化**
- 🛑 **全力修复稳定性债务**

**示例**：
- Error Budget 剩余：10%
- 禁止所有新功能发布，只允许 bug fix 和稳定性优化

### KYC 项目的错误预算示例

**月度计算**：
- **总请求数**：1000 文档/月
- **SLO**：99%（成功率）
- **Error Budget**：1% = 10 个文档/月 可以失败

**当前状态**（PoV 阶段）：
- **成功率**：95%
- **实际失败数**：50 个文档/月（5% × 1000）
- **Error Budget 消耗**：50 / 10 = 500%（严重超标）

**决策**：
- 🛑 **冻结发布**，专注稳定性修复
- **优先级**：
  1. 修复 Schema Validation Failures（2% = 20 个文档/月）
  2. 优化错误处理逻辑
  3. 增加重试机制

### 错误预算策略（Error Budget Policy）

**规则**：

1. **错误预算消耗 > 50%** → 正常状态，可以快速发布
2. **错误预算消耗 25% - 50%** → 警告状态，限制高风险发布
3. **错误预算消耗 < 25%** → 冻结状态，禁止新功能发布

**实施**：

```python
# 伪代码示例
def can_release_new_feature(error_budget_remaining_percent):
    if error_budget_remaining_percent > 50:
        return True  # 正常状态，可以发布
    elif error_budget_remaining_percent > 25:
        return "require_strict_review"  # 警告状态，需要严格审查
    else:
        return False  # 冻结状态，禁止发布
```

---

## 🧪 第三部分：如何设计测试来测量 p95/p99？

### 大厂思维：测试的时机和策略

#### 1. 测试的时机（When to Test）

**什么时候测试 p95/p99？**

| 时机 | 测试类型 | 目的 | 频率 |
|------|---------|------|------|
| **开发阶段** | Unit Test | 验证单文档处理时间 | 每次提交 |
| **集成阶段** | Integration Test | 验证 E2E 流程延迟 | 每次 PR |
| **发布前** | Performance Test | 验证 p95/p99 是否达标 | 每次发布前 |
| **生产环境** | Continuous Monitoring | 实时监控 p95/p99 | 实时 |
| **定期** | Load Test | 验证系统容量 | 每月/每季度 |

**KYC 项目的测试时机**：

1. **开发阶段**（每次代码提交）
   - **Unit Test**：`tests/test_pipeline.py`（单文档处理时间）
   - **目的**：快速反馈，验证代码变更没有引入性能退化
   - **频率**：每次 `git commit`

2. **集成阶段**（每次 PR）
   - **Integration Test**：`tests/test_integration.py`（E2E 流程）
   - **目的**：验证完整流程的延迟
   - **频率**：每次 Pull Request

3. **发布前**（每次发布前）
   - **Performance Test**：专门的性能测试脚本
   - **目的**：验证 p95/p99 是否达标（SLO：p95 < 15s, p99 < 30s）
   - **频率**：每次发布前（必须通过才能发布）

4. **生产环境**（实时监控）
   - **Continuous Monitoring**：基于 `_summary.json` 的实时分析
   - **目的**：实时监控生产环境的 p95/p99
   - **频率**：实时（每次 batch 运行后）

5. **定期**（每月/每季度）
   - **Load Test**：压力测试
   - **目的**：验证系统容量（如 1000 文档/小时的处理能力）
   - **频率**：每月/每季度

---

### 2. 测试的设计（How to Design Tests）

#### 测试设计原则

1. **真实场景**：使用真实的数据分布（不是均匀分布）
2. **样本量足够**：至少 100+ 样本才能计算 p95/p99
3. **覆盖边界情况**：包括正常、异常、边界场景
4. **隔离环境**：避免外部因素影响（如网络波动）

#### KYC 项目的测试设计

**测试场景设计**：

```python
# tests/test_performance.py (伪代码示例)

class TestLatency:
    """测试延迟指标（p50/p95/p99）"""
    
    def test_single_document_latency(self):
        """单文档处理时间（Unit Test）"""
        # 测试单个文档的处理时间
        # 目标：p95 < 10s, p99 < 15s
        
        latencies = []
        for i in range(100):  # 100 个样本
            start_time = time.time()
            result = pipeline.process_document(test_image)
            latency = time.time() - start_time
            latencies.append(latency)
        
        # 计算 p95/p99
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        assert p95 < 10, f"p95 {p95}s exceeds 10s threshold"
        assert p99 < 15, f"p99 {p99}s exceeds 15s threshold"
    
    def test_batch_processing_latency(self):
        """Batch 处理时间（Integration Test）"""
        # 测试 100 个文档的 batch 处理时间
        # 目标：每个文档的平均时间 < 8s
        
        start_time = time.time()
        results = pipeline.process_batch(test_images, batch_size=100)
        total_time = time.time() - start_time
        
        avg_latency_per_doc = total_time / 100
        
        assert avg_latency_per_doc < 8, f"Average latency {avg_latency_per_doc}s exceeds 8s threshold"
    
    def test_p95_p99_distribution(self):
        """测试延迟分布（Performance Test）"""
        # 测试不同场景的延迟分布
        # 正常场景、边界场景、异常场景
        
        scenarios = {
            "normal": 50,  # 50 个正常场景
            "edge": 30,    # 30 个边界场景
            "anomaly": 20  # 20 个异常场景
        }
        
        all_latencies = []
        for scenario, count in scenarios.items():
            for i in range(count):
                image = load_test_image(scenario, i)
                start_time = time.time()
                result = pipeline.process_document(image)
                latency = time.time() - start_time
                all_latencies.append(latency)
        
        # 计算 p50/p95/p99
        p50 = np.percentile(all_latencies, 50)
        p95 = np.percentile(all_latencies, 95)
        p99 = np.percentile(all_latencies, 99)
        
        # 验证是否达标
        assert p50 < 5, f"p50 {p50}s exceeds 5s threshold"
        assert p95 < 15, f"p95 {p95}s exceeds 15s threshold"
        assert p99 < 30, f"p99 {p99}s exceeds 30s threshold"
```

**测试数据设计**：

```python
# 测试数据分布（模拟真实场景）

test_data_distribution = {
    "normal_cases": {
        "count": 50,  # 50% 正常场景
        "examples": ["清晰ID", "标准格式", "高分辨率"]
    },
    "edge_cases": {
        "count": 30,  # 30% 边界场景
        "examples": ["模糊图片", "部分遮挡", "低分辨率"]
    },
    "anomaly_cases": {
        "count": 20,  # 20% 异常场景
        "examples": ["版式变化", "多页文档", "特殊字符"]
    }
}
```

---

### 3. 测试的实现（Implementation）

#### CI/CD 集成（发布前必跑）

```yaml
# .github/workflows/performance_test.yml (示例)

name: Performance Test

on:
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  performance_test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run Performance Test
        run: |
          python -m pytest tests/test_performance.py -v
          # 验证 p95 < 15s, p99 < 30s
      
      - name: Performance Test Report
        run: |
          # 生成性能测试报告
          python scripts/generate_performance_report.py
```

#### 生产环境监控（实时）

```python
# scripts/monitor_latency.py (示例)

import json
import numpy as np
from pathlib import Path

def calculate_p95_p99_from_summary(summary_path):
    """从 _summary.json 计算 p95/p99"""
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    latencies = []
    for result in summary.get("results", []):
        if result.get("status") == "success":
            latency = result.get("latency_ms", 0) / 1000  # 转换为秒
            latencies.append(latency)
    
    if len(latencies) < 10:
        print("Not enough samples for p95/p99 calculation")
        return None
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"Latency Statistics:")
    print(f"  p50: {p50:.2f}s")
    print(f"  p95: {p95:.2f}s (SLO: < 15s)")
    print(f"  p99: {p99:.2f}s (SLO: < 30s)")
    
    # 检查是否达标
    if p95 > 15:
        print(f"⚠️  WARNING: p95 {p95:.2f}s exceeds SLO threshold 15s")
    if p99 > 30:
        print(f"🛑 CRITICAL: p99 {p99:.2f}s exceeds SLO threshold 30s")
    
    return {"p50": p50, "p95": p95, "p99": p99}

if __name__ == "__main__":
    summary_path = Path("output_results/_summary.json")
    calculate_p95_p99_from_summary(summary_path)
```

---

### 4. 测试的门禁（Release Gate）

**发布前必须通过的性能测试**：

```python
# tests/test_release_gate.py (示例)

class TestReleaseGate:
    """发布门禁：性能测试（Release Gate）"""
    
    def test_p95_latency_gate(self):
        """p95 延迟门禁（必须 < 15s 才能发布）"""
        latencies = self.run_performance_test(sample_size=100)
        p95 = np.percentile(latencies, 95)
        
        assert p95 < 15, f"p95 {p95}s exceeds release gate threshold 15s. Cannot release."
    
    def test_p99_latency_gate(self):
        """p99 延迟门禁（必须 < 30s 才能发布）"""
        latencies = self.run_performance_test(sample_size=100)
        p99 = np.percentile(latencies, 99)
        
        assert p99 < 30, f"p99 {p99}s exceeds release gate threshold 30s. Cannot release."
```

---

## 📝 总结

### 三层指标的关系

```
L0 稳定性（实时监控）
    ↓
L1 业务收益（价值证明）
    ↓
L2 长期健康（可持续发展）
```

### 错误预算的作用

- **平衡"发布速度 vs 稳定性"**
- **量化稳定性成本**
- **客观的决策机制**

### 测试的设计原则

1. **真实场景**：使用真实的数据分布
2. **样本量足够**：至少 100+ 样本
3. **覆盖边界情况**：正常、异常、边界场景
4. **隔离环境**：避免外部因素影响

### 测试的时机

- **开发阶段**：Unit Test（每次提交）
- **集成阶段**：Integration Test（每次 PR）
- **发布前**：Performance Test（每次发布前，Release Gate）
- **生产环境**：Continuous Monitoring（实时）
- **定期**：Load Test（每月/每季度）

---

## 🎯 下一步行动

1. **理解你的 KYC 项目的当前指标**
   - 分析 `_summary.json` 计算当前的 p50/p95/p99
   - 计算当前的 L0/L1/L2 指标

2. **设计性能测试**
   - 创建 `tests/test_performance.py`
   - 实现 p95/p99 测试（至少 100 个样本）

3. **集成到 CI/CD**
   - 添加 Performance Test 到 GitHub Actions
   - 设置 Release Gate（p95 < 15s, p99 < 30s）

4. **建立监控 Dashboard**
   - 从 `_summary.json` 实时计算 p95/p99
   - 设置告警（p95 > 15s → Warning, p99 > 30s → Critical）

---

## 📚 参考

- KYC 项目：https://github.com/Nickcp39/kyc_pov/tree/main
- Google SRE Book: [SLO, SLI, SLAs](https://sre.google/workbook/slo/)
- Error Budget Policy: https://sre.google/workbook/error-budget-policy/
- Performance Testing: https://sre.google/workbook/testing-for-reliability/
