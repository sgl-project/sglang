# Senior 能力栈：1个月强化计划（AI Infra 实战版）

**目标**：从"写代码"转变为"把 LLM 变成可靠产品"的 AI Infra 工程师  
**标准**：大厂 AI Infra / LLMOps / Model Serving 岗位的真实能力栈（L5-L6，8-12年工作经验）  
**载体**：用你的核心项目（KYC、SGLang）实战训练

---

## 🎯 核心转变

### Junior/Mid → Senior 的鸿沟

**不再是**：
- ❌ 被动接任务的"码农"
- ❌ "我写了 1000 行代码"
- ❌ "我的系统很稳"

**而是**：
- ✅ 能够承载业务确定性的"技术 Owner"
- ✅ "我降低了团队 50% 的联调时间"
- ✅ "我的错误预算（Error Budget）是多少"

### 9 根支柱（面试官抽样验证）

按"面试最常抽查、且你最容易被误判缺失"的维度：

1. **可运营与可靠性**（SRE 思维）：SLO/Error Budget + Canary/Rollback
2. **可观测性与线上排障**（Metrics/Logs/Traces）：坏了能发现、能定位、能止血
3. **测试与回归文化**（测试金字塔/回归门禁）：改动不会把系统搞坏
4. **Code Review 与可维护性**（把复杂变简单）：代码库更健康、更易维护
5. **交付能力与工程效率指标**（DORA / Four Keys）：跑得快还不翻车
6. **影响力与带人**（Senior/Staff 的分水岭）：放大器能力——让别人/别的团队也变强
7. **需求澄清 + Trade-off + 决策 ROI**（产品与工程的翻译）：为什么做、选 A 不选 B、怎么验证收益
8. **安全与隐私**（Threat Model / PII 处理）：数据/权限/审计/泄露风险
9. **写作与设计评审**（Design Doc 能力）：结构化写清楚假设、方案、风险、迁移与回滚

---

## 🎯 你到底要补什么（按重要性排序）

### 主线 = AI Infra System Design（可运营闭环）

**不是"研究模型本身"，而是"把 LLM 变成可控系统"**

1. ✅ **SLO / Error Budget**（用错误预算平衡发布速度与稳定性）
2. ✅ **Canary / Rollback**（低风险进化）
3. ✅ **Observability**：metrics/logs/traces（坏了能发现、能定位）
4. ✅ **回归与发布门禁**（任何改动都能证明"没变差"）

### 地基 = 只补跟 SD 强绑定的基础（20%）

**只学高频、且能立刻转化成面试得分的**

1. ✅ **网络**：TCP/HTTP、连接复用、超时与重试语义
2. ✅ **并发**：互斥锁/死锁/竞态、线程池/协程、背压
3. ✅ **缓存/存储**：缓存命中、穿透/雪崩、读写路径
4. ✅ **性能指标**：吞吐、延迟、p95/p99、容量估算、瓶颈定位

### 过线 = LeetCode / SQL

**LC**：稳定写出中等题、不卡边界/复杂度  
**SQL**：只有投数据岗才加大比重

---

## 📅 1个月训练计划（4周）

### Week 1｜可运营闭环（SLO + 可观测 + 低风险发布）

**目标**：建立"把 LLM 变成可控系统"的思维

---

### Day 1-2：SLO + Error Budget（用错误预算管理风险）

#### Day 1：为 KYC 项目定义 SLO 和 Error Budget（实战）

**实战动作**：

1. **在 KYC 项目中定义 SLO**
   - **打开**：`src/pipeline.py` 和 `src/errors.py`
   - **定义 SLO**：
     - 成功率：99%（SLO）
     - p95 延迟：< 15 秒（SLO）
     - 错误率：< 1%（SLO）
   - **计算 Error Budget**：
     - Error Budget = 100% - SLO = 1%（成功率）
     - 月度 Error Budget = 1000 文档中允许 10 个失败
   - **评估当前状态**：
     - 成功率：95%（PoV 阶段）
     - Error Budget 消耗：500%（严重超标）
     - **决策**：冻结发布，专注稳定性修复

2. **在代码中实现 Error Budget 检查**
   - **创建**：`src/error_budget.py`
   - **实现**：
     ```python
     class ErrorBudgetPolicy:
         def __init__(self, slo_percent=99.0):
             self.slo = slo_percent
             self.error_budget = 100.0 - slo_percent
         
         def can_release(self, current_success_rate):
             """基于 Error Budget 决定是否允许发布"""
             budget_remaining = current_success_rate - self.slo
             if budget_remaining > 50:
                 return "normal"  # 可以快速发布
             elif budget_remaining > 25:
                 return "warning"  # 限制高风险发布
             else:
                 return "freeze"  # 冻结发布
     ```
   - **集成**：在 `main.py` 中调用，每次 batch 后检查

3. **在 `_summary.json` 中记录 Error Budget 状态**
   - **修改**：`src/io_utils.py`
   - **添加**：Error Budget 状态字段到 `_summary.json`

**输出**：
- ✅ `src/error_budget.py`（Error Budget Policy 实现）
- ✅ `KYC_SLO_ERROR_BUDGET.md`（SLO 定义、Error Budget 计算、决策机制）
- ✅ 更新 `_summary.json` 结构（包含 Error Budget 状态）

**面试表达模板**：
```
"我为 KYC 项目设定了 99% 的成功率 SLO，并建立了 Error Budget 机制。
当错误预算剩余 > 50% 时，我们继续快速发布；当 < 25% 时，我们冻结发布，专注稳定性修复。
我在代码中实现了 Error Budget Policy，每次 batch 后自动检查，决定是否允许发布。"
```

#### Day 2：Canary 发布 + 自动回滚（实战）

**实战动作**：

1. **在 KYC 项目中实现 Feature Flag 机制**
   - **创建**：`src/feature_flags.py`
   - **实现**：
     ```python
     class FeatureFlags:
         def __init__(self):
             self.flags = {
                 "model_version": {"enabled": False, "percentage": 0},
                 "prompt_version": {"enabled": False, "percentage": 0},
                 "validator_strictness": {"enabled": False, "percentage": 0}
             }
         
         def should_use_feature(self, flag_name, request_id):
             """基于百分比决定是否使用新特性"""
             flag = self.flags.get(flag_name, {})
             if not flag.get("enabled"):
                 return False
             # 简单的哈希取模决定百分比
             hash_val = hash(request_id) % 100
             return hash_val < flag.get("percentage", 0)
     ```
   - **集成**：在 `src/pipeline.py` 中调用，根据 Feature Flag 选择模型/prompt

2. **实现 Canary 发布的观察和回滚机制**
   - **创建**：`src/canary_monitor.py`
   - **实现**：
     ```python
     class CanaryMonitor:
         def check_canary_metrics(self, old_metrics, new_metrics):
             """检查 Canary 指标是否达标"""
             # Schema Fail Rate 检查
             if new_metrics["schema_fail_rate"] > old_metrics["schema_fail_rate"] * 2:
                 return "rollback"  # Schema Fail × 2 立即回滚
             
             # p95 Latency 检查
             if new_metrics["p95_latency"] > old_metrics["p95_latency"] * 1.2:
                 return "rollback"  # p95 + 20% 立即回滚
             
             # Error Rate 检查
             if new_metrics["error_rate"] > 0.05:
                 return "rollback"  # Error Rate > 5% 立即回滚
             
             return "continue"  # 继续 Canary
     ```
   - **集成**：在 batch processing 后调用，自动决定是否回滚

3. **实现自动回滚机制**
   - **创建**：`src/rollback.py`
   - **实现**：
     ```python
     class RollbackManager:
         def rollback_feature(self, flag_name):
             """回滚 Feature Flag 到 0%"""
             flag = self.flags.get(flag_name, {})
             flag["percentage"] = 0
             flag["enabled"] = False
             # 保存到配置文件
             self.save_flags()
     ```
   - **集成**：在 Canary Monitor 触发回滚时自动调用

**输出**：
- ✅ `src/feature_flags.py`（Feature Flag 实现）
- ✅ `src/canary_monitor.py`（Canary 观察和回滚检查）
- ✅ `src/rollback.py`（自动回滚机制）
- ✅ `KYC_CANARY_ROLLBACK.md`（Canary 流程、回滚条件、实现细节）

**面试表达模板**：
```
"我实现了 Canary 发布机制。新版本先在 1% 流量上验证，每步观察 L0/L1 指标。
如果 Schema Fail Rate 增加 2 倍或 p95 增加 20%，系统会自动回滚到旧版本。
我在代码中实现了 Feature Flags 和 Canary Monitor，可以在生产环境自动控制发布流程。"
```

---

### Day 3-4：可观测性（Metrics/Logs/Traces）

#### Day 3：为 KYC 项目实现 Metrics 收集（实战）

**实战动作**：

1. **在 KYC 项目中实现 Metrics 收集**
   - **创建**：`src/metrics_collector.py`
   - **实现**：
     ```python
     class MetricsCollector:
         def __init__(self):
             self.metrics = {
                 "requests_total": 0,
                 "requests_success": 0,
                 "requests_fail": 0,
                 "latencies": [],  # 用于计算 p95/p99
                 "schema_fails": 0,
                 "rate_limits": 0
             }
         
         def record_request(self, status, latency_ms):
             """记录请求指标"""
             self.metrics["requests_total"] += 1
             if status == "success":
                 self.metrics["requests_success"] += 1
                 self.metrics["latencies"].append(latency_ms)
             else:
                 self.metrics["requests_fail"] += 1
         
         def calculate_percentiles(self):
             """计算 p50/p95/p99"""
             if len(self.metrics["latencies"]) < 10:
                 return None
             sorted_latencies = sorted(self.metrics["latencies"])
             p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
             p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
             p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
             return {"p50": p50, "p95": p95, "p99": p99}
     ```
   - **集成**：在 `src/pipeline.py` 中调用，记录每个请求的指标

2. **实现 Metrics 导出（JSON 格式）**
   - **修改**：`src/io_utils.py`
   - **添加**：Metrics 导出到 `_summary.json` 的 `metrics` 字段

**输出**：
- ✅ `src/metrics_collector.py`（Metrics 收集实现）
- ✅ 更新 `_summary.json` 结构（包含 Metrics 字段）
- ✅ `KYC_METRICS.md`（Metrics 设计、收集方式、导出格式）

**面试表达模板**：
```
"我实现了 Metrics 收集机制。在 pipeline 中记录每个请求的状态、延迟、错误类型等指标。
batch 处理后自动计算 p50/p95/p99 延迟，并导出到 _summary.json。
这样可以在生产环境实时监控系统健康状态。"
```

#### Day 4：结构化日志 + Trace ID（实战）

**实战动作**：

1. **在 KYC 项目中实现结构化日志**
   - **创建**：`src/structured_logger.py`
   - **实现**：
     ```python
     import logging
     import json
     import uuid
     
     class StructuredLogger:
         def __init__(self):
             self.logger = logging.getLogger(__name__)
             # 配置 JSON 格式
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(message)s')
             handler.setFormatter(formatter)
             self.logger.addHandler(handler)
             self.logger.setLevel(logging.INFO)
         
         def log_request(self, trace_id, request_id, status, latency_ms, error_code=None):
             """记录结构化日志"""
             log_entry = {
                 "timestamp": datetime.now().isoformat(),
                 "trace_id": trace_id,
                 "request_id": request_id,
                 "status": status,
                 "latency_ms": latency_ms,
                 "error_code": error_code
             }
             # 注意：不记录 PII（base64 image、prompt content、extracted PII）
             self.logger.info(json.dumps(log_entry))
     ```
   - **集成**：在 `src/pipeline.py` 中调用，记录每个请求的日志

2. **实现 Trace ID 生成和传递**
   - **修改**：`src/pipeline.py`
   - **实现**：为每个请求生成 `trace_id`，在所有阶段传递

3. **实现 Trace ID 关联（日志聚合）**
   - **创建**：`scripts/aggregate_traces.py`
   - **实现**：从日志中提取 `trace_id`，关联所有相关日志

**输出**：
- ✅ `src/structured_logger.py`（结构化日志实现）
- ✅ 更新 `src/pipeline.py`（Trace ID 生成和传递）
- ✅ `scripts/aggregate_traces.py`（Trace ID 关联脚本）
- ✅ `KYC_OBSERVABILITY.md`（结构化日志设计、Trace ID 关联、PII 脱敏）

**面试表达模板**：
```
"我实现了结构化日志和 Trace ID 机制。每个请求生成唯一的 trace_id，
在所有处理阶段传递。日志采用 JSON 格式，包含 trace_id、request_id、status、latency_ms 等字段。
注意：我们不记录 PII（base64 image、prompt content、extracted PII），只记录 trace_id。
这样可以在生产环境快速定位问题，同时保护用户隐私。"
```

---

### Week 2｜测试 + 可维护性（质量）

**目标**：建立"改动不会把系统搞坏"的机制

---

### Day 5-6：测试金字塔 + 回归门禁（实战）

#### Day 5：为 KYC 项目实现测试金字塔（实战）

**实战动作**：

1. **完善 Unit Tests（70%）**
   - **修改**：`tests/test_rules.py`
   - **添加**：
     - 测试 expiry check 逻辑
     - 测试 review decision 逻辑
     - 测试 fraud markers 生成
   - **运行**：`pytest tests/test_rules.py -v`

2. **添加 Integration Tests（20%）**
   - **创建**：`tests/test_pipeline_integration.py`
   - **实现**：
     ```python
     def test_pipeline_e2e_flow():
         """测试 E2E 流程：preprocess → api → validation → rules"""
         # 模拟输入
         test_image = load_test_image("test_data/sample_id.jpg")
         
         # 运行 pipeline
         result = pipeline.process_document(test_image, trace_id="test_trace_001")
         
         # 验证结果
         assert result["status"] == "success"
         assert "trace_id" in result
         assert "extracted_fields" in result
     ```
   - **运行**：`pytest tests/test_pipeline_integration.py -v`

3. **添加 E2E Tests（10% - Golden Set）**
   - **创建**：`tests/test_golden_set.py`
   - **实现**：
     ```python
     GOLDEN_SET = [
         {"file": "test_data/normal_001.jpg", "expected": {...}},
         {"file": "test_data/edge_001.jpg", "expected": {...}},
         {"file": "test_data/anomaly_001.jpg", "expected": {...}}
     ]
     
     def test_golden_set():
         """测试 Golden Set（50-200 条测试用例）"""
         for case in GOLDEN_SET:
             result = pipeline.process_document(case["file"])
             assert result["status"] == case["expected"]["status"]
             # 验证关键字段
     ```
   - **运行**：`pytest tests/test_golden_set.py -v`

**输出**：
- ✅ 更新 `tests/test_rules.py`（完善 Unit Tests）
- ✅ `tests/test_pipeline_integration.py`（Integration Tests）
- ✅ `tests/test_golden_set.py`（E2E Tests - Golden Set）
- ✅ `KYC_TESTING_PYRAMID.md`（测试金字塔结构、覆盖范围、运行方式）

**面试表达模板**：
```
"我实现了测试金字塔（70/20/10）。70% 是 Unit Tests（测试确定性规则和 Schema 验证），
20% 是 Integration Tests（测试 E2E 流程），10% 是 E2E Tests（Golden Set，50-200 条测试用例）。
这样平衡了测试成本和覆盖范围。"
```

#### Day 6：回归门禁（实战）

**实战动作**：

1. **实现回归门禁检查脚本**
   - **创建**：`scripts/regression_gate.py`
   - **实现**：
     ```python
     def check_regression_gate(golden_set_results):
         """检查是否通过回归门禁"""
         thresholds = {
             "schema_pass_rate": 0.95,  # Schema Pass Rate > 95%
             "field_accuracy": 0.90,    # 字段准确率 > 90%
             "fallback_rate": 0.05,      # Fallback Rate < 5%
             "cost_per_request": 0.002   # 成本 < $0.002 / request
         }
         
         # 计算指标
         metrics = calculate_metrics(golden_set_results)
         
         # 检查阈值
         if metrics["schema_pass_rate"] < thresholds["schema_pass_rate"]:
             return False, "Schema Pass Rate below threshold"
         if metrics["field_accuracy"] < thresholds["field_accuracy"]:
             return False, "Field Accuracy below threshold"
         # ... 其他检查
         
         return True, "All gates passed"
     ```
   - **集成**：在 CI/CD 中调用，发布前必须通过

2. **在 CI/CD 中集成回归门禁**
   - **创建**：`.github/workflows/regression_gate.yml`
   - **实现**：
     ```yaml
     name: Regression Gate
     on: [pull_request]
     jobs:
       regression-gate:
         runs-on: ubuntu-latest
         steps:
           - uses: actions/checkout@v3
           - name: Run Golden Set
             run: pytest tests/test_golden_set.py
           - name: Check Regression Gate
             run: python scripts/regression_gate.py
     ```

**输出**：
- ✅ `scripts/regression_gate.py`（回归门禁检查脚本）
- ✅ `.github/workflows/regression_gate.yml`（CI/CD 集成）
- ✅ `KYC_REGRESSION_GATE.md`（回归门禁设计、阈值、CI/CD 集成）

**面试表达模板**：
```
"我实现了回归门禁机制。每次发布前，系统会自动运行 Golden Set，
只有所有指标通过阈值（Schema Pass Rate > 95%、准确率 > 90%）才能发布。
我在 CI/CD 中集成了回归门禁，发布前必须通过检查。"
```

---

### Day 7-8：Code Review + 可维护性（实战）

#### Day 7：为 KYC 项目制定 Code Review 标准（实战）

**实战动作**：

1. **创建 Code Review 检查清单**
   - **创建**：`docs/CODE_REVIEW_CHECKLIST.md`
   - **内容**：
     - ✅ 代码复杂度 < 10（圈复杂度）
     - ✅ 代码重复率 < 5%
     - ✅ 所有公共 API 都有文档字符串
     - ✅ 错误处理覆盖边界情况
     - ✅ 日志记录（结构化、PII 脱敏）

2. **使用工具自动化检查**
   - **创建**：`.pre-commit-config.yaml`
   - **实现**：
     ```yaml
     repos:
       - repo: local
         hooks:
           - id: complexity-check
             name: Check Code Complexity
             entry: python scripts/check_complexity.py
           - id: duplicate-check
             name: Check Code Duplication
             entry: python scripts/check_duplication.py
     ```

**输出**：
- ✅ `docs/CODE_REVIEW_CHECKLIST.md`（Code Review 标准）
- ✅ `.pre-commit-config.yaml`（自动化检查配置）
- ✅ `KYC_CODE_REVIEW.md`（Code Review 流程、标准、工具）

**面试表达模板**：
```
"我建立了 Code Review 标准和流程。我们关注 Code Health（代码复杂度 < 10、重复率 < 5%），
不只是找 Bug。我们使用自动化工具检查代码质量和复杂度，确保所有代码都符合标准。"
```

#### Day 8：可维护性指标（实战）

**实战动作**：

1. **实现代码复杂度检查脚本**
   - **创建**：`scripts/check_complexity.py`
   - **实现**：
     ```python
     import radon
     from radon.complexity import cc_visit
     
     def check_complexity(file_path, threshold=10):
         """检查代码复杂度"""
         with open(file_path) as f:
             code = f.read()
         complexity = cc_visit(code)
         for func in complexity:
             if func.complexity > threshold:
                 print(f"WARNING: {func.name} complexity {func.complexity} > {threshold}")
     ```

2. **实现代码重复率检查脚本**
   - **创建**：`scripts/check_duplication.py`
   - **实现**：使用 `pylint` 或 `flake8` 检查代码重复

3. **生成可维护性报告**
   - **创建**：`scripts/generate_maintainability_report.py`
   - **实现**：汇总复杂度、重复率、测试覆盖率等指标

**输出**：
- ✅ `scripts/check_complexity.py`（代码复杂度检查）
- ✅ `scripts/check_duplication.py`（代码重复率检查）
- ✅ `scripts/generate_maintainability_report.py`（可维护性报告）
- ✅ `KYC_MAINTAINABILITY.md`（可维护性指标、工具、报告）

**面试表达模板**：
```
"我建立了可维护性指标体系。通过代码复杂度检查（< 10）、重复率检查（< 5%）、
测试覆盖率（> 70%），我们将新同学的上手时间从 5 天降低到 2 天。"
```

---

### Week 3｜工程效率 + 影响力（放大）

**目标**：用数据说话 + 成为"团队放大器"

---

### Day 9-10：DORA / Four Keys 指标（实战）

#### Day 9：为 KYC 项目计算 DORA 指标（实战）

**实战动作**：

1. **实现 DORA 指标计算脚本**
   - **创建**：`scripts/calculate_dora_metrics.py`
   - **实现**：
     ```python
     def calculate_dora_metrics():
         """计算 DORA / Four Keys 指标"""
         # 从 Git 历史计算部署频率
         deployment_frequency = calculate_deployment_frequency()
         
         # 从 Git 历史计算变更前置时间
         lead_time = calculate_lead_time()
         
         # 从 _summary.json 计算变更失败率
         change_failure_rate = calculate_change_failure_rate()
         
         # 从事故记录计算 MTTR
         mttr = calculate_mttr()
         
         return {
             "deployment_frequency": deployment_frequency,  # 每周几次
             "lead_time": lead_time,  # 从提交到生产多久
             "change_failure_rate": change_failure_rate,  # 变更失败率
             "mttr": mttr  # 平均恢复时间
         }
     ```

2. **实现变更失败率计算**
   - **修改**：`scripts/calculate_dora_metrics.py`
   - **实现**：从 `_summary.json` 和历史记录计算变更失败率

**输出**：
- ✅ `scripts/calculate_dora_metrics.py`（DORA 指标计算脚本）
- ✅ `KYC_DORA_METRICS.md`（DORA 指标、当前值、目标值、改进计划）

**面试表达模板**：
```
"我关注的是 DORA / Four Keys 指标：部署频率（每周 1 次）、变更前置时间（2-3 天）、
变更失败率（3%）、恢复时间（2 小时）。通过引入自动化回归门禁和可观测性，
我们将变更失败率从 15% 降低到 3%，MTTR 从 4 小时降低到 2 小时。"
```

#### Day 10：工程效率改进计划（实战）

**实战动作**：

1. **分析工程效率瓶颈**
   - **创建**：`docs/ENGINEERING_EFFICIENCY_ANALYSIS.md`
   - **内容**：
     - 部署频率瓶颈（手动部署 → CI/CD 自动化）
     - 变更前置时间瓶颈（测试时间长 → 并行测试）
     - 变更失败率瓶颈（缺少回归门禁 → 添加门禁）
     - 恢复时间瓶颈（缺少可观测性 → 添加 Metrics/Logs/Traces）

2. **制定改进计划**
   - **创建**：`docs/IMPROVEMENT_PLAN.md`
   - **内容**：
     - 部署频率：从每周 1 次 → 每天 1 次（CI/CD 自动化）
     - 变更前置时间：从 2-3 天 → < 1 天（并行测试、发布门禁）
     - 变更失败率：从 3% → < 1%（回归门禁、Canary 发布）
     - 恢复时间：从 2 小时 → < 1 小时（可观测性、Runbook）

**输出**：
- ✅ `docs/ENGINEERING_EFFICIENCY_ANALYSIS.md`（工程效率瓶颈分析）
- ✅ `docs/IMPROVEMENT_PLAN.md`（改进计划）
- ✅ `KYC_ENGINEERING_EFFICIENCY.md`（工程效率指标、瓶颈、改进计划）

**面试表达模板**：
```
"我分析了工程效率瓶颈，并制定了改进计划。通过引入 CI/CD 自动化、并行测试、回归门禁、
可观测性，我们将部署频率从每周 1 次提升到每天 1 次，变更失败率从 15% 降低到 3%。"
```

---

### Day 11-12：影响力 + 跨团队协作（实战）

#### Day 11：为 KYC 项目写跨团队协作案例（实战）

**实战动作**：

1. **准备跨团队协作案例**
   - **创建**：`docs/CROSS_TEAM_COLLABORATION.md`
   - **案例 1**：推动统一的 Schema 契约
     - **问题**：多个团队使用不同的 Schema 格式，联调时间很长
     - **方案**：推动统一的 Schema 格式（Pydantic models + JSON schema）
     - **结果**：下游团队联调时间减少 50%
   - **案例 2**：跨团队技术选型
     - **问题**：不同团队使用不同的验证库
     - **方案**：推动统一使用 Pydantic
     - **结果**：验证代码减少 50%

2. **量化影响力量化**
   - **创建**：`docs/IMPACT_METRICS.md`
   - **内容**：
     - 联调时间减少 50%
     - 验证代码减少 50%
     - 团队效率提升 30%

**输出**：
- ✅ `docs/CROSS_TEAM_COLLABORATION.md`（跨团队协作案例）
- ✅ `docs/IMPACT_METRICS.md`（影响力量化）
- ✅ `KYC_CROSS_TEAM_COLLABORATION.md`（跨团队协作、影响力量化）

**面试表达模板**：
```
"我在跨团队协作中扮演'胶水'角色。通过推动统一的 Schema 契约（Pydantic models + JSON schema），
我减少了下游团队 50% 的联调时间。另外，通过推动统一使用 Pydantic，
我减少了 50% 的验证代码。这样不仅提升了效率，还传播了最佳实践。"
```

#### Day 12：Mentor/Sponsor 案例（实战）

**实战动作**：

1. **准备 Mentor 案例**
   - **创建**：`docs/MENTOR_CASES.md`
   - **案例**：指导新同学
     - **场景**：新同学不知道如何设计 Schema
     - **方法**：手把手教他使用 Pydantic，分享最佳实践
     - **结果**：新同学 2 周内独立完成 Schema 设计

2. **准备 Sponsor 案例**
   - **案例**：支持下属的技术决策
     - **场景**：下属提出使用新的验证库
     - **方法**：支持他的决策，帮助他推动落地
     - **结果**：验证代码减少 30%，团队效率提升

**输出**：
- ✅ `docs/MENTOR_CASES.md`（Mentor/Sponsor 案例）
- ✅ `KYC_MENTOR_SPONSOR.md`（Mentor/Sponsor 案例、影响力量化）

**面试表达模板**：
```
"我通过 Mentor 新同学，帮助他们快速上手。通过手把手教他们使用 Pydantic 和分享最佳实践，
新同学的上手时间从 5 天降低到 2 周内能独立完成设计。另外，我通过 Sponsor 下属的技术决策，
支持他们推动新的验证库落地，验证代码减少 30%，团队效率提升。"
```

---

### Week 4｜产品思维 + 安全 + 设计能力（深度）

**目标**：能说清"为什么做" + 处理安全隐私 + 写出 Design Doc

---

### Day 13-14：Trade-off + ROI 分析（实战）

#### Day 13：为 KYC 项目写 Trade-off 分析（实战）

**实战动作**：

1. **为 KYC 项目写 Trade-off 分析文档**
   - **创建**：`docs/TRADE_OFF_ANALYSIS.md`
   - **Trade-off 1**：准确性 vs 成本
     - **方案 A**：使用大模型（Qwen2.5-72B，成本高，准确率高）
     - **方案 B**：使用小模型（Qwen2.5-7B，成本低，准确率低）
     - **决策**：选择方案 A（高准确率）+ 降级策略（低成本备份）
     - **理由**：KYC 是强监管领域，准确性优先，但用降级策略控制成本

2. **为 KYC 项目写 ROI 分析文档**
   - **创建**：`docs/ROI_ANALYSIS.md`
   - **投入**：
     - 开发时间：2 个月
     - 人力成本：2 人 × 2 月 = 4 人月
   - **收益**：
     - 每单节省人审时间：5 分钟 → 3-5 秒（节省 99%）
     - 成本节省：$7.5 / request → $0.0015 / request（节省 99.98%）
     - 自动化率：60-70%（减少人工审核工作量）
   - **ROI**：投入 4 人月，节省 $7500/月（人力成本）+ 降低合规风险

**输出**：
- ✅ `docs/TRADE_OFF_ANALYSIS.md`（Trade-off 分析）
- ✅ `docs/ROI_ANALYSIS.md`（ROI 分析）
- ✅ `KYC_TRADE_OFF_ROI.md`（Trade-off 分析、ROI 分析、决策理由）

**面试表达模板**：
```
"我们在准确性 vs 成本之间做 Trade-off。虽然方案 B 理论成本低 50%，但它引入了非标准依赖，
会增加团队未来的维护成本。在当前阶段，我选择了更稳健、生态更好的方案 A（大模型 + 降级策略），
既保证了准确性（强监管要求），又控制了成本（降级策略）。"

"我们量化了 KYC 项目的 ROI。投入 4 人月，实现了每单节省人审时间 99%（5 分钟 → 3-5 秒）、
成本节省 99.98%（$7.5 → $0.0015 / request）。这带来了 $7500/月的成本节省，同时降低了合规风险。"
```

---

### Day 15-16：安全 + 隐私（实战）

#### Day 15：为 KYC 项目做 Threat Model（实战）

**实战动作**：

1. **为 KYC 项目写 Threat Model 文档**
   - **创建**：`docs/THREAT_MODEL.md`
   - **威胁识别**：
     - 数据泄露（PII 数据被非法访问）
     - 数据篡改（提取结果被修改）
     - 服务攻击（DDoS、API 滥用）
   - **防护措施**：
     - 数据加密（PII 数据在入库前加密）
     - 访问控制（最小权限原则）
     - 审计日志（所有访问都有 trace_id）

2. **实现最小权限检查脚本**
   - **创建**：`scripts/check_permissions.py`
   - **实现**：检查代码中是否有不必要的权限

**输出**：
- ✅ `docs/THREAT_MODEL.md`（Threat Model、防护措施）
- ✅ `scripts/check_permissions.py`（最小权限检查）
- ✅ `KYC_SECURITY_THREAT_MODEL.md`（Threat Model、最小权限、防护措施）

**面试表达模板**：
```
"我们在设计之初就坚持最小权限原则。所有 PII 数据在入库前必须加密，
只有授权人员才能访问。所有访问都有 trace_id 审计日志。
通过这种方式，我们降低了攻击面，满足了合规要求。"
```

#### Day 16：PII 处理与合规（实战）

**实战动作**：

1. **实现 PII 脱敏检查脚本**
   - **创建**：`scripts/check_pii_leakage.py`
   - **实现**：
     ```python
     def check_pii_leakage(log_file):
         """检查日志中是否有 PII 泄漏"""
         pii_keywords = ["base64", "prompt", "full_name", "date_of_birth", "document_number"]
         with open(log_file) as f:
             for line in f:
                 for keyword in pii_keywords:
                     if keyword in line.lower():
                         print(f"WARNING: Potential PII leakage: {keyword}")
     ```

2. **更新日志记录，确保 PII 脱敏**
   - **修改**：`src/structured_logger.py`
   - **实现**：确保不记录 PII（base64 image、prompt content、extracted PII）

**输出**：
- ✅ `scripts/check_pii_leakage.py`（PII 泄漏检查）
- ✅ 更新 `src/structured_logger.py`（PII 脱敏）
- ✅ `KYC_PII_HANDLING.md`（PII 识别、保护、合规要求）

**面试表达模板**：
```
"我们严格遵守 PII 处理规范。在设计之初，我们就明确了 PII 字段（姓名、出生日期、证件号等），
并实现了日志脱敏（Never log base64 image、prompt content、extracted PII）。
所有 PII 数据在入库前必须加密，只有授权人员才能访问。
所有 PII 访问都有 trace_id 审计日志，满足合规要求。"
```

---

### Day 17-18：Design Doc 写作（实战）

#### Day 17：为 KYC 项目写 Design Doc（实战）

**实战动作**：

1. **为 KYC 项目写完整的 Design Doc**
   - **创建**：`docs/KYC_DESIGN_DOC.md`
   - **结构**：
     - **Goals & Non-Goals**（我们要解决什么，不解决什么）
     - **System Architecture**（带有监控和失败处理的架构）
     - **Rollout Plan**（如何灰度、如何回滚）
     - **Security/Privacy**（如何处理医疗敏感数据）
     - **Trade-offs**（准确性 vs 成本、功能 vs 稳定性）

2. **使用 Mermaid 画架构图**
   - **创建**：`docs/KYC_ARCHITECTURE.md`
   - **实现**：用 Mermaid 画出系统架构图

**输出**：
- ✅ `docs/KYC_DESIGN_DOC.md`（完整的 Design Doc）
- ✅ `docs/KYC_ARCHITECTURE.md`（系统架构图）
- ✅ `KYC_DESIGN_DOC.md`（Design Doc、架构图、Trade-offs）

**面试表达模板**：
```
"我们坚持先写 Design Doc 再开工。Design Doc 包含：
- Goals & Non-Goals（明确解决的问题和不解决的问题）
- System Architecture（带有监控和失败处理的架构）
- Rollout Plan（Canary 发布、回滚策略）
- Security/Privacy（PII 处理、合规要求）

通过这种方式，我们在开工前就对齐了团队，降低了返工风险。"
```

#### Day 18：Design Doc 评审与迭代（实战）

**实战动作**：

1. **创建 Design Doc 评审模板**
   - **创建**：`docs/DESIGN_REVIEW_TEMPLATE.md`
   - **内容**：
     - 评审清单（架构、Trade-offs、风险、迁移与回滚）
     - 评审流程（谁来评审、评审标准）
     - 评审记录（发现的问题、改进方案）

2. **准备 Design Doc 评审案例**
   - **创建**：`docs/DESIGN_REVIEW_CASES.md`
   - **案例**：Design Doc 评审中发现的问题
     - **问题**：初始设计缺少降级策略
     - **改进**：补充降级策略（小模型/规则/转人工）
     - **结果**：避免了生产环境的事故

**输出**：
- ✅ `docs/DESIGN_REVIEW_TEMPLATE.md`（Design Doc 评审模板）
- ✅ `docs/DESIGN_REVIEW_CASES.md`（Design Doc 评审案例）
- ✅ `KYC_DESIGN_REVIEW.md`（Design Doc 评审、案例、迭代）

**面试表达模板**：
```
"我们建立了 Design Doc 评审流程。在评审中，我们发现初始设计缺少降级策略。
通过补充降级策略（小模型/规则/转人工），我们避免了生产环境的事故。
Design Doc 评审不仅发现了设计问题，还帮助团队对齐，降低了返工风险。"
```

---

## 🔧 基础八股实战补全（Week 2-3 穿插）

**不是"学理论"，而是"用项目实战基础"**

---

### 网络基础（Day 3 穿插）

**实战动作**：

1. **在 KYC 项目中理解超时与重试**
   - **打开**：`src/fw_client.py`
   - **理解**：为什么设置 `API_TIMEOUT=60` 秒？
   - **理解**：为什么重试时使用指数退避？

2. **在代码中实现连接复用**
   - **修改**：`src/fw_client.py`
   - **实现**：使用 `requests.Session()` 实现连接复用

**输出**：
- ✅ `KYC_NETWORK_NOTES.md`（超时、重试、连接复用、面试表达）

---

### 并发基础（Day 4 穿插）

**实战动作**：

1. **在 KYC 项目中理解并发处理**
   - **打开**：`main.py`
   - **理解**：为什么使用 `concurrent.futures.ThreadPoolExecutor`？
   - **理解**：如何控制并发数量？

2. **实现背压机制**
   - **修改**：`src/rate_limiter.py`
   - **实现**：当队列满时，阻塞新请求

**输出**：
- ✅ `KYC_CONCURRENCY_NOTES.md`（并发、背压、面试表达）

---

### 缓存基础（Day 7 穿插）

**实战动作**：

1. **在 KYC 项目中理解缓存策略**
   - **思考**：什么时候需要缓存？（重复请求、稳定结果）
   - **思考**：如何防止缓存雪崩？（随机过期时间）

2. **实现幂等性检查**
   - **修改**：`src/pipeline.py`
   - **实现**：使用 `request_id` 去重，避免重复处理

**输出**：
- ✅ `KYC_CACHE_NOTES.md`（缓存策略、幂等性、面试表达）

---

## 🎯 每日检查清单

### 每天必须回答的三个问题

**1. "如果这个系统今天半夜挂了，我如何能不被叫醒（自愈/告警）？"**

**答案要点**：
- ✅ Error Budget + Canary 发布（自动回滚）
- ✅ 告警围绕 SLO（可行动告警）
- ✅ 可观测性（Metrics/Logs/Traces，快速定位）

**2. "如果我要把这个项目交给一个新来的同学，他需要多久能上手（可维护性）？"**

**答案要点**：
- ✅ 代码可读性（代码复杂度 < 10、代码重复率 < 5%）
- ✅ 文档完整性（所有公共 API 都有文档）
- ✅ 测试覆盖率（> 70% Unit Tests）
- ✅ 目标：新同学上手时间 < 2 天

**3. "如果公司要缩减一半算力成本，我的系统如何活下去（Trade-off/降级）？"**

**答案要点**：
- ✅ 降级策略（小模型/规则/转人工）
- ✅ 成本优化（模型量化、缓存优化、Batch 优化）
- ✅ Trade-off 分析（牺牲什么换来什么）

---

## 📋 1个月训练检查清单

### Week 1｜可运营闭环（代码实现）
- [ ] Day 1：SLO + Error Budget（`src/error_budget.py` + `KYC_SRE_SLO_ERROR_BUDGET.md`）
- [ ] Day 2：Canary 发布 + 自动回滚（`src/feature_flags.py` + `src/canary_monitor.py` + `KYC_CANARY_ROLLBACK.md`）
- [ ] Day 3：Metrics 收集（`src/metrics_collector.py` + `KYC_METRICS.md`）
- [ ] Day 4：结构化日志 + Trace ID（`src/structured_logger.py` + `KYC_OBSERVABILITY.md`）

### Week 2｜测试 + 可维护性（代码实现）
- [ ] Day 5：测试金字塔（`tests/test_rules.py` + `tests/test_pipeline_integration.py` + `tests/test_golden_set.py` + `KYC_TESTING_PYRAMID.md`）
- [ ] Day 6：回归门禁（`scripts/regression_gate.py` + `.github/workflows/regression_gate.yml` + `KYC_REGRESSION_GATE.md`）
- [ ] Day 7：Code Review 标准（`.pre-commit-config.yaml` + `docs/CODE_REVIEW_CHECKLIST.md` + `KYC_CODE_REVIEW.md`）
- [ ] Day 8：可维护性指标（`scripts/check_complexity.py` + `scripts/check_duplication.py` + `KYC_MAINTAINABILITY.md`）

### Week 3｜工程效率 + 影响力（脚本实现 + 文档）
- [ ] Day 9：DORA 指标计算（`scripts/calculate_dora_metrics.py` + `KYC_DORA_METRICS.md`）
- [ ] Day 10：工程效率改进计划（`docs/ENGINEERING_EFFICIENCY_ANALYSIS.md` + `KYC_ENGINEERING_EFFICIENCY.md`）
- [ ] Day 11：跨团队协作（`docs/CROSS_TEAM_COLLABORATION.md` + `KYC_CROSS_TEAM_COLLABORATION.md`）
- [ ] Day 12：Mentor/Sponsor（`docs/MENTOR_CASES.md` + `KYC_MENTOR_SPONSOR.md`）

### Week 4｜产品思维 + 安全 + 设计（文档 + 脚本）
- [ ] Day 13：Trade-off + ROI 分析（`docs/TRADE_OFF_ANALYSIS.md` + `docs/ROI_ANALYSIS.md` + `KYC_TRADE_OFF_ROI.md`）
- [ ] Day 14：Threat Model（`docs/THREAT_MODEL.md` + `scripts/check_permissions.py` + `KYC_SECURITY_THREAT_MODEL.md`）
- [ ] Day 15：PII 处理（`scripts/check_pii_leakage.py` + `KYC_PII_HANDLING.md`）
- [ ] Day 16：Design Doc 写作（`docs/KYC_DESIGN_DOC.md` + `docs/KYC_ARCHITECTURE.md` + `KYC_DESIGN_DOC.md`）
- [ ] Day 17：Design Doc 评审（`docs/DESIGN_REVIEW_TEMPLATE.md` + `docs/DESIGN_REVIEW_CASES.md` + `KYC_DESIGN_REVIEW.md`）

### 基础八股（穿插在 Week 2-3）
- [ ] 网络基础（超时、重试、连接复用）- Day 3 穿插
- [ ] 并发基础（并发、背压）- Day 4 穿插
- [ ] 缓存基础（缓存策略、幂等性）- Day 7 穿插

---

## 🎯 核心要点

### 1. 不是"学技术"，而是"用项目实战技术"

**每个模块都要有代码实现，不是纯理论**

- ✅ 在 KYC 项目中实现 SLO/Error Budget
- ✅ 在 KYC 项目中实现 Canary 发布
- ✅ 在 KYC 项目中实现 Metrics 收集
- ✅ 在 KYC 项目中实现回归门禁

### 2. 每个模块都有可执行的输出物

**不是"看完文档就完事"，而是"有代码可以展示"**

- ✅ 代码实现（`src/error_budget.py`、`src/feature_flags.py`）
- ✅ 脚本工具（`scripts/regression_gate.py`、`scripts/calculate_dora_metrics.py`）
- ✅ CI/CD 集成（`.github/workflows/regression_gate.yml`）
- ✅ 文档输出（`KYC_XXX.md`）

### 3. 面试时可以展示实际代码

**不是"我学过"，而是"我做过"**

- ✅ 展示 `src/error_budget.py`：Error Budget Policy 实现
- ✅ 展示 `src/canary_monitor.py`：Canary 监控和回滚
- ✅ 展示 `scripts/regression_gate.py`：回归门禁检查
- ✅ 展示 `_summary.json`：包含 Metrics 和 Error Budget 状态

---

## 🚀 开始实战训练

**第 1 步**：从 Week 1 Day 1 开始，每天完成一个模块  
**第 2 步**：在 KYC 项目中实现代码，不是纯理论  
**第 3 步**：每天产出可执行的代码和文档  
**第 4 步**：准备面试表达，展示实际代码

**记住**：不是"学完所有内容"，而是"用项目实战技术"。

**加油！** 🎉
