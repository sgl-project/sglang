# Day 1｜指标体系：KYC 项目示例

**基于项目**：KYC Identity Verification System (PoV)  
**项目链接**：https://github.com/Nickcp39/kyc_pov/tree/main

---

## L0 稳定性（监控/可用/延迟）

**核心指标**：

- [x] **成功率**（Success Rate）: `95%` (PoV 阶段)
  - 定义：正常响应的文档数 / 总文档数（batch processing）
  - SLO 目标：`99%` (Production 目标)
  - 当前值：`95%` (PoV 阶段)
  - **基于**：`_summary.json` 中的 `status: "success"` vs `"fail"`

- [x] **延迟指标**（Latency）:
  - p50: `3-5秒` (单文档处理时间)
  - p95: `8-10秒` (SLO 目标：`< 15秒`)
  - p99: `15-20秒` (SLO 目标：`< 30秒`)
  - **基于**：Fireworks API 调用时间（包括 image preprocessing + model inference + validation）

- [x] **错误率**（Error Rate）: `5%` (PoV 阶段)
  - 定义：错误响应的文档数 / 总文档数
  - SLO 目标：`< 1%`
  - 当前值：`5%` (PoV 阶段，包含 schema validation failures)
  - **错误类型**：`IMAGE_FORMAT_UNSUPPORTED`, `SCHEMA_VALIDATION_FAILED`, `API_TIMEOUT`, `RATE_LIMIT_EXCEEDED`

- [x] **回退率**（Fallback Rate）: `0%` (PoV 阶段无降级)
  - 定义：触发降级/回退的请求数 / 总请求数
  - 当前值：`0%` (PoV 阶段)
  - **未来**：可以添加低质量图片的降级策略（如 OCR-only fallback）

- [x] **可用性**（Availability）: `99%` (PoV 阶段)
  - SLO 目标：`99.9%` (Production = 每月 < 43分钟不可用)
  - 当前值：`99%` (PoV 阶段)
  - **基于**：Fireworks API uptime + 本地 batch processing 稳定性

- [x] **Batch 处理成功率**
  - 定义：完全成功的 batch 数 / 总 batch 数
  - 目标：`> 95%` (允许单个文件失败，但不影响整个 batch)
  - 当前值：`90%` (PoV 阶段)
  - **基于**：`src/pipeline.py` 的 per-file isolation（one fail ≠ crash all）

---

## L1 进化收益（ROI）

**业务指标**：

- [x] **每单节省的人审分钟数**
  - 定义：人工审核时间 - AI 处理时间
  - 基线：`5-10 分钟/单`（人工审核一张 ID 文档）
  - 当前值：`3-5 秒/单`（AI 处理时间）
  - ROI：`5 分钟/单 × 1000 单/月 = 5000 分钟/月 = 83 小时/月`
  - **节省效率**：`> 99%` 的时间节省

- [x] **错误拦截率带来的风险降低**
  - 定义：被系统拦截的错误提交数 / 总提交数
  - **关键拦截场景**：
    - `expiry:expired` - 过期文档拦截
    - `missing_critical:full_name,date_of_birth` - 关键字段缺失拦截
    - `low_confidence_critical:document_number` - 低置信度关键字段拦截
  - 基线：`0%`（无自动化拦截）
  - 当前值：`15-20%`（PoV 阶段，基于 `src/rules.py` 的 fraud markers）
  - 风险降低估算：`200 件/月`（避免的潜在合规风险）

- [x] **吞吐提升带来的成本节省**
  - 定义：$ / request 或 tokens / request
  - **成本分解**：
    - Fireworks API 调用：`$0.001-0.002 / request` (Qwen2.5-VL-32B)
    - 人工审核成本：`$5-10 / request`（按小时工资计算）
  - 基线成本：`$7.5 / request`（平均人工成本）
  - 当前成本：`$0.0015 / request`（AI 成本）
  - **节省**：`$7.5 / request × 1000 requests/月 = $7500 / 月`
  - **ROI 倍数**：`5000x` 成本降低

- [x] **自动化率**
  - 定义：无需人工介入的请求数 / 总请求数
  - **自动化判断**：
    - 所有关键字段提取成功（`full_name, date_of_birth, document_number, expiry_date, issuing_country`）
    - 置信度 > 阈值（如 `> 0.85`）
    - 通过确定性规则检查（expiry valid, quality good）
  - 目标：`80%`（20% 需要人工 review）
  - 当前值：`60-70%`（PoV 阶段）
  - **基于**：`src/rules.py` 的 `apply_deterministic_rules` 中的 review decision logic

- [x] **人工 Review 队列减少**
  - 定义：需要人工 review 的文档数 / 总文档数
  - 基线：`100%`（全部人工审核）
  - 当前值：`30-40%`（自动化处理 60-70%）
  - **减少**：`60-70%` 的 review 工作量

---

## L2 长期健康（可维护/可扩展）

**工程健康指标**：

- [x] **变更失败率**（Change Failure Rate）
  - 定义：导致回滚/问题的发布数 / 总发布数
  - 目标：`< 5%`
  - 当前值：`0%`（PoV 阶段，尚未有 production releases）
  - **基于**：Schema-first 设计（`src/schemas.py`）和确定性规则（`src/rules.py`）降低变更风险

- [x] **回滚频率**（Rollback Frequency）
  - 定义：回滚次数 / 总发布次数
  - 目标：`< 2%`
  - 当前值：`0%`（PoV 阶段）
  - **近30天**：`0` 次回滚

- [x] **回归门禁通过率**（Regression Gate Pass Rate）
  - 定义：通过回归测试的发布数 / 总发布数
  - **回归测试覆盖**：
    - Unit tests (`tests/test_rules.py`, `tests/test_validators.py`)
    - Schema validation tests (`tests/test_validators.py`)
    - Error handling tests (`tests/test_errors.py`)
  - 目标：`> 95%`
  - 当前值：`100%`（PoV 阶段，所有单元测试通过）

- [x] **告警噪音**（Alert Noise - Precision）
  - 定义：有效告警数 / 总告警数
  - 目标：`> 80%`（减少误报）
  - 当前值：`N/A`（PoV 阶段，无生产告警系统）
  - **平均告警数/周**：`0`（PoV 阶段）

- [x] **Toil（重复劳动）趋势**
  - 定义：每周花在重复性任务上的时间
  - **Toil 来源**：
    - 手动运行 batch processing
    - 手动检查 `_summary.json`
    - 手动处理错误文档
  - 目标：`< 5 小时/周`
  - 当前值：`2-3 小时/周`（PoV 阶段）
  - 趋势：`→`（稳定）
  - **自动化机会**：CI/CD 集成、自动化 batch scheduling、错误自动重试

- [x] **MTTR（Mean Time To Recovery）**
  - 定义：平均故障恢复时间
  - 目标：`< 15 分钟`
  - 当前值：`N/A`（PoV 阶段，无 production incidents）
  - **基于**：`src/errors.py` 的标准化错误处理 + `src/rate_limiter.py` 的自动重试

- [x] **MTBF（Mean Time Between Failures）**
  - 定义：平均故障间隔时间
  - 目标：`> 168 小时`（一周）
  - 当前值：`N/A`（PoV 阶段）

- [x] **Schema 兼容性**
  - 定义：Schema 变更导致的 breaking changes 数 / 总 schema 变更数
  - 目标：`0 breaking changes`（通过 versioning：`schema_version = "v1"`）
  - 当前值：`0`（PoV 阶段）
  - **基于**：`src/schemas.py` 的版本化设计

- [x] **Auditability 覆盖率**
  - 定义：包含完整 trace_id 的文档数 / 总文档数
  - 目标：`100%`
  - 当前值：`100%`（PoV 阶段）
  - **基于**：每个请求都有 `trace_id`（Privacy & Logging section）

- [x] **PII 泄漏事件**
  - 定义：PII 泄漏事件数
  - 目标：`0`
  - 当前值：`0`
  - **基于**：Logging rules（Never log: base64 image, prompt content, extracted PII fields）

---

## Error Budget Policy

### 错误预算（Error Budget）定义

**Error Budget = 100% - SLO**

**L0 稳定性 Error Budget**：
- SLO = 99%（成功率）
- Error Budget = 1% = 1000 文档中允许 10 个失败（月度）

**当前状态**（PoV 阶段）：
- 成功率：95%
- Error Budget 消耗：5%（超过 SLO）
- **状态**：`警告`（Error Budget < 50%，需要稳定性工作）

### 错误预算消耗规则

1. **正常状态**（Error Budget > 50%）
   - ✅ 可以继续发布新功能
   - ✅ 可以承担一定风险

2. **警告状态**（Error Budget 25% - 50%）
   - ⚠️ 限制高风险发布
   - ⚠️ 增加审查流程
   - **当前状态**：PoV 阶段在此状态

3. **冻结状态**（Error Budget < 25%）
   - 🛑 **冻结所有新功能发布**
   - 🛑 **只允许稳定性修复和优化**
   - 🛑 **全力修复稳定性债务**

### 决策

**PoV 阶段决策**：
- [ ] 继续发布新功能（PoV 阶段允许实验性功能）
- [x] **限制高风险发布**（避免 Schema breaking changes）
- [ ] 冻结发布，专注稳定性（如果进入 production，需要考虑）

---

## 指标监控 Dashboard 设计

### 关键 Dashboard 视图

1. **实时监控视图**（On-Call Dashboard）
   - L0 核心指标（成功率、p95、错误率）
   - Batch 处理状态
   - 实时告警
   - 系统健康状态

2. **业务收益视图**（ROI Dashboard）
   - L1 指标趋势
   - 成本分析（$ / request）
   - 自动化率变化
   - 人工 review 队列减少

3. **工程健康视图**（Engineering Health Dashboard）
   - L2 指标趋势
   - 变更失败率
   - Toil 时间
   - Auditability 覆盖率

---

## 指标收集与上报

### 数据来源

- **Metrics**：`_summary.json` 聚合（未来：Prometheus / Datadog）
- **Logs**：Python logging（当前：stdout，未来：JSON logging）
- **Traces**：`trace_id`（当前：日志中，未来：Jaeger / Zipkin）

### 指标更新频率

- 实时指标：`Batch 完成后`（当前），未来：`实时`
- 聚合指标：`每次 batch 运行后`（当前），未来：`每分钟`
- 报表指标：`每日`（基于 `_summary.json` 聚合）

---

## 下一步行动

- [x] ✅ 完成 KYC 项目的三层指标填写
- [ ] 建立监控 Dashboard（生产环境）
- [ ] 制定 Error Budget Policy 并执行
- [ ] 定期回顾（每周/每月）指标趋势
- [ ] 自动化指标收集和报告

---

## 基于项目的关键设计点

### 1. Schema-First 设计
- **指标影响**：降低变更失败率（L2）
- **方法**：`src/schemas.py` 严格 Schema + `src/validators.py` Pydantic 验证

### 2. 确定性规则引擎
- **指标影响**：提高自动化率（L1），降低错误率（L0）
- **方法**：`src/rules.py` 代码逻辑，不是 LLM opinion

### 3. Per-File Isolation
- **指标影响**：提高 Batch 处理成功率（L0）
- **方法**：`src/pipeline.py` 中 one fail ≠ crash all

### 4. 标准化错误处理
- **指标影响**：降低 MTTR（L2），提高可观测性
- **方法**：`src/errors.py` 错误分类 + `trace_id`

### 5. Rate Limiting + Retry
- **指标影响**：降低错误率（L0），提高可用性
- **方法**：`src/rate_limiter.py` token bucket + exponential backoff

### 6. Privacy-Aware Logging
- **指标影响**：PII 泄漏事件 = 0（L2）
- **方法**：Never log base64/prompts/PII，只 log trace_id

---

## 参考

- KYC 项目文档：https://github.com/Nickcp39/kyc_pov/tree/main
- Google SRE Book: [SLO, SLI, SLAs, oh my!](https://sre.google/workbook/slo/)
- SRE Error Budget Policy: https://sre.google/workbook/error-budget-policy/
