# A1_B5：如何平衡 Error Rate 与 fragile system（严格校验 vs 宽松校验）？

---
doc_type: glossary
layer: L2
scope_in:  严格校验 vs 宽松校验的 trade-off、分层校验设计、可配置阈值、监控反馈循环
scope_out:  具体 rules.py / validators.py 的实现（见代码）；Feature Flag 配置（见 howto）
inputs:   (设计) Error Rate 目标、output 稳定性要求、业务风险容忍度；(运行时) 校验结果、output 质量
outputs:  分层校验策略、可配置阈值、Error Rate 与 output 质量的平衡点
entrypoints: [ Definition, 核心问题 ]
children: []
related: [ Error Rate, SCHEMA_VALIDATION_FAILED, src/rules.py, src/validators.py, 自动化率, Feature Flag ]
---

## Definition（定义）

**Fragile system** = 系统对输入/条件变化**过于敏感**，容易因严格校验而失败（高 Error Rate），但 output 质量稳定。

**核心问题**：KYC preprocessing 太多（严格校验）→ 容易爆 error（正常请求也被拒绝）；preprocessing 少（宽松校验）→ error 少但 output 不稳定（错误数据通过了）。如何设计系统平衡这两者？

---

## 核心问题：严格校验 vs 宽松校验的 trade-off

### 两种极端及其问题

| 策略 | Error Rate | Output 稳定性 | 问题 |
|------|------------|---------------|------|
| **严格校验**（preprocessing 多） | ❌ **高**（正常请求也被拒绝） | ✅ **稳定**（只有高质量数据通过） | 误杀正常请求，用户体验差；Error Rate 超标 |
| **宽松校验**（preprocessing 少） | ✅ **低**（大部分请求通过） | ❌ **不稳定**（错误数据也通过） | 下游处理失败、业务风险、需要更多人工 review |

**KYC 场景**：
- **严格**：在 preprocessing 阶段就拒绝格式不完美、置信度 < 0.9 的图片 → Error Rate 高，但通过的都质量好
- **宽松**：只做基本格式检查，让所有图片都进入推理 → Error Rate 低，但 output 可能包含错误数据

---

## 设计原则：分层校验 + 可配置阈值

### 1. 分层校验（不是「全在 preprocessing」或「全不校验」）

**设计**：把校验分散到多个阶段，而不是集中在 preprocessing。

| 阶段 | 校验内容 | 失败后果 | 目的 |
|------|----------|----------|------|
| **Preprocessing**（入口） | 基础格式检查（图片格式、大小、是否损坏） | `IMAGE_FORMAT_UNSUPPORTED` | 避免明显无效数据进入 pipeline |
| **Post-inference**（推理后） | Schema 验证（必填字段、类型、格式） | `SCHEMA_VALIDATION_FAILED` | 捕获 LLM 输出不符合预期的情况 |
| **Post-processing**（后处理） | 业务规则检查（expiry valid、置信度阈值） | 转人工 review 或标记 `needs_review=true` | 不直接失败，而是标记需要人工介入 |

**KYC 设计**（基于 `src/rules.py`、`src/validators.py`）：
- **Preprocessing**：只检查图片格式、大小（避免明显无效数据）
- **Post-inference**：Schema 验证（`src/validators.py`）→ 如果不符合 Schema，标记 `SCHEMA_VALIDATION_FAILED`
- **Post-processing**：业务规则（`src/rules.py`）→ 如果置信度低或过期，标记 `needs_review=true`（**不直接失败**，而是转人工）

### 2. 可配置阈值（Feature Flag / 配置）

**设计**：校验的严格程度可配置，而不是硬编码。

| 配置项 | 严格模式 | 宽松模式 | 说明 |
|--------|----------|----------|------|
| **置信度阈值** | `> 0.9` | `> 0.7` | 严格模式：只有高置信度才通过；宽松模式：允许较低置信度 |
| **Schema 校验** | 严格（所有字段必填） | 宽松（允许部分字段缺失） | 通过 Feature Flag 切换 |
| **业务规则** | 严格（过期文档直接拒绝） | 宽松（过期文档转人工） | 通过 `validator_strictness` 配置 |

**KYC 实现**（学习指南提到）：
- `validator_strictness`：`high` / `medium` / `low`
- 通过 Feature Flag 或环境变量控制
- 可以根据 Error Rate 和 output 质量动态调整

### 3. 监控反馈循环（根据实际数据调优）

**设计**：持续监控 Error Rate 和 output 质量，根据数据调整阈值。

| 指标 | 监控内容 | 调优策略 |
|------|----------|----------|
| **Error Rate** | `SCHEMA_VALIDATION_FAILED` 占比 | 如果 > 2%，考虑放宽 Schema 校验（允许部分字段缺失） |
| **Output 质量** | 人工 review 率、下游失败率 | 如果 review 率 > 30%，考虑收紧置信度阈值 |
| **自动化率** | `needs_review=false` 的比例 | 如果 < 60%，说明校验太宽松，需要收紧 |

**KYC 示例**：
- **当前**：Error Rate = 5%（包含 2% SCHEMA_VALIDATION_FAILED），自动化率 = 60-70%
- **调优**：如果 Error Rate 降到 3% 但自动化率降到 50%，说明**校验太严格**，误杀了正常请求 → 放宽置信度阈值
- **平衡点**：Error Rate < 1%，自动化率 > 80%

---

## 具体设计：KYC 的分层校验策略

### Preprocessing（入口校验）

**只做「必须拒绝」的检查**，避免明显无效数据进入 pipeline：

```python
# src/pipeline.py - preprocessing 阶段

def preprocess_image(file_path: str) -> dict:
    """预处理：只做基础格式检查"""
    
    # 1. 检查文件格式（必须）
    if not is_supported_format(file_path):
        raise ImageFormatError("IMAGE_FORMAT_UNSUPPORTED")
    
    # 2. 检查文件大小（必须，避免 OOM）
    if file_size > MAX_SIZE:
        raise ImageFormatError("IMAGE_TOO_LARGE")
    
    # ❌ 不做：置信度检查、业务规则检查（这些放在后处理）
    # ✅ 只做：格式、大小等「硬性要求」
    
    return {"file_path": file_path, "format": "jpg", "size": file_size}
```

### Post-inference（Schema 验证）

**验证 LLM 输出是否符合 Schema**，但**允许部分字段缺失**（通过配置控制）：

```python
# src/validators.py - post-inference 阶段

def validate_output(llm_output: dict, strictness: str = "medium") -> dict:
    """Schema 验证：根据 strictness 调整严格程度"""
    
    if strictness == "high":
        # 严格模式：所有字段必填
        required_fields = ["full_name", "date_of_birth", "document_number", "expiry_date"]
    elif strictness == "medium":
        # 中等模式：核心字段必填
        required_fields = ["full_name", "date_of_birth", "document_number"]
    else:  # low
        # 宽松模式：只要求 name
        required_fields = ["full_name"]
    
    missing = [f for f in required_fields if f not in llm_output]
    if missing:
        raise ValidationError("SCHEMA_VALIDATION_FAILED", missing_fields=missing)
    
    return llm_output
```

### Post-processing（业务规则）

**不直接失败，而是标记 `needs_review`**，让系统继续运行：

```python
# src/rules.py - post-processing 阶段

def apply_deterministic_rules(extracted_data: dict, confidence: float) -> dict:
    """业务规则：不直接失败，而是标记需要 review"""
    
    needs_review = False
    review_reasons = []
    
    # 1. 置信度检查（可配置阈值）
    confidence_threshold = get_config("confidence_threshold", default=0.85)
    if confidence < confidence_threshold:
        needs_review = True
        review_reasons.append("low_confidence")
    
    # 2. 过期检查（不直接失败，而是标记）
    if extracted_data.get("expiry_date") and is_expired(extracted_data["expiry_date"]):
        needs_review = True
        review_reasons.append("expired_document")
    
    # 3. 返回结果（不抛异常，而是标记）
    return {
        "status": "success",  # ✅ 不失败
        "needs_review": needs_review,  # ⚠️ 标记需要人工介入
        "review_reasons": review_reasons,
        "extracted_data": extracted_data
    }
```

**关键**：Post-processing 的规则**不直接导致 `status: "fail"`**，而是标记 `needs_review=true`，这样：
- Error Rate 不会因为「置信度低」而飙升
- 但 output 质量通过「转人工 review」保证

---

## 平衡点的设计

### 目标：Error Rate < 1%，自动化率 > 80%

| 阶段 | 校验策略 | 失败后果 | 对 Error Rate 的影响 |
|------|----------|----------|---------------------|
| **Preprocessing** | 严格（格式、大小） | `IMAGE_FORMAT_UNSUPPORTED` | 必须的，避免无效数据 |
| **Post-inference** | 中等（核心字段必填） | `SCHEMA_VALIDATION_FAILED` | 可配置，通过 `strictness` 调整 |
| **Post-processing** | 宽松（标记 review，不直接失败） | `needs_review=true` | **不影响 Error Rate**，只影响自动化率 |

**关键设计**：
- **Preprocessing**：必须严格（格式检查）
- **Post-inference**：可配置（通过 Feature Flag 调整 `strictness`）
- **Post-processing**：宽松（不直接失败，而是转人工）→ **这是平衡的关键**：通过「转人工」而不是「直接失败」来保证 output 质量，同时不推高 Error Rate

---

## 监控与调优

### 持续监控两个指标

| 指标 | 目标 | 如果超标，如何调优 |
|------|------|-------------------|
| **Error Rate** | < 1% | 如果 > 1%：检查 `SCHEMA_VALIDATION_FAILED` 占比 → 如果高，放宽 `strictness` |
| **自动化率** | > 80% | 如果 < 80%：检查 `needs_review=true` 占比 → 如果高，收紧置信度阈值或业务规则 |

### 反馈循环

```
监控 Error Rate 和自动化率
    ↓
如果 Error Rate > 1% 且 SCHEMA_VALIDATION 多
    → 放宽 post-inference 的 strictness（允许部分字段缺失）
    ↓
如果自动化率 < 80% 且 review 率高
    → 收紧 post-processing 的置信度阈值（但保持「转人工」而不是「直接失败」）
    ↓
重新监控，找到平衡点
```

---

## 设计要点小结

1. **分层校验**：Preprocessing（严格）→ Post-inference（可配置）→ Post-processing（宽松，转人工）
2. **Post-processing 不直接失败**：通过 `needs_review=true` 保证 output 质量，同时不推高 Error Rate
3. **可配置阈值**：通过 Feature Flag / `validator_strictness` 动态调整
4. **监控反馈循环**：持续监控 Error Rate 和自动化率，根据数据调优

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1 指标与测试（KYC_Day01_A1_详细讲解_指标与测试.md）— Error Rate 与自动化率 |
| **Related** | Error Rate、SCHEMA_VALIDATION_FAILED、自动化率、`src/rules.py`、`src/validators.py`、Feature Flag |
