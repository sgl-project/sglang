# A1_B1：错误分类的设计是怎么做的？

---
doc_type: glossary
layer: L2
scope_in:  错误分类的设计目标、层次、分类维度、常见 error_code、与指标/告警的衔接
scope_out:  src/errors.py 的具体实现（见代码）；如何新增/修改某一类（见 howto 或 ADR）
inputs:   (设计) 失败场景、可恢复性、告警/响应需求；(运行时) Exception、HTTP 状态、业务校验结果
outputs:  统一的 error_code 枚举、error_breakdown 指标、告警/重试/降级策略的输入
entrypoints: [ Definition, 设计层次 ]
children: [ KYC_Day01_A1_B2_unknown_error.md（未知/致命错误与快速定位） ]
related: [ src/errors.py, Error Rate, error_breakdown, A4 告警响应, A2 指标计算 ]
---

## Definition（定义）

**错误分类** = 在 `src/errors.py` 里定义的、**标准化且可枚举的 error_code 体系**，用来：

- **统计**：按类型聚合为 `error_breakdown`，支撑 Error Rate 及「各类型占比」
- **告警与响应**：按类型决定是否重试、降级、转人工（见 A4）
- **可观测与 Postmortem**：日志、trace、事后分析都使用同一套 code，便于定位根因

**边界**：只规定「有哪些类、怎么记、怎么用」；**各类型的占比（如 1%、2%）是 PoV 的观测值，不是设计出来的目标**。

---

## 设计层次（四层）

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 定义层（src/errors.py）                                        │
│    定义所有 error_code 常量/枚举；Exception → error_code 的映射    │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 记录层（pipeline / 调用链）                                    │
│    catch 时取标准 error_code，写入 result.error_code、error_msg   │
│    写入 _summary.json 的 results[].error_code                     │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 汇总层（A2 MetricsCalculator）                                 │
│    error_breakdown = { error_code: count } 对 status=="fail" 聚合  │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. 消费层（A4 告警、重试、降级）                                   │
│    按 error_breakdown 判断：API_TIMEOUT 多 → 重试；SCHEMA 多 → 查数据/规则 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 分类维度（为何是这些类）

### 按来源/根因

| 类型 | 含义 | 典型来源 |
|------|------|----------|
| `IMAGE_FORMAT_UNSUPPORTED` | 输入图片格式不支持 | 预处理 / 解码失败 |
| `SCHEMA_VALIDATION_FAILED` | 输出不符合 Schema 或必填缺失 | `src/validators.py`、`src/schemas.py` |
| `API_TIMEOUT` | 调用 Fireworks API 超时 | 网络、上游慢 |
| `RATE_LIMIT_EXCEEDED` | 触发上游或自研限流 | `src/rate_limiter.py`、Fireworks 429 |
| `API_SERVER_ERROR` | 上游 5xx 等 | Fireworks / 网关 |
| `API_CONNECTION_ERROR` | 连接失败、断连 | 网络、DNS |

### 按是否可恢复（用于告警响应）

| 可恢复（可重试） | 不可恢复或需改输入/逻辑 |
|------------------|--------------------------|
| `API_TIMEOUT` | `IMAGE_FORMAT_UNSUPPORTED` |
| `API_CONNECTION_ERROR` | `SCHEMA_VALIDATION_FAILED` |
| `API_SERVER_ERROR` | `RATE_LIMIT_EXCEEDED`（需限流/退避，而非简单重试） |

A4：对可恢复类型做**自动重试**；对不可恢复类型做**告警 + 人工/规则排查**。

---

## 常见 error_code 与 A1 中的「错误分类」示例

A1 第 902–906 行的**错误分类**示例：

- `IMAGE_FORMAT_UNSUPPORTED`：1%
- `SCHEMA_VALIDATION_FAILED`：2%
- `API_TIMEOUT`：1%
- `RATE_LIMIT_EXCEEDED`：1%

这些数字是 **PoV 阶段观测到的占比**，用来示意 error_breakdown 长什么样；**不是设计阶段规定的目标比例**。设计只保证：**所有失败都能归到某一类，并写入 `result.error_code`**；未识别时用 `UNKNOWN_ERROR`（A2 的 `error_breakdown` 逻辑）。

---

## 与 _summary.json、A2、A4 的衔接

| 环节 | 用法 |
|------|------|
| **_summary.json** | `results[].status=="fail"` 时必有 `error_code`（及可选 `error_msg`） |
| **A2 指标计算** | `calculate_error_breakdown()`：对 `status=="fail"` 按 `error_code` 计数 → `error_breakdown` |
| **A4 告警响应** | 读 `error_breakdown`：`API_TIMEOUT` / `API_CONNECTION_ERROR` 多 → 触发自动重试；其余类型 → 告警与人工/规则处理 |

---

## 设计要点小结

1. **单一来源**：所有 code 在 `src/errors.py` 定义，避免散落 `str(e)` 或魔术字符串。
2. **全失败必填**：pipeline 里 catch 后必须映射到某一 `error_code`，不落空的 `fail`。
3. **为响应服务**：分类要能支持「可重试 / 不可重试」、「限流 / 非限流」等判断，否则告警难以自动化。
4. **可扩展**：新增一类 = 在 `errors.py` 加常量 + 映射逻辑 + 必要时在 A4 增加分支；`error_breakdown` 自动包含新类。
5. **未知兜底**：无法映射时用 `UNKNOWN_ERROR`，并走专门告警与快速定位流程 → 见 [A1_B2：未知/致命错误与快速定位](./KYC_Day01_A1_B2_unknown_error.md)。

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1 详细讲解（KYC_Day01_A1_详细讲解_指标与测试.md）— 错误率与错误分类 |
| **Related** | `src/errors.py`、Error Rate、error_breakdown、A2 指标计算、A4 告警响应、`tests/test_errors.py` |
