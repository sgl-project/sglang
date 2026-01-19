# A1_B2：未知 / 致命错误时，如何设计系统以快速定位？

---
doc_type: glossary
layer: L2
scope_in:  未知/未枚举错误的兜底、记录、告警与快速定位设计
scope_out:  src/errors.py 和日志的具体实现；Postmortem 模板（见 A4 或 runbook）
inputs:   未被 errors.py 映射的 Exception、或致命/未预期的失败
outputs:  统一的 UNKNOWN_ERROR 记录、可检索的上下文、告警与 Runbook 入口
entrypoints: [ Definition, 设计原则 ]
children: []
related: [ A1_B1 错误分类, trace_id, 结构化日志, A4 告警, src/errors.py ]
---

## Definition（定义）

**未知 / 致命错误** = 未被 `src/errors.py` 枚举的异常，或无法映射到现有 error_code 的失败（含未 catch 的 Python Exception、上游返回的未知 5xx/4xx、OOM、段错误等）。

**快速定位** = 在告警后，通过 **trace_id + 上下文 + 日志 + 可选 error 存储**，在较短时间内找到根因（哪次请求、哪一阶段、什么异常、哪段代码）。

**边界**：本节点只讲「如何设计」；具体 log 格式、采样、存储保留策略见 howto / runbook。

---

## 设计原则（四条）

1. **绝不静默**：未知错误必须进日志、进 `_summary.json`（或等价），并有一个**统一兜底 error_code**，不能因为「没枚举」就丢。
2. **上下文一次打齐**：在 catch-all 处一次记录：trace_id、file_id、batch_id、**step**、exception 类型与消息、栈、时间，便于事后一条 trace_id 查全。
3. **未知即告警**：`UNKNOWN_ERROR` / `UNHANDLED_EXCEPTION` 出现即视为异常，需要告警与 Runbook，不能只当普通 fail。
4. **可检索、不堆 PII**：日志与存储可按 trace_id / file_id / error_code / 时间检索；不落盘 base64、prompt、提取出的 PII。

---

## 兜底：统一 error_code

在 `src/errors.py` 中：

- 增加 **`UNKNOWN_ERROR`**（或 `UNHANDLED_EXCEPTION`）：用于「能 catch 但无法映射到已知类」的异常。
- pipeline 最外层 **catch-all**：`except Exception as e`（或等价）时：
  - 先用现有映射逻辑尝试得到已知 `error_code`；
  - 若得不到，则 `error_code = "UNKNOWN_ERROR"`；
  - `error_msg` = 脱敏后的 `str(e)` 或 `type(e).__name__ + ": " + str(e)`（长度可截断，避免超大 blob）；
  - 同时写**完整栈**到日志（见下），不入 `error_msg` 字段以免膨胀。

这样：**所有失败**都会进入 `error_breakdown`，未知的单独成类，便于监控和告警。

---

## 代码示例：未知错误如何被捕获并记录

### 场景：一个未被 errors.py 枚举的异常

假设 pipeline 在处理 `doc_001.jpg` 时，在 `validate` 阶段抛出了一个**未预期的异常**（例如：`KeyError: 'document_number'`，但 `errors.py` 里没有 `KeyError` 的映射，只有 `SCHEMA_VALIDATION_FAILED` 等）。

### 代码流程

```python
# src/pipeline.py

import traceback
from src.errors import (
    map_exception_to_error_code,  # 已知异常的映射函数
    UNKNOWN_ERROR,  # 兜底 error_code
    IMAGE_FORMAT_UNSUPPORTED,
    SCHEMA_VALIDATION_FAILED,
    API_TIMEOUT,
    # ... 其他已知 error_code
)

def process_file(file_id: str, batch_id: str, trace_id: str, step: str = None):
    """处理单个文件（示例：展示 catch-all 如何捕获未知错误）"""
    
    result = {
        "file_id": file_id,
        "status": "success",
        "trace_id": trace_id,
        "batch_id": batch_id,
        "step": step or "unknown",
        "error_code": None,
        "error_msg": None,
    }
    
    try:
        # ... 实际的 pipeline 逻辑（preprocess / inference / validate / save）
        # 假设在 validate 阶段，代码访问了一个不存在的 key：
        extracted_data = {"name": "John"}  # 缺少 document_number
        doc_number = extracted_data["document_number"]  # ← 这里会抛 KeyError
        
    except Exception as e:
        # 【关键】catch-all：捕获所有异常（包括未知的）
        result["status"] = "fail"
        
        # 1. 尝试映射到已知 error_code
        error_code = map_exception_to_error_code(e)
        # map_exception_to_error_code 的逻辑：
        #   - 如果是 requests.Timeout → API_TIMEOUT
        #   - 如果是 ValidationError → SCHEMA_VALIDATION_FAILED
        #   - 如果是 ImageFormatError → IMAGE_FORMAT_UNSUPPORTED
        #   - 如果是 KeyError → None（未映射，未知）
        
        # 2. 如果映射失败（返回 None），使用兜底
        if error_code is None:
            error_code = UNKNOWN_ERROR  # "UNKNOWN_ERROR"
            result["error_code"] = error_code
            result["error_msg"] = f"{type(e).__name__}: {str(e)[:200]}"  # 截断，避免超大
            result["exception_type"] = type(e).__name__  # "KeyError"
            
            # 3. 【关键】记录完整栈到日志（结构化 JSON）
            import logging
            logger = logging.getLogger(__name__)
            
            logger.error(
                "Unknown error caught",
                extra={
                    "trace_id": trace_id,
                    "file_id": file_id,
                    "batch_id": batch_id,
                    "step": step or "unknown",
                    "error_code": UNKNOWN_ERROR,
                    "exception_type": type(e).__name__,
                    "error_msg": str(e)[:200],
                    "stack_trace": traceback.format_exc(),  # 完整栈（仅日志）
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            # 日志输出示例（JSON）：
            # {
            #   "level": "ERROR",
            #   "message": "Unknown error caught",
            #   "trace_id": "trace_abc123",
            #   "file_id": "doc_001.jpg",
            #   "batch_id": "batch_20250101_120000",
            #   "step": "validate",
            #   "error_code": "UNKNOWN_ERROR",
            #   "exception_type": "KeyError",
            #   "error_msg": "document_number",
            #   "stack_trace": "Traceback (most recent call last):\n  File ...",
            #   "timestamp": "2025-01-01T12:00:05Z"
            # }
        else:
            # 已知错误：正常处理（不写完整栈到日志，只写 error_code）
            result["error_code"] = error_code
            result["error_msg"] = str(e)[:200]
            logger.warning(
                f"Known error: {error_code}",
                extra={
                    "trace_id": trace_id,
                    "file_id": file_id,
                    "error_code": error_code,
                    "error_msg": str(e)[:200],
                }
            )
    
    # 4. 无论成功或失败，都写入 _summary.json（通过 io_utils.py）
    return result

# 调用示例
result = process_file(
    file_id="doc_001.jpg",
    batch_id="batch_20250101_120000",
    trace_id="trace_abc123",
    step="validate"
)
# result 会包含：
# {
#   "file_id": "doc_001.jpg",
#   "status": "fail",
#   "error_code": "UNKNOWN_ERROR",  ← 未知错误
#   "error_msg": "KeyError: document_number",
#   "exception_type": "KeyError",  # 可选：扩展字段
#   "trace_id": "trace_abc123",
#   "batch_id": "batch_20250101_120000",
#   "step": "validate"
# }
```

### 关键点

1. **"未知"的含义**：不是"系统不知道有错误"，而是**"错误类型不在 `errors.py` 的枚举映射里"**。系统通过 `try/except` 能 catch 到异常，但 `map_exception_to_error_code(e)` 返回 `None`，说明这是未预期的类型。
2. **如何进日志**：在 catch-all 的 `except Exception as e` 里，用 `logger.error()` 写结构化 JSON，包含 trace_id、stack_trace 等。
3. **如何进 `_summary.json`**：`result` 字典（含 `error_code="UNKNOWN_ERROR"`）会被 `io_utils.py` 写入 `_summary.json` 的 `results[]`，后续 A2 的 `calculate_error_breakdown()` 会统计 `error_breakdown["UNKNOWN_ERROR"]`。

---

## 快速定位：必须记录的上下文

在 catch-all（以及各层 catch 后决定「转为 UNKNOWN」）时，**一次写齐**以下字段（结构化日志 + 可选 `_summary.json` 的扩展字段）：

| 字段 | 用途 | 示例 |
|------|------|------|
| **trace_id** | 串联同一次请求的所有日志、trace、metric | `trace_abc123` |
| **file_id** | 哪一个输入文件 | `doc_001.jpg` |
| **batch_id** | 哪一批 | `batch_20250101_120000` |
| **step** | 失败发生在哪一阶段 | `preprocess` / `inference` / `validate` / `save` |
| **error_code** | 兜底为 `UNKNOWN_ERROR` | `UNKNOWN_ERROR` |
| **exception_type** | 原始异常类型 | `KeyError`, `ConnectionResetError` |
| **error_msg** | 脱敏后的异常信息（截断） | `KeyError: 'document_number'` |
| **stack_trace** | 完整 traceback（仅日志，不写进 _summary 的通用字段） | 多行文本 |
| **timestamp** | 发生时间 | `2025-01-01T12:00:05Z` |

**step 的设计**：在 pipeline 各阶段（预处理、推理、校验、落库）的 try/except 里显式设置 `step=xxx`，这样看到 `UNKNOWN_ERROR` 时能立刻缩小到「某一步」，加速定位。

---

## 结构化日志与检索

- **格式**：JSON，包含上述字段；不写 base64、prompt、提取出的 PII。
- **检索**：按 `trace_id` 查单次请求全链路；按 `error_code=UNKNOWN_ERROR` + 时间范围查所有未知错误；按 `exception_type` 或 `error_msg` 关键词做模式发现。
- **保留**：按现有日志策略（如 30 天）；若存在「error 专题存储」，可对 UNKNOWN / fatal 做更长保留与采样。

---

## 告警与 Runbook

| 动作 | 说明 |
|------|------|
| **告警** | `error_breakdown["UNKNOWN_ERROR"] > 0` 或超过阈值 → 触发 P2/P1；或对未在 `errors.py` 枚举的 error_code 统一当未知处理并告警。 |
| **Runbook** | 1) 用 trace_id 查日志 / trace；2) 看 step、exception_type、error_msg、stack；3) 若是新异常类型，在 `errors.py` 增加映射或新 code，并排期修复；4) 若为上游/环境问题，按 A4 走升级或降级。 |

---

## 与 A1_B1 错误分类的衔接

- **A1_B1** 定义已知的 error_code 集合；**A1_B2** 定义「已知之外的兜底」与「快速定位」。
- `UNKNOWN_ERROR` 作为**正式的一类**进入 `error_breakdown` 和 Error Rate；同时走**单独告警与 Runbook**，而不是混在其它类里。
- 随着排查，新的失败模式应**沉淀回** `errors.py`：加新 code、加映射，使 `UNKNOWN_ERROR` 占比逐步下降。

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1 指标与测试（KYC_Day01_A1_详细讲解_指标与测试.md） |
| **Related** | [A1_B1 错误分类](./KYC_Day01_A1_B1_error_classification.md)、trace_id、结构化日志、A4 告警、`src/errors.py`、Privacy-Aware Logging |
