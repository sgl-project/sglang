# A3_B1：什么叫做完全成功的 batch？

---
doc_type: glossary
layer: L1
scope_in:  per-file 的 file、file 处理成功的含义；完全成功的 batch 的定义、判定、与 Batch 处理成功率的关系
scope_out: 怎么统计（见 howto）；per-file isolation 的实现细节（见 src/pipeline.py）
inputs:   一个 batch（若干文件）、每个文件的 status（success/fail）
outputs:  该 batch 是否计入「完全成功的 batch 数」；Batch 处理成功率 = 完全成功的 batch 数 / 总 batch 数
entrypoints: [ 前置概念, Definition ]
children: []
related: [ Per-File Isolation, KYC_Day01_A3_METRICS_CARD_EXAMPLE.md, KYC_Day01_A2_指标计算脚本示例.md ]
---

## 前置概念：per-file 的 file 与 file 处理成功

### per-file 的 file 指什么？

**file** = **一个输入文档/文件**，即 batch 里的**单次 KYC 处理对象**。

- 在 KYC 场景：通常是**一张待核验的证件图像**（如 `doc_001.jpg`、`doc_002.jpg`），对应 `_summary.json` 里 `results[]` 的一个元素，用 `file_id` 标识。
- 一个 **batch** = 一批这样的 file；pipeline 逐 file 处理，每个 file 一条 result（含 `status`、`latency_ms` 等）。

### file 处理成功 指什么？

**file 处理成功** = 该 file 对应的 `result.status == "success"`。

- **数据来源**：`_summary.json` 的 `results[].status`：`"success"` 表示成功，`"fail"` 表示失败。
- **含义**：KYC 流程（image preprocessing + model inference + validation）对该 file 跑完，且未因 `IMAGE_FORMAT_UNSUPPORTED`、`SCHEMA_VALIDATION_FAILED`、`API_TIMEOUT`、`RATE_LIMIT_EXCEEDED` 等导致 `status: "fail"`。
- **判定**：见 A2 指标计算脚本里对 `results[].status` 的用法；`_summary.json` 结构见 A2 文档。

---

## Definition（定义）

**完全成功的 batch**：在一个 batch 内，**该 batch 中所有文件（file）都处理成功**的 batch。

- **边界**：只要该 batch 中有**任意一个文件**失败，该 batch **不算**完全成功。
- **不计入**：单个文件的 status 如何解析、`_summary.json` 结构（见 Reference / A2）。

---

## 与 Per-File Isolation 的关系

- **Per-File Isolation**（`src/pipeline.py`）：某个文件失败 **不会** 让整个 batch 崩掉，其他文件照常处理（one fail ≠ crash all）。
- **完全成功的 batch**：看的是 **batch 级别** 的计数——只有 batch 内 **100% 文件都成功**，才计入「完全成功的 batch 数」；有任一文件失败，该 batch 就不算完全成功。

---

## 公式与目标（来自 A3 指标卡）

- **公式**：Batch 处理成功率 = **完全成功的 batch 数** / 总 batch 数
- **目标**：> 95%（允许单个文件失败**不影响整个 batch 继续跑**，但该 batch 仍**不算**完全成功）
- **基于**：`src/pipeline.py` 的 per-file isolation。

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A3 指标卡（KYC_Day01_A3_METRICS_CARD_EXAMPLE.md） |
| **Related** | Per-File Isolation、Batch 处理成功率、A2 指标计算（_summary.json / status） |
