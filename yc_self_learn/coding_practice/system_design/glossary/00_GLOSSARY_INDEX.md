---
doc_type: index
layer: L0
scope_in:  字典入口、概念树、按问题/关键词定位
scope_out: 具体概念定义见各节点文件；操作见 howto/；决策见 adr/
inputs:   (读者) 问题 / 关键词 / 复习路线
outputs:  概念树 + 搜索关键词 + 下钻路径
entrypoints:
  - 按层浏览（下方 L0–L4）
  - 按关键词搜索（见「搜索关键词」）
  - 按项目（如 KYC）下钻
children: [ SYSTEM_DESIGN_DICTIONARY_RULES.md（分层字典规则）, FILE_CONTRACT_AND_STRUCTURE_RULES.md（文件契约规则）, Structure_rule.md（结构规则）, teaching_rules.md（教学规则）, 见下方各层链接 ]
related:  [ 00_INDEX.md（主导航）, howto/, reference/, adr/ ]
owner: you
last_updated: 2025-01-01
---

# 字典入口：概念树 + 搜索关键词

> **用法**：问任何问题 → 从本页按层或按关键词找到节点 → 点进去；想深挖再沿 `children:` 下钻。

---

## TL;DR

- **按层**：L0 概览 → L1 指标 → L2 权衡 → L3 机制 → L4 实现
- **按关键词**：见下表，搜到后点链接
- **按项目**：KYC 等见 `children:` 下的项目子节点

---

## Quick Jump

- [按层浏览 L0–L4](#按层浏览-l0l4)
- [搜索关键词](#搜索关键词)
- [项目子树（如 KYC）](#项目子树如-kyc)

---

## 按层浏览（L0–L4）

### L0 概览（One-liner + Scope）

| 概念 | 链接 | 一句话 |
|------|------|--------|
| 分层字典规则 | [SYSTEM_DESIGN_DICTIONARY_RULES](./SYSTEM_DESIGN_DICTIONARY_RULES.md) | 分层知识图谱、概念节点结构、AI 输出规则 |
| 文件契约规则 | [FILE_CONTRACT_AND_STRUCTURE_RULES](./FILE_CONTRACT_AND_STRUCTURE_RULES.md) | 文件契约头、大文件导航、Diátaxis 分类、ADR 规则 |
| 结构规则 | [Structure_rule](./Structure_rule.md) | 文件结构、导航块、目录、Section Map |
| 教学规则 | [teaching_rules](./teaching_rules.md) | Senior System Design 教学规则、输出契约、Trade-off 表 |
| （示例）系统概览 | [L0_system_overview](./L0_system_overview.md) | 待补 |

### L1 目标与指标（SLO / Success Metrics）

| 概念 | 链接 | 一句话 |
|------|------|--------|
| （示例）指标与 SLO | [L1_metrics_slo](./L1_metrics_slo.md) | 待补 |

### L2 设计与权衡（Trade-offs / Alternatives）

| 概念 | 链接 | 一句话 |
|------|------|--------|
| （示例）权衡与备选 | [L2_tradeoffs_alternatives](./L2_tradeoffs_alternatives.md) | 待补 |

### L3 机制与模式（Mechanism / Patterns）

| 概念 | 链接 | 一句话 |
|------|------|--------|
| （示例）限流与背压 | [L3_rate_limit_backpressure](./L3_rate_limit_backpressure.md) | 待补 |
| （示例）缓存 | [L3_caching](./L3_caching.md) | 待补 |

### L4 实现与验证（Code / Test / Runbook）

| 概念 | 链接 | 一句话 |
|------|------|--------|
| （示例）回归测试 | [L4_regression_testing](./L4_regression_testing.md) | 待补 |

---

## 搜索关键词

| 关键词 | 对应节点 | 层 |
|--------|----------|-----|
| 分层字典 / 知识图谱 / 概念节点 / AI 输出规则 | [SYSTEM_DESIGN_DICTIONARY_RULES](./SYSTEM_DESIGN_DICTIONARY_RULES.md) | L0 |
| 文件契约 / YAML front matter / 大文件导航 / Diátaxis / ADR | [FILE_CONTRACT_AND_STRUCTURE_RULES](./FILE_CONTRACT_AND_STRUCTURE_RULES.md) | L0 |
| 结构规则 / 文件结构 / 导航块 / TOC / Section Map | [Structure_rule](./Structure_rule.md) | L0 |
| 教学规则 / 输出契约 / Trade-off 表 / SRE DoD | [teaching_rules](./teaching_rules.md) | L0 |
| （示例）batch / 批量 / 成功率 | [KYC_Day01_A3_B1_good_batch](../KYC%20project/KYC_Day01_A3_B1_good_batch.md) | L1 |
| 错误分类 / error_code / errors.py | [KYC_Day01_A1_B1_error_classification](../KYC%20project/KYC_Day01_A1_B1_error_classification.md) | L2 |
| 未知错误 / UNKNOWN_ERROR / 快速定位 | [KYC_Day01_A1_B2_unknown_error](../KYC%20project/KYC_Day01_A1_B2_unknown_error.md) | L2 |
| 告警阈值 / Alert Threshold / 告警疲劳 | [KYC_Day01_A1_B3_alert_threshold](../KYC%20project/KYC_Day01_A1_B3_alert_threshold.md) | L2 |
| 重试 / 概率累积 / MAX_RETRIES / 幂等 | [KYC_Day01_A1_B4_retry_error_rate](../KYC%20project/KYC_Day01_A1_B4_retry_error_rate.md) | L2 |
| 校验 / validation / strictness / fragile system | [KYC_Day01_A1_B5_validation_tradeoff](../KYC%20project/KYC_Day01_A1_B5_validation_tradeoff.md) | L2 |
| 分层容错 / Defense in Depth / 多层保护 / 系统不 down / Chrome 设计 | [KYC_Day01_A1_B6_layered_fault_tolerance](../KYC%20project/KYC_Day01_A1_B6_layered_fault_tolerance.md) | L2 |
| 错误影响 / 损失评估 / 公司损失 / 客户损失 / 客户流失 / on-call | [KYC_Day01_A1_B7_error_impact_classification](../KYC%20project/KYC_Day01_A1_B7_error_impact_classification.md) | L2 |
| A/B Test / AB测试 / 实验验证 / 用户体验阈值 / 统计显著性 | [KYC_Day01_A1_B8_ab_testing_validation](../KYC%20project/KYC_Day01_A1_B8_ab_testing_validation.md) | L2 |
| 流失率 / 指标计算 / 实时计算 / 批处理 | [KYC_Day01_A1_B8_C1_metric_calculation_methods](../KYC%20project/KYC_Day01_A1_B8_C1_metric_calculation_methods.md) | L3 |
| A/B Test 决策 / 什么时候进行 A/B Test / 决策标准 / 考虑因素 | [KYC_Day01_A1_B8_C2_ab_test_decision_criteria](../KYC%20project/KYC_Day01_A1_B8_C2_ab_test_decision_criteria.md) | L3 |
| A/B Test 成本 / 成本计算 / 时间成本 / 资源成本 / 人力成本 | [KYC_Day01_A1_B8_C2_D1_ab_test_cost_calculation](../KYC%20project/KYC_Day01_A1_B8_C2_D1_ab_test_cost_calculation.md) | L4 |
| cron / 定时 / Cron | [KYC_Day01_A2_B1_cron](../KYC%20project/KYC_Day01_A2_B1_cron.md) | L3 |
| CI/CD / 门禁 / 持续集成 / 持续部署 | [KYC_Day01_A2_B2_ci_cd](../KYC%20project/KYC_Day01_A2_B2_ci_cd.md) | L3 |
| （示例）SLO / 指标 / 成功率 | L1_metrics_slo | L1 |
| （示例）限流 / 背压 / rate limit | L3_rate_limit_backpressure | L3 |
| （示例）缓存 / cache | L3_caching | L3 |
| （示例）回归 / regression | L4_regression_testing | L4 |

> 按你实际沉淀的节点**增删改**本表；AI 在「动作清单」里应注明：需在 00_GLOSSARY_INDEX 的搜索关键词中新增某某。

---

## 项目子树（如 KYC）

| 父级 | 子概念（children） |
|------|---------------------|
| KYC_Day01_A1 指标与测试 | [KYC_Day01_A1_B1_error_classification](../KYC%20project/KYC_Day01_A1_B1_error_classification.md)（错误分类）、[KYC_Day01_A1_B2_unknown_error](../KYC%20project/KYC_Day01_A1_B2_unknown_error.md)（未知/致命错误）、[KYC_Day01_A1_B3_alert_threshold](../KYC%20project/KYC_Day01_A1_B3_alert_threshold.md)（告警阈值）、[KYC_Day01_A1_B4_retry_error_rate](../KYC%20project/KYC_Day01_A1_B4_retry_error_rate.md)（重试与错误率）、[KYC_Day01_A1_B5_validation_tradeoff](../KYC%20project/KYC_Day01_A1_B5_validation_tradeoff.md)（校验严格度权衡）、[KYC_Day01_A1_B6_layered_fault_tolerance](../KYC%20project/KYC_Day01_A1_B6_layered_fault_tolerance.md)（分层容错设计）、[KYC_Day01_A1_B7_error_impact_classification](../KYC%20project/KYC_Day01_A1_B7_error_impact_classification.md)（错误影响分级与损失评估）、[KYC_Day01_A1_B8_ab_testing_validation](../KYC%20project/KYC_Day01_A1_B8_ab_testing_validation.md)（A/B Test 验证设计思路，含 [C1：指标计算方法](../KYC%20project/KYC_Day01_A1_B8_C1_metric_calculation_methods.md)、[C2：决策标准](../KYC%20project/KYC_Day01_A1_B8_C2_ab_test_decision_criteria.md)） |
| KYC_Day01_A2 指标计算 | [KYC_Day01_A2_B1_cron](../KYC%20project/KYC_Day01_A2_B1_cron.md)（Cron）、[KYC_Day01_A2_B2_ci_cd](../KYC%20project/KYC_Day01_A2_B2_ci_cd.md)（CI/CD） |
| KYC_Day01_A3 指标卡 | [KYC_Day01_A3_B1_good_batch](../KYC%20project/KYC_Day01_A3_B1_good_batch.md)（完全成功的 batch） |

> 父级文档的 `children:` 与上表一致；新下级概念按 `SYSTEM_DESIGN_DICTIONARY_RULES.md` 的 A3 创建并在此登记。

---

## Where to look

| 你想… | 去 |
|-------|-----|
| 查概念定义 | 本页 → 按层/关键词 → 点进节点 |
| 查「怎么做 X」 | [howto/](../howto/) |
| 查「字段/公式/参数」 | [reference/](../reference/) |
| 查「为什么选 A 不选 B」 | [adr/](../adr/) |
| 复习主线 / 路线 | [00_INDEX.md](../00_INDEX.md) |

---

## 维护约定

- **新增概念节点**：在对应层的表 + 搜索关键词表各加一行；若为某父级下级，在「项目子树」加一行。
- **AI 动作清单**：每次回答若产生新节点，必须写明「在 00_GLOSSARY_INDEX 的 X 表新增 Y」。
