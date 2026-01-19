---
doc_type: explanation
layer: L0
scope_in:  文件契约头（YAML front matter）、大文件导航规则、Diátaxis 分类规则、ADR 规则
scope_out:  具体文件实现（见各项目文件）；字典规则（见 SYSTEM_DESIGN_DICTIONARY_RULES.md）；结构规则（见 Structure_rule.md）
inputs:   (设计) 文件内容、文件类型、文件大小；(运行时) 文件创建、文件更新
outputs:  文件契约头、导航结构、文档分类
entrypoints:
  - B1 文件「契约头」（File Contract Header）
  - B2 大文件导航规则（什么时候需要导航）
  - B3 Diátaxis 分类规则（如何选择 doc_type）
  - B4 ADR 规则（什么时候写 ADR）
children: []
related: [ SYSTEM_DESIGN_DICTIONARY_RULES.md（字典规则）, Structure_rule.md（结构规则）, 00_GLOSSARY_INDEX.md（字典入口） ]
owner: you
last_updated: 2025-01-01
---

# File Contract & Structure Rule（文件契约 / AI 输出约束）

> **目标**：每个文件有「输入/输出/讨论范围」；大文件必有目录和 Quick Jump；AI 输出按契约，不乱发散。  
> **借鉴**：Diátaxis 文档分型 | RFC 2119 | 你现有 Structure_rule 的 200/800/2000 行规则

---

## TL;DR

- **契约头**：> 200 行或关键设计文档 **MUST** 有 YAML front matter（doc_type, layer, scope_in/out, inputs/outputs, entrypoints, children, related）
- **大文件导航**：> 800 行或 > 5 概念 **MUST** 有 Quick Jump + Where to look；> 2000 行 **MUST** 拆分，保留 entrypoint
- **Diátaxis**：新内容先选 doc_type，**禁止** 四类混在一篇
- **ADR**：出现「为什么选 A 不选 B」**MUST** 写 ADR（Context / Decision / Consequences）

---

## B1 文件「契约头」（File Contract Header）

任何 **> 200 行** 的 .md，或任何**关键设计文档**：**MUST** 在开头放：

```yaml
---
doc_type: tutorial | howto | reference | explanation | adr | index | glossary
layer: L0 | L1 | L2 | L3 | L4
scope_in:  (本文件覆盖什么)
scope_out: (本文件不讨论什么，指向哪里)
inputs:    (输入：需求/数据/schema/依赖)
outputs:   (输出：决策/接口/脚本/指标/结论)
entrypoints:
  - (读者入口：先看哪一节)
children:
  - (下钻文件列表)
related:
  - (平级对照概念)
owner: you | ai
last_updated: YYYY-MM-DD
---
```

打开任意文件，立刻知道：**该干嘛、不该干嘛、下一步去哪**。

---

## B2 大文件导航规则

满足**任一**即触发：

| 条件 | **MUST** |
|------|----------|
| **> 800 行** 或 **> 5 个概念** | Quick Jump + Where to look |
| **> 2000 行代码** | 拆分；保留 1 个 **entrypoint** 只做组装 |

### B2.1 Markdown 必备导航块（固定格式）

```md
## TL;DR（5 行）
## Quick Jump
## Where to look（按问题定位）
## Table of Contents
```

### B2.2 代码文件必备「地图注释」

文件顶部 docstring **MUST** 含：

- `ENTRY`：入口函数
- `FLOW`：主路径（如 request → validate → infer → postprocess → persist）
- `Section Map`：分区及行号范围（大重构时更新）

---

## B3 Diátaxis 分类规则

写新内容前，**MUST** 先选 `doc_type`：

| doc_type | 用途 |
|----------|------|
| **tutorial** | 主线学习路径 |
| **howto** | 操作步骤（怎么做 X） |
| **reference** | 速查（字段/公式/参数表） |
| **explanation** | 原理/背景 |
| **adr** | 决策记录 |
| **index** | 主导航 / 路线 |
| **glossary** | 概念节点（字典条目） |

**禁止**把 tutorial / howto / reference / explanation 混在一篇里。

---

## B4 ADR 规则：锁住 trade-off

当出现 **「为什么选 A 不选 B」**：

**MUST** 写 ADR，结构至少包含（Nygard 模板）：

- **Context**：背景与约束
- **Decision**：选了啥
- **Consequences**：后果、可回滚条件

---

## 附录：与 Structure_rule 的关系

- 本规则 = **File Contract & 导航** 的细化与可执行版
- `Structure_rule.md` 中的 200/800/2000、TOC、Module Docstring、Inbox 等：与本规则**对齐**，冲突时以本规则为准
- 下级概念拆分：见 `SYSTEM_DESIGN_DICTIONARY_RULES.md` 的 A3
