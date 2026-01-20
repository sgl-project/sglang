---
doc_type: explanation
layer: L0
scope_in:  触发条件（什么时候必须加结构）、Markdown/设计文档头部规范、2000+ 行代码导航规则、项目级导航规则、Inbox 规则
scope_out:  具体文件实现（见各项目文件）；文件契约规则（见 FILE_CONTRACT_AND_STRUCTURE_RULES.md）；字典规则（见 SYSTEM_DESIGN_DICTIONARY_RULES.md）
inputs:   (设计) 文件大小、概念数量、文件类型；(运行时) 文件创建、文件更新
outputs:  文件结构、导航块、目录、Section Map
entrypoints:
  - §0 触发条件（什么时候必须加目录/导航/结构）
  - §1 单文件头部规范（Markdown / 设计文档）
  - §2 2000+ 行代码：文件内导航规则
  - §3 项目级导航规则（README + INDEX）
  - §4 Inbox 规则（新问题放哪）
  - §5 最小执行建议（今天就能开始用）
  - 6 下级概念拆分（问答驱动的文档派生）
children: []
related: [ FILE_CONTRACT_AND_STRUCTURE_RULES.md（文件契约规则）, SYSTEM_DESIGN_DICTIONARY_RULES.md（字典规则）, 00_GLOSSARY_INDEX.md（字典入口） ]
owner: you
last_updated: 2025-01-01
---

# Structure Rule（结构规则）

> **1 句话**：本规则规定何时、如何为 Markdown / 设计文档 / 大文件（含 2000+ 行代码）添加「目录 + 导航 + 结构」，保证复习不迷路、维护可连续。

---

## TL;DR

- **> 200 行**：SHOULD 有 TL;DR + Quick Jump + TOC  
- **> 800 行或概念 > 5 个**：MUST 有 Section Map + Where to look  
- **> 2000 行代码**：MUST 拆成多文件，保留 1 个 entrypoint 做组装  
- **单文件头部**：Title + TL;DR + Quick Jump + TOC + Where to look（Markdown/设计文档）  
- **代码文件**：Module Docstring 地图 + 分区标记 + ENTRY/FLOW  
- **新问题**：先入 `03_questions_inbox/`，答完再按 Diátaxis 归档  

---

## Quick Jump

- **触发条件（什么时候必须加）**：§0  
- **Markdown/设计文档头部五块**：§1  
- **2000+ 行代码：地图 + 分区 + 入口**：§2  
- **项目级导航（README + INDEX）**：§3  
- **Inbox 规则（新问题放哪）**：§4  
- **最小执行建议**：§5  

---

## Table of Contents

- [§0 触发条件](#0-触发条件什么时候必须加目录导航结构)
- [§1 单文件头部规范（Markdown / 设计文档）](#1-单文件头部规范markdown--设计文档)
- [§2 2000+ 行代码：文件内导航规则](#2-适用于-2000-行代码文件内导航规则)
- [§3 项目级导航规则](#3-目录太复杂时项目级导航规则readme--index)
- [§4 Inbox 规则](#4-新问题来了放哪inbox-规则)
- [§5 最小执行建议](#5-最小执行建议)

---

## Where to look（按问题定位）

| 你现在的问题是… | 直接去看 |
|-----------------|----------|
| 什么时候必须加目录/导航？ | §0 触发条件 |
| Markdown/设计文档头部该写哪五块？ | §1 单文件头部规范 |
| 大代码文件怎么加“地图”和分区？ | §2 文件内导航规则 |
| 项目/目录多了怎么导航？ | §3 README + 00_INDEX |
| 新问题、新想法先放哪？ | §4 Inbox 规则 |
| 今天就能落地的动作？ | §5 最小执行建议 |

---

## §0 触发条件（什么时候必须加“目录+导航+结构”）

| 条件 | 要求 | 关键词 |
|------|------|--------|
| 文件 **> 200 行** | **SHOULD** 有 TL;DR + Quick Jump + TOC | 宜有 |
| 文件 **> 800 行** 或 **概念 > 5 个** | **MUST** 有 Section Map（模块地图）+ Where to look（去哪看什么） | 必须有 |
| **> 2000 行代码** | **MUST** 拆分成多文件/模块；保留 1 个 **entrypoint** 文件作为入口（只做组装，不做细节） | 必须拆 |

> **说明**：> 2000 行单文件基本不可复习；拆成多文件后，entrypoint 负责“组装”，细节进模块，复习能连续。

**MUST / SHOULD 的用法**：建议按 **RFC 2119** 解释，避免团队理解不一致：
- **MUST**：不可违反，否则视为不符合本规则  
- **SHOULD**：强烈建议，有充分理由时可例外，但需说明  

---

## §1 单文件头部规范（Markdown / 设计文档）

### 1.1 适用于 Markdown / 设计文档：必须有这五块

这套结构跟 **Diátaxis**（Tutorial / How-to / Reference / Explanation）兼容，用来保证「复习不迷路」。

在文件**最顶部**固定放：

1. **Title + 1 句定位**（这文件解决什么）  
2. **TL;DR**（5 行内）：结论 / 决策 / 最重要的 3 点  
3. **Quick Jump**：入口段落、核心流程、关键 trade-off、失败模式&回滚、参考/附录  
4. **TOC（目录）**：自动生成（推荐 `doctoc`，GitHub anchor 兼容）  
5. **Where to look**：按「问题类型 → 章节」映射（非常适合复习）  

### 1.2 模板（可直接复制）

```markdown
# <TITLE>

> 1 句话：这份文档解决什么问题 / 覆盖什么范围

## TL;DR
- 结论 1
- 结论 2
- 结论 3

## Quick Jump
- 入口/背景：见 §1
- 核心流程（主干）：见 §2
- Trade-offs（为什么选A不选B）：见 §3
- 失败模式 / 回滚 / 告警：见 §4
- 速查/附录：见 §A

## Table of Contents
<!-- START doctoc -->
<!-- END doctoc -->

## Where to look（按问题定位）
| 你现在的问题是… | 直接去看 |
|---|---|
| “这玩意到底要干嘛？” | §1 背景与目标 |
| “主流程怎么走？” | §2 端到端流程 |
| “为什么这样设计？” | §3 Trade-offs & Alternatives |
| “挂了怎么办？” | §4 Reliability：SLO/告警/回滚 |
| “字段/接口/公式是什么？” | §A Reference |
```

TOC 推荐用 **doctoc** 自动维护，与 GitHub 锚点兼容。

---

## §2 适用于 2000+ 行代码：文件内导航规则

> 目标：在代码里也能快速找到「去那几行看哪些信息」——顶部“地图注释” + 分区标记 + 导出清单。

### 2.1 Python / 通用代码文件顶部：Module Docstring 作为“地图”

PEP 8 建议：公共模块/类/函数要写 docstring；Google Python Style Guide 强调模块级文档与可读性。

**强制格式（MUST）**：

```python
"""
<TITLE>: 1 句话说明本文件的职责（边界清晰）

Quick Jump (search keywords):
- ENTRY: main() / run() / serve()   # 入口
- FLOW: request -> validate -> infer -> postprocess -> persist
- API: public functions/classes list
- CONFIG: config keys
- ERRORS: error handling strategy
- TESTS: how to test

Section Map (approx ranges, update when large refactor):
1) Types & Schemas ............ (L1-L120)
2) Core Pipeline .............. (L121-L520)
3) Integrations (db/queue) .... (L521-L980)
4) Observability .............. (L981-L1150)
5) CLI/Entry .................. (L1151-L1300)
"""
```

列出模块导出的类/函数并给一行摘要，便于读者快速找入口。

### 2.2 分区标记（MUST）：让编辑器可折叠、可搜索

在代码里用**统一标记**（任何语言均可）：

```
# =========================
# 1) Types & Schemas
# =========================

# =========================
# 2) Core Pipeline
# =========================
```

每个区块的**第一段**必须写：这一段负责什么 + 输入/输出。

### 2.3 “入口”与“关键路径”必须显式

- **ENTRY**：标出唯一入口函数（如 `main()` / `serve()` / `handler()`）  
- **FLOW**：用 1 行写主流程（方便复习）  

任何人打开文件 **10 秒内**能找到入口，即达到「可维护性」目标。

---

## §3 目录太复杂时：项目级导航规则（README + INDEX）

在 `system_design` 或类似目录下，固定两个文件：

| 文件 | 用途 |
|------|------|
| **README.md** | 给人看的导航与复习路线 |
| **00_INDEX.md** | 给机器/自己查的「链接清单」 |

与「先选结构再填内容」的文档工程建议一致；Read the Docs 也推荐用 **Diátaxis** 做结构。

---

## §4 新问题来了放哪？：Inbox 规则（防止越写越乱）

**MUST**：任何新问题**先进入** `03_questions_inbox/`，回答后再归档到四类之一（Diátaxis）：

- **Tutorial**（教程）  
- **How-to**（操作指南）  
- **Reference**（参考）  
- **Explanation**（原理解释）  

这样主线不会断裂，新问题有固定落脚点。

---

## §5 最小执行建议（今天就能开始用）

| 对象 | 动作 |
|------|------|
| **所有 > 200 行的 .md** | 加上 **TL;DR + Quick Jump + TOC + Where to look** |
| **所有 > 800 行代码** | 加 **Module Docstring 地图 + 分区标记 + ENTRY/FLOW** |
| **所有 > 2000 行代码** | **拆分**（保留 1 个 entrypoint 文件 + 若干模块文件） |

---

## 6 下级概念拆分（问答驱动的文档派生）

**规则**：当对某父级文档（如 A2）中的**概念/术语**单独提问时（例如「什么是成功的 batch」），说明该概念是该父级的下级，应拆分为独立文件讲解。

**命名格式**：`KYC_day01_<父级ID>_B1_<简短主题>.md`

**示例**：
- 父级：`KYC_Day01_A2_指标计算脚本示例.md`
- 提问：「什么是完全成功的 batch 数」→ 下级概念
- 新建：`KYC_Day01_A2_B1_good_batch.md`

---

## 附录：与本规则相关的文件

- **KYC 教学规则**：[KYC_teaching_rules.md](KYC%20project/KYC_teaching_rules.md)（输出契约、Trade-off、SRE DoD、示例隔离）  
- **Diátaxis**：Tutorial / How-to / Reference / Explanation 四型文档结构  
- **RFC 2119**：MUST / SHOULD / MAY 等关键词的标准化含义  
- **System Design 字典规则**：[SYSTEM_DESIGN_DICTIONARY_RULES.md](SYSTEM_DESIGN_DICTIONARY_RULES.md)（分层字典、可追溯知识图谱、AI 输出限制）  
- **文件契约与结构规则**：[FILE_CONTRACT_AND_STRUCTURE_RULES.md](FILE_CONTRACT_AND_STRUCTURE_RULES.md)（契约头、大文件导航、Diátaxis、ADR）  