---
doc_type: explanation
layer: L0
scope_in:  分层字典规则、概念节点结构、下级概念派生规则、AI 输出限制规则
scope_out:  具体概念节点实现（见各节点文件）；文件契约规则（见 FILE_CONTRACT_AND_STRUCTURE_RULES.md）；结构规则（见 Structure_rule.md）
inputs:   (设计) 系统设计知识、概念、问题；(运行时) 用户提问、概念查询
outputs:  分层知识图谱、概念节点结构、AI 输出规则
entrypoints:
  - A0 目标（为什么需要分层字典）
  - A1 层级定义（L0-L4 是什么）
  - A2 字典条目（概念节点必须长什么样）
  - A3 下级概念派生规则（什么时候创建新节点）
  - Rule C：AI 输出限制规则（AI 如何输出）
children: []
related: [ FILE_CONTRACT_AND_STRUCTURE_RULES.md（文件契约规则）, Structure_rule.md（结构规则）, 00_GLOSSARY_INDEX.md（字典入口） ]
owner: you
last_updated: 2025-01-01
---

# System Design Dictionary Rule（分层字典 / 可追溯知识图谱）

> **目标**：任何问题都能一路追到最底层，又允许在任意层停下；提问 = 可管理的知识库增量。  
> **借鉴**：Diátaxis 文档分型 | RFC 2119 约束词 | Zettelkasten 原子化+链接 | ADR 决策记录

---

## TL;DR

- **5 层**：L0 概览 → L1 指标 → L2 权衡 → L3 机制 → L4 实现
- **概念节点**：Definition + Layer + Inputs/Outputs + Parent/Children/Related + Stop Points
- **下级派生**：被单独提问的概念 MUST 拆成节点；操作→How-to；决策→ADR
- **AI 输出**：先问/默认层级 → 只输出该层该输出的 → 必须写动作清单（新增哪些节点/How-to/ADR）

---

## A0 目标

把系统设计知识组织成**可分层停靠**的知识图谱：

- 问任何问题，AI 先判断/让你选：停在哪一层
- 想深挖：有明确的**下钻链接**一路追到 L4
- 不想深挖：在当前层得到完整答案（不被细节淹没）

---

## A1 层级定义（Layer Model）

| 层 | 名称 | 典型问题 |
|----|------|----------|
| **L0** | 概览（One-liner + Scope） | 这是什么、解决什么 |
| **L1** | 目标与指标（SLO / Success Metrics） | 怎么定义成功 |
| **L2** | 设计与权衡（Trade-offs / Alternatives） | 为什么选 A 不选 B |
| **L3** | 机制与模式（Mechanism / Patterns） | 限流、缓存、队列、幂等、回滚等 |
| **L4** | 实现与验证（Code / Test / Runbook） | 接口、代码入口、回归测试、演练 |

> 例：「confidence 90% 怎么来的」→ 通常 L2/L3（校准/不确定性机制），可下钻 L4（评测/校准实现）。

---

## A2 字典条目（Concept Entry）必须长什么样

每个概念是一个**节点文件**，**MUST** 包含：

| 字段 | 含义 |
|------|------|
| **Definition** | 一句话 + 边界（不包括什么） |
| **Layer Anchor** | 在 L0–L4 的哪一层 |
| **Inputs / Outputs** | 接收什么、产出什么 |
| **Links** | `Parent` / `Children` / `Related` |
| **Stop Points** | 「看到这里够了」的版本 & 「想更深看哪里」 |

> 对应 Zettelkasten：原子化 + 链接，不迷路。

---

## A3 下级概念派生规则（可执行）

| 触发条件 | **MUST** 动作 |
|----------|---------------|
| 对父级文档中的**概念/术语**提「定义/为什么/怎么判断」 | 创建下级概念节点，父文档加 `children:` 链接 |
| 问题是「怎么做 X」（操作步骤） | 写到 How-to，概念节点链接过去；**禁止**把操作塞进概念定义 |
| 问题是「字段/公式/参数表」 | 写到 Reference |
| 问题是「为什么选文件/批量/不实时」等**决策** | 写 ADR（Context / Decision / Consequences），概念节点链接 ADR |

**命名与路径**：

- 通用概念：`glossary/L{n}_{topic}.md`（如 `L3_rate_limit_backpressure.md`）
- 父级下的子概念：`glossary/<父级ID>_B1_<简短主题>.md`（如 `glossary/KYC_Day01_A2_B1_good_batch.md`）
- 父文档的 `children:` 必须列出上述路径

---

## Rule C：AI 输出限制规则

每次你提问，AI **MUST** 按以下流程输出：

### C1 先问层级（或按默认策略）

- 显式问：「你要停在 L1 指标层，还是下钻到 L3/L4？」
- **默认策略**（可配置）：
  - **S（面试向）**：默认给 **L1 + L2**（指标 + trade-off），附「下钻入口链接/关键词」
  - **B（学习向）**：默认给 **L0 → L3**（概览到机制），L4 仅在你点名要时给

> 当前默认：**S**

### C2 只输出该层该输出的东西

- 问 L1 → 指标定义 + 怎么衡量，不展开实现
- 问 L4 → 入口函数 / 测试策略 / 回归点

### C3 必须写出「动作清单」

- 这次问题会**新增**哪个 Concept 节点？
- 会**新增**哪个 How-to / Reference？
- 是否需要 **ADR**？

把「聊天」变成**可管理的知识库增量**。

---

## 附录：参考

- [Diátaxis](https://diataxis.fr/)：Tutorial / How-to / Reference / Explanation
- [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119)：MUST / SHOULD / MAY
- [Zettelkasten](https://zettelkasten.de/introduction/)：原子化 + 链接
- [ADR](https://adr.github.io/)：Architectural Decision Records
