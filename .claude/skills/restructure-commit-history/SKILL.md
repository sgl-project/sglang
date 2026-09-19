---
name: restructure-commit-history
description: >-
  Restructure and clean up a linear git commit history (typically a cherry-picked
  patch-migration range) by folding fix commits back into the commits that
  introduced the defect, deleting byte-inverse commit pairs, squashing fragmented
  same-topic series into self-contained units, optionally splitting grab-bag
  commits, and rewriting merged commit messages per the project template.
  Produces a written plan with full per-commit accounting first, then executes it
  via a two-pass interactive rebase under a safety-net branch with rerere enabled,
  and verifies byte-identical tree content (git diff old-head new-head must be
  empty). Use when the user mentions commit history restructuring, folding fixes
  into their origin commits, removing introduce-defect-then-fix-later traces,
  merging scattered duplicate commits, cleaning up a patch migration branch, or
  rewriting/reorganizing an existing commit range.
---

# Restructure Commit History

## Overview

对一段**线性**提交历史（典型场景：cherry-pick 迁移后的补丁链）做质量重构：
消除"先破坏后修复"痕迹、合并零散碎提交、删除无效提交、按模板重写合并后的
提交信息。

核心约束（不可违反）：**重构不改任何代码逻辑，最终树与重构前逐字节一致**
——`git diff <旧HEAD> <新HEAD>` 必须为空（等价判据：两个 HEAD 的 tree hash
相同）。提交历史变干净，代码内容一个字节都不动。

## 关键概念

| 概念 | 说明 |
|------|------|
| 基线 baseline | 重构范围的起点（范围外第一个提交），rebase 的 onto 目标 |
| 互逆对 inverse pair | 两个提交对同一文件的改动逐字节互逆、净效果为零 → 双双删除 |
| fix→origin 折叠 fold | 修复提交（fix）并入引入该缺陷的原始提交（origin），历史不再出现"引入→修复"轨迹 |
| 系列合并 squash | 同主题的多个碎提交（内核+使能+崩溃修复+接入层）合并为一个自洽功能单元 |
| grab-bag 拆分 split | 多主题混合提交拆成单职责提交（**可选项**，用户可随时取消） |
| 按文件分发 distribute | fix 的不同文件分别并入不同提交（fix 文件在 origin 时点不存在时使用） |
| 树不变量 | 重构前后最终树逐字节一致；唯一合法例外须在计划中显式声明并经用户同意 |
| 核算 accounting | 每个原提交都有明确处置（保留/删除/被吸收/被拆分），总数公式精确闭合 |

## 执行流程

```
Phase 1: 范围与安全网 → Phase 2: 审计与分类 → Phase 3: 可行性核查（零干扰验证）
→ Phase 4: 计划文档（核算闭合，用户确认）→ Phase 5: 两轮 rebase 执行
→ Phase 6: 验证（树哈希/diff/计数/range-diff/抽查）
```

**计划先行**：Phase 1–4 产出完整计划文档，经用户确认（或用户裁剪范围，如
取消拆分）后才进入 Phase 5 执行。执行中的任何偏差都要回写到计划文档。

---

## Phase 1: 范围与安全网

### 1.1 接收输入

向用户确认：

```
- 基线分支或 commit（重构范围起点，范围外第一个提交）
- 范围终点（通常为当前 HEAD）
- 重构目标：折叠哪些 fix、合并哪些系列、是否拆分 grab-bag
```

### 1.2 前置检查（不满足则停止）

```bash
git status --porcelain                    # 工作树必须干净（未跟踪文档除外）
git log --merges --oneline <base>..HEAD   # 必须为空（历史必须线性，rebase -i 前提）
git rev-list --count <base>..HEAD         # 记录原始提交数 N（注意口径：基线顶端提交是否计入）
git log --reverse --format="%H %s" <base>..HEAD   # 全量提交清单（后续审计的输入）
```

### 1.3 安全网（任何历史改写操作之前，必做）

```bash
git branch backup-restructure-<N> HEAD    # 冻结重构前现状（最终验证的对照物）
git config rerere.enabled true            # 冲突解法自动复用
git config rerere.autoUpdate true
```

---

## Phase 2: 审计与分类

通读全部提交（Phase 1.2 的清单），逐个 `git show --stat` + 读 message，
按四类处置分类。重点识别信号：

### 2.1 互逆提交对 → 删除

信号：A 改了某文件，隔若干提交后 B 又把它改回去（message 常含 restore/
revert/undo 语义）。验证逐字节互逆：

```bash
git show A -- <file>      # 与
git show B -- <file>      # 两个 diff 逐行互逆（+↔−），净效果为零
git diff A B -- <file>    # 若 B 只回退 A 的改动且无其他改动，接近空
```

### 2.2 fix→origin 折叠对

信号（满足其一即候选）：
- fix message 自述修复来源："repair ... left by X"、"fix crash introduced
  in ..."、"follow-up to ..."
- fix 的 diff 直接改写 origin 引入的行（新增行回退/修正）
- origin 与 fix 主题相同、文件相同或高度重叠

每对记录：fix hash、origin hash、涉及文件、缺陷机理（为什么 origin 有缺陷）。

### 2.3 系列合并组（M 组）

信号：同文件或同主题的 2–5 个邻近提交构成一个功能的碎片化演进
（内核落地 → 崩溃修复 → dtype 修复 → 接入层路由 → 测试）。整组合并为
一个自洽提交。**N>1 前移合并（如 5→1）标记为高风险组**，须预设降级方案
（拆成两个组，总数 +1）。

### 2.4 grab-bag 提交（可选拆分，用户可取消）

信号：message 为 "chore(misc): ..." 等笼统主题，diff 跨 ≥3 个互不相干的
主题/文件。拆分方案须列出每个新提交的文件清单与主题。**默认先列入计划，
用户明确取消则保持原样单提交并更新核算**。

---

## Phase 3: 可行性核查（零干扰验证）

对每个折叠对（fix → origin），验证两者之间无第三方提交触碰 fix 涉及的
文件——这是"预期零冲突"的依据，也是高风险组的风险评估输入。

### 3.1 常规方向（fix 晚于 origin）

```bash
for f in $(git diff-tree --no-commit-id --name-only -r <fix>); do
  echo "== $f"
  git log --oneline <origin>..<fix>^ -- "$f"   # 除 origin 自身外应为空
done
```

有第三方触碰不等于不可折叠：进一步比对 hunk 区域是否重叠
（`git show <fix> -- <f> | grep "^@@"` vs 第三方提交的同文件 hunk）。

### 3.2 两个必查陷阱（实战踩过）

1. **fix 早于 origin（时间倒置）**：如"守卫提交先于重写提交"——fix 给旧代码
   加守卫，origin 之后重写该文件。折叠方向仍是 fix 并入 origin，但核查范围
   方向要写对：`git log --oneline <fix>..<origin>^ -- <file>`
2. **fix 修改的文件在 origin 时点不存在**（文件由 origin 之后的提交创建）：
   整提交折叠不可行 → 改为**按文件分发**（Phase 5.3）。先查：
   ```bash
   git ls-tree <origin> -- <file>              # 输出为空 → origin 时点不存在
   git log --oneline --diff-filter=A -- <file> # 找到创建该文件的提交（分发目标）
   ```

### 3.3 核算闭合（硬约束）

```
目标提交数 = N − 互逆对删除数 ×2 − 折叠吸收数 ± 拆分净增减
```

附录映射表必须覆盖**每一个**原提交（保留 → 新编号 / 删除 / 被吸收 → 目标 /
被拆分 → 新编号），总数对不上就说明审计有遗漏，禁止进入执行。

---

## Phase 4: 计划文档

用 [assets/plan-template.md](assets/plan-template.md) 撰写计划，保存到
`docs/<project>-commit-restructure-plan-<version>.md`，包含：

- 总账表（N → M，核算公式）
- 每对折叠的 hash/文件/缺陷机理/验证结论
- 冲突热点预警（按风险排序，最高风险组放最后执行）
- 高风险组降级方案
- 最终提交序列（以原时序为骨架，标注 ⊕ 吸收关系）
- 执行手册（安全网、分阶段顺序、todo 骨架、验证清单）
- 附录：全量映射表

**计划完成后向用户展示摘要，等用户确认或裁剪**（如"取消拆分、执行其余"），
按裁剪结果更新核算再执行。

---

## Phase 5: 两轮 rebase 执行

总体策略：**Pass 1 只动内容（折叠/删除，todo 重排），Pass 2 只做分发折叠
与消息重写（edit 停靠）**。两轮分离使每轮的操作单一、冲突域可控；全程以
原时序为骨架，只做折叠所需的局部移动。

### 5.1 Pass 1 —— 内容折叠

预生成 todo 文件（**LF 行尾、无 CR**），规则：

| 处置 | todo 写法 |
|------|-----------|
| 整提交折叠（fix 全部文件可并入 origin） | fix 行改为 `fixup <fix-hash>`，**移动到 origin 的 pick 行之后**（不相邻也要移） |
| 按文件分发的折叠 | origin 标 `edit`；fix 的 pick 行删除（内容 Pass 2 分发） |
| 互逆对删除 | 两行 pick 直接删除 |
| 系列合并（组内相邻，基提交不变） | 基提交 `pick`，其余成员 `fixup` 移到其后 |
| grab-bag（未取消拆分时） | 标 `edit`，`git reset HEAD^` 后分块暂存逐个提交 |
| 其余提交 | `pick` 原样 |

```bash
# 注入预生成的 todo 文件启动 rebase：
GIT_SEQUENCE_EDITOR="cp <todo-file>" git rebase -i <baseline>

# 后续 continue 一律用 no-op editor（不再弹 todo）：
GIT_SEQUENCE_EDITOR=: git rebase --continue
```

fixup 行移动到 origin 之后意味着 fix 的 diff 通过三路合并应用到 origin 的
结果上——即使 fix 与 origin 不相邻甚至 fix 早于 origin，只要 Phase 3 验证
过零干扰（hunk 不重叠），即可干净合并且终态不变。

### 5.2 edit 停靠点操作（Pass 1 与 Pass 2 通用）

```bash
# 应用 fix 的指定文件到当前停靠提交：
git diff <fix>^ <fix> -- <file1> <file2> | git apply --3way --index -v

# 校验该文件与重构前终态一致（推荐）：
git diff --cached backup-restructure-<N> -- <file>   # 应为空

# 继续 rebase：
GIT_SEQUENCE_EDITOR=: git rebase --continue
```

### 5.3 按文件分发（fix 文件在 origin 时点不存在时）

fix 的每个文件分别并入"语义最合适"的提交：

- fix 改的是提交 X **引入**的代码且该文件由 X 创建 → 该文件并入 X
- helper 的**定义**与**消费**分属两个提交 → 定义并入提供方提交、消费并入
  使用方提交，保持"先提供后消费"的依赖顺序
- 测试文件的环境变量名修正 → 随对应测试的落地提交走

### 5.4 Pass 2 —— 分发折叠 + 消息重写

对每个 edit 停靠点（分发折叠的 origin + 全部 M 组基提交）：

1. 应用分发的 fix 文件补丁（同 5.2）
2. 合并提交信息：以基提交 message 为底，按
   `../patch-refinement/assets/commit-message-template.txt`
   （Why/How/Dependency/Test 四段）吸收被折叠提交的对应内容
   - Why：合并双方的问题陈述（去重）
   - How：合并双方的实现要点（按逻辑分组，非简单拼接）
   - Test：合并双方的 Unit/E2E 条目（去重）
3. amend：

```bash
git commit --amend --no-verify -F - <<'MSGEOF'
<合并后的完整提交信息>
MSGEOF
```

**amend 一律加 `--no-verify`**：pre-commit hooks 可能修改文件，破坏树一致性
（hooks 校验留给 Phase 6 之后的独立环节跑）。

署名规范：
- 基提交作者/日期不变（rebase 自然保留）
- 被吸收提交的作者不同时，补 `Co-authored-by: <name> <email>`
- 保留原 `(cherry picked from ...)` trailer（逐条保留，不合并改写）

### 5.5 高风险组降级

N>1 前移合并组（如 5→1）若冲突失控：拆成两个语义自洽的组（如"内核+使能"
与"hardening 修复"），总数 +1，并在计划文档记录偏差与理由。**不允许**为了
省事丢弃任何 hunk——树不变量优先于合并组数。

---

## Phase 6: 验证（全部必须通过，缺一不可）

```bash
# 1. 树不变量（最强判据）：两个树哈希必须完全相同
git rev-parse "HEAD^{tree}" "backup-restructure-<N>^{tree}"

# 2. diff 为空
git diff backup-restructure-<N> HEAD          # 期望：无输出

# 3. 提交数核算
git rev-list --count <baseline>..HEAD          # = N − 删除 − 吸收 ± 拆分

# 4. range-diff 逐条对照（= 内容等价 / ! 预期修改组 / 无未配对项即无丢失）
git range-diff <baseline> backup-restructure-<N> HEAD

# 5. 被吸收内容抽查：每个被折叠 fix 的关键符号/逻辑在最终树中存在
git show HEAD:<file> | grep -c "<symbol>"

# 6. 语法抽查（被触及的 .py 文件）
python -m py_compile <touched .py files>

# 7. （环境允许时）pre-commit hooks 与单测——在树一致性验证之后单独跑
```

验证通过后，把执行结果回写到计划文档（实际范围、偏差、验证数据、最终
提交序列），完成闭环。

---

## 常见陷阱与注意事项

1. **Windows autocrlf 幻影改动**：rebase 后 `git status` 显示若干 M 但
   `git diff` 为空 → `git add <files>` 刷新即可；确认
   `git diff --cached HEAD` 为空后继续，勿误判为真实改动
2. **pre-commit hooks 破坏树一致性**：所有 `git commit --amend` 一律
   `--no-verify`；hooks 校验作为独立环节在树验证后执行
3. **todo 文件行尾**：必须 LF、无 CR（Windows 下生成后 `grep -c $'\r'`
   应为 0），否则 todo 解析异常
4. **空 commit message**：`-F <file>` 读到空文件会 abort；长消息用
   heredoc `git commit --amend -F - <<'MSGEOF'` 最稳，内容含特殊字符时
   定界符加引号防展开
5. **fix 早于 origin**：干扰核查范围方向别写反（3.2 陷阱 1）
6. **文件时点不存在**：折叠前先 `git ls-tree <origin> -- <file>`，不存在
   则按文件分发（3.2 陷阱 2）
7. **核算是硬约束**：映射表缺一个提交就说明审计遗漏，禁止带着缺口执行
8. **拆分计划可被用户中途取消**：取消后核算更新、拆分专属的树清理一并
   跳过（保证树逐字节一致）
9. **口径注意**：`git rev-list --count <base>..HEAD` 与"范围提交数"可能
   差 1（基线顶端提交是否计入），全程统一口径并在文档写明
10. **fixup 跨距离移动是安全的**：前提是 Phase 3 验证过 hunk 不重叠；
    冲突出现时优先核对是否触碰了 Phase 3 判定"不重叠"的区域
11. **每个 edit 停靠点做完立即校验**：`git diff --cached <backup> -- <file>`
    为空再做 amend，不要把问题攒到最后
12. **中断恢复**：rebase 中断（超时/人为）后先 `git status` 确认停靠点与
    暂存区状态，从中断处继续，勿 reset 重来（rerere 已记录解法）

## 参考文件

- 命令速查与实战案例：[reference.md](reference.md)
- 重构计划文档模板：[assets/plan-template.md](assets/plan-template.md)
- 提交信息模板：`../patch-refinement/assets/commit-message-template.txt`
- 实战计划文档（v0.5.19_rel，81→65）：`docs/ppu-commit-restructure-plan-v0519.md`
