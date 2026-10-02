---
name: cherry-pick-ppu-patches
description: >-
  Cherry-pick PPU adaptation patches from a source branch (based on sglang release)
  to the target branch (latest sglang community branch). Scans source branch for
  PPU-specific changes, analyzes each commit's message and diff, applies patches
  one-by-one in original commit order with conflict resolution, and performs
  interface compatibility checks, data type extension adaptation, and Hook
  mechanism maintenance. Use when migrating PPU patches between sglang versions,
  porting hardware adaptation code, or when the user mentions cherry-pick, PPU,
  patch migration, branch rebase, or version upgrade.
---

# Cherry-pick PPU Patches

## Overview

将源分支（如 `0.5.17`）上的 PPU 适配补丁迁移到目标分支（如 `0.5.18`）。
PPU ZW 系列硬件，通过 CUDA 兼容 API 接入 sglang，设备名含 `"ZW"`，
device_type 保持 `"cuda"`，通过 `is_ppu()` / `is_cuda_alike()` 识别。

## 关键概念

| 概念 | 说明 |
|------|------|
| PPU 平台识别 | 统一以 `torch.cuda.get_device_name()` 含 `"ZW"` 判定（`is_ppu()` / `is_ppu_available()` / `is_ppu_runtime()`）|
| CUDA 兼容 | PPU `is_cuda()` 返回 True，复用社区 CUDA 路径 |
| 数据类型 | 社区默认 FP8，PPU 需同时支持 FP16/BF16/INT8 等 |
| Hook 机制 | PPU 通过 REPLACE/AROUND hook 替换社区函数行为 |
| pyproject | PPU 独立构建配置 `pyproject_ppu.toml`，kernel AOT 也有 `pyproject_ppu.toml` |

## 执行流程

```
Phase 1: 确定提交范围 → Phase 2: 分析提交内容 → Phase 3: 确认提交顺序
→ Phase 4: 逐个 Cherry-pick + 冲突处理 + 逐 commit Review → Phase 5: 七大适配检查 → Phase 6: 验证
```

---

## Phase 1: 确定提交范围

### 1.1 接收输入

向用户确认以下信息：

```
- 源分支（如 release/0.5.17 或 origin/release/0.5.17）
- 目标分支（如 release/0.5.18 或 origin/release/0.5.18）
- 起始 commit hash（包含）
- 结束 commit hash（包含）
- 可选：需要排除的 commit hash 列表
```

### 1.2 获取提交列表

```bash
# 确保两个分支的最新代码已拉取
git fetch origin

# 获取范围内的所有提交（从旧到新）
git log --reverse --oneline <起始commit>^..<结束commit> --no-merges
```

### 1.3 过滤 PPU 相关提交

扫描每个 commit message，关键词包括但不限于：
`ppu`, `PPU`, `ZW`, `flashmla`, `is_ppu`, `is_cuda_alike`,
`pyproject_ppu`, `sm80a`, `device_name`

**输出**：生成过滤后的 PPU 相关 commit 列表，记录 hash、message、日期。

---

## Phase 2: 分析提交内容

### 2.1 逐个分析 commit message

对每个 PPU 相关 commit 执行：

```bash
git show --stat <commit_hash>
git log -1 --format="%H%n%an%n%ad%n%s%n%b" <commit_hash>
```

### 2.2 理解修改目的和影响范围

为每个 commit 记录：

```
Commit: <hash>
Message: <subject line>
目的: <修改意图>
影响范围:
  - 涉及文件: <file list>
  - 涉及模块: <platforms / kernels / quantization / utils / ...>
  - 修改类型: <新增 / 修改 / 删除>
  - 是否涉及接口变更: <是/否>
  - 是否涉及数据类型: <是/否>
  - 是否涉及 Hook: <是/否>
```

### 2.3 查看 diff 内容

```bash
git show <commit_hash> -- <specific_file_path>
```

重点关注：
- 函数签名变化（参数增减、类型注解变更）
- import 路径调整
- 新增的条件分支（如 `if is_ppu()` / `if is_cuda_alike()`）
- 数据类型相关代码（如 `torch.float8_e4m3fn`）

---

## Phase 3: 确认提交顺序

**原则**：按原始提交时间顺序（从旧到新）逐个 cherry-pick，不重新分组或重排。
原始顺序已隐含依赖关系（后提交的 commit 基于前一个），保持原序可最大程度避免冲突。

### 3.1 生成有序提交列表

将 Phase 1.2 获取的 commit 列表（`--reverse` 已保证从旧到新）直接作为执行顺序。

### 3.2 标注关键属性（用于 Phase 5 适配检查）

为每个 commit 标注是否涉及以下五类检查，便于后续快速定位：

```
Commit: <hash>
Message: <subject>
涉及接口兼容检查: <是/否>  — 修改了社区核心函数签名/import 路径
涉及数据类型适配: <是/否>  — 新增/修改 FP8 或其他数据类型路径
涉及 Hook 维护:    <是/否>  — 修改了被 PPU Hook 替换的社区函数
涉及 C/CUDA 源码:  <是/否>  — 新增/修改 kernels/{jit,aot}/csrc 下的 .cu/.cuh（见 5.6）
涉及能力守卫:      <是/否>  — 新增 is_ppu() 守卫规避底层 kernel 能力缺失（见 5.7）
```

### 3.3 可选：排除不相关 commit

如果范围内存在非 PPU 相关 commit（如纯社区 bug 修复），经用户确认后排除。

---

## Phase 4: 逐个 Cherry-pick 与冲突处理（按原始顺序）

### 4.1 应用前准备

```bash
# 切换到目标分支
git checkout <目标分支>
git pull origin <目标分支>

# 创建工作分支
git checkout -b ppu-migration-<源版本>-to-<目标版本>
```

### 4.2 逐个 Cherry-pick（按原始时间顺序）

对每个 commit，按 Phase 1.2 获取的原始顺序（从旧到新）依次执行：

```bash
git cherry-pick <commit_hash>
```

**不重排、不分组**：直接按 `git log --reverse` 输出顺序逐个应用。

### 4.3 冲突处理

如果出现冲突：

1. **查看冲突文件**：`git status` 和 `git diff`
2. **分析冲突原因**：
   - 社区新版本已重构该文件 → 需手动重新适配
   - 上下文偏移（行号变化）→ 手动调整 hunk 位置
   - 同一行被双方修改 → 需理解双方意图后合并
3. **解决冲突**：编辑冲突文件，保留正确的 PPU 适配逻辑
4. **标记已解决**：`git add <file>`
5. **继续 cherry-pick**：`git cherry-pick --continue`

### 4.4 零冲突 ≠ 正确落地：结构锚点漂移与主动移位

**cherry-pick 的决策集不是二选一**（apply：三路合并成功即采纳 / abandon：解不动
冲突就放弃）——存在第三种动作：**移位**。三路合并只保证 hunk 文本层面可融合，
不保证语义落点正确：社区重构过补丁触碰的结构区域（类体边界移动、方法抽为混入
类、类/函数之间插入新定义、函数搬家）后，hunk 的文本锚点漂移，合并会把新增代码
**零冲突**地塞进错误的类体/函数体——无任何提示，且符号存在、语法合法，
§5.4 的存在性检查全部通过，炸点后移到运行时（完整形态分析见 5.4 形态 5）。

历史案例（v0.5.19，`is_acext`）：源补丁在 `MoeRunnerBackend` 枚举体内、紧跟
`is_aiter` 之后新增 `is_acext` 谓词；目标基线已把全部谓词重构进
`_MoeRunnerBackendPredicates` 混入类，并在该枚举与 `DeepEPMode` 之间插入了两个
新类。三路合并以漂移后的「`class DeepEPMode` 前的类体边界」为锚，把 `is_acext`
零冲突塞进了 `DeepEPv2Fp8ScaleFormat`（NamedTuple）体内。PPU 实机调用
`moe_runner_backend.is_acext()` 才 AttributeError——apply 看似成功，落点是错的。

**执行时快速判别**（对「向类体/函数体内新增代码」的 hunk，pick 完立即做）：

```bash
# 源补丁中新增块的上方上下文（应是同类兄弟成员，如 def is_aiter）
git show <源commit> -- <file>
# 落地结果中同一新增块的上方上下文（若变成数据字段/无关成员 → 落错类了）
git show HEAD -- <file>
```

判据：新增 `def`（或代码块）**上方的兄弟成员**应与源补丁语义一致。整类归属
存疑或需要全量复核时，跑类成员归属审计（源所属类 vs 目标所属类 + 兄弟方法集
重叠度投票，自动区分 MISPLACED / ADAPTED）：

```bash
python .claude/skills/cherry-pick-ppu-patches/assets/audit_class_placement.py <base_ref>
```

**处置**：既不能保留（语义错误会在运行时炸），也不该 abandon 整个补丁（新增
内容本身是对的，问题只在落点）——**移位**：编辑文件，把新增块搬进语义正确的
类/函数体。正确归属的定位方法：以源补丁中的兄弟成员为参照；若社区把方法抽为
混入类/基类，则搬进对应混入类并沿用其代码风格（如谓词混入类用
`self.value == X.value` 比较）。完成后 `git add <file> && git cherry-pick --continue`
（pick 已结束的用 `git commit --amend` 折入，不留 fix-later 痕迹）。
### 4.5 逐 commit Review 被修改文件（强制）

**每完成一个 cherry-pick，立即 review 该 commit 修改的所有文件**，不要等全部 pick
完再统一检查——后续 commit 会叠加在问题之上，越晚发现修复成本越高（极端情况需要
历史改写）。

```bash
# 查看刚应用的 commit 修改了哪些文件
git show --stat HEAD

# 对被修改的 .py 文件做名称解析检查（import 完整性，见 5.4）
python .git/scan_missing_imports.py <changed .py files>

# 快速语法检查
python -m py_compile <changed .py files>
```

**Review 要点**（逐文件人工过一遍 diff：`git diff HEAD^ HEAD -- <file>`）：

1. **Import 完整性**：本 commit 新增的代码引用了哪些名字？这些名字的 import 在文件
   中是否存在（cherry-pick 迁移盲区，见 5.4）
2. **上下文正确性**：新增代码引用的函数/类/模块路径在目标基线中是否存在、签名是否
   一致（接口兼容，见 5.1）
3. **落点正确性**：hunk 是否应用到了正确的类体/函数/分支内——上下文偏移（社区重构
   类边界/函数搬家）可能导致代码落到错误位置且**零冲突**；判别与移位处置见 4.4
4. **配置读法正确性**：本 commit 新增的 `server_args.<field>` 读取点——字段是否
   标注 `resolvable=True`？是则必须用分层访问器（get_schedule() 等，同文件既有
   惯例为参照）；CLI 直填字段直读合法（见 5.4 形态 6）
5. **遗留冲突标记**：`<<<<<<<` / `>>>>>>>` / `=======` 是否清理干净
6. **构建配置引用**：commit 若涉及 `kernels/aot/csrc/` 下文件的删除/移动/重命名，
   立即审计 PPU 构建文件中的静态引用是否悬空（见 5.5）
7. **C/CUDA 源码基线对账**：commit 若新增/修改 `kernels/jit/csrc/` 或 `kernels/aot/csrc/`
   下的 `.cu`/`.cuh`，逐个核对命名空间归属、共享宏/`constexpr`、`struct` 字段数、
   helper 函数签名是否仍与目标基线一致（见 5.6）——5.4 的全部工具链只覆盖 `.py`，
   完全不看 C++ 源码
8. **PPU 能力守卫覆盖**：commit 若新增 `is_ppu()` 守卫来规避某个底层 kernel 的能力
   缺失（split-k、smem 上限、dtype 支持…），立即在目标基线中搜索该 kernel 的**全部
   调用点**，确认社区新增的调用路径（尤其是把两步合并的“融合/快速路径”）同样被
   守卫覆盖（见 5.7）

发现问题当场修复（`git commit --amend`），并记录到 4.7 的日志中。

### 4.6 跳过无效补丁

如果某个 commit 的修改已被社区新版本直接包含（社区已自行修复）：
```bash
git cherry-pick --skip
```
记录跳过原因到执行日志。

### 4.7 Cherry-pick 日志模板

每个 commit 应用后记录：

```
Commit: <hash>
  Message: <subject>
  状态: ✅ 成功 / ⚠️ 有冲突(已解决) / 🔀 零冲突但落点错误(已移位, 见 4.4) / ⏭️ 跳过
  冲突文件: <list or N/A>
  冲突原因: <description or N/A>
  解决方式: <description or N/A>
  适配修改: <description or N/A>
  Review: ✅ 无问题 / ⚠️ 发现问题(已修复): <description>
```

---

## Phase 5: 七大适配检查

全部 Cherry-pick 完成后，执行以下七项检查。如中途遇到严重冲突，可在解决该 commit 后立即对涉及文件做局部检查。

### 5.1 接口兼容性检查

**目标**：PPU 复用了社区核心函数，需确认这些函数在新版本中未发生破坏性变更。

**检查范围**：

| 检查项 | 方法 |
|--------|------|
| 函数签名变化 | 对比源/目标分支中 PPU 复用的社区函数签名 |
| Import 路径调整 | 搜索 PPU 代码中的 import 语句，确认目标分支中路径仍有效 |
| 类/方法重命名 | 搜索 PPU 引用的类名/方法名，确认目标分支中仍存在 |
| 参数语义变更 | 确认 PPU 调用社区函数时传参方式与新版本一致 |

**执行步骤**：

1. 从 cherry-pick 的 diff 中提取 PPU 代码引用的所有社区函数/类
2. 在目标分支中搜索这些符号：

```bash
# 搜索函数定义
grep -rn "def <function_name>" python/sglang/
# 搜索类定义
grep -rn "class <class_name>" python/sglang/
# 搜索 import
grep -rn "from.*import.*<symbol>" python/sglang/
```

3. 对比签名是否一致，如不一致则适配 PPU 调用代码

**典型场景**：
- 社区将 `some_func(a, b)` 改为 `some_func(a, b, c=None)` → PPU 调用通常兼容
- 社区将 `some_func(a, b)` 改为 `some_func(a, b, *, strict=True)` → 检查 PPU 是否用了位置参数
- 社区将模块从 `sglang.srt.utils.common` 迁移到 `sglang.srt.utils.device` → 更新 PPU import
- 社区迁移 + 改签名同时发生（如 `get_req_to_token_extra_context_len(server_args)`
  从 `mem_cache/common.py` 迁到 `mem_cache/allocation_sizing.py` 且改为无参 `()`）
  → import 路径与调用实参**同步**适配，二者只改其一仍会炸

**隐形引用源（易漏）**：步骤 1 提取“PPU 引用的社区函数/类”时，不要只看 diff
对社区文件的修改——**PPU 新建的文件**（如从社区文件提取出的 mixin、新模块）会
整体携带老基线的 import 与调用，同样属于社区接口引用；对 diff 中每个新增的
`.py` 文件，机械核对其中全部 `from sglang.* import` 与社区函数调用的实参列表
（对照目标分支定义处），不能依赖人工“记得”哪些是对社区的引用。

### 5.2 数据类型扩展适配

**目标**：社区为 FP8 新增功能时，PPU 需同步适配其他数据类型（FP16, BF16, INT8 等）。

**检查方法**：

1. 在 cherry-pick 的社区 diff 中搜索 FP8 相关代码：

```bash
# 搜索新增的 FP8 路径
grep -n "fp8\|float8\|FP8\|e4m3\|e5m2" <cherry-picked files>
```

2. 分析新增 FP8 代码路径是否需要为其他数据类型添加分支：

```python
# 社区可能新增的模式（仅 FP8）:
if dtype == torch.float8_e4m3fn:
    do_fp8_path()

# PPU 需适配为:
if dtype == torch.float8_e4m3fn:
    do_fp8_path()
elif dtype == torch.bfloat16:
    do_bf16_path()  # PPU 扩展
elif dtype == torch.float16:
    do_fp16_path()  # PPU 扩展
```

3. 检查量化配置中的数据类型条件：

```python
# 搜索数据类型判断分支
grep -n "supports_fp8\|is_float8\|dtype.*==.*float8" <cherry-picked files>
```

4. 确认 PPU 的 `supports_fp8()` 和数据类型路由逻辑仍与社区一致

**关键文件**：
- `python/sglang/srt/layers/quantization/fp8.py`
- `python/sglang/srt/platforms/ppu.py`（`supports_fp8()` 定义）
- 任何新增 `torch.float8_*` 引用的文件

### 5.3 Hook 机制维护

**目标**：PPU 通过注册 Hook（REPLACE/AROUND）替换社区函数行为，需确认被替换的社区函数本身未发生变化。

**检查方法**：

1. 在 cherry-pick 的 diff 中识别所有被 Hook 替换的社区函数
2. 对比这些函数在源/目标分支中的实现：

```bash
# 查看目标分支中的函数实现
git show <目标分支>:<file_path> | grep -A 30 "def <hooked_function>"
# 查看源分支中的函数实现
git show <源分支>:<file_path> | grep -A 30 "def <hooked_function>"
```

3. 判断是否需要调整 PPU Hook：
   - **社区函数签名变更** → 更新 PPU Hook 的参数传递
   - **社区函数逻辑变更** → 评估 PPU Hook 是否仍需替换该函数
   - **社区函数已自行处理 PPU 场景** → 移除 PPU Hook（参考 FlashMLA 清理经验）
   - **新增参数/分支** → PPU Hook 需同步处理新参数/分支

4. Hook 清理原则（来自项目经验）：
   - 当社区函数已自行处理 PPU 所需逻辑时，直接移除对应 Hook 定义
   - 同步更新 Hook 文件的模块文档字符串和 section 注释
   - 更新后续 Hook 的编号

**当前已知 PPU FlashMLA Hooks（6 个，供参考）**：

| # | Hook 目标函数 | 类型 | 说明 |
|---|------------|------|------|
| 1 | `get_mla_metadata` | REPLACE | PPU MLA metadata 计算 |
| 2 | `flash_mla_with_kvcache` | REPLACE | PPU MLA forward |
| 3 | `flash_mla_sparse_fwd` | REPLACE | PPU sparse forward |
| 4 | `init_cuda_graph_state` | AROUND | cap mla_metadata buffer to 320 sm_parts |
| 5 | `_apply_decode_target_verify_metadata` | REPLACE | PPU metadata handling |
| 6 | `init_forward_metadata_in_graph` | REPLACE | reset have_initialized for lazy init |

### 5.4 Import 与符号引用完整性检查（AST 名称解析 + 符号对账）

**目标**：堵住 cherry-pick 的符号迁移盲区（六种形态，同根源——基线漂移）：
1. **模块级名字悬空**：源文件中某个 import 由范围外 commit 引入，而目标基线
   （社区新版本）恰好已重构掉该 import，则“使用”会落地但“import”缺失 →
   运行时 NameError
2. **envs 类属性悬空**：`envs.XXX` 属性访问落地，但 environ.py 中类属性已被
   社区清理 commit 删除 → 运行时 AttributeError
3. **from-import 符号悬空（社区函数/类/常量接口）**：`from sglang.xxx import
   <func/class>` 落地，但目标符号已被社区重构删除（重命名/合并/移入子包）→
   import 语句执行时 ImportError
4. **接口迁移 + 签名漂移（符号仍在，只是搬家/改签名）**：社区把函数迁到新模块
   （老 import 路径失效 → ImportError）**且**修改签名（如删参数）；源基线上合法的
   调用落地后双重断裂。即使 import 路径碰巧正确、符号存在性对账通过，带参调用
   无参函数仍是运行时 TypeError（`takes 0 positional arguments but 1 was given`）
5. **类成员归属错位（方法落错类，零冲突静默漂移）**：源补丁在类 A 体内新增
   `def m(self)`，社区基线把类 A 重构过（方法抽出为混入类 / 类体间插入新类），
   三方合并以类体边界作锚点，锚点漂移后该 hunk 被**无冲突**塞进相邻类 B 体内。
   符号存在性对账、语法检查、AST 名称解析全部通过（`m` 确实有定义），炸点后移到
   `instance.m()` 运行时 AttributeError。v0.5.19 实例：`is_acext` 落进
   `DeepEPv2Fp8ScaleFormat`（NamedTuple）而非 `_MoeRunnerBackendPredicates`
   （`MoeRunnerBackend.is_acext()` AttributeError）。识别特征：**hunk 落点上方
   不是源补丁里的类体边界**——源头 `is_acext` 紧跟 `is_aiter`（同类兄弟谓词），
   落点上方却是 `ue8m0: bool`（数据字段）。检查方法：对迁移范围内新增的每个
   `def`，用 AST 取「源补丁中所属类 → 目标终态中所属类」，不一致即告警；再以
   「兄弟方法集重叠度」投票确认正确归属（见下方类成员归属对账）
6. **访问惯用法漂移（配置读取契约变更，零冲突静默漂移）**：源基线在解析结束把声明「物化」回 ServerArgs 字段（如 materialize_declarations），目标基线已删除物化、改为声明进 stash 投影进 config bag——`server_args.<field>` 原始直读在源上是合法读法，cherry-pick 逐字搬运后在目标上读到 None/旧值。符号存在、语法合法、AST 解析通过、源文本逐字一致（保真度检查全部通过），炸点后移到运行时（TypeError/AttributeError/静默行为分叉）。判别判据：字段在 server_args.py 标注 `resolvable=True`（值可能来自模型覆盖/后处理 pass 声明）→ 必须走分层访问器（get_schedule()/get_disagg()/...）；CLI 直填字段（resolvable=False）原始直读合法。v0.5.19 实例：`_estimate_req_to_token_pool_bytes` 读 `self.server_args.page_size`（None）传入 `page_size % 128` 断言炸 TypeError——同函数的 max_mamba_cache_size 已适配 get_schedule()、唯独 page_size 漏改（同函数适配不一致是最强信号）；one_batch.py 的 disable_overlap_schedule/page_size 同病（bench 静默分叉）。检查方法：读法分歧对账（见下方 server_args 读法对账）

**为什么常规手段发现不了**：
- 零冲突：PPU commit 新增的使用代码（如 `einops.rearrange(...)`）不与目标基线的
  import 区重叠，cherry-pick 干净合并，无人工检查点
- py_compile 只查语法，NameError 是运行时名称解析错误
- 逐 commit 身份对账（author/date/subject）不校验文件内容
- **AST 名称解析也有盲区**：`envs.SGLANG_XXX` 是合法的属性访问（`envs` 本身已正确
  import，名字已绑定），悬空的是 `Envs` 的**类属性**——解析层面完全正常，需单独对账
  （见下方 envs 属性对账）
- **from-import 符号悬空同理**：import 语句把名字绑定得完好无损，AST 名称解析
  不报错——炸点后移到 import 语句执行时（ImportError），需单独对账（见下方
  from-import 符号对账）
- **符号对账只查存在性、不查签名**：from-import 符号对账能抓住路径悬空，但抓
  不住签名漂移——`f(server_args)` 调用无参函数在名字解析层面完全合法，炸点后移
  到函数调用时（TypeError）；必须对照**定义处的参数列表**逐个核对调用实参
  （见 5.1 接口兼容性检查，二者互补：5.4 查“符号在不在”，5.1 查“签名对不对”）
- **符号对账也不查类内归属**：`def m` 名字存在、语法合法、AST 解析通过，
  但可能挂在错误的类体里（形态 5）——存在性检查对「落错类」完全免疫；必须用
  「源所属类 vs 目标所属类」AST 对账才能发现（见下方类成员归属对账）
- **保真度检查对「读取契约变更」完全免疫**：迁移代码与源补丁逐字节一致（冲突检查、源文本对齐全通过），但源/目标基线对同一字段的读取契约已分叉（物化直写 vs 声明投影）——缺陷不在文本层，在两个基线的语义差里。必须做「读法分歧对账」：新增的 `server_args.<field>` 直读 vs 同文件既有访问器惯例 + 字段 resolvable 标注（见下方 server_args 读法对账）

**检查方法**（对终态所有被修改的 .py 文件执行 AST 名称解析）：

1. 收集文件内所有名字绑定（bindings）：import / def / class / 赋值目标 / 函数参数 /
   except as / global / walrus / lambda 参数
2. 收集所有名字使用（loads）：Load 上下文的 Name 节点
3. `undefined = loads - bindings - BUILTINS`
4. **与源终态交叉验证消除误报**：取源 tip 同名文件的顶层 import 名集合，
   `CONFIRMED = undefined ∩ src_imports` —— 源终态有 import 而目标终态没有，
   才是迁移引入的缺失

**工具**（本仓库已内置，位于 `.git/`）：

```bash
# 单文件扫描（无参则扫描内置默认清单）
python .git/scan_missing_imports.py <file1> <file2> ...

# 全量扫描：自动取 pick 范围内全部 .py 文件逐个检查，CONFIRMED 必须为 0
python .git/run_full_scan.py

# envs 属性对账：全部 envs.<ATTR> 使用 vs environ.py 类定义，DANGLING 必须为 0
python .git/check_envs_attrs.py

# from-import 符号对账：全部 from sglang.* import <sym> vs 目标模块顶层绑定，
# DANGLING 必须为 0（社区函数/类/常量接口引用）
python .git/check_import_refs.py

# server_args 读法对账（见 5.4 形态 6）：迁移提交新增的 server_args.<field>
# 原始直读 vs parent 同文件访问器惯例 + 字段 resolvable 标注，
# DIVERGENT(resolvable=True) 必须人工甄别为 0
python .claude/skills/cherry-pick-ppu-patches/assets/audit_serverargs_reads.py
```

**已知误报**：match-case 捕获绑定（`case NextNEnabledConfig(nextn_layer_prefix=layer_prefix)`）
在扫描器未处理 ast.MatchAs 时会被误报为 undefined，需人工排除。
envs 对账侧：注释行（如社区 TODO `# if envs.VLLM_...`）、独立模块的
`lora_envs` / `multimodal_gen` 的 `envs` 需排除（工具已内置过滤）。
from-import 对账侧：子模块导入（`from pkg import submodule` 是合法 Python
回退机制，工具已内置跳过）、try/except ImportError 可选依赖（工具已内置
跳过）、`__getattr__` 动态导出（如 dsa/__init__.py 的
CuteDSLPagedMQALogitsRunner，工具已跳过）、star re-export（
`from .x import *`，如 srt/distributed/__init__.py 的几十个函数——工具标记
但不确定符号，需人工确认符号来自被 star 的子模块）。

**envs 属性对账方法**（check_envs_attrs.py 的逻辑）：
1. 使用侧：提取全部 `.py` 活代码（剔除注释）中的 `envs.<ATTR>` 属性访问
2. 定义侧：`environ.py` 中 `class Envs` 的全部 `XXX = EnvBool/EnvInt/...` 属性
3. 差集（使用但未定义）必须为 0——否则运行时 `AttributeError`

**from-import 符号对账方法**（check_import_refs.py 的逻辑，覆盖对社区
函数/类/常量等一切接口的 `from sglang.* import <symbol>` 引用）：
1. 使用侧：全部 `.py` 文件中 `from sglang.<mod> import <sym>` 的 sym
2. 定义侧：目标模块源文件 AST 解析出的顶层绑定（def/class/赋值/import 再导出）
3. 差集（引用但目标模块没有）必须为 0——否则运行时 ImportError
4. 误报排除见上方“已知误报”

**实际事故案例**（53 commit 迁移后人工发现 3 处 import，均由本机制确认）：
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`：使用 `einops` 无 import
- `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`：使用 `envs` 无 import
- `python/sglang/srt/mem_cache/memory_pool.py`：使用 `index_buf_accessor` 无 import

v0.5.18_rel 迁移后另发现 2 处 envs 属性悬空（同一元凶 bc312d185d 清理 commit）：
- `envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE`（arg_groups/overrides.py）：PPU guard 使用，
  社区清理 commit 删除定义——**任意 server 启动必炸**：`_a2a_backend_overrides`
  是 `@register_post_process` 无条件执行，且 `or` 短路求值使右侧
  `envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get()` 必然求值（条件是否满足都炸）
- `envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION`（moe_runner/deep_gemm.py 4 处）：PPU
  DeepGEMM backend 使用，同被 bc312d185d 删除——分支内炸（MoE runner 路径触发）

v0.5.19 迁移后另发现访问惯用法漂移 2 处（形态 6，同一元凶——源基线有
materialize_declarations 物化、目标基线已删除改为声明投影）：
- `mem_cache/kv_cache_configurator.py` `_estimate_req_to_token_pool_bytes`
  （847757ea14 新增）：`self.server_args.page_size` 读到 None 传入
  `DSV4PoolConfigurator.calculate_pool_sizes` 的 `page_size % 128` 断言，
  DSV4+EAGLE+dp-attention 启动必炸 TypeError。铁证：同函数紧邻的
  max_mamba_cache_size 已适配 `get_schedule().`（源同为 server_args.* 读法），
  唯独 page_size 漏改——**同函数适配不一致是此形态的最强信号**
- `benchmark/one_batch.py`（43b83be12e8a）：`server_args.disable_overlap_schedule`
  与 `server_args.page_size` 原始直读——同文件 508/570 行已有
  `get_schedule().` 正确惯例，bench 判定的 batch 类型与真实 worker 分支静默分叉

v0.5.18_rel 迁移后另发现接口迁移 + 签名漂移 1 例（形态 4，两处调用 + 一处 import）：
- `get_req_to_token_extra_context_len`：社区 e789ca24a7 将其从
  `mem_cache/common.py` 迁到 `mem_cache/allocation_sizing.py`，d2bc697396 又把
  签名从 `(server_args, *, max_draft_tokens=None)` 改为无参 `()`（配置改从
  bags 全局取）。源链（老基线）上两处调用都合法，cherry-pick 到 v0.5.18
  基线后双重断裂：
  - `model_executor/model_runner_kv_cache_mixin.py`（4fe9f70b59 新建文件，从
    ModelRunner 提取 KV 初始化逻辑时整体带入老基线 import）：
    `from ...mem_cache.common import ...` → ImportError（common 里已无此函数）
  - `mem_cache/kv_cache_configurator.py` 尾部 `estimate_req_to_token_pool_bytes`
    （aea6da6cc3 新增函数）：`get_req_to_token_extra_context_len(server_args)` →
    TypeError（同文件 import 路径碰巧正确、符号存在，from-import 对账通过，
    签名断裂漏网）
- 教训：**PPU 新建文件是隐形引用源**——从社区文件提取出的 mixin/新模块会整体
  携带老基线的 import 与调用，它们同样是对社区接口的引用，必须在 5.1 的
  接口兼容性检查范围内（对新增文件的每个 `from sglang.* import` 与每个社区
  函数调用，对照目标基线定义处签名核对）

修复方式（按形态区分，原则：**社区已删除的接口 = 社区已废弃的路线**）：
- **模块级名字悬空（import 缺失）**：历史改写，在第一个使用该名字的 commit 中
  补 import（插入位置与源终态 import 块中的相邻 import 对齐）——模块本身仍在，
  补 import 只是恢复引用
- **envs 属性悬空 / from-import 符号悬空（社区已删除的接口）**：**禁止复活**
  ——不得在 environ.py 恢复定义、不得在目标模块补回符号。社区删除即社区废弃，
  PPU 应跟随社区清理方向，在**使用侧消除依赖**：按语义折叠分支（保留 PPU 实际
  需要的行为路径、删除死分支），或改用社区新接口/等价实现
- **接口迁移/签名漂移（形态 4，符号仍在、社区仍维护）**：与“社区已删除”相反，
  **适配新接口**——import 路径与调用签名同步更新到目标基线的定义处；参数被
  删除的，先确认新实现内部已从全局配置（如 bags）取到等价值，再去掉实参
- **访问惯用法漂移（形态 6，字段被声明式管线接管）**：把原始直读改为目标基线的分层访问器——字段 NS 归属哪个 bag 就用哪个 getter（page_size/disable_overlap_schedule 属 schedule → `get_schedule().page_size`，disaggregation_* 属 disagg → `get_disagg().xxx`）；同文件既有访问器读法即为参照。注意：解析期内（@register_post_process / resolving_view）的读法是另一套合法惯例，不要改

**消除使用示范**（envs 事故的实际修复——按语义折叠，而非机械取默认值）：
- 条件折叠（默认 False）：`is_ppu() and (backend == "megamoe" or
  envs.X.get())` → `is_ppu() and backend == "megamoe"`
  （overrides.py：删掉 `or envs.X.get()`，保留 False 语义路径）
- 分支折叠（默认 True 且该路径可用）：`if envs.X.get(): A else: B` → 直接 `A`
  （deep_gemm.py 三处：`swiglu_limit_arg = self.swiglu_limit`，删 else 回退）
- 默认路径不可用时选可工作路径（默认 True 但该分支是 raise
  NotImplementedError）：保留 else 回退路径而非默认分支（deep_gemm.py 第一处
  保留了 rearrange 非融合路径）
- 折叠依据是 PPU 实际行为需求与代码可用性，不是变量默认值本身

### 5.5 构建配置完整性检查（AOT 源列表与算子注册对账）

**目标**：堵住 PPU 构建配置的“静态引用悬空”。PPU 专属构建文件——`setup_ppu.py`
的 `common_sources` 源列表与 `common_extension_ppu.cc` 的算子注册——是引入时点的
csrc 目录结构快照。社区对 `kernels/aot/csrc/` 的重构（删除/移动/合并 kernel 源
文件）不会通过 cherry-pick 传导到这些 PPU 文件：迁移后源列表可能引用不存在的
文件，算子注册可能引用头文件中已无声明的符号。

**为什么常规验证发现不了**：
- py_compile / import 检查 / AST 名称解析均只覆盖 .py 文件，完全不触及构建脚本
  与 C++ 文件
- ninja 报错（`missing and no known rule to make it`）发生在实际构建时；若迁移
  验证阶段不编译 kernel，问题远端才暴露
- 悬空有两层：源文件引用（ninja 层，先报错）+ 注册符号无声明（.cc 编译层，删了
  源引用后紧接着报错）——只修一层必然撞下一层

**检查方法**（迁移完成后、以及任何涉及 `kernels/aot` 变更的 commit 应用后）：

1. **源文件存在性审计**：提取 `setup_ppu.py` 中全部 `"csrc/..."` 引用，逐一核对
   文件系统存在性
2. **符号声明审计**：提取 `common_extension_ppu.cc` 中全部 `&func` 引用，核对每个
   符号在 `include/` 头文件中是否有声明
3. **Python 调用对账**：对每个待删除的算子，确认 Python 侧已无
   `torch.ops.sgl_kernel.<op>` 调用——社区可能已改走 flashinfer pip 包、JIT 或
   Triton 路径（如 `bmm_fp8` 现由 `from flashinfer import bmm_fp8` 提供）

**工具**（本仓库已内置，位于 `.git/`）：

```bash
python .git/check_sources.py   # 源引用 vs 文件系统，MISSING 必须为 0
python .git/audit_symbols.py   # .cc 符号 vs include/ 声明，NOT DECLARED 必须为 0
```

**修复三层联动原则**（同一次修改完成，缺一会在下一层报错）：

1. `setup_ppu.py`：删除悬空的源文件条目（ninja 层）
2. `common_extension_ppu.cc`：删除对应的 `m.def`/`m.impl` 注册块（编译/链接层），
   同 section 中仍有声明与定义的算子（如 flashinfer 提供的 `top_k_renorm_probs`）
   必须保留
3. Python 侧：确认无调用后放行删除（调用层）

**实际事故案例**（v0.5.18_rel 迁移后远端编译才发现，共 3 处同源问题）：
- `csrc/gemm/bmm_fp8.cu`：社区 csrc 重构后该文件在此链上从未存在；Python 侧
  `bmm_fp8` 已改走 flashinfer pip 包，AOT 注册属死代码
- `csrc/gemm/qserve_w4a8_per_chn_gemm.cu` / `qserve_w4a8_per_group_gemm.cu`：
  qserve kernel 已被社区整体移除，Python 侧零调用
- 三个算子符号（`bmm_fp8` / `qserve_w4a8_per_chn_gemm` / `qserve_w4a8_per_group_gemm`）
  在 `sgl_kernel_ops.h` 中均已无声明，注册块不删则 .cc 编译必失败

### 5.6 PPU 自研 kernel 源码与基线对账（C/CUDA 层）

**目标**：堵住 PPU 自研/改写的 `.cu`/`.cuh` 源码的“基线漂移”。这些源码是引入时点
社区 C++ 基线的快照——命名空间布局、共享宏/`constexpr`、`struct` 字段、helper 函数
签名都可能被社区重构，而 cherry-pick 只搬“相对源基线的增量”，不会把漂移传导过来。

**为什么常规验证发现不了**：
- 5.4 的全部机制（AST 名称解析 / envs 对账 / from-import 对账 / py_compile）与
  `.git/*.py` 工具链只覆盖 `.py`；5.5 只审计 AOT 构建配置里的**文件与符号存在性**，
  两者都不读 C++ 源码内容
- JIT kernel 是**运行时首次调用才编译**：报错发生在 PPU 实机启动/推理途中
  （`error: command 'nvcc' failed` / `ninja: build stopped`），且一次只暴露一个——
  修完一个再撞下一个
- **PPU-only 编译盲区（最关键）**：社区默认关闭、只有 PPU 会走的 JIT 模块在社区 CI 与
  任何 CUDA 机器上**永不编译**，源码可以长期腐坏而无人发现。两类来源：
  - PPU 在 `server_args.py` 反转社区默认值（如 PPU 默认 `SGLANG_OPT_USE_TOPK_V2=False`
    → 只有 PPU 编译 `topk_v1.cuh`，社区走 `topk_v2.cuh`）
  - 模块本身 `if not is_ppu(): raise RuntimeError(...)`（如
    `kernels/ops/elementwise/silu_mul_quant.py` 的三个 PPU-only 模块）

**三项检查**：

| # | 检查项 | 方法 | 判据 |
|---|--------|------|------|
| 1 | 命名空间归属 | `grep -rn "^namespace {" python/sglang/kernels/jit/csrc/` | 顶格匿名 namespace 且外层无 `namespace sglang` → 必然编译失败 |
| 2 | 共享标识符/签名对账 | 见下方清单，逐文件机械核对 | 引用的宏/constexpr/struct 字段数/函数签名必须与目标基线一致 |
| 3 | PPU-only JIT 冒烟编译 | 见 6.3 | 清单内每个模块必须能编译通过 |

**检查 2 的对账清单**（对 diff 中每个 `.cu`/`.cuh` 逐项过）：
- **宏/编译期常量**：源里用到的 `kXxx` / `SGL_XXX` 是否仍在目标基线中定义？社区常把
  编译期常量改成运行时参数（典型：`topk` 从 `-DSGL_TOPK` 宏 → `TopKParams.topk` 字段），
  此时必须把常量改成运行时形参层层透传，`__shared__` 数组维度改用 `kMaxXxx` 上界
- **结构体字段数**：所有 `const auto& [a, b, c] = params;` 结构化绑定的名字个数必须等于
  struct 当前字段数——社区加字段即报 `decomposes into 9 elements but 8 names provided`
- **helper 函数签名**：源里调用的社区 device/host helper（如 `radix_topk`）参数个数/顺序
  是否变化
- **命名空间内的 helper 可见性**：`host::` / `device::` / `bf16_t` / `TensorMatcher` /
  `LaunchKernel` / `RuntimeCheck` 等非限定名必须在 `namespace sglang` 作用域内使用

**正确的文件骨架**（house style，见 `deepseek_v4/store.cuh`、`diffusion/*.cuh`）：

```cpp
namespace sglang {
namespace my_kernel {    // 可选子命名空间
namespace {              // 可选匿名命名空间：必须内嵌，不能顶格
...
}  // namespace
}  // namespace my_kernel
}  // namespace sglang
```

生成的 JIT wrapper（`jit/utils/compile/spec.py::_wrapper_source`）是
`#include "xxx.cuh"` + `namespace sglang { TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, (KernelClass::method)); }`,
导出符号在 `namespace sglang` 内做非限定查找，所以 kernel 源码必须落在该命名空间内。

**匿名命名空间要保留的场景**（不要为了统一而删掉）：同一份源码按宏编译成多个 module
（如 `topk_prefill.cuh` 按 `-DSGL_TOPK=<k>` 一个 k 一个 module）时，外部链接会让
`setup_kernel_smem_once()` 的 function-local static 变成 STB_GNU_UNIQUE 被 loader 跨
`.so` 合并，第二个 module 跳过 `cudaFuncSetAttribute` 后以 64KB 动态 smem 启动失败
（`invalid argument`）——此时用 `namespace sglang { namespace { ... } }` 内嵌写法。

**实际事故案例**（v0.5.18_rel，DeepSeek-V4 在 PPU 实机上逐个暴露）：
- **形态 A（编译期常量 → 运行时字段）**：`deepseek_v4/topk_v1.cuh` 的 v3 kernel
  （PPU 提交 a8c6bf78ba 新增，并把 `TopKKernel::kernel` 切到 v3）仍引用社区已删除的
  `kTopK` 宏、结构化绑定只写 8 个名字（`TopKParams` 已 9 个字段）、调用 3 参数
  `radix_topk`（已改 4 参数）→ 9 个编译错误。因社区默认 topk_v2，该文件只有 PPU 编译，
  CI 侧完全无感
- **形态 B（命名空间迁移）**：社区 4ad5bb5d9a 把 JIT helper 全部移入 `namespace sglang`；
  PPU 提交 f21bff9436 / 20ca033dd2 / 7807449b0e / b9de885fb4 新增的 6 个 `.cuh`
  （`dequantize_k_cache` / `topk_prefill` / `topk_bf16` / `topk_prefill_bf16` /
  `silu_and_mul_post_per_token_quant_fp8` / 两个 `silu_and_mul_*_mxfp4`）停留在顶格匿名
  namespace → `no namespace named 'host'` / `unknown type name 'bf16_t'`
  （clang 的 typo-correction 会“纠正后继续”，往往只报 2 个错，但仍是硬错误）

### 5.7 PPU 能力守卫覆盖检查（guard 覆盖漂移）

**目标**：PPU 用 `if is_ppu():` 规避硬件/kernel 能力缺失（split-k 不支持、smem 上限、
dtype 缺失…）。守卫是按**引入时点的调用点**加的；社区之后新增/切换到同一底层 kernel 的
第二条调用路径时，cherry-pick 只带来社区新路径的代码，**不会**把 PPU 守卫复制过去 →
新路径在 PPU 上裸调不支持的能力。

**为什么常规验证发现不了**：守卫是语义约束而非符号——import、签名、名称解析全部正常，
5.4/5.6 都查不到；只在 PPU 实机命中该路径时炸，且常常要特定 batch/序列长度才走到
（小 batch 冒烟测试反而绕过）。

**检查方法**：

1. 从每个 PPU commit 的 diff 中提取所有 `is_ppu()` / `is_ppu_runtime()` 守卫，记录
   “守卫保护的底层能力”：被限制的调用（`deep_gemm.xxx` / `torch.ops.sgl_kernel.xxx` /
   TileLang kernel）与被限制的取值（如 `n_splits == 1`）
2. 在目标基线中搜索该底层调用的**全部调用点**：

```bash
grep -rn "tf32_hc_prenorm_gemm" python/sglang/
```

3. 逐一确认每个调用点都有等价守卫；社区新增的调用点必须补齐（守卫落点与社区原守卫
   同层——在同一个函数内部，避免只在外层入口拦、内层新路径绕过）
4. 同一 kernel 的“老路径 + 新融合路径”并存时，两条路径的守卫要写成同一语义
   （必要时抽成同一个 helper），后续社区再加路径时容易 grep 到

**实际事故案例**（v0.5.18_rel，DeepSeek-V4 W8A8-INT8）：`deep_gemm.tf32_hc_prenorm_gemm`
的 PPU 版无 split-k（断言 `d.shape == (1, m, n)`）。PPU 提交 7320e4c7bb 只在当时唯一的
调用点 `mhc_pre`（`_mhc_pre_impl`）加了 `is_ppu()` → `assert n_splits == 1`；社区随后新增
`mhc_fused_post_pre` 融合路径（`SGLANG_OPT_FUSE_MHC_POST_PRE` 默认开启），自己
`_compute_num_split_for_mhc_pre(...)` 算出 `n_splits > 1` 直接调 DeepGEMM → PPU 上
`--cuda-graph-max-bs 64` 的 decode 图捕获（`num_tokens=64 > 32` 走大 batch DeepGEMM 分支）
必然 AssertionError；`bs <= 32` 走 TileLang FMA 小 batch 路径不报错，进一步掩盖了问题。
同一函数还漏了 b1d7886aa8 的 `torch.zeros` 精度修复（PPU SM80 partial buffer 必须置零）
——**同一 kernel 的多个 PPU 修复要一起复查新调用点**。

---

## Phase 6: 验证

### 6.1 代码验证

```bash
# 确认 PPU 平台文件完整
python -c "from sglang.srt.platforms.ppu import PPUSRTPlatform; print('OK')"

# 确认 is_ppu 逻辑存在
python -c "from sglang.srt.utils.common import is_ppu, is_cuda_alike; print('OK')"

# 确认 pyproject_ppu.toml 存在
ls python/pyproject_ppu.toml python/sglang/kernels/aot/pyproject_ppu.toml
```

### 6.2 名称解析与终态对账

```bash
# 全量 AST 名称解析扫描（见 5.4），CONFIRMED 必须为 0
python .git/run_full_scan.py

# envs 属性对账（见 5.4），DANGLING 必须为 0
python .git/check_envs_attrs.py

# from-import 符号对账（见 5.4，社区函数/类/常量接口引用），DANGLING 必须为 0
python .git/check_import_refs.py

# 类成员归属对账（见 5.4 形态 5，方法落错类）：对迁移范围内新增的每个 def，
# 比较源补丁中所属类与目标终态中所属类，MISPLACED 必须为 0
python .claude/skills/cherry-pick-ppu-patches/assets/audit_class_placement.py

# server_args 读法对账（见 5.4 形态 6）：新增 server_args.<field> 原始直读 vs
# 同文件访问器惯例 + resolvable 标注，DIVERGENT(resolvable=True) 必须甄别为 0
python .claude/skills/cherry-pick-ppu-patches/assets/audit_serverargs_reads.py

# 语法检查所有被修改的 .py 文件
python -m py_compile <modified files>

# 终态对账：与迁移前备份分支的 diff 应只包含预期的适配差异
git diff <backup-branch> HEAD --numstat
```

注意：终态对账前先确认 backup 分支确实指向迁移前的 HEAD（曾发生 backup 误指向
源链 `v0.5.18_rel` 标签 commit，导致 762 文件假差异——源链与目标链的社区基线差异
被误读为迁移错误）。判断方法：`git rev-list --count <backup>..HEAD` 应等于重放的
commit 数。

### 6.3 C/CUDA 源码静态审计与 PPU-only JIT 冒烟编译

```bash
# 1) 命名空间归属审计（见 5.6）：顶格匿名 namespace 且外层无 namespace sglang → 必须为 0
grep -rn "^namespace {" python/sglang/kernels/jit/csrc/

# 2) 结构化绑定字段数对账（见 5.6）：逐个核对名字个数 == struct 字段数
grep -rn "const auto& \[" python/sglang/kernels/jit/csrc/
```

**PPU-only JIT 模块冒烟编译**（必须在 PPU 机器上执行；这些模块在社区 CI 与 CUDA 机器上
永不编译，不跑这一步就只能靠实机启动逐个撞）：

| 模块 loader | 源文件 | 为什么只有 PPU 编译 |
|-------------|--------|-------------------|
| `dsv4.topk._jit_topk_v1_module()` | `deepseek_v4/topk_v1.cuh` | PPU 默认 `SGLANG_OPT_USE_TOPK_V2=False` |
| `dsv4.topk._jit_topk_bf16_module()` | `deepseek_v4/topk_bf16.cuh` | PPU 专用 BF16 topk 路径 |
| `dsv4.topk._jit_topk_prefill_module(512)` | `deepseek_v4/topk_prefill.cuh` | PPU chunked prefill topk |
| `dsv4.topk._jit_topk_prefill_bf16_module(512)` | `deepseek_v4/topk_prefill_bf16.cuh` | PPU BF16 prefill topk |
| `dsv4.attn._jit_dequantize_k_cache_module()` | `deepseek_v4/dequantize_k_cache.cuh` | PPU 替代 Triton dequant |
| `elementwise.silu_mul_quant._jit_fp8_module(False)` | `elementwise/silu_and_mul_post_per_token_quant_fp8.cuh` | `if not is_ppu(): raise` |
| `elementwise.silu_mul_quant._jit_tp_module(False, 512)` | `elementwise/silu_and_mul_post_quant_mxfp4.cuh` | `if not is_ppu(): raise` |
| `elementwise.silu_mul_quant._jit_ep_module(False)` | `elementwise/silu_and_mul_masked_post_quant_mxfp4.cuh` | `if not is_ppu(): raise` |

```bash
python - <<'PY'
from sglang.kernels.ops.attention.dsv4 import attn, topk
from sglang.kernels.ops.elementwise import silu_mul_quant as smq

for name, fn, args in [
    ("topk_v1", topk._jit_topk_v1_module, ()),
    ("topk_bf16", topk._jit_topk_bf16_module, ()),
    ("topk_prefill", topk._jit_topk_prefill_module, (512,)),
    ("topk_prefill_bf16", topk._jit_topk_prefill_bf16_module, (512,)),
    ("dequantize_k_cache", attn._jit_dequantize_k_cache_module, ()),
    ("silu_mul_fp8", smq._jit_fp8_module, (False,)),
    ("silu_mul_mxfp4_tp", smq._jit_tp_module, (False, 512)),
    ("silu_mul_mxfp4_ep", smq._jit_ep_module, (False,)),
]:
    fn(*args)
    print(name, "JIT OK")
PY
```

新增 PPU 自研 kernel 时，同步把其 loader 补进上表与脚本。

### 6.4 单元测试

```bash
# 运行 PPU 平台接口测试
python -m pytest test/registered/unit/platforms/test_platform_interface.py -k "Ppu" -v
```

### 6.5 构建验证

```bash
# 先做构建配置静态审计（见 5.5），两项 MISSING / NOT DECLARED 必须为 0
python .git/check_sources.py
python .git/audit_symbols.py

# 使用 PPU 构建配置验证安装
pip install -e python --config-settings pyproject.toml=python/pyproject_ppu.toml
```

### 6.6 生成迁移报告

汇总所有记录，输出最终报告：

```
# PPU 补丁迁移报告

## 迁移范围
- 源分支: <source_branch>
- 目标分支: <target_branch>
- Commit 范围: <start>.. <end>

## 迁移结果
- 总 Commit 数: N
- 成功应用: N
- 有冲突(已解决): N
- 跳过(社区已包含): N
- 失败: N

## 适配检查结果
### 接口兼容性
- 检查函数数: N
- 需适配函数: <list>
- 适配状态: ✅/⚠️

### 数据类型扩展
- 新增 FP8 路径: <list>
- 已适配数据类型: FP16/BF16/INT8
- 适配状态: ✅/⚠️

### Hook 机制
- Hook 总数: N
- 需调整 Hook: <list>
- 已移除 Hook: <list>
- 适配状态: ✅/⚠️

### Import 与符号引用完整性
- 扫描文件数: N
- 缺失 import (CONFIRMED): <list or 0>
- envs 属性悬空 (DANGLING): <list or 0>
- 接口符号悬空 (DANGLING): <list or 0>
- 适配状态: ✅/⚠️

### 构建配置完整性
- 源引用审计 MISSING: <list or 0>
- 符号声明审计 NOT DECLARED: <list or 0>
- 适配状态: ✅/⚠️

### C/CUDA 源码基线对账
- 顶格匿名 namespace: <list or 0>
- 宏/结构体字段数/签名漂移: <list or 0>
- PPU-only JIT 冒烟编译: ✅ PASS / ❌ FAIL
- 适配状态: ✅/⚠️

### PPU 能力守卫覆盖
- 守卫总数: N
- 社区新增调用点需补守卫: <list or 0>
- 适配状态: ✅/⚠️

## 验证结果
- 单元测试: ✅ PASS / ❌ FAIL
- PPU-only JIT 冒烟编译: ✅ PASS / ❌ FAIL
- 构建验证: ✅ PASS / ❌ FAIL
```

---

## 常见陷阱与注意事项

1. **`lru_cache` 与 `torch.compile` 不兼容**：PPU 代码中如果使用了 `@lru_cache`，需确认是否在 `torch.compile` 路径上，如冲突则改用 `@cache_once`（参考 `python/sglang/kernels/jit/utils/common.py`）

2. **PPU `is_cuda()` 返回 True**：PPU 平台 override 了 `is_cuda()` 返回 True（CUDA 兼容 API），社区新增的 `is_cuda()` 分支 PPU 会自动命中，需确认行为正确

3. **`device_type` 保持 `"cuda"`**：PPU 的 `device_type` 不变，`torch.device("cuda")` 是 PPU 唯一有效的设备类型字符串

4. **JIT arch 后缀**：PPU 810/810e 设备需要 `"sm80a"` arch 变体，由 `get_jit_cuda_arch_suffix()` 返回 `"a"`

5. **tc_piecewise CUDA graph 不兼容**：PPU 环境下 `tc_piecewise` 可能与 CUDA graph 不兼容，cherry-pick 时需注意相关代码路径

6. **Commit message 规范**：sglang 社区要求 commit message 包含 Test 模块和 E2E 占位信息，cherry-pick 时保留原始 message

7. **cherry-pick 的 import 迁移盲区**：cherry-pick 只迁移“相对源基线的增量”。若源文件
   某个 import 由范围外 commit 引入、且目标基线（社区新版本）已重构掉它，迁移结果是
   “使用”落地而“import”缺失（运行时 NameError）。零冲突、py_compile 查不出，必须做
   AST 名称解析检查（见 5.4）。历史案例：deep_gemm.py 的 `einops`、fused_moe.py 的
   `envs`、memory_pool.py 的 `index_buf_accessor`

8. **PPU 构建配置的静态引用悬空**：`setup_ppu.py` 的 `common_sources` 与
   `common_extension_ppu.cc` 的算子注册是引入时点的 csrc 目录结构快照，社区重构
   `kernels/aot/csrc/`（删除/合并 kernel 源文件）后不会自动更新。Python 层检查
   完全不覆盖构建脚本与 C++ 文件，ninja 报错发生在远端构建环境。必须跑
   `check_sources.py` / `audit_symbols.py` 审计（见 5.5），修复需三层联动：源列表
   条目 + 算子注册块 + Python 侧调用确认。历史案例：`bmm_fp8.cu`（社区改走
   flashinfer pip 包）与两个 qserve kernel（Python 侧零调用）在 v0.5.18_rel 上
   从未存在，远端编译时才暴露

9. **`envs.<ATTR>` 属性悬空（Envs 类属性迁移盲区）**：社区清理 commit（如
   bc312d185d "Clean deprecated DeepSeek V4 Environs"）删除 env 定义、而 PPU
   commit（基于老基线，使用时定义尚在）的代码被 cherry-pick 落地后，
   `envs.XXX.get()` 运行时 AttributeError。AST 名称解析不可见（`envs` 已正确
   import，悬空的是类属性）。加重因素：server args 后处理管线
   （`@register_post_process` 如 `_a2a_backend_overrides`）无条件执行 + `or`
   短路求值使右侧必然求值 → **任意 server 启动必炸**（而非特定路径触发）。
   必须跑 `check_envs_attrs.py` 对账（见 5.4），修复必须在 PPU 使用侧消除依赖
   （按语义折叠分支，**禁止在 environ.py 恢复定义复活废弃变量**——社区删除即
   废弃）。历史案例：`SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE`
   （overrides.py，启动必炸，删 `or envs.X.get()`）、
   `SGLANG_OPT_SWIGLU_CLAMP_FUSION`（deep_gemm.py，
   分支内炸，四处按语义折叠）——元凶与 import 案例（einops/envs/index_buf_accessor）
   同为 bc312d185d/31c1e5943f 社区清理 commit

10. **社区函数/类接口引用悬空（from-import 符号迁移盲区）**：与 import/envs 悬空
    同根源——基线漂移。`from sglang.xxx import <func/class>` 的目标符号被社区
    重构删除（重命名/合并/移入子包）时，import 语句完整落地且名字绑定正常，
    AST 名称解析与 py_compile 均不报错，import 执行时才 ImportError。必须跑
    `check_import_refs.py` 对账（见 5.4）。误报需排除：子模块导入、
    try/except ImportError 可选依赖、`__getattr__` 动态导出、star re-export。
    修复同 envs 悬空：**禁止补回符号**，在使用侧消除依赖（改社区新接口/等价
    实现/删除死分支）

11. **PPU 自研 kernel 源码的基线漂移（C/CUDA 层盲区）**：PPU 提交里的 `.cu`/`.cuh`
    是引入时点社区 C++ 基线的快照：命名空间布局（社区 4ad5bb5d9a 将 JIT helper 全部
    移入 `namespace sglang`）、编译期常量（`-DSGL_TOPK` 宏 → 运行时 `TopKParams.topk`
    字段）、`struct` 字段数（结构化绑定 8 vs 9）、helper 签名（`radix_topk` 3 → 4 参）
    都可能被社区重构，而 cherry-pick 不会传导。Python 层检查（5.4）与 AOT 构建审计
    （5.5）均不读 C++ 源码，JIT 又是运行时首次调用才编译 → 只能在 PPU 实机启动途中
    一个一个撞（`nvcc failed`）。必须做 5.6 的三项对账 + 6.3 的冒烟编译。历史案例：
    `topk_v1.cuh` 的 `kTopK`/8-vs-9 解构/`radix_topk` 签名（6 个 `.cuh` 同时停留在
    顶格匿名 namespace）

12. **PPU-only 代码路径的编译/执行盲区**：社区默认关闭、只有 PPU 走的路径
    （PPU 在 `server_args.py` 反转默认值，如 `SGLANG_OPT_USE_TOPK_V2=False`；或模块
    `if not is_ppu(): raise`）在社区 CI 与任何 CUDA 机器上永不编译/执行，源码可以
    长期腐坏而无人发现。迁移验证阶段必须显式跑一遍 PPU-only JIT 模块清单
    （见 6.3），不能依赖“服务能启动”作为覆盖证据

13. **PPU 能力守卫只覆盖老调用点（guard 覆盖漂移）**：`is_ppu()` 守卫是按引入时点的
    调用点加的，社区新增/切换到同一底层 kernel 的第二条路径（尤其是把两步合并的
    “融合路径”）时，cherry-pick 不会把守卫复制过去。守卫是语义约束而非符号，所有
    静态检查都看不到，且往往需特定 batch/序列长度才命中（小 batch 冒烟测试绕过）。
    必须按 5.7 对每个守卫保护的底层调用 grep 全部调用点。历史案例：
    `deep_gemm.tf32_hc_prenorm_gemm` 无 split-k 的守卫只加在 `mhc_pre`，社区新增的
    `mhc_fused_post_pre` 融合路径在 PPU 上 decode 图捕获（bs=64）必炸；同函数还漏了
    `torch.zeros` 精度修复

14. **零冲突的类成员归属错位（结构锚点漂移，apply 成功但落点错误）**：社区把类 A
    重构过（谓词抽为混入类、类体间插入新类、函数搬家）后，源补丁向类 A 体内新增
    的方法会被三路合并以漂移后的文本锚点**零冲突**塞进相邻类 B——apply 看似成功，
    语义落点错误。决策集不是 apply/abandon 二选一，而是需要**移位**（见 4.4）。
    符号存在性、py_compile、AST 名称解析、from-import 对账全部免疫（见 5.4 形态 5），
    运行时 `instance.m()` 才 AttributeError。pick 时对「向类体内新增 def」的 hunk
    必须比对源补丁与落地结果的上方上下文（见 4.4 快速判别），终验跑类成员归属审计
    （见 6.2，MISPLACED 必须为 0）。历史案例：`is_acext` 落进
    `DeepEPv2Fp8ScaleFormat`（v0.5.19 迁移，PPU 实机批量炸于 `w8a8_int8.py` 的
    `moe_runner_backend.is_acext()`；上游基线自身的 `is_intel_xpu` 悬空死代码为
    同构先例）

15. **零冲突的配置读取契约变更（访问惯用法漂移，文本保真但语义漂移）**：源基线
    在解析结束把声明物化回 ServerArgs 字段、目标基线改为声明投影进 config bag 时，
    源上合法的 `server_args.<field>` 直读搬过来后读到 None/旧值——迁移代码与源
    逐字节一致，冲突检查、源文本对齐、AST 名称解析、符号对账全部通过（它们校验的
    都是「文本/结构保真」，而缺陷在「读取契约」维度）。判据：字段标注
    `resolvable=True`（server_args.py）→ 运行期必须走 `get_<bag>().<field>`；
    CLI 直填字段（resolvable=False）直读合法。**同函数适配不一致是最强信号**：
    兄弟字段已改访问器而它没改，基本可断定漏改。pick 时对每个新增的
    `server_args.` 读取点查 resolvable 标注，终验跑读法对账（见 6.2）。
    历史案例：`page_size`（kv_cache_configurator.py，DSV4 启动必炸
    `TypeError: NoneType % int`）、`disable_overlap_schedule`/`page_size`
    （one_batch.py，bench 静默分叉）——两个基线的 `materialize_declarations`
    存废差异是元凶
## 参考文件

- PPU 平台定义: `python/sglang/srt/platforms/ppu.py`
- 平台发现机制: `python/sglang/srt/platforms/__init__.py`
- 设备 Mixin: `python/sglang/srt/platforms/device_mixin.py`
- 工具函数: `python/sglang/srt/utils/common.py`（`is_ppu()`, `is_cuda_alike()`）
- JIT 编译: `python/sglang/kernels/jit/utils/arch.py`, `common.py`
- 构建配置: `python/pyproject_ppu.toml`, `python/sglang/kernels/aot/pyproject_ppu.toml`
- PPU kernel 构建: `python/sglang/kernels/aot/setup_ppu.py`, `python/sglang/kernels/aot/csrc/common_extension_ppu.cc`
- 平台测试: `test/registered/unit/platforms/test_platform_interface.py`

详细代码模式参考见 [reference.md](reference.md)。
