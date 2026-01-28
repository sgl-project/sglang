# A01_B01 Plan: GitHub 上修 Bug 的标准流程（以 SGLang Diffusion 依赖缺失为例）

> 目标：你在 GitHub 上修 bug 时做到 **可复现、可证伪、可 review、可合并**，并且避免“捣乱/改坏”。

---

## 1) 复现与分层定位（先别急着改代码）

- **1.1 复现最小用例**
  - 用最短命令复现（能 1 分钟跑出 traceback 的那种）。
  - 记录：镜像/版本/commit/hash、模型、命令行、完整 traceback。

- **1.2 分层判断问题属于哪一类**
  - **镜像层**：docker 镜像里缺依赖/缺文件（所有 OS 都会复现）
  - **依赖/打包层**：`sglang[diffusion]` extras 声明不完整（pip/uv 装不全）
  - **代码层**：函数重命名残留、逻辑 bug
  - **资源层**：OOM、WSL2 内存、磁盘配额
  - **模型层**：HF repo 不完整、模型结构不兼容

> 对本次问题：你已经实锤属于 **依赖/打包层 + 镜像层**（B20 + traceback）。

---

## 2) 收集“硬证据”（让维护者无法反驳）

- **2.1 运行时 traceback**
  - 重点截取关键错误行（如 `device_map requires accelerate`）。

- **2.2 打包元数据证据（最关键）**
  - 用 `importlib.metadata` 打印 `Provides-Extra` 与 `Requires-Dist`（你已完成，见 B20）。

- **2.3 源码侧对照**
  - 检查 repo 内 `python/pyproject.toml` 是否能追溯到 extras 定义（你已完成）。

---

## 3) 在 Issue 里先发 Comment（先沟通，再动刀）

- **3.1 Comment 内容结构**
  - **What I observed**：按文档装 `sglang[diffusion]` 仍缺 `accelerate/ftfy` 导致崩溃
  - **Evidence**：贴 B20 的 metadata 输出 + traceback
  - **Suggested fix**：diffusion extra 补齐 `accelerate/ftfy`；dev 镜像也补齐

- **3.2 Cross-reference**
  - 可轻描淡写提一句：最近 diffusion issues 较多（#17618/#17671/#17874），这更像“基础设施缺陷”。

---

## 4) 开 PR：坚持“最小可合并改动”

### PR-1（推荐先做）：只修依赖（diffusion extra 补齐）

- **目标**：让 `uv/pip install "sglang[diffusion]"` 能把运行时关键依赖装齐（`accelerate`、`ftfy`）。
- **原则**：只补“必需依赖”，别顺手大改一堆版本锁，避免引发平台冲突。

### PR-2（可选）：只修代码命名残留

- **目标**：修 `set_default_dtype` → `set_default_torch_dtype` 的残留引用。
- **原则**：单独 PR，避免和依赖 PR 混在一起。

---

## 5) Fork → Branch → Commit → Push → PR

- **Fork**：fork `sgl-project/sglang` 到你账号
- **Branch**：从 upstream 最新分支切新分支
- **Commit**：1 PR = 1–2 commits（消息清晰）
- **PR 描述**：必须包含证据（B20）与复现方式
- **请求 review**：@ 相关维护者，问清“extras 定义/发布流程应该改在哪”

---

## 6) 合并后的跟进

- 如果维护者说“发布链路不在这个文件”：你就把 PR 按他们指定路径移动（你已经有证据链，方向不会错）。
- 如果维护者要求加 CI：加一个最小 sanity test（导入 accelerate/ftfy + 确认 diffusion extra 声明存在）。

---

## 7) 本次问题的“最终交付”清单（你该产出哪些链接）

- **Issue comment（英文）**：`A01_B18_issue_reply_en.md`
- **硬证据**：`A01_B20_distribution_metadata_proves_diffusion_extra_missing_accelerate.md`
- **依赖/脱节分析**：`A01_B19_sglang_diffusion_extras_missing_accelerate.md`
- **PR 流程**：`A02_PR_workflow_and_review_guide.md`

