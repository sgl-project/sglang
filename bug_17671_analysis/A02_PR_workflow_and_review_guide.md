# A02: 如何提交 PR（Pull Request）并让维护者 Review（以 Diffusion 依赖修复为例）

> 目标：用**最小风险**的方式，把你已验证的结论（`sglang[diffusion]` 缺 `accelerate/ftfy`、以及 `set_default_dtype` 命名残留）变成一个可 review、可合并的 PR。

---

## 0. 先确定你要走哪条“最小改动”路线

你现在发现的问题分两层：

- **A. 打包/依赖层**：发布包 `diffusion` extra 里缺 `accelerate/ftfy`（证据见 `A01_B20_...`）
- **B. 代码层**：`set_default_dtype` → `set_default_torch_dtype` 的引用残留（见 B15）

为了避免“改太多被拒”，建议拆成两个 PR（推荐）：

- **PR-1（强烈推荐先做）**：只修依赖（diffusion extra 补齐 `accelerate/ftfy` + 文档一句话）
- **PR-2（可选）**：只修命名残留（`set_default_dtype` 调用点改为 `set_default_torch_dtype` 或加兼容 alias）

如果你只想开一个 PR，也可以合并，但**合并 PR 会降低合并概率**（维护者更难 review）。

---

## 1. 在本地准备工作区（Windows + GitHub）

- **确认你有 fork 权限**：
  - 去 `sgl-project/sglang` 点 `Fork`，fork 到你自己的账号。
- **添加 upstream**（让你能随时 rebase 到官方最新）：

```bash
git remote -v
git remote add upstream https://github.com/sgl-project/sglang.git
git fetch upstream
```

- **创建新分支**（不要在 main/master 上改）：

```bash
git checkout -b fix/diffusion-extras-accelerate-ftfy upstream/main
```

> 如果官方默认分支不是 `main`，改成对应名字（如 `master`）。

---

## 2. PR-1：依赖修复（在源码中新增 `diffusion` extra，并补齐 `accelerate` / `ftfy`）

### 2.1 你需要改哪些文件？

根据你当前 repo 快照：`python/pyproject.toml` 里确实没有 `diffusion` 定义，但分发包里却有（B20 实锤），并且分发包的 `diffusion` extra 依赖列表缺 `accelerate/ftfy`。这意味着“真实的依赖定义来源”可能在：

- `python/pyproject.toml`（你当前看到的）
- 或者某个发布/打包脚本会动态注入 extras（需要维护者确认）

**因此 PR-1 的策略应该是：**

- **先在源码里补齐 `diffusion` extra（让配置可追溯）**
- 并且在 `diffusion` 里**明确加入 `accelerate` 和 `ftfy`（运行时必需）**
- 同时**不要大改其他 extras**，避免引入不必要的 dependency 变更

### 2.2 建议的最小 patch（保守版，推荐）

在 `python/pyproject.toml` 的 `[project.optional-dependencies]` 下新增：

- `diffusion = [...]`（只补齐你已实锤缺失、且运行时必需的两项）

建议最小集合（推荐）：

```toml
diffusion = [
  "accelerate",
  "ftfy",
]
```

> 为什么不把所有 diffusion 依赖都写进去？  
> 因为你已经从 B20 看到发布包里有很多 diffusion 依赖，贸然在源码里一次性 “全量同步” 容易引发版本/平台冲突；最稳的做法是**只补齐缺失的 runtime 关键依赖**。

> PR 描述里务必引用 B20：证明当前分发包的 `diffusion` extra 没声明 `accelerate/ftfy`。

### 2.3 本地自检（不要求跑大模型）

你只需要验证 “安装后不再报缺库”：

在容器/虚拟环境里：

```bash
python -c "import accelerate, ftfy; print('ok')"
```

然后用一个你已复现的命令确认不会再卡在 `device_map` 缺 accelerate 的错误（例如你之前的 Z-Image-Turbo 流程）。

---

## 3. PR-2：命名残留修复（可选）

### 3.1 改动目标

把 `set_default_dtype` 的引用改为 `set_default_torch_dtype`（或加 alias），避免运行时报错。

### 3.2 风险控制建议

优先选择“兼容 alias”的方式（最不容易 break 其他调用方）：

- 在定义 `set_default_torch_dtype` 的模块里增加：
  - `set_default_dtype = set_default_torch_dtype`

如果你不确定定义位置，就只修已知报错的文件（例如 comfyui pipeline）做最小改动。

---

## 4. Commit 规范（让 review 更顺）

建议每个 PR 只有 1–2 个 commit：

- PR-1：
  - `fix(diffusion): add accelerate/ftfy to diffusion extra`
- PR-2：
  - `fix(diffusion): update set_default_dtype callsites`

提交命令：

```bash
git status
git add python/pyproject.toml
git commit -m "fix(diffusion): add accelerate/ftfy to diffusion extra"
```

---

## 5. Push 到你的 fork

```bash
git push -u origin fix/diffusion-extras-accelerate-ftfy
```

---

## 6. 在 GitHub 上创建 PR

打开你的 fork 页面，GitHub 会提示 “Compare & pull request”。

### 6.1 PR 标题建议

- PR-1：`fix(diffusion): ensure diffusion extra installs accelerate/ftfy`
- PR-2：`fix(diffusion): fix set_default_dtype naming regression`

### 6.2 PR 描述（建议结构）

- **Problem**
  - `sglang[diffusion]` install still fails at runtime due to missing `accelerate`/`ftfy`
- **Evidence**
  - 贴 B20 的 `importlib.metadata` 输出（证明 diffusion extra 没声明 accelerate/ftfy）
  - 贴你复现的 traceback（device_map requires accelerate）
- **Fix**
  - Added `accelerate` and `ftfy` to diffusion extra (source metadata)
- **Impact**
  - Users can follow docs without manual pip installs

> 关联 issue：不要一上来 `Fixes #17618 #17671`。  
> 建议先写 `Relates to #17671`，避免维护者觉得你“强行关闭 issue”。合并时他们自己会决定是否 close。

---

## 7. 请求维护者 Review（让人愿意点开）

### 7.1 怎么找 reviewer？

最简单：

- 在 PR 描述里 @ 维护 diffusion 的人（你看到最近 diffusion issue 很多，可以在对应文件 git blame/最近 PR 里找活跃作者）
- 或者在 PR 评论里写一句：
  - “Could someone familiar with the packaging/release pipeline confirm this is the right place to define diffusion extras?”

### 7.2 怎么写才显得“不捣乱”

关键句式：

- “I’m not sure where the release pipeline injects extras, but the installed distribution metadata shows diffusion extra missing accelerate/ftfy (see evidence). This PR makes the dependency explicit and traceable in source.”

---

## 8. 合并前后的维护

- 如果 CI 挂了：
  - 不要 panic，先看是不是 lint/format 或依赖冲突
- 如果维护者要求改动位置：
  - 很正常：说明他们的发布流程确实不是从你看到的 `python/pyproject.toml` 生成
  - 你只要把 PR 调整到他们指定位置即可（你已经有证据链）

---

## 9. 你可以直接复制粘贴的 “最短 PR 说明”（英文）

```text
Problem: Following the docs, diffusion/DiT models fail at runtime because `accelerate` (device_map) and `ftfy` (prompt cleaning) are missing.

Evidence: `importlib.metadata` shows the installed `sglang` distribution provides `diffusion` extra, but `Requires-Dist` for extra=="diffusion" does not include `accelerate` or `ftfy` (see attached output). This matches the NotImplementedError from diffusers when `device_map="cuda"/"auto"`.

Fix: Add `accelerate` and `ftfy` to the diffusion extra so `pip/uv install "sglang[diffusion]"` satisfies runtime requirements out of the box.

Relates to: #17671
```

