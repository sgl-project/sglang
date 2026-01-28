# A01_B19: `sglang[diffusion]` extra 与源码 `pyproject.toml` 脱节，且 extra 依赖不完整（缺 `accelerate` / `ftfy`）

## 概述

**发现时间**: 2026-01-28  
**测试环境**: `lmsysorg/sglang:dev` 容器内  
**问题类型**: 🔴 **致命缺陷** - 已发布分发包的 `diffusion` extra 与源码 `python/pyproject.toml` 配置脱节，且 extra 依赖不完整  
**严重程度**: 🔴🔴🔴 **极严重** - 用户按文档执行 `uv/pip install "sglang[diffusion]"` 后仍会在运行时遇到缺 `accelerate` / `ftfy` 的崩溃

## 🚨 核心发现（已验证）

### A. 源码侧：`python/pyproject.toml` 中没有 `diffusion` 定义

**`python/pyproject.toml` 中根本没有定义 `diffusion` 这个 optional-dependencies！**

当前 `pyproject.toml` 中只有：
- `decord = ["decord"]`
- `test = [...]`
- `tracing = [...]`
- `all = ["sglang[test]", "sglang[decord]"]`
- `blackwell = [...]`
- `dev = [...]`

**完全没有 `diffusion` 的定义！**

### B. 分发包侧：installed distribution 确实提供 `diffusion` extra，但依赖列表缺关键库

通过 `importlib.metadata` 读取已安装 `sglang` 的分发包元数据，`Provides-Extra` 中 **包含** `diffusion`，但该 extra 的 `Requires-Dist` 列表 **不包含**：

- `accelerate`（diffusers `device_map` 的运行时必需）
- `ftfy`（部分 prompt 文本清洗路径必需）

证据见：`A01_B20_distribution_metadata_proves_diffusion_extra_missing_accelerate.md`。

---

## 测试过程

### 步骤 1: 安装 `sglang[diffusion]`

```bash
pip install "sglang[diffusion]"
```

**结果**: ✅ 安装成功，所有依赖显示已满足

```
Requirement already satisfied: sglang[diffusion] in /usr/local/lib/python3.12/dist-packages (0.0.0.dev1+g93423ff78)
Requirement already satisfied: diffusers==0.36.0 in /usr/local/lib/python3.12/dist-packages (from sglang[diffusion]) (0.36.0)
...
```

**关键发现**: 安装过程中**没有任何关于 `accelerate` 的提示或安装**。

### 步骤 2: 尝试运行 Diffusion 模型

```bash
sglang generate --model-path Tongyi-MAI/Z-Image-Turbo \
    --backend diffusers \
    --prompt "A tech-style owl logo, futuristic, 4k" \
    --num-inference-steps 1 \
    --save-output
```

**结果**: ❌ **失败** - 报错缺少 `accelerate`

---

## 错误信息

### 完整 Traceback

```
[01-28 17:14:02] Loading diffusers pipeline with dtype=torch.bfloat16, device_map=cuda
Cannot initialize model with low cpu memory usage because `accelerate` was not found in the environment. 
Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install `accelerate` for faster 
and less memory-intense model loading. You can do so with:
```
pip install accelerate
```

[01-28 17:14:02] Worker 0: Shutdown complete.
Process sglang-diffusionWorker-0:
Traceback (most recent call last):
  File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py", line 316, in run_scheduler_process
    scheduler = Scheduler(
                ^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/scheduler.py", line 71, in __init__
    worker = GPUWorker(
             ^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py", line 61, in __init__
    self.init_device_and_model()
  File "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py", line 90, in init_device_and_model
    self.pipeline = build_pipeline(self.server_args)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/__init__.py", line 74, in build_pipeline
    pipeline = pipeline_cls(model_path, server_args)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py", line 376, in __init__
    self.diffusers_pipe = self._load_diffusers_pipeline(model_path, server_args)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py", line 424, in _load_diffusers_pipeline
    pipe = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/diffusers/pipelines/pipeline_utils.py", line 793, in from_pretrained
    raise NotImplementedError(
NotImplementedError: Using `device_map` requires the `accelerate` library. Please install it using: `pip install accelerate`.
```

---

## 问题分析

### 核心矛盾（三重打击）

1. **官方文档推荐**: 多处文档建议使用 `pip install "sglang[diffusion]"`
2. **源码/发布物脱节**: 源码 `python/pyproject.toml` 中没有 `diffusion`，但已发布分发包元数据里存在 `diffusion`
3. **extra 依赖不完整**: 已发布分发包的 `diffusion` extra 未声明 `accelerate` / `ftfy`，导致运行时崩溃

### 为什么 `pip install "sglang[diffusion]"` 没有解决问题？

因为 **已发布分发包确实有 `diffusion` extra**（见 B20 证据），但它的依赖列表不完整：

- 能带来 `diffusers` 等部分依赖
- 但缺少 `accelerate` / `ftfy` 这种运行时关键依赖

因此“安装成功/依赖满足”的输出会给用户造成错觉，但运行到 `device_map=cuda/auto` 时仍然必崩。

### 为什么这是一个严重问题？

- **用户信任度**: 用户按照官方文档/推荐方式安装，期望"开箱即用"
- **错误信息误导**: `pip install "sglang[diffusion]"` 显示"所有依赖已满足"，但实际上缺少关键运行时依赖
- **双重问题**: 
  - Docker 镜像层面：缺少 `accelerate`（已在 B13 记录）
  - 依赖定义层面：`sglang[diffusion]` extras 不完整（本文件记录）

---

## 根本原因

### 1. 源码侧：`pyproject.toml` 中缺失 `diffusion` extra 定义（或被拆到其他打包配置）

**文件位置**: `python/pyproject.toml`  
**当前状态**: `[project.optional-dependencies]` 部分**完全没有 `diffusion` 的定义**

**需要添加的完整定义**:

```toml
[project.optional-dependencies]
# ... 现有的 decord, test, tracing 等 ...

diffusion = [
    "diffusers>=0.36.0",
    "accelerate>=0.20.0",  # ← 必需：device_map 功能依赖
    "ftfy",                # ← 必需：文本处理依赖
    "imageio>=2.36.0",     # ← 图像 I/O
    "imageio-ffmpeg>=0.5.1",  # ← FFmpeg 支持
    "opencv-python-headless>=4.10.0",  # ← 图像处理
    "cache-dit>=1.2.0",    # ← Cache-DiT 加速器（如果使用）
]
```

### 2. 发布分发包侧：`diffusion` extra 依赖不完整（缺 `accelerate` / `ftfy`）

证据见：`A01_B20_distribution_metadata_proves_diffusion_extra_missing_accelerate.md`。

### 3. 文档与实现脱节

文档推荐的 `sglang[diffusion]` 并没有保证安装 `accelerate` / `ftfy`，导致用户按文档操作仍失败。

### 4. `diffusers` 的依赖关系理解不足

`diffusers` 库虽然可以独立安装，但**使用 `device_map` 参数时，`accelerate` 是必需的运行时依赖**。这不是一个"可选优化"，而是**核心功能依赖**。

---

## 影响范围

### 受影响的用户场景

1. **通过 `pip install "sglang[diffusion]"` 安装的用户**
   - 期望：安装后即可使用
   - 实际：运行时崩溃，需要手动补装 `accelerate`

2. **Docker 镜像用户**
   - 即使镜像内已预装 `sglang[diffusion]`，仍然缺少 `accelerate`
   - 需要手动 `pip install accelerate` 才能运行

3. **CI/CD 环境**
   - 自动化测试可能因为缺少 `accelerate` 而失败
   - 需要额外的依赖安装步骤

---

## 修复建议

### 1. 立即修复（高优先级）⭐

**确保 `diffusion` extra 在“源码侧可见且可追溯”，并且在“分发包侧依赖完整”**：

- 源码侧：在 `python/pyproject.toml` 的 `[project.optional-dependencies]` 中补上（或明确迁移/引用）`diffusion = [...]`
- 分发包侧：确保最终 wheel 的 `diffusion` extra 至少包含 `accelerate` / `ftfy`

一个最小可行的 `diffusion` extra 示例（供维护者参考）：

```toml
[project.optional-dependencies]
# ... 现有定义 ...

diffusion = [
    "diffusers>=0.36.0",
    "accelerate>=0.20.0",      # ← 必需：device_map 功能
    "ftfy",                     # ← 必需：文本清洗
    "imageio>=2.36.0",          # ← 图像 I/O
    "imageio-ffmpeg>=0.5.1",   # ← FFmpeg 支持
    "opencv-python-headless>=4.10.0",  # ← 图像处理
    "cache-dit>=1.2.0",        # ← Cache-DiT 加速器（可选，但推荐）
]
```

**文件位置**: `python/pyproject.toml`，在第 93 行 `all = [...]` 之前添加。

### 2. 验证修复

修复后，执行以下测试：

```bash
# 1. 清理环境
pip uninstall -y sglang accelerate

# 2. 仅安装 sglang[diffusion]
pip install "sglang[diffusion]"

# 3. 验证 accelerate 是否自动安装
python -c "import accelerate; print('accelerate version:', accelerate.__version__)"

# 4. 运行 Diffusion 模型测试
sglang generate --model-path <test-model> --backend diffusers --prompt "test" --save-output
```

### 3. 文档更新

在官方文档中明确说明：
- `sglang[diffusion]` 会自动安装 `accelerate`
- 如果遇到 `device_map` 相关错误，检查 `accelerate` 是否已安装

---

## 相关文件

- **依赖定义**: `python/pyproject.toml`
- **相关 Issue**: #17671
- **相关文档**: 
  - `A01_B13_missing_accelerate_dependency.md` - Docker 镜像层面缺失
  - `A01_B17_final_issue_reply_summary.md` - 综合总结

---

## 测试记录

**测试时间**: 2026-01-28 17:03-17:14  
**测试镜像**: `lmsysorg/sglang:dev`  
**测试模型**: `Tongyi-MAI/Z-Image-Turbo`  
**测试命令**: 
```bash
pip install "sglang[diffusion]"
sglang generate --model-path Tongyi-MAI/Z-Image-Turbo --backend diffusers --prompt "A tech-style owl logo, futuristic, 4k" --num-inference-steps 1 --save-output
```

**结果**: ❌ 失败 - `NotImplementedError: Using device_map requires the accelerate library`

---

## 结论

这是一个严重 Bug，核心是：

1. **源码与发布分发包 extras 脱节**：源码 `python/pyproject.toml` 无 `diffusion` 定义，但分发包声明了 `diffusion`
2. **发布分发包的 `diffusion` extra 依赖不完整**：缺 `accelerate` / `ftfy`（证据见 B20）
3. **Docker dev 镜像也缺依赖**：镜像层面未预装 `accelerate` / `ftfy`（B13 已记录）

### 修复优先级

1. **最高优先级**：在 `python/pyproject.toml` 中添加 `diffusion` extra 定义
2. **高优先级**：更新 Dockerfile，确保 `accelerate` 和 `ftfy` 被安装
3. **中优先级**：更新文档，确保与代码一致

只有**同时修复这三个问题**，才能确保用户能够顺利使用 SGLang 的 Diffusion 功能。
