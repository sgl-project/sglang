# A01_B13: Missing `accelerate` Dependency - 教科书级别的"打脸"现场

## 概述

这是一个**教科书级别的"打脸"现场**。这段 Traceback 完美实证了 SGLang 的 Docker 镜像（dev 版）在 Diffusion 模块上确实是个"半成品"。

**错误类型**: 依赖缺失 - `accelerate` 库未安装
**严重程度**: 🔴 **致命** - 导致 Diffusion 模块完全无法运行
**复现状态**: ✅ **已复现**

---

## 错误信息

### 完整 Traceback

```
Loading diffusers pipeline with dtype=torch.bfloat16, device_map=cuda
Cannot initialize model with low cpu memory usage because `accelerate` was not found in the environment. 
Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install `accelerate` for faster 
and less memory-intense model loading. You can do so with:
```
pip install accelerate
```

[01-28 04:18:09] Worker 0: Shutdown complete.
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
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/diffusers/pipelines/pipeline_utils.py", line 793, in from_pretrained
    raise NotImplementedError(
NotImplementedError: Using `device_map` requires the `accelerate` library. Please install it using: `pip install accelerate`.
[rank0]:[W128 04:18:09.073114819 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[01-28 04:18:10] Rank 0 scheduler is dead. Please check if there are relevant logs.
[01-28 04:18:11] Exit code: 1
Traceback (most recent call last):
  File "/usr/local/bin/sglang", line 7, in <module>
    sys.exit(main())
             ^^^^^^
  File "/sgl-workspace/sglang/python/sglang/cli/main.py", line 42, in main
    args.func(args, extra_argv)
  File "/sgl-workspace/sglang/python/sglang/cli/generate.py", line 29, in generate
    generate_cmd(parsed_args)
  File "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/cli/generate.py", line 116, in generate_cmd
    generator = DiffGenerator.from_pretrained(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py", line 106, in from_pretrained
    return cls.from_server_args(server_args, local_mode=local_mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py", line 126, in from_server_args
    instance.local_scheduler_process = instance._start_local_server_if_needed()
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py", line 143, in _start_local_server_if_needed
    processes = launch_server(self.server_args, launch_http_server=False)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/launch_server.py", line 153, in launch_server
    data = reader.recv()
           ^^^^^^^^^^^^^
  File "/usr/lib/python3.12/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
          ^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/multiprocessing/connection.py", line 430, in _recv_bytes
    buf = self._recv(4)
          ^^^^^^^^^^^^^
  File "/usr/lib/python3.12/multiprocessing/connection.py", line 399, in _recv
    raise EOFError
EOFError
```

---

## 报错深度拆解：为什么说这是"打脸"现场？

### 核心矛盾点

**SGLang 试图调用一个它声称支持的专业功能，却在执行时发现自己连最基础的工具包都没装。**

### 致命的缺失

```
NotImplementedError: Using `device_map` requires the `accelerate` library. 
Please install it using: `pip install accelerate`.
```

### 荒谬之处

1. **`device_map` 是将模型加载到 GPU 的核心技术**
   - `device_map=cuda` 是 SGLang 代码中明确使用的参数
   - 这是将模型加载到 RTX 4090 (GPU) 的必需技术

2. **SGLang 作为高性能推理引擎，其默认的 dev 镜像里居然没有预装 `accelerate`**
   - `accelerate` 是管理显存最基础的库
   - 这是 Hugging Face 生态系统的核心依赖

3. **官方文档 vs. 现实货不对板**
   - 文档里写着"Broad Model Support"和"Ease of Use"
   - 但官方推荐的镜像连 `pip install accelerate` 都没做

### 连锁反应

1. **第一步警告**:
   ```
   Cannot initialize model with low cpu memory usage because `accelerate` was not found
   Defaulting to `low_cpu_mem_usage=False`
   ```

2. **第二步尝试**:
   - SGLang 尝试使用 `device_map=cuda` 加载模型
   - 但 `diffusers` 库检测到没有 `accelerate`，拒绝执行

3. **第三步崩溃**:
   ```
   NotImplementedError: Using `device_map` requires the `accelerate` library
   ```

4. **最终结果**:
   - Rank 0 scheduler is dead
   - 整个推理进程崩毁
   - 报出 `EOFError`

---

## 实验结论总结

通过这次 1.3B 模型的"自杀式"运行，我们获得了以下结论：

| 维度 | 实验发现 |
|------|---------|
| **镜像完整性** | ✅ **实锤缺失**。`lmsysorg/sglang:dev` 镜像根本没有准备好跑 Diffusion 任务 |
| **官方文档 vs. 现实** | ❌ **货不对板**。文档里写着"Broad Model Support"和"Ease of Use"，但官方推荐的镜像连 `pip install accelerate` 都没做 |
| **你的环境** | ✅ **100% 正常**。下载链路通畅，显卡驱动正常，是软件层面的依赖断裂杀死了进程 |

---

## 解决方案

### 临时解决方案（Workaround）

**在容器内安装缺失的依赖**:

```bash
# 进入容器
docker exec -it <container_id> bash

# 安装 accelerate
pip install accelerate

# 或者使用 uv（如果容器内有）
uv pip install accelerate
```

### 完整修复方案

**在 Dockerfile 中添加 `accelerate` 依赖**:

```dockerfile
# 在构建镜像时添加
RUN uv pip install accelerate
```

**或者在 `pyproject.toml` 中添加依赖**:

```toml
[project]
dependencies = [
    # ... 其他依赖 ...
    "accelerate",  # 添加这一行
]
```

---

## 验证步骤

### 1. 安装 accelerate

```bash
docker run --rm -it --gpus all \
  lmsysorg/sglang:dev \
  bash -c "pip install accelerate && sglang generate --model-path <model> --prompt 'test' --save-output"
```

### 2. 验证修复

如果安装 `accelerate` 后可以正常运行，说明问题确实在依赖缺失。

---

## 下一步行动

### 选项 1: 提交 GitHub Issue（推荐）

**你现在手里握着的这段 Traceback，就是最硬的证据。**

可以提交一个详细的 Issue，包括：
- 完整的错误信息
- 复现步骤
- 环境信息
- 临时解决方案

### 选项 2: 继续测试

**如果你想看 4090 真正跑出这张图**，只需要在容器里做这一件事：

```bash
# 补上镜像里缺失的那个"零件"
pip install accelerate
```

补完之后，再运行刚才那个 1.3B 的命令，你的 4090 就会立刻开始真正的计算逻辑。

---

## 相关文档

- [A01_B01_original_issue.md](./A01_B01_original_issue.md) - 原始问题
- [A01_B12_complete_troubleshooting_marathon.md](./A01_B12_complete_troubleshooting_marathon.md) - 完整的排障马拉松记录
- [A01_B09_issue_reply_to_kevin.md](./A01_B09_issue_reply_to_kevin.md) - 给 Kevin 的回复

---

## Issue 描述模板

如果你想提交 GitHub Issue，可以使用以下模板：

```markdown
## Bug Report: Missing `accelerate` dependency in SGLang dev image

### Description
SGLang dev Docker image (`lmsysorg/sglang:dev`) is missing the `accelerate` library, 
which is required for Diffusion models to use `device_map=cuda`.

### Error Message
```
NotImplementedError: Using `device_map` requires the `accelerate` library. 
Please install it using: `pip install accelerate`.
```

### Steps to Reproduce
1. Run: `docker run --gpus all lmsysorg/sglang:dev sglang generate --model-path <diffusion-model> --prompt "test" --save-output`
2. Observe the error

### Expected Behavior
The model should load successfully with `device_map=cuda`.

### Actual Behavior
The process crashes with `NotImplementedError` due to missing `accelerate` library.

### Environment
- Docker image: `lmsysorg/sglang:dev`
- GPU: RTX 4090
- CUDA: 12.9

### Workaround
Install `accelerate` inside the container:
```bash
pip install accelerate
```

### Suggested Fix
Add `accelerate` to the Docker image dependencies or `pyproject.toml`.
```

---

**最后更新**: 2025年1月28日
**测试环境**: Windows + Docker + RTX 4090 + CUDA 12.9
**状态**: ✅ **已复现并确认**
