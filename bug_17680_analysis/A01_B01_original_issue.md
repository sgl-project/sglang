# A01_B01: Issue #17680 原始内容

## 相关文档
- [A01: MoE Tensor Parallelism Bug 详解](./A01_moe_tp_bug.md) - 了解整体问题
- [A01_B02: 修复分析](./A01_B02_fix_analysis.md) - 修复前后代码对比与详细解释
- [A01_B03: 代码变更](./A01_B03_code_changes.md) - 修复的具体代码位置

---

## Issue 链接
https://github.com/sgl-project/sglang/issues/17680

## 问题标题
[Bug] RuntimeError with MoE model MedAIBase/AntAngelMed-INT4 on tp=2

## 提交者
@applebomb

## 问题描述

### 1. Describe the bug
When trying to run the MoE model MedAIBase/AntAngelMed-INT4 with tensor parallelism (--tp-size 2), the server fails to load. It throws a RuntimeError: start (8) + length (8) exceeds dimension size (8). on the second GPU (TP1). The same model also causes an OOM error when trying to run on a single GPU, so tp=2 is the only viable option.

### 2. To Reproduce
Run the following command on a machine with 2x NVIDIA A100 (80GB) GPUs.

**Command:**
```bash
python3 -m sglang.launch_server  \
    --model-path MedAIBase/AntAngelMed-INT4 \
    --host 0.0.0.0 --port 30012  \
    --trust-remote-code  \
    --attention-backend fa3  \
    --mem-fraction-static 0.9 \
    --tp-size 2
```

### 3. Expected behavior
The model server should launch successfully without any errors.

### 4. Environment

**Hardware**: 2x NVIDIA A100 (80GB)

**Software versions**:
- sglang version: 0.5.6 (and also tested with the latest version after pip install --upgrade sglang)
- Python: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0]
- CUDA available: True
- GPU 0,1: NVIDIA A100 80GB PCIe
- GPU 0,1 Compute Capability: 8.0
- CUDA_HOME: /usr/local/cuda-12.3
- NVCC: Cuda compilation tools, release 12.3, V12.3.103
- CUDA Driver Version: 560.35.03
- PyTorch: 2.9.1+cu128
- sglang: 0.5.8
- sgl_kernel: 0.3.21
- flashinfer_python: 0.6.1
- flashinfer_cubin: 0.6.1
- flashinfer_jit_cache: Module Not Found
- triton: 3.5.1
- transformers: 4.57.1
- torchao: 0.9.0
- numpy: 2.4.1
- aiohttp: 3.13.3
- fastapi: 0.128.0
- hf_transfer: 0.1.9
- huggingface_hub: 0.36.0
- interegular: 0.3.3
- modelscope: 1.9.5
- orjson: 3.11.5
- outlines: 0.1.11
- packaging: 26.0
- psutil: 7.2.1
- pydantic: 2.12.5
- python-multipart: 0.0.21
- pyzmq: 27.1.0
- uvicorn: 0.40.0
- uvloop: 0.22.1
- vllm: Module Not Found
- xgrammar: 0.1.27
- openai: 2.6.1
- tiktoken: 0.12.0
- anthropic: 0.76.0
- litellm: Module Not Found
- decord2: 3.0.0

**NVIDIA Topology:**
```
        GPU0    GPU1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      SYS     0,2,4,6,8,10    0               N/A
GPU1    SYS      X      1,3,5,7,9,11    1               N/A

Legend:
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

**ulimit soft**: 65535

### 5. Full error log

```
[2026-01-25 01:19:07 TP0] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/transformers/__init__.py)
[2026-01-25 01:19:07 TP1] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/transformers/__init__.py)
[2026-01-25 01:19:07 TP1] Load weight begin. avail mem=78.77 GB
[2026-01-25 01:19:07 TP0] Load weight begin. avail mem=78.77 GB
[2026-01-25 01:19:07 TP1] Using CompressedTensorsWNA16MarlinMoEMethod
[2026-01-25 01:19:07 TP0] Using CompressedTensorsWNA16MarlinMoEMethod
Loading safetensors checkpoint shards:   0% Completed | 0/12 [00:00<?, ?it/s]
[2026-01-25 01:19:08 TP1] Scheduler hit an exception: Traceback (most recent call last):
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py", line 2937, in run_scheduler_process
    scheduler = Scheduler(
                ^^^^^^^^^^
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py", line 346, in __init__
    self.init_model_worker()
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py", line 535, in init_model_worker
    self.init_tp_model_worker()
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py", line 497, in init_tp_model_worker
    self.tp_worker = TpModelWorker(
                     ^^^^^^^^^^^^^^
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/managers/tp_worker.py", line 246, in __init__
    self._init_model_runner()
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/managers/tp_worker.py", line 329, in _init_model_runner
    self._model_runner = ModelRunner(
                         ^^^^^^^^^^^^
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/model_executor/model_runner.py", line 383, in __init__
    self.initialize(min_per_gpu_memory)
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/model_executor/model_runner.py", line 460, in initialize
    self.load_model()
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/model_executor/model_runner.py", line 889, in load_model
    self.model = self.loader.load_model(
                 ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/model_loader/loader.py", line 662, in load_model
    self.load_weights_and_postprocess(
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/model_loader/loader.py", line 670, in load_weights_and_postprocess
    model.load_weights(weights)
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/models/bailing_moe.py", line 979, in load_weights
    weight_loader(
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton/layer.py", line 607, in weight_loader
    self._weight_loader_physical(
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton/layer.py", line 637, in _weight_loader_physical
    self._weight_loader_impl(
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton/layer.py", line 794, in _weight_loader_impl
    self._load_model_weight_or_group_weight_scale(
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton/layer.py", line 332, in _load_model_weight_or_group_weight_scale
    self._load_w2(
  File "/home/zhangfan/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton/layer.py", line 501, in _load_w2
    loaded_weight = loaded_weight.narrow(
                    ^^^^^^^^^^^^^^^^^^^^^
RuntimeError: start (8) + length (8) exceeds dimension size (8).

[2026-01-25 01:19:08] Received sigquit from a child process. It usually means the child failed.
./Ant-INT4-1gpu.run: line 7: 2502678 Killed                  python3 -m sglang.launch_server --model-path MedAIBase/AntAngelMed-INT4 --host 0.0.0.0 --port 30012 --trust-remote-code --attention-backend fa3 --mem-fraction-static 0.6 --tp-size 2
```

**Warning:**
```
[W125 01:22:28.490457241 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead
```

### Issue 状态
- **状态**: Open
- **提交时间**: opened yesterday · edited by applebomb

### 关键信息
1. **错误位置**: `sglang/srt/layers/moe/fused_moe_triton/layer.py`, line 501, in `_load_w2`
2. **错误信息**: `RuntimeError: start (8) + length (8) exceeds dimension size (8)`
3. **触发条件**: 
   - MoE模型 `MedAIBase/AntAngelMed-INT4`
   - 使用 `--tp-size 2` (tensor parallelism)
   - 在第二个GPU (TP1) 上加载权重时出错
4. **问题原因**: 
   - 当 `tp_rank=1` 时，代码尝试从 `shard_size * tp_rank = 8` 开始，取长度为 `shard_size = 8` 的切片
   - 但权重张量的维度大小只有8，所以 `start (8) + length (8) = 16` 超出了维度大小
   - 这是因为权重维度没有正确对齐，需要添加padding逻辑
