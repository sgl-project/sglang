# 通信操作性能测试指南

## 📋 概述

本文档说明如何正确测试自定义通信操作（PyNccl）的性能提升，以及哪些 diffusion 模型会用到这些操作。

## 🔍 哪些模型会使用通信操作？

### 1. 使用 Tensor Parallelism (TP) 的模型

**模型**: Flux 2, QwenImage

**通信场景**:
- `all_reduce`: 在 linear 层的输出聚合
- `all_gather`: 在 embedding 层收集结果

**测试命令**:
```bash
# Flux 2 with TP=2
sglang generate \
  --model-path black-forest-labs/FLUX.1-dev \
  --prompt "A beautiful landscape" \
  --tp-size 2 \
  --perf-dump-path flux_tp2.json

# Flux 2 with TP=4
sglang generate \
  --model-path black-forest-labs/FLUX.1-dev \
  --prompt "A beautiful landscape" \
  --tp-size 4 \
  --perf-dump-path flux_tp4.json
```

### 2. 使用 Sequence Parallelism (SP) 的模型

**模型**: WanVideo, HunyuanVideo, FastWan

**通信场景**:
- `all_to_all_4D`: 在 attention 层重新分布序列和头维度
- `all_gather`: 收集 replicated token 的结果

**代码位置**: `runtime/layers/attention/layer.py` 的 `UlyssesAttention`

**测试命令**:
```bash
# Wan 2.1 with SP (Ulysses)
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A curious raccoon" \
  --ulysses-degree 2 \
  --perf-dump-path wan_sp2.json

# Wan 2.1 with SP (Ulysses) + Ring
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A curious raccoon" \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --perf-dump-path wan_usp22.json
```

### 3. 使用 CFG Parallel 的模型

**模型**: 所有支持 Classifier-Free Guidance 的模型

**通信场景**:
- `broadcast`: 广播条件和非条件的 prompt embeddings
- `all_gather`: 收集 CFG 并行的结果

**测试命令**:
```bash
# 任何模型 with CFG parallel
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A beautiful scene" \
  --enable-cfg-parallel \
  --num-gpus 2 \
  --perf-dump-path wan_cfg.json
```

## 🚀 正确的性能测试方法

### 方法 1: 端到端推理测试（推荐）

对比使用和不使用 device_communicator 的性能：

```bash
# 1. 使用 PyNccl (device_communicator)
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "test prompt" \
  --ulysses-degree 2 \
  --perf-dump-path baseline_pynccl.json

# 2. 使用 PyTorch distributed (需要修改代码禁用 device_communicator)
# 在 group_coordinator.py 中暂时设置 use_device_communicator=False
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "test prompt" \
  --ulysses-degree 2 \
  --perf-dump-path baseline_pytorch.json

# 3. 对比性能
python python/sglang/multimodal_gen/benchmarks/compare_perf.py \
  baseline_pytorch.json baseline_pynccl.json
```

### 方法 2: Attention 层微基准测试

创建专门的 attention 层测试：

```python
# test_attention_communication.py
import torch
import torch.distributed as dist
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    initialize_model_parallel
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_to_all_4D
)

def benchmark_attention_comm():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    initialize_model_parallel(sequence_parallel_degree=2)
    
    # 模拟 attention 层的通信模式
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    
    # 大张量：模拟真实的 attention 输入 (类似 Wan 模型)
    # [3(qkv), seq_len, num_heads, head_dim]
    batch_size = 1
    seq_len = 4096  # 长序列
    num_heads = 16
    head_dim = 64
    
    qkv = torch.randn(3, seq_len, num_heads, head_dim, device=device)
    
    # 预热
    for _ in range(10):
        _ = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
    torch.cuda.synchronize()
    
    # 测试
    import time
    start = time.perf_counter()
    for _ in range(100):
        result = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    if rank == 0:
        print(f"Average time per all-to-all: {elapsed/100*1000:.3f} ms")
        print(f"Tensor size: {qkv.nbytes / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    benchmark_attention_comm()
```

运行：
```bash
torchrun --nproc_per_node=2 test_attention_communication.py
```

## 📊 预期的性能提升

### 小张量（< 1MB）
- **预期提升**: 0-10%
- **原因**: 开销占主导，PyNccl 优势不明显

### 中等张量（1-10MB）
- **预期提升**: 10-30%
- **原因**: 通信时间开始占主导

### 大张量（> 10MB）
- **预期提升**: 30-50%
- **原因**: PyNccl 的优化完全发挥作用

### 实际模型推理
- **Wan 2.1 with SP (ulysses_degree=2)**: 预期整体提升 5-15%
- **Flux 2 with TP (tp_size=2)**: 预期整体提升 3-10%
- **复杂并行配置**: 预期整体提升 10-25%

## 🎯 关键测试场景

### 场景 1: Wan 视频生成 + Sequence Parallel

这是最能体现通信优化的场景：

```bash
# 测试脚本
#!/bin/bash

MODEL="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
PROMPT="A curious raccoon peers through a vibrant field of yellow sunflowers"

# 不同的并行配置
for SP in 1 2 4; do
  echo "Testing SP=$SP..."
  sglang generate \
    --model-path $MODEL \
    --prompt "$PROMPT" \
    --ulysses-degree $SP \
    --num-inference-steps 50 \
    --perf-dump-path "wan_sp${SP}.json"
done

# 对比性能
python python/sglang/multimodal_gen/benchmarks/compare_perf.py \
  wan_sp1.json wan_sp2.json > sp_comparison.txt
```

### 场景 2: Flux 图像生成 + Tensor Parallel

```bash
#!/bin/bash

MODEL="black-forest-labs/FLUX.1-dev"
PROMPT="A professional photo of a cat wearing a tiny hat"

# 不同的 TP 配置
for TP in 1 2 4; do
  echo "Testing TP=$TP..."
  sglang generate \
    --model-path $MODEL \
    --prompt "$PROMPT" \
    --tp-size $TP \
    --num-inference-steps 28 \
    --perf-dump-path "flux_tp${TP}.json"
done
```

## 🔬 通信操作的调用位置

### 1. `all_reduce`
- **位置**: `runtime/layers/linear.py` - `RowParallelLinear`
- **频率**: 每个 attention 块后
- **大小**: 取决于 hidden_dim 和 sequence length

### 2. `all_gather`
- **位置**: `runtime/layers/vocab_parallel_embedding.py`
- **频率**: Embedding 层输出
- **大小**: 取决于 vocab_size 和 batch_size

### 3. `all_to_all_4D`
- **位置**: `runtime/layers/attention/layer.py` - `UlyssesAttention`
- **频率**: 每个 attention 层前后各一次
- **大小**: [batch, seq_len, num_heads, head_dim]

### 4. `broadcast`
- **位置**: `runtime/pipelines_core/executors/parallel_executor.py`
- **频率**: 每次 forward 开始时
- **大小**: 整个 batch 的元数据

## 💡 调试建议

### 1. 验证 PyNccl 是否启用

```python
# 在 group_coordinator.py 中添加日志
def all_reduce(self, input_, ...):
    if self.device_communicator is not None:
        logger.info(f"✅ Using device_communicator (PyNccl)")
        result = self.device_communicator.all_reduce(input_, op=op)
    else:
        logger.info(f"⚠️  Fallback to PyTorch distributed")
        # ...
```

### 2. 检查通信耗时

```python
import time

def all_reduce(self, input_, ...):
    start = time.perf_counter()
    result = self.device_communicator.all_reduce(input_, op=op)
    elapsed = time.perf_counter() - start
    logger.info(f"all_reduce time: {elapsed*1000:.3f} ms, size: {input_.nbytes/1024/1024:.2f} MB")
    return result
```

### 3. 对比不同实现

临时修改代码，对比两种实现：

```python
def all_reduce(self, input_, ...):
    # 方法 1: PyNccl
    input_pynccl = input_.clone()
    start1 = time.perf_counter()
    result1 = self.device_communicator.all_reduce(input_pynccl, op=op)
    time1 = time.perf_counter() - start1
    
    # 方法 2: PyTorch
    input_torch = input_.clone()
    start2 = time.perf_counter()
    torch.distributed.all_reduce(input_torch, group=self.device_group, op=op)
    time2 = time.perf_counter() - start2
    
    logger.info(f"PyNccl: {time1*1000:.3f} ms, PyTorch: {time2*1000:.3f} ms, Speedup: {time2/time1:.2f}x")
    return result1
```

## 📈 性能分析工具

### 使用 NCCL 环境变量

```bash
# 启用详细日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 性能调优
export NCCL_IB_DISABLE=0  # 启用 InfiniBand (如果有)
export NCCL_P2P_DISABLE=0  # 启用 P2P
export NCCL_SHM_DISABLE=0  # 启用共享内存

sglang generate --model-path ... --ulysses-degree 2 ...
```

### 使用 PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    result = generator.generate(sampling_params_kwargs=dict(prompt="test"))

prof.export_chrome_trace("trace.json")
# 在 chrome://tracing 中查看
```

## ✅ 检查清单

- [ ] 确认使用了 `device_communicator` (查看日志)
- [ ] 确认 PyNccl 初始化成功
- [ ] 使用多 GPU 场景 (TP > 1 或 SP > 1)
- [ ] 使用足够大的模型和序列长度
- [ ] 在端到端推理中测试，而不是单独测试通信操作
- [ ] 对比不同并行配置的性能
- [ ] 检查 NCCL 版本和配置

## 🎓 总结

1. **单独测试通信操作性能提升不明显是正常的**
2. **真正的性能提升体现在实际模型推理中**
3. **使用 SP 或 TP 的模型才会用到这些通信操作**
4. **推荐测试场景**: Wan + Ulysses SP 或 Flux + TP

