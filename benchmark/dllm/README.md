# dLLM Post-Processing Triton Kernel

This directory contains benchmarks for the optimized dLLM post-processing Triton kernel.

## Background

The dLLM (Diffusion LLM) decoding process requires multiple iterations of:
1. Model forward pass
2. Post-processing (softmax, argmax, threshold comparison, token update)

The post-processing step involves multiple PyTorch operations that launch separate CUDA kernels, causing:
- Kernel launch overhead
- Multiple memory round-trips
- CPU-GPU synchronization points

## Optimization

The Triton kernel (`sglang.srt.dllm.kernels.post_process`) fuses all post-processing operations into a single kernel:

```python
# Original PyTorch (multiple kernel launches):
x = torch.argmax(logits, dim=-1)
probs = F.softmax(logits, dim=-1)
p = torch.gather(probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
x = torch.where(mask_index, x, input_ids)
confidence = torch.where(mask_index, p, -inf)
transfer_index = confidence > threshold
# ... more operations

# Optimized Triton (single kernel, in-place update):
from sglang.srt.dllm.kernels import dllm_post_process_fused
dllm_post_process_fused(logits, input_ids, mask_id=128000, threshold=0.95)
# input_ids is modified in-place; no return value
```

### Key Optimizations

1. **Online Softmax Algorithm**: Compute argmax and softmax probability in a single pass
2. **Early Exit**: Skip computation for non-masked positions
3. **On-Device Fallback**: When no token exceeds threshold, a lightweight fallback kernel forces the highest-confidence position — no GPU-CPU sync needed
4. **Autotuning**: Multiple configurations for different vocab sizes

## Performance

### Kernel-Level (single call)

Benchmark environment: NVIDIA GeForce RTX 5070 Laptop GPU, PyTorch 2.9.1+cu130, Triton 3.6.0

Configuration: `block_size=32, vocab_size=128000`

| Implementation | Latency | Speedup |
|----------------|---------|---------|
| PyTorch Reference | ~1.20 ms | 1.0x |
| Triton Fused | ~0.13 ms | **9.3x** |

Measured with `benchmark/dllm/bench_triton_post_process.py`.

### End-to-End (LLaDA2.0-mini, H20 GPU)

Server config: `max_running_requests=4, cuda_graph_bs=1,2,3,4`. Measured with `test/registered/dllm/test_llada2_mini.py`.

Single request (bs=1, `test_bs_1_speed`):

| Branch | Latency | Speed | Speedup |
|--------|---------|-------|---------|
| main (baseline) | 2.333s | 303.01 token/s | 1.0x |
| Triton kernel | 2.230s | 317.01 token/s | **+4.6%** |

GSM8K (200 questions, 128 parallel, `test_gsm8k`):

| Branch | Accuracy | Latency | Throughput | Speedup |
|--------|----------|---------|------------|---------|
| main (baseline) | 0.920 | 84.53s | 298.74 token/s | 1.0x |
| Triton kernel | 0.925 | 75.98s | 321.24 token/s | **+7.5%** |

The Triton path also eliminates per-block `.item()` GPU-CPU synchronization (moved to a single global sync per iteration), which is expected to yield larger gains under concurrent batching (e.g., with FDFO scheduling #17770).

## Usage

### Kernel-Level Benchmark

```bash
python benchmark/dllm/bench_triton_post_process.py

# Skip correctness checks
python benchmark/dllm/bench_triton_post_process.py --skip-correctness

# NCU profiling
ncu --set full -o dllm_post_process python benchmark/dllm/bench_triton_post_process.py --ncu
```

### End-to-End Benchmark

```bash
# Single request speed (bs=1)
python test/registered/dllm/test_llada2_mini.py TestLLaDA2Mini.test_bs_1_speed

# GSM8K throughput (max_running_requests=4)
python test/registered/dllm/test_llada2_mini.py TestLLaDA2Mini.test_gsm8k
```

### Unit Tests

```bash
python -m pytest test/registered/dllm/test_dllm_triton_kernel.py
```

### In Code

```python
from sglang.srt.dllm.kernels import dllm_post_process_fused

logits = torch.randn(32, 128000, dtype=torch.float16, device='cuda')
input_ids = torch.full((32,), mask_id, dtype=torch.int64, device='cuda')

# Fused operation (modifies input_ids in-place)
dllm_post_process_fused(logits, input_ids, mask_id=128000, threshold=0.95)
```

## Files

- `bench_triton_post_process.py`: Kernel-level benchmark (Triton vs PyTorch)
- `../../test/registered/dllm/test_dllm_triton_kernel.py`: Correctness unit tests
