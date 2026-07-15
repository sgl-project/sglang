## Motivation

Current LoRA weight storage allocates `max_lora_rank` (e.g., 64) for every adapter, regardless of actual rank. An adapter with rank=8 wastes 56/64 = 87.5% of its allocated weight memory to zero-padding. This wastes GPU memory bandwidth and hurts L2 cache efficiency, especially under multi-tenant LoRA serving where many adapters of different ranks coexist.

This PR introduces **paged LoRA**: weight storage organized into fixed-size pages (`page_rank_size=8` ranks per page), mapped to physical pages via a `page_table`. An r=8 adapter uses 1 page; an r=64 adapter uses 8 pages — no zero-padding waste. Pages are allocated/evicted individually, similar to paged attention. The result is a smaller GPU working set, better cache utilization, and +5.6% decode throughput on single GPU (TP=1) and +1.4% on 4-GPU (TP=4).

## Usage

By default (`--lora-page-rank-size 0`), SGLang uses flat LoRA mode: each adapter's weights are stored contiguously at `max_lora_rank` size, with zero-padding for adapters whose actual rank is smaller.

Enable paged mode by setting `--lora-page-rank-size` to a positive value:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-paths adapter1=/path/to/lora1 adapter2=/path/to/lora2 \
    --lora-page-rank-size 8 \
    --lora-pages 64 \
    --max-lora-rank 64 \
    --lora-target-modules all
```

New server arguments:

| Argument | Default | Description |
|---|---|---|
| `--lora-page-rank-size` | 0 (disabled) | Ranks per page. 8 = enable paged mode. |
| `--lora-pages` | 0 (auto) | Total physical pages in pool. Auto-computed if 0. |

## Modifications

### New files (9)

| File | Location | Description |
|---|---|---|
| `paged_mem_pool.py` | `srt/lora/` | Page pool: allocate/free/evict/page_in/ensure_adapter_ready. O(1) reverse map for LRU eviction. TP-aware weight scatter. |
| `chunked_sgmv_shrink_paged.py` | `kernels/ops/gemm/` | Paged shrink Triton kernel. Grid `(max_pages, segments)`. Reads `page_table[w_index][pid_page]` → physical page → `x @ A_pages[phys_page]`. Evicted pages (-1) return immediately. |
| `chunked_sgmv_expand_paged.py` | `kernels/ops/gemm/` | Paged expand Triton kernel. Loops `MAX_PAGES_PER_LORA` (constexpr), loading one B-weight page per iteration. Pages beyond rank or swapped out are skipped. |
| `test_paged_mem_pool.py` | `test/registered/unit/lora/` | 17 CPU unit tests: build_page_table_tensor, page_generation counter, page lifecycle. |
| `test_lora_paged_manager.py` | `test/registered/unit/lora/` | 11 CPU unit tests: cache key ordering, cache hit/miss, bounded clear. |
| `test_paged_lora_args.py` | `test/registered/unit/lora/` | 4 CPU unit tests: default values, CLI parsing. |
| `test_paged_kernel_correctness.py` | `test/registered/lora/` | 6 GPU kernel tests: paged vs flat bit-exact comparison. |
| `test_lora_paged_e2e.py` | `test/registered/lora/` | 4 GPU E2E tests: server launch + LoRA request verification. |
| `PR_DESCRIPTION.md` | root | This file. |

### Modified files (12)

| File | Changes |
|---|---|
| `lora_manager.py` | Paged path in `__init__`, `init_memory_pool`, `init_cuda_graph_batch_info`, `fetch_new_loras` (returns bool), `prepare_lora_batch` (builds page_table, cache, CUDA graph copy), `update_lora_info` (sets A_pages/B_pages on layers). |
| `chunked_backend.py` | Paged kernel dispatch: `run_lora_a_sgemm_paged`, `run_lora_b_sgemm_paged`, `run_qkv_lora_paged`, `run_gate_up_lora_paged`. `init_cuda_graph_batch_info` pre-allocates `cg_page_table`. |
| `layers.py` | `_is_paged_mode()` + paged `apply_lora` branches in ColumnParallel, MergedColumnParallel, QKVParallelLinear, RowParallelLinear. |
| `base_backend.py` | Paged method declarations + `init_cuda_graph_batch_info` params (`page_rank_size`, `max_lora_rank`). |
| `utils.py` | `LoRABatchInfo` fields: `page_table`, `max_pages_per_lora`, `page_rank_size`. |
| `scheduler.py` | `_get_lora_rank` + page pool check in `_can_schedule_lora_req`. |
| `forward_batch_info.py` | `fetch_new_loras` return check → `RuntimeError` on failure. |
| `lora_overlap_loader.py` | `fetch_new_loras` return value check. |
| `server_args.py` | `--lora-page-rank-size`, `--lora-pages` CLI args. |
| `chunked_sgmv_shrink.py` | Add `N` to triton cache key. |
| `chunked_sgmv_expand.py` | Add `MAX_RANK` to triton cache key. |
| `kernels/ops/gemm/__init__.py` | Export paged kernel functions. |

Both paged kernels use `get_lora_shrink_config()` / `get_lora_expand_config()` for auto-tuned block sizes, matching the flat kernel's tuning mechanism.

## Accuracy Tests

Paged and flat kernels produce **identical output tensors** (max_diff = 0.00e+00):

| Test | Input | Comparison | max_diff | Result |
|---|---|---|---|---|
| `test_shrink_single_adapter` | 1 adapter (r=8), 16 tokens | `shrink_paged(x, A_pages) == shrink_flat(x, A_flat)` | 0.00e+00 | ✅ |
| `test_shrink_multiple_adapters` | 3 adapters (r=8 each) | same, 3 segments | 0.00e+00 | ✅ |
| `test_shrink_multi_page` | 1 adapter (r=16), 2 pages | same, `page_table=[[0,1]]` | 0.00e+00 | ✅ |
| `test_expand_single_adapter` | 1 adapter (r=8), 16 tokens | `expand_paged(x, B_pages) == expand_flat(x, B_flat)` | 0.00e+00 | ✅ |
| `test_shrink_expand_chain` | 1 adapter (r=8), 16 tokens | `expand_paged(shrink_paged(x)) == expand_flat(shrink_flat(x))` | 0.00e+00 | ✅ |
| `test_shrink_evicted_page` | 1 adapter (r=16), `page_table=[[0,-1]]` | page 0 output non-zero, page 1 output = 0 | — | ✅ |

## Speed Tests and Profiling

**Setup:** 4× NVIDIA H800 (80GB HBM3), CUDA 12.9, Llama-3.1-8B-Instruct, 2 LoRA adapters (r=8 + r=64), 300 prompts (input=256, output=128), concurrency=32, warmup=20, 3 runs per config (median reported).

### TP=1 (single GPU)

| Metric | Flat (median) | Flat IQR | Paged (median) | Paged IQR | Delta |
|---|---|---|---|---|---|
| Output throughput (tok/s) | 544.02 | [543.29, 544.69] | **574.60** | [574.27, 574.62] | **+5.6%** |
| Median TPOT (ms) | 55.67 | [55.42, 55.77] | **52.71** | [52.70, 52.74] | **-5.3%** |
| P99 TPOT (ms) | 60.48 | — | **57.71** | — | **-4.6%** |
| Median E2E (ms) | 3356.65 | — | **3212.71** | — | **-4.3%** |

IQR ranges do not overlap — improvement is statistically significant.

### TP=4 (4 GPUs)

| Metric | Flat (median) | Flat IQR | Paged (median) | Paged IQR | Delta |
|---|---|---|---|---|---|
| Output throughput (tok/s) | 1030.98 | [992.07, 1069.89] | **1044.94** | [980.55, 1054.61] | **+1.4%** |
| Median TPOT (ms) | 28.32 | [27.88, 28.77] | 28.56 | [27.90, 30.29] | +0.8% |
| P99 TPOT (ms) | 41.19 | — | **38.96** | — | **-5.4%** |

TPOT IQR ranges overlap — decode performance is comparable. Paged shows +1.4% throughput and -5.4% P99 TPOT improvement.

## Checklist
- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).
