# hisa_triton

Triton re-implementations of HISA kernels (one at a time), plus a bench harness to compare them against the tilelang originals in `../hisa/custom_ops.py`.

## Why

The two stacks have different compilers and different code-gen. We want one clean apples-to-apples kernel-level data point: on the same shapes, same input tensors, does triton beat tilelang or vice versa? That informs whether a larger port is worth pursuing.

## Layout

```
hisa_triton/
├── __init__.py
├── kernels.py      # triton kernel defs (grow as we add more)
├── benchmark.py    # side-by-side vs tilelang with correctness + latency
└── README.md
```

## Kernels covered

| Kernel | tilelang source | triton impl | Status |
|---|---|---|---|
| `batch_pool_mqa` (contiguous block-MQA, fp8×fp8) | `hisa/custom_ops.py:batch_decode_pool_mqa_attn_return_logits_fp8` | `kernels.py:batch_pool_mqa_triton` | ✅ ported |
| `sparse_paged_mqa` (paged indirect, the 80% hotspot) | `hisa/custom_ops.py:fp8_native_paged_block_sparse_mqa_attn_return_logits` | — | TODO |

## Run

```bash
# Bench the contiguous block-MQA kernel across a sweep.
python -m sglang.srt.layers.attention.nsa.hisa_triton.benchmark \
    --kernel batch_pool_mqa \
    --batch-sizes 1 8 32 64 \
    --num-pool 128 512 1024
```

Output columns: `B`, `nb`, tilelang time (ms ± stdev), triton time, speedup, correctness check summary.

## Notes on the fp8 path

- Triton 3.0+ required. We're on 3.5.1.
- `tl.dot` with fp8 operands → f32 accumulator: supported on H100+.
- We keep the fp8 scales applied post-GEMM (same as tilelang kernel), so the numerical recipe matches.

## Correctness tolerance

The bench uses `atol=1e-3, rtol=5e-3` on finite positions. Tilelang and triton may differ by ~1 fp8 ULP at a few positions due to different accumulation order — that's expected and OK.

Separate: ±inf positions (the `force_maintain` slot at pos=0 and last-valid, and the -inf masked tail) must match bit-exactly across the two impls.
