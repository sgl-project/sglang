# Stage-3 PoC — split-source unified_kv decode

Proves the core hard part of the FP8 main-KV plan: a single online-softmax
decode K-loop that dispatches each slot to one of TWO physical buffers by
`slot >= swa_pages`, reading BF16 SWA rows or dequantizing packed
(448 fp8 + 64 bf16 + ue8m0 scale, 584 B) compressed rows in-kernel.

## Files
- `python/.../unified_kv_kernels/paged_decode_split_src.py`
  PoC Triton kernel `sparse_attn_v4_paged_decode_split_src`. Byte-identical
  arithmetic to the existing `_paged_decode_fused_kernel`; only the KV load is
  replaced with per-slot source dispatch. Fused single-pass path only
  (split-K is a mechanical copy of the same load block — deferred).
- `poc/poc_split_src_decode.py`  (needs GPU + torch/triton/aiter)
  Self-test. Oracle = production `dequantize_k_cache_paged` -> bf16 unified
  buffer -> existing `sparse_attn_v4_paged_decode`. PoC reads split buffers and
  must match to fp32-accum tolerance. Covers ratio_comp 0.0/0.5/0.7/1.0,
  T=1/16/32, H=16/64/128, page_size 1/64/128.
- `poc/addr_math_check.py`  (runs anywhere, numpy only)
  Independent numpy port of BOTH the production dequant ref AND the PoC
  kernel's exact address expressions; asserts they compute identical offsets.
  STATUS: PASS for page_size 1/64/128 (max_abs_diff 0.0).

## How to run the GPU self-test (on an MI355 node with the sglang runtime)
    cd unified_kv_fp8/sglang
    python -m poc.poc_split_src_decode
Expect: every case `OK`, final `ALL OK`.

## Status
- addr_math_check.py: PASS locally (no GPU needed).
- poc_split_src_decode.py: NOT yet run — this box has no torch/triton/aiter and
  the GPU is low-power; requires dispatch to a GPU runtime.

## CUDAGraph-safety note
Source dispatch is a runtime `tl.where` over slot VALUES + masked loads; launch
grid depends only on capture-time (T, H). No host-side control flow, no
shape-varying allocation. Safe to capture.
