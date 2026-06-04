# LoRA MoE per-kernel testbeds (EP8 bs64 Kimi-K2.5-NVFP4 decode)

Self-contained **perf-bench + correctness-test** scripts for the individual MoE kernels on the
Kimi-K2.5-NVFP4 LoRA decode path, so each kernel can be benchmarked and checked in isolation
(outside a full 2-node serving run). Built to drive per-kernel optimization.

## Scenario these reproduce

Per-rank shapes of an **EP8, batch-size-64 decode** step (Kimi-K2.5-NVFP4, `tp8 ep8 dp1 nnodes2`,
LoRA on, `modelopt_fp4`): 48 experts/rank, `top_k=8`, `hidden=7168`, `inter=2048`,
`num_tokens=64`, `max_num_padded_tokens=3200`. These were captured from a real e2e run (one-off
print instrumentation, since reverted) and are hard-coded as the script defaults; per-kernel in/out
shapes/dtypes are documented inline in each script's docstring and asserted in `--mode correctness`.

## Kernels covered (8 kernels, 3 scripts)

| script | kernels | how it calls them |
|---|---|---|
| `bench_triton_gemm_prep.py` | `_fused_virtual_topk_ids_kernel`, `moe_align_block_size_kernel`, `count_and_sort_expert_tokens_kernel` | `_fused_virtual_topk_ids` (triton) + `moe_align_block_size` (the native `sgl_kernel::moe_align_block_size`, which launches both the align and the count_and_sort kernels) |
| `bench_kimi_gate.py` | `kimi_k2_moe_fused_gate` | `sgl_kernel.kimi_k2_moe_fused_gate` (the Kimi routing gate) |
| `bench_fp4_lora_moe_kernels.py` | `permuteKernel`, `nvfp4QuantAndPerTokenScaleKernel` (×2: gate_up + down), `activationKernel` | the standalone runner shim (see below) |

The "triton-gemm prep" three kernels are benched as **one combined pipeline** (one script, one
number), not individually — they run back-to-back as the LoRA virtual-experts routing prep, so a
future fused replacement can be compared 1:1.

## The standalone-runner shim (for the fp4-LoRA compute kernels)

`permuteKernel` / `nvfp4QuantAndPerTokenScaleKernel` / `activationKernel` have **no standalone
Python binding** — they only ever run inside `FP4BlockScaleLoraLauncher::run`. To bench them in
isolation, three thin C++ runners are exported from the flashinfer-trtllm-moe overlay module
(`python/sglang/jit_kernel/flashinfer_trtllm_moe/data/csrc/trtllm_fused_moe_kernel_launcher.cu`):

```
bench_permute(hidden_in, idx_map, total_pad, permuted_out, num_tokens, top_k, hidden_size)
bench_nvfp4_quant(in_bf16, idx_map?, out_fp4, out_sf, out_ptsf, m, n, tile)
bench_activation(gate_up, lora_delta, idx_map, total_pad, activated_out, lora_input_out, inner_dim, num_tokens, top_k)
```

Each takes **pre-allocated** in/out tensors and only builds the kernel `Data` struct + launches the
kernel (no device allocation → safe to capture in a CUDA graph for timing). The `Data` setup mirrors
`FP4BlockScaleLoraLauncher::run` exactly. They are accessed from Python via
`get_sgl_trtllm_moe_sm100_raw_module()` (see `bench_fp4_lora_moe_kernels.py`).

## Timing methodology (cold-L2, in `common_bench.py`)

`bench_kernel()` captures one sweep over **N rotation buffer sets** into a CUDA graph via
`triton.testing.do_bench_cudagraph` and divides by N:

- **Amortize launch overhead:** the graph replays all N calls back-to-back, so the per-replay
  launch/dispatch overhead (which alone floors a single tiny op at ~8-10 µs) is divided away —
  exposing true steady-state device time. Matches graph-on e2e; do **not** use per-iter
  `cudaSynchronize` CPU timing (it inflates ~µs kernels and dilutes speedups).
- **Cold L2 (matters for the memory-bound kernels):** each call reads a *different* buffer set.
  These kernels are memory-bound, so if you reuse one buffer the data stays resident in L2 and you
  measure an unrealistically fast warm-L2 number (the fp4 kernels read ~15-20% faster warm). The e2e
  sees cold/HBM (each kernel reads freshly-written HBM). `pick_n_sets` auto-sizes N to FILL a memory
  budget (default 16 GB) so the footprint vastly exceeds the GB200 L2 (135 MB) — and it **auto-grows
  if a future optimization shrinks the per-call working set**, so you never hand-tune it up. Each run
  prints `n_sets`, footprint, and an `L2-COLD` / `WARN: footprint<L2` flag. (For the sub-L2,
  latency-bound prep/gate kernels the footprint can't exceed L2, but their time is launch/compute
  bound and L2-state-independent anyway — the WARN says so.)

## How to run (single GPU)

The scripts import `sglang` / `sgl_kernel` / the flashinfer-trtllm-moe JIT module, so run them on a
machine with the same build (e.g. one GPU of the serving pod, with no server occupying it):

```bash
cd /path/to/sglang
CUDA_VISIBLE_DEVICES=0 python3 benchmark/kernels/lora_moe_triton_prep/bench_triton_gemm_prep.py   --mode correctness
CUDA_VISIBLE_DEVICES=0 python3 benchmark/kernels/lora_moe_triton_prep/bench_triton_gemm_prep.py   --mode bench
CUDA_VISIBLE_DEVICES=0 python3 benchmark/kernels/lora_moe_triton_prep/bench_kimi_gate.py           --mode correctness
CUDA_VISIBLE_DEVICES=0 python3 benchmark/kernels/lora_moe_triton_prep/bench_kimi_gate.py           --mode bench
CUDA_VISIBLE_DEVICES=0 python3 benchmark/kernels/lora_moe_triton_prep/bench_fp4_lora_moe_kernels.py --mode correctness
CUDA_VISIBLE_DEVICES=0 python3 benchmark/kernels/lora_moe_triton_prep/bench_fp4_lora_moe_kernels.py --mode bench
```

Each script defaults to the decode-bs64 production shapes; shape knobs are CLI flags
(`--bs/--num-tokens/--hidden/...`). `bench_fp4_lora_moe_kernels.py` triggers a one-time JIT rebuild
of the overlay module on first import (the shim lives there).

## Measured results (GB200, decode bs64)

`--mode bench` device time (cold-L2) vs the e2e profile (per-kernel times observed in production
traces):

| kernel | testbed (cold-L2) | e2e profile | correctness (ref-based) |
|---|---|---|---|
| triton-gemm prep (combined) | 5.98 µs | 1.5 + 2.7 + 4.7 | see below |
| └ `_fused_virtual_topk_ids` | (in combined) | 1.5 | **bitwise** vs torch ref |
| └ `moe_align` + `count_and_sort` | (in combined) | 2.7 + 4.7 | vs torch ref: post_pad exact + expert_ids multiset |
| `kimi_k2_moe_fused_gate` | 5.87 µs | 5 | expert-ids **bitwise**; weights vs torch ref (err 3e-8) |
| `permuteKernel` | 4.21 µs | 7 | **bitwise** gather vs torch ref |
| `nvfp4 quant #1` (gate_up, m=3200) | 13.40 µs | 14 | dequant vs input, rel 9.5e-2 (e2m1+8x4-sf decoded) |
| `activationKernel` | 13.89 µs | 14-16 | vs SwiGLU+lora torch ref, rel 2.5e-3 |
| `nvfp4 quant #2` (down, m=512) | 3.68 µs | (part of the 14) | dequant (same as #1) |

- **Correctness** is ref-based for every kernel: a torch reference is computed and compared. Integer /
  copy kernels (`_fused_virtual_topk_ids`, `permute`, gate expert-ids) assert **bitwise** equality;
  `moe_align` asserts post_pad + the per-expert block multiset; arithmetic kernels assert a small
  numerical error (gate weights 3e-8; activation SwiGLU+lora rel 2.5e-3; nvfp4 quant round-trips the
  output back through a torch dequant — e2m1 codes × swizzled-8x4 e4m3 block scale × per-token scale —
  and asserts rel error within fp4 precision, 9.5e-2). All shapes match the captured e2e shapes.
- **Speed vs profile:** the fp4 quant/activation match the profile closely now that L2 is cold
  (warm-L2 measured ~15-20% faster). `permute` is still below the profile (its e2e cost includes the
  scatter/index work). The prep + gate are below/near the profile and are latency-bound (sub-L2).

## Key finding (optimization lead)

The nvfp4 quant is invoked twice. **quant #1** (gate_up input) processes
`max_num_padded_tokens = 3200` rows → **13.40 µs**, but only `num_tokens*top_k = 512` of those are
real tokens (the rest are padding). **quant #2** (down input) uses the
`expanded_idx_to_permuted_idx` map to process only the **512** real rows → **3.68 µs**.

→ If quant #1 also used the index map to skip padding rows, it would scale toward quant #2 / the
baseline cutlass NVFP4 quantize (`<3 µs`). The ~6.25× padding amplification (3200 vs 512) is the
root cause of the slow quant, and the same amplification applies to `permute` and `activation`
(all three run on the 3200-row padded buffer).

## References

- Timing template: `benchmark/kernels/lora_moe_expand/bench_expand_add_down.py`.
- Kernel sources: `sgl_kernel` (`kimi_k2_moe_fused_gate`, `moe_align_block_size`),
  `python/sglang/srt/lora/triton_ops/virtual_experts.py` (`_fused_virtual_topk_ids`), and
  `python/sglang/jit_kernel/flashinfer_trtllm_moe/data/csrc/` (permute / nvfp4-quant / activation,
  exposed via the `bench_*` runners in `trtllm_fused_moe_kernel_launcher.cu`).
</content>
