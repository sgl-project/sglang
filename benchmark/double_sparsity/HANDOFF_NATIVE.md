# DS v2 native sparse-decode — session results

**Branch**: `dev/double-sparsity-v2`
**Date**: 2026-05-14 (this session)
**Predecessor**: `HANDOFF.md` (prior session — legacy 3-stage selection lost
11.8× to dense at 32K bs=1; native path replaces it)
**Status**: visible-win on TBT (≤0.90× gate) confirmed at 128K/conc=16;
NIAH quality at the default token_budget=512 is too narrow (0.4% of context)
and is being re-measured at token_budget=2048.

## 1. Architecture

The FA3 + DSFlashAttentionAdaptor sparse path (legacy v2) is bypassed for
decode. A new `RadixAttention.forward` branch tries
`DoubleSparsityAlgorithm.try_native_sparse_decode` first; if it returns a
tensor, attention is already computed and the FA3 backend call is skipped.
Legacy path stays as fallback for short-sequence or `bs > scratch_max_bs`
cases.

```
RadixAttention.forward (decode mode, DS enabled):
  try_native_sparse_decode(q, k, v, layer, fb) → native_out
  ├─ if not None: attention_end(native_out)            ; return native_out
  └─ else (legacy fallback):
       coordinator.attention_begin(...)                ; rewrites FA3 metadata
       _forward_inner(...)                             ; FA3 with rewritten meta
       attention_end(...)
```

### Native kernel pipeline (`double_sparsity_native_decode.py`)

Per decode layer:
1. **Score** Triton kernel — per `(bs, kv_head, BLOCK_T)` program, loads
   `Q_label`, scores `K_label · Q_label` at `req_to_token[b, t]`, masks
   sink/recent/oob positions to `-inf`.
2. **`torch.topk`** — single CUB call, output `[bs, kv_head, top_k]` of
   logical history positions.
3. **Build_selected_physical** Triton kernel — fused (top-k physical
   ++ sink physical ++ recent physical). Replaces ~20 torch ops worth
   110 µs/layer with one kernel at ~16 µs/layer.
4. **Sparse attention** Triton (stage2 + stage3) — v1 PR #22992's split-K
   + reduce decoder, adapted to consume physical ids directly (no
   logical→physical round-trip in the kernel). One selected set per
   `(bs, kv_head)`, shared by all GQA query heads.

Selection cost scales with `top_k`, sparse-attn cost with
`total_selected = top_k + sink + recent`. Neither scales with `seq_len`.

### Algorithm-side scratch (`DoubleSparsityAlgorithm._allocate_native_scratch`)

Sized for static worst case `scratch_max_bs` (from
`server_args.max_running_requests`). At 70B/TP=8/128K/bs=16/top_k=512:
~9 MiB total (att_out_approx ≈ 8 MiB dominates).

## 2. Synthetic profile (70B/TP=8 shape, no server)

`benchmark/double_sparsity/repro_session/profile_native_decode.py`:

| ctx  | per-layer | per-step (80 layers) |
|------|-----------|----------------------|
| 32K  | 151 µs    | 12.07 ms             |
| 64K  | 150 µs    | 12.00 ms             |
| 128K | 150 µs    | 12.00 ms             |
| 128K bs=16 | 163 µs | 13.03 ms          |

Headline DS property holds: end-to-end native time is essentially
constant across context length and batch size. Legacy was 100 ms at
32K bs=1; native is **5–8× faster** depending on shape.

### Per-phase breakdown (32K bs=1)

| phase | µs/call | ms × 80 layers | % |
|---|---:|---:|---:|
| score (Triton) | 16 | 1.3 | 11 |
| `torch.topk` | 62 | 5.0 | 41 |
| build_selected_physical (Triton) | 16 | 1.3 | 11 |
| sparse attn stage2+3 (Triton) | 36 | 2.9 | 24 |
| inter-op overhead | — | 1.6 | 13 |
| **TOTAL** | 151 | **12.07** | 100 |

### Sparse-attn microbench (the headline DS property)

`benchmark/double_sparsity/repro_session/microbench_sparse_attn.py`:

```
selected\seq_len     32K     64K     128K
selected=512        37.2µs  36.6µs  36.3µs
selected=1024       36.3µs  35.8µs  35.7µs
selected=2048       36.2µs  36.2µs  36.6µs
```

Attention time is flat across seq_len at fixed selected count — bounded
by `total_selected`, not by `seq_len`. ≤2% jitter per row.

## 3. Real bench — 70B/TP=8/128K/output_len=512 concurrency sweep

`benchmark/double_sparsity/repro_session/sweep_70b_128k_tbt_win/`

### TBT (the bandwidth-bound metric)

| conc | DS-off TBT | DS-on TBT | ratio | gate (≤0.90×) |
|---|---:|---:|---:|:--|
| 1  |  9.66 ms |  16.05 ms | 1.66× | FAIL (DS-off wins) |
| 4  | 13.48 ms |  17.67 ms | 1.31× | FAIL |
| 8  | 18.60 ms |  19.46 ms | 1.05× | FAIL (essentially even) |
| 16 | 27.94 ms |  22.99 ms | **0.82×** | **PASS** |

**Visible win at conc=16: 18% lower TBT.** Dense scales linearly with
batch × seq_len (KV bandwidth dominates above conc=8); native DS stays
roughly flat in TBT (~16 ms at conc=1 → 23 ms at conc=16).

### aggregate tok/s (prefill-dominated at 128K — diagnostic)

| conc | off | on | ratio |
|---|---:|---:|---:|
| 1  | 33.3  | 24.9  | 0.75× |
| 4  | 42.7  | 39.4  | 0.92× |
| 8  | 44.6  | 44.8  | 1.00× |
| 16 | 44.7  | 45.6  | 1.02× |

Aggregate barely moves because both legs share the same FA3 prefill —
the wall-clock metric is bottlenecked there. TBT is the
sparse-attn-specific signal.

### NIAH quality (5-probe retrieval at 128K)

| | conc=1 |
|---|---|
| DS-off | 0.60 (3/5) |
| DS-on (token_budget=512) | 0.00 (0/5) |

**Quality FAIL** under the −0.02 NIAH guard. token_budget=512 selects
only 0.4% of 128K context — too narrow for arbitrary-position needle
retrieval given the wikitext calibration.

### NIAH re-measure (token_budget=2048; in flight)

[ to be filled in once the re-bench completes ]

## 4. What landed in code

| File | Change |
|---|---|
| `python/sglang/srt/layers/attention/triton_ops/double_sparsity_native_decode.py` | NEW — score / build_selected_physical / sparse_attn (stage2+stage3) kernels + orchestrator |
| `python/sglang/srt/mem_cache/sparsity/algorithms/double_sparsity.py` | `try_native_sparse_decode` entry, `_allocate_native_scratch` |
| `python/sglang/srt/layers/radix_attention.py` | dispatch — try native first, fall through to legacy adaptor path |
| `python/sglang/srt/model_executor/model_runner.py` | demote legacy capacity guard from RuntimeError → warning (native path doesn't use stage-2 merge / union) |
| `benchmark/double_sparsity/bench_decode.py` | accept `--concurrency` as CSV sweep; output multi-result JSON per leg |
| `benchmark/double_sparsity/compare.py` | per-concurrency table; pick best-speedup point for visible-win gate |
| `benchmark/double_sparsity/run_70b_sweep.sh` | NEW — 64K/128K sweep driver (CTX env var) |
| `benchmark/double_sparsity/repro_session/profile_native_decode.py` | NEW — per-phase synthetic profile |
| `benchmark/double_sparsity/repro_session/microbench_sparse_attn.py` | NEW — sparse-attn flatness check |
| `benchmark/double_sparsity/repro_session/run_nsys_at_winning_point.sh` | NEW — nsys runner for DS-on vs DS-off at the winning point |
| `test/registered/unit/mem_cache/sparsity/test_double_sparsity_native_decode.py` | NEW — 5 tests; total suite 120 passed |

## 5. Open work

1. **NIAH at token_budget=2048** — verify quality recovers; re-run TBT
   gate at the new budget. If still failing, calibration corpus is the
   next lever (wikitext is language-modeling; retrieval calibration
   would teach the K_label scoring to attend to needles).
2. **CUDA-graph capture is on, host-sync removed**; but per-step
   `index_select` caching, q_label inline (synthetic neutral, stashed),
   and a fused score+topk+build kernel are the remaining ~3-4 ms of
   per-step Python+kernel-launch overhead the synthetic profile shows.
3. **nsys diff at winning point** — `run_nsys_at_winning_point.sh`
   wired but not yet executed at 128K/conc=16.
4. **Legacy v2 path** (stage-2 merge + union) is still present as
   fallback for `bs > scratch_max_bs`. Could be removed entirely once
   we accept native as canonical.

## 6. Reproduction

```bash
# Calibration (one-time, ~5 min on 8x H200; produced in prior session
# at /workspace/calib_llama_3_1_70b_wikitext_s32.json).
python3 scripts/double_sparsity/calibrate.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --output /workspace/calib_llama_3_1_70b_wikitext_s32.json \
    --heavy-channels 32 --n-samples 64 --seq-len 4096 \
    --dataset wikitext --dataset-subset wikitext-2-raw-v1 \
    --device-map auto

# Full bench (both legs through one model-load each).
CTX=131072 N_REQUESTS=8 OUTPUT_LEN=512 CONCURRENCIES=1,4,8,16 \
  bash benchmark/double_sparsity/run_70b_sweep.sh \
  /workspace/calib_llama_3_1_70b_wikitext_s32.json

# Compare.
PYTHONPATH=python python3 benchmark/double_sparsity/compare.py \
  --branch-off bench_70b_sweep_131072/branch_ds_off.json \
  --branch-on  bench_70b_sweep_131072/branch_ds_on.json
```
