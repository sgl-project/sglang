# Double Sparsity v2 Session Handoff

**Branch**: `dev/double-sparsity-v2`
**Date**: 2026-05-14
**Status**: Visible-win at 70B/TP=8/128K not achieved; root cause is now understood empirically.

This document captures the full state for a fresh session to pick up. It is descriptive,
not prescriptive — the next session decides whether to keep, refactor, or restart.

---

## 1. Goal that was being pursued

Produce one reproducible "visible win" of Double Sparsity ON vs OFF on this branch:

- Model: `meta-llama/Llama-3.1-70B-Instruct`
- Hardware: 8× H200, TP=8
- Backend: FA3, page_size=1, bf16 KV
- Primary bench: 128K context, output_len=1024, **concurrency=1**, n_requests=4
- Secondary bench: 64K with the same setup
- Visible-win threshold: `decode_tok/s(on) / decode_tok/s(off) ≥ 1.10` **or** `tbt_p50(on) ≤ 0.90 × tbt_p50(off)`
- Quality guard: NIAH(on) − NIAH(off) ≥ −0.02

---

## 2. Repository state at handoff

### Files changed in this session (all committed and pushed)

| Area | Files | Purpose |
|---|---|---|
| DS core (Commit 0) | `python/sglang/srt/mem_cache/sparsity/algorithms/double_sparsity_config.py` | added `scratch_max_bs` field |
| | `python/sglang/srt/mem_cache/sparsity/algorithms/double_sparsity.py` | preallocate selection scratch in `initialize_representation_pool` |
| | `python/sglang/srt/mem_cache/sparsity/triton_ops/select_kernels.py` | thread scratch buffers through |
| | `test/registered/unit/mem_cache/sparsity/test_double_sparsity_preallocation.py` | NEW regression test |
| | `test/registered/unit/mem_cache/sparsity/test_double_sparsity_klabel_extend_lifecycle.py` | minor fixture fix |
| Bench harness | `benchmark/double_sparsity/bench_decode.py` | `--tp-size`, `--mem-fraction-static`, `--max-running-requests`, `--token-budget`, `--recent-tokens`, `--sink-tokens`, `--min-seq-len`, `--max-selected-per-request`, `--niah-n-samples` CLI flags; `calibration_mode` field |
| | `benchmark/double_sparsity/compare.py` | `VISIBLE_WIN` + `STRETCH_1_30X` + `quality_guard` + `calibration_mode` reporting |
| | `benchmark/double_sparsity/README.md` | 70B TP=8 invocation docs |
| | `benchmark/double_sparsity/run_70b_128k.sh` | NEW 3-rep driver |
| Calibrate | `scripts/double_sparsity/calibrate.py` | `--device-map auto`, lazy per-device accumulators, `use_cache=False` |
| Session repro | `benchmark/double_sparsity/repro_session/*` | NEW — scripts + JSON artifacts from this session |

### Unit tests after Commit 0

`pytest test/registered/unit/mem_cache/sparsity/ -q` → **115 passed, 20 warnings in 12.59s** on the H200 box.

### Calibration JSON

Real wikitext calibration generated at `/workspace/calib_llama_3_1_70b_wikitext_s32.json`:
- `dataset`: `wikitext`
- 80 layers, 8 KV heads, head_dim=128, heavy_channels=32
- Used as the DS-on calibration for every bench JSON cited below

### Environment (H200 box)

- `sglang-kernel==0.4.2.post1` from PyPI (the cu13-compatible wheel)
- torch 2.11.0+cu130, CUDA 13.2, H200 SM90
- 8× H200, ~141 GB HBM each
- `flashinfer-cubin==0.6.11.post1`

---

## 3. Benchmarks run and results

### Smoke (16K, single request, DS-on, real calibration)

`benchmark/double_sparsity/repro_session/` — equivalent to the 16K invocation in `README.md`.

| metric | value |
|---|---|
| `decode_tok_per_s` | 10.72 |
| `tbt_ms_p50` | 93.23 |
| `calibration_mode` | `wikitext` |

Confirms server boots, DS coordinator stamps every layer, capacity guard passes, end-to-end decode completes.

### 128K attempts

Three attempts to run the **primary** benchmark (CTX = 131072 → CTX = 129792 → CTX = 127744):

- **CTX=131072**: failed at server startup with
  > DS stage-2 merge candidates = num_blocks * k_block = 65 * 64 = 4160 > merge_safe_threshold=4096

  Cause: `req_to_token.shape[1]` is one larger than `server_ctx_len`, so `ceil((131072+1024+256+1)/2048) = 65`.

- **CTX=129792**: same capacity-guard failure (server_ctx still rounds up to 65 blocks). With `k_block=32` to bypass the guard, **the stage-2-merge Triton kernel compile hung past 6 minutes in LLVM `optimize_module O3`** (`triton/backends/nvidia/compiler.py:417`). Killed at ~7 min.

- **CTX=127744**: same compile-time hang in stage-2 merge LLVM optimization. Killed at ~7 min.

  py-spy confirmed all 8 TP rank scheduler processes were active in `make_llir` at line 378 → 417, not deadlocked, but the LLVM IR is too large for O3 to converge in reasonable time.

  Diagnosis (in §4 below): the stage-2-merge Triton kernel takes `num_blocks` as a **constexpr**, which causes IR unrolling to scale superlinearly with `num_blocks`. At `block_t=2048, ctx=131072` → 64 blocks the kernel is too big for LLVM O3 to compile in <10 min. The smoke kernel (`block_t=1024, ctx=16384` → 17 blocks) compiles in ~2 min.

**No 128K JSON was ever produced.**

### 32K downscale (the one config that completed end-to-end)

After three 128K compile failures, switched to CTX=30720 (32K) with `block_t=1024, k_block=64, token_budget=512` — exactly the parameter shape that worked in the 16K smoke. Single trial per leg.

JSONs at `benchmark/double_sparsity/repro_session/70b_32k_{off,on}_1.json`.

| metric | DS-off | DS-on | ratio |
|---|---:|---:|---:|
| `decode_tok_per_s` | 117.41 | 9.99 | **0.085×** |
| `tbt_ms_p50` (ms) | 8.52 | 100.14 | 11.76× slower |
| `aggregate_tok_per_s` | 99.09 | 9.84 | |
| `ttft_ms_p50` (ms) | 1620 | 1599 | |
| `niah_accuracy` | 1.00 (5/5) | 1.00 (5/5) | identical |
| `calibration_mode` | `null` | `wikitext` | |

`compare.py` output:
```
VISIBLE_WIN:           FAIL
decode_tok_s_speedup: 0.085x  threshold: >=1.10x  FAIL
p50_tbt_ratio:        11.757x  threshold: <=0.90x  FAIL
STRETCH_1_30X:         NO
quality_guard:         PASS  (niah_on - niah_off = +0.000, min -0.02)
calibration_mode:      wikitext
```

**No 64K JSON was ever produced** (Triton compile hangs at that scale too, same root cause as 128K).

---

## 4. DS-on overhead — empirical profiling

### 4.1 Per-phase synthetic profile

`benchmark/double_sparsity/repro_session/profile_ds_selection.py` calls the DS selection kernels in isolation at exactly the 32K bench shape (bs=1, H_kv=1, S=32, seq=30720, block_t=1024, k_block=64, eff_budget=512, max_sel=8192). 80 iterations simulate one decode step.

| phase | µs/call | ms × 80 layers | % of selection |
|---|---:|---:|---:|
| K_label write (Triton) | 15.4 | 1.23 | 1.5% |
| Stage-1 block-topk (Triton) | 49.8 | 3.98 | 4.7% |
| **Stage-2 merge (Triton)** | **338.2** | **27.05** | **32.2%** |
| **`ds_union_per_batch` (torch-on-CUDA)** | **646.9** | **51.75** | **61.6%** |
| **Selection total / decode step** | | **84.02 ms** | 100% |

Selection accounts for 84 / 91.6 ms of the observed DS-on overhead (= 100.14 ms TBT − 8.52 ms dense baseline) → **92% match**.

### 4.2 Full nsys profile (DS-off vs DS-on at 32K)

`benchmark/double_sparsity/repro_session/compare_nsys.py` extracts per-kernel CUDA totals from both reports.

Raw reports in `/workspace/nsys_reports/{ds_off_32k,ds_on_32k}.nsys-rep` (not committed; regenerate with `run_70b_32k_single.sh` followed by `nsys profile … ` per the script).

| | DS-off | DS-on | ratio |
|---|---:|---:|---:|
| Total GPU time over same workload | 14.23 s | 48.52 s | 3.41× |
| Unique kernels | 24 | 28 | |

**Pure DS-only kernels (don't exist in DS-off)**:

| kernel | GPU time | % of DS-on total | instances |
|---|---:|---:|---:|
| `_ds_select_stage2_merge_kernel` | 24.90 s | **51.3%** | 31,418 |
| `_ds_select_stage1_block_topk_kernel` | 2.81 s | 5.8% | 31,424 |
| Three nvjet matmul variants (DS-related projections) | 0.53 s | 1.1% | 2,560 |
| `_ds_k_label_write_kernel` | 0.05 s | 0.1% | 33,976 |
| **DS-only total** | **28.28 s** | **58.3%** | |

**Truly identical between modes (model kernels)** — within 2-8% wall-clock:

| kernel | DS-off | DS-on | ratio |
|---|---:|---:|---:|
| NCCL AllReduce | 3.21 s | 3.16 s | 0.98× |
| nvjet matmul 320x128 (FFN) | 2.02 s | 2.02 s | 1.00× |
| nvjet matmul 256x128 | 2.52 s | 2.32 s | 0.92× |
| FA3 FwdSm90 (2 variants) | 2.33 s | 2.18 s | 0.93× |
| nvjet 64x8 splitK | 0.99 s | 0.99 s | 1.00× |
| nvjet 64x8 | 0.73 s | 0.73 s | 1.00× |
| FA3 combine | 0.31 s | 0.28 s | 0.91× |

The Q/K/V projections, FFN matmuls, FA3 attention, AllReduce, norms, rotary are essentially unchanged. The 3.41× total GPU-time blow-up is entirely **additional DS kernels stacked on top of an unchanged forward pass**.

**Union-pass torch ops** are split across many generic-named PyTorch kernels (radixSort, scatter, gather, etc.) and show up as outsized ratios on "shared" kernels:

| pattern | DS-off | DS-on | ratio | meaning |
|---|---:|---:|---:|---|
| `void at::native::...` (radixSort, etc.) | 0.07 s | 6.35 s | 92.7× | union argsort/topk/gather |
| flashinfer fused RMSNorm (extra calls) | 0.35 s | 1.02 s | 2.93× | union normalisation |
| `void flash::prepare_varlen_num_blocks` | 0.01 s | 0.11 s | 10.3× | FA3 metadata rewrite per layer |
| `void at_cuda_detail::index_put` | 0.00 s | 0.07 s | 356× | union scatter |

Adding the ~7 s of "shared-named but DS-driven" overhead to the 28.28 s of DS-only kernels gives **~35 s of DS-related GPU time = 72% of DS-on total**.

### 4.3 Bandwidth model (sanity-check, not measured)

At 70B/TP=8, bs=1, H200:

| | 32K | 128K |
|---|---:|---:|
| Per-rank weights | 17.5 GB | 17.5 GB |
| Per-rank KV (80 layers × 8 KV × 128 dim × 2 × 2 B / 8 ranks) | 1.3 GB | 5.2 GB |
| Weight read @ 3 TB/s HBM eff. | 5.8 ms | 5.8 ms |
| KV read @ 3 TB/s | 0.4 ms | 1.7 ms |
| Observed dense TBT | 8.5 ms | ~32 ms est |
| Observed DS-on TBT | 100.1 ms | not measured (compile failure) |

Dense decode at bs=1 is dominated by weight reads, not KV reads. DS only saves on the KV portion. At bs=1 the absolute KV-read time is much smaller than current DS selection overhead.

---

## 5. Architectural comparison

### 5.1 Paper reference: `andy-yang-1/DoubleSparse` (MLSys 2024)

Source: `models/triton_kernels/heavy.py`, `models/triton_kernels/sparse.py`.

Per layer, **2 Triton kernels total**:

1. `get_heavy_kernel`: grid=(B, H). Inside the kernel: `att_value = sum(Q_label · K_label)` then `argsort(att_value)` (using Triton's own argsort op) → writes sorted indices to `Heavy_List`. **No torch.topk, no Python orchestration.**
2. `fwd_sparse_kernel`: grid=(B, H). Loads K/V only at `Heavy_List` indices, computes Q·K + softmax + V in one kernel, writes output.

No FA3, no page table, no multi-stage merge, no separate adapt-metadata step.

### 5.2 v1 reference: SGLang PR 22992 (`dev/double-sparsity-reintro`, currently open)

Branch fetched into local repo as `refs/pr/22992`. Author: same as paper (Andy Yang) → original PR #1459 from Oct 2024; PR 22992 is the restore. Files:

- `python/sglang/srt/layers/attention/double_sparsity_backend.py` — Triton-only backend
- `python/sglang/srt/layers/attention/double_sparsity_fa3_backend.py` — FA3 for dense paths + Triton sparse for top-K path
- `python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py` — 1106 lines of Triton (3-stage sparse decode kernel)

Per layer in `forward_decode`, **5 ops total**:

1. `torch.gather(q, ..., sorted_channels)` → q_label
2. `_sparse_fwd_kernel_flash_decode_stage1` (Triton) — BGEMV Q_label·K_label → `[H, B, S]` score tensor
3. `torch.topk(scores, heavy_token_num, dim=-1)` — selection in one torch op
4. `_sparse_fwd_kernel_flash_decode_stage2` (Triton) — sparse attention: loads K/V at top-K indices via `req_to_token`, computes Q·K + softmax + V in tile blocks
5. `_sparse_fwd_kernel_flash_decode_stage3` (Triton) — reduces partial outputs across blocks

The FA3 backend comment explicitly states: *"Keeps the Triton sparse-decode kernel for the top-K / heavy-token path — FA3 has no fused equivalent."* The FA3 backend uses FA3 for dense extend + short-sequence fallback only; sparse decode is the same custom Triton kernel.

v1's own bench results (from PR description, H100, Llama-3.1-8B):
- Short decode (in=512, out=128): DS = 2856 tok/s vs Triton baseline = 2950 tok/s → **3% regression**
- Long decode (in=512, out=1024): DS = 4565 tok/s vs Triton baseline = 5194 tok/s → **12% regression**
- The PR text says: *"DS is expected to be more beneficial on bandwidth-constrained GPUs, very long contexts, and larger models. Performance optimization is planned for follow-up work."*

### 5.3 v2 current branch (`dev/double-sparsity-v2`)

Per layer in `attention_begin` + `attention_end`, ~35 ops total:

1. `effective_sparse_mask` (torch ops): ~5 kernels
2. `retrieve_topk` → `ds_select_tokens_triton`:
   - Stage-1 block-topk Triton: 1 kernel
   - **Stage-2 merge Triton: 1 kernel (51% of all DS-on GPU time)**
   - `ds_union_per_batch` (torch-on-CUDA): ~20 kernels (argsort×2, gather×4, topk, sort, cat, where×many, arange, copy_, sum)
3. `adapt_for_attn_metadata` (page-table rewrite torch ops): ~5 kernels
4. FA3 sparse attention via page table: 1 kernel
5. `construct_representations` + `update_representations` → K_label write Triton: 1 kernel
6. Misc bookkeeping: ~3 kernels

### 5.4 Comparison table

| dimension | paper | v1 PR 22992 | v2 (this branch) |
|---|---|---|---|
| Selection kernels / layer | 1 Triton (fused score + argsort) | 1 Triton score + 1 `torch.topk` | 1 Triton stage-1 + 1 Triton stage-2 merge + ~20 torch union ops |
| Attention kernel | own Triton sparse | own Triton sparse | FA3 with sparse page-table |
| Total ops / layer | 2 | 5 | ~35 |
| Page-table mgmt | none | none | per-layer rewrite |
| Stage-2 merge constexpr `num_blocks`? | n/a | n/a | yes — causes 128K compile-time hang |
| Triton compile time at 32K (stage-2 merge) | n/a | n/a | ~5 min cold |
| Triton compile time at 128K (stage-2 merge) | n/a | n/a | did not converge in 7+ min |
| Selection cost / decode step at 32K | not measured | not measured | 84 ms |

### 5.5 What v2 added that v1/paper did not

v2's multi-stage selection (block-topk → merge → union with sink/recency window) was designed to:
- Make per-layer selection capture-safe under CUDA graphs (no dynamic shapes)
- Add sink + recency window semantics on top of pure top-K
- Scale to 128K where a single `torch.topk` on `[H, B, 128K]` was assumed to be too slow
- Support score-aware deduplication across kv-heads

Empirically: `torch.topk` on `[8, 1, 128K]` is one optimized CUB call (single-digit-µs to tens-of-µs). v2's replacement runs ~20 torch ops × 80 layers = 1600 small kernels per decode step to accomplish the same thing, plus a Triton stage-2 merge that doesn't compile at the target shape.

v2 also switched the attention backend from "own Triton sparse kernel" (v1) to "FA3 with page-table sparse". FA3 does not have a fused sparse-decode that takes a top-K index list; the page-table mechanism requires the metadata adaptor to rewrite per-layer FA3 metadata each step, which is the source of `void flash::prepare_varlen_num_blocks` (10.3× ratio) and `void at_cuda_detail::index_put` (356× ratio) in the nsys diff.

---

## 6. Decision points the next session faces

These are open questions, not recommendations:

1. **Is the bs=1 single-request workload still the right bench target?** The bandwidth analysis shows dense decode at bs=1 is weight-bandwidth-bound, with KV reads at ~5% of TBT (32K) or ~5% (128K). Even with zero-cost selection, DS-on at bs=1 would save ≤1 ms / step. Larger-batch decode (bs ≥ 8) shifts the dense path toward KV-bandwidth-bound and gives DS room to win.
2. **Keep v2's FA3-page-table architecture?** It adds the per-layer page-table rewrites that account for the union-pass torch-op explosion. Reverting to v1's "own Triton sparse-decode kernel" eliminates that overhead entirely. v1's kernel already exists in PR 22992.
3. **Keep v2's multi-stage selection?** A single `torch.topk` (v1) or a single fused score+argsort Triton kernel (paper) both reduce selection from ~35 ops/layer to 1-2 ops/layer.
4. **Stage-2 merge compile-time bug**: even at 32K it takes ~5 min, and at 128K it doesn't converge. Refactoring `num_blocks` from constexpr to runtime would make it scalable. Independently of the architectural question, this blocks any 128K test on the current code path.
5. **Sink + recency window semantics**: v2 added these on top of pure top-K. They produced NIAH=1.0 at 32K (perfect retrieval), but pure top-K (v1/paper) was also retrieving correctly in the literature. The added complexity may or may not be worth the cost.
6. **Restart vs. incremental fix**: v2 has ~550 LoC of new code in the modified files. v1's full implementation is 1873 LoC across the PR's 12 files. Either can be the base going forward.

---

## 7. Artifacts to inspect

- `benchmark/double_sparsity/repro_session/70b_32k_on_1.json` — DS-on at 32K result
- `benchmark/double_sparsity/repro_session/70b_32k_off_1.json` — DS-off at 32K result
- `benchmark/double_sparsity/repro_session/profile_ds_selection.py` — synthetic per-phase profiler (run via `PYTHONPATH=python python3`)
- `benchmark/double_sparsity/repro_session/compare_nsys.py` — DS-on vs DS-off kernel diff
- `benchmark/double_sparsity/repro_session/run_70b_*_single.sh` — single-trial bench drivers at multiple context lengths
- `benchmark/double_sparsity/run_70b_128k.sh` — 3-trial driver (not used; 128K never reached this stage)
- `/workspace/calib_llama_3_1_70b_wikitext_s32.json` — real wikitext calibration JSON (not committed; regenerable via `scripts/double_sparsity/calibrate.py` with `--device-map auto`)

Fetch v1 PR 22992 locally:
```bash
git fetch https://github.com/Jiminator/sglang.git dev/double-sparsity-reintro:refs/pr/22992
git show refs/pr/22992:python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py
```

Paper repo: `https://github.com/andy-yang-1/DoubleSparse`
Paper: arXiv 2408.07092

---

## 8. Reproducing the 32K result from a cold start

```bash
# 1. Generate calibration (one-time, ~5 min on 8× H200)
python3 scripts/double_sparsity/calibrate.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --output /workspace/calib_llama_3_1_70b_wikitext_s32.json \
    --heavy-channels 32 \
    --n-samples 64 --seq-len 4096 \
    --dataset wikitext --dataset-subset wikitext-2-raw-v1 \
    --device-map auto

# 2. Run the 32K bench (DS-on first to fail fast on any kernel issue)
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
PYTHONPATH=/workspace/sglang/python \
bash benchmark/double_sparsity/repro_session/run_70b_32k_single.sh

# 3. Profile each DS phase in isolation (synthetic; ~10 s)
PYTHONPATH=/workspace/sglang/python \
python3 benchmark/double_sparsity/repro_session/profile_ds_selection.py
```

Expected: DS-off ≈ 117 tok/s, DS-on ≈ 10 tok/s, NIAH=1.0 on both, `VISIBLE_WIN: FAIL`.
