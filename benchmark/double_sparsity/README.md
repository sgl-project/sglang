# Double Sparsity benchmark

Three-way comparison driver for the DS ship-gate. Measures decode
tok/s, TTFT, TBT (p50/p95), and a NIAH-style retrieval probe across:

1. **`main_dense`** — clean `origin/main` checkout (regression baseline).
2. **`branch_ds_off`** — DS branch with DS disabled (proves the dense
   path is unchanged on the branch).
3. **`branch_ds_on`** — DS enabled (the speedup measurement).

The `(1)↔(2)` comparison verifies "DS off → byte-for-byte unchanged"; the
`(2)↔(3)` comparison is the actual sparse vs dense win.

## Hardware

- **8B baseline:** 1× H200, single-node TP=1. Useful for correctness
  and capacity-guard measurements; not the architectural target for
  perf wins.
- **70B ship-gate:** 8× H200, TP=8 (or 4× H200 TP=4 with reduced KV).
  This is the DS architecture's intended winning configuration —
  H_kv_local=1 saturates the two-stage `(bs, 1, num_blocks)` grid and
  avoids the union-pass capacity limit that gates 8B (see "v1.1
  results").

## Prerequisites

A working SGLang install with the matching `sglang-kernel` version,
FA3 backend, CUDA. Generate a calibration first:

```bash
python scripts/double_sparsity/calibrate.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output calib_8b.json \
    --heavy-channels 32 \
    --n-samples 64 --seq-len 4096
```

For 70B, do the same with `--model meta-llama/Meta-Llama-3.1-70B-Instruct`.

## Running

### Long-context concurrency sweep (post-v2-native pivot — RECOMMENDED)

For the 70B/TP=8/128K target, the native-decode path's win lives at
high concurrency where dense decode is KV-bandwidth bound. The
`run_70b_sweep.sh` driver launches one DS-off and one DS-on server
(not 10 separate launches) and sweeps concurrency in `{1,4,8,16}` per
leg:

```bash
CTX=131072 N_REQUESTS=8 OUTPUT_LEN=512 CONCURRENCIES=1,4,8,16 \
  bash benchmark/double_sparsity/run_70b_sweep.sh \
    /workspace/calib_llama_3_1_70b_wikitext_s32.json
```

The driver writes `bench_70b_sweep_131072/branch_ds_{off,on}.json` and
runs `compare.py` to render the per-concurrency table + visible-win
gate at the best speedup point. See `HANDOFF_NATIVE.md` for the
results landed this session.

### Single-point comparison (legacy harness)

Each config is a separate process. Easiest is a small driver shell
script that sets up the right git checkout for `main_dense`:

```bash
# 1. Clean main baseline (in a separate worktree)
git worktree add /tmp/sglang-main origin/main
cd /tmp/sglang-main
python benchmark/double_sparsity/bench_decode.py \
    --config main_dense \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --context-len 65536 --output-len 1024 \
    --output-json /tmp/results_main.json
cd -

# 2. Branch DS off
python benchmark/double_sparsity/bench_decode.py \
    --config branch_ds_off \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --context-len 65536 --output-len 1024 \
    --output-json /tmp/results_off.json

# 3. Branch DS on
python benchmark/double_sparsity/bench_decode.py \
    --config branch_ds_on \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --calibration calib_8b.json \
    --context-len 65536 --output-len 1024 \
    --niah --niah-context-tokens 32768 \
    --output-json /tmp/results_on.json

# 4. Compare
python benchmark/double_sparsity/compare.py \
    --main /tmp/results_main.json \
    --branch-off /tmp/results_off.json \
    --branch-on /tmp/results_on.json
```

## Two-tier gate (v1.2 → 70B TP=8 work)

- **VISIBLE_WIN** (the current goal): `decode_tok_per_s(DS_on) >= 1.10 ×
  decode_tok_per_s(DS_off)` OR `tbt_ms_p50(DS_on) <= 0.90 × tbt_ms_p50
  (DS_off)`, on Llama-3.1-70B-Instruct at 128K context, output_len=1024,
  batch=1, H200 TP=8, across 3 repeated runs.
- **STRETCH_1_30X** (original v1 ship-gate, still tracked): `decode_tok
  _per_s(DS_on) >= 1.30 × decode_tok_per_s(DS_off)`. Separate badge in
  `compare.py` output. A PR-quality ship-claim requires `STRETCH_1_30X=YES`.
- **Quality guard**: `niah_accuracy(DS_on) >= niah_accuracy(DS_off) -
  0.02`. **Both DS-off and DS-on legs must pass `--niah`** — otherwise
  the guard cannot be computed and `compare.py` reports
  `quality_guard: UNKNOWN`.
- **Dense-path regression vs main** (structural sanity, optional):
  `|delta| <= 2%` decode tok/s between branch DS-off and a clean
  `origin/main` worktree. Used in the 3-way harness; not required for
  the discovery run.
- **Calibration cite**: README-cited result JSONs must have
  `extra.calibration_mode != "synthetic"` (real wikitext or real prompts
  file). Synthetic calibration is plumbing-only.

If `VISIBLE_WIN=PASS` and `quality_guard=PASS`: ship the result. If only
`STRETCH_1_30X=YES`: cite the stronger ship gate. If neither passes:
ship off-by-default and run the pivot path (profile → Triton union →
custom sparse-decode kernel; FlashInfer sparse last).

## 70B TP=8 H200 invocation (the primary visible-win workload)

Generate calibration on the H200 box first (real text required for the
headline claim; `--device-map auto` is mandatory for 70B in bf16
because the weights don't fit on a single H200 once forward activations
land on top — `calibrate.py --device-map auto` triggers Accelerate
multi-GPU loading with `use_cache=False` to skip HF's KV cache):

```bash
python scripts/double_sparsity/calibrate.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --output /workspace/calib_llama_3_1_70b_wikitext_s32.json \
    --heavy-channels 32 \
    --n-samples 64 --seq-len 4096 \
    --dataset wikitext --dataset-subset wikitext-2-raw-v1 \
    --device-map auto
```

DS-off baseline (3 repeats, NIAH on both legs):

```bash
for i in 1 2 3; do
  python benchmark/double_sparsity/bench_decode.py \
      --config branch_ds_off --tp-size 8 \
      --model meta-llama/Meta-Llama-3.1-70B-Instruct \
      --context-len 131072 --output-len 1024 \
      --n-requests 4 --concurrency 1 \
      --mem-fraction-static 0.85 --max-running-requests 4 \
      --niah --niah-context-tokens 131072 --niah-n-samples 5 \
      --output-json /workspace/70b_128k_off_${i}.json
done
```

DS-on (real calibration; `--block-t 2048 --k-block 64` is the only
size-1 knob slot that satisfies the merge capacity guard at 128K —
`num_blocks * k_block = 64 * 64 = 4096`, exactly at the threshold):

```bash
for i in 1 2 3; do
  python benchmark/double_sparsity/bench_decode.py \
      --config branch_ds_on --tp-size 8 \
      --model meta-llama/Meta-Llama-3.1-70B-Instruct \
      --calibration /workspace/calib_llama_3_1_70b_wikitext_s32.json \
      --context-len 131072 --output-len 1024 \
      --n-requests 4 --concurrency 1 \
      --block-t 2048 --k-block 64 --token-budget 1024 \
      --mem-fraction-static 0.85 --max-running-requests 4 \
      --niah --niah-context-tokens 131072 --niah-n-samples 5 \
      --output-json /workspace/70b_128k_on_${i}.json
done
```

Compare (use median run of each by inspecting `decode_tok_per_s` in
each JSON, then pass the medians):

```bash
python benchmark/double_sparsity/compare.py \
    --main /workspace/70b_128k_off_2.json \
    --branch-off /workspace/70b_128k_off_2.json \
    --branch-on /workspace/70b_128k_on_2.json
```

(`--main` and `--branch-off` are the same file in the discovery run;
the dense-path regression check is the structural sanity at the end
of `compare.py` output.)

64K corroboration: same as above with `--context-len 65536` and
`--niah-context-tokens 65536`.

## v1.2 follow-ups (deferred from v1.1)

- **Chunked-union path** to lift the `union_safe_threshold=4096` limit
  (see "v1.1 results" below — this is the gating issue for 8B-class GQA
  models at long context).
- Wrap `ds_select_tokens_triton` as a `register_custom_op +
  register_split_op` so torch.compile / piecewise CUDA graph can run
  with DS enabled. Currently auto-disabled.
- Per-Q-head + per-batch union scoring (v2 path from the plan), and
  K_label fp8 (e4m3 with per-channel scale) — both deferred to v1.2
  pending v1.1 measurements.
- Custom sparse-decode Triton kernel (drop FA3 on the sparse path).
- MLA support (DeepSeek-V2/V3, Llama-4) and SWA composition.

## v1.1 results — honest record

**Llama-3.1-8B-Instruct, single H200, bfloat16, page_size=1, FA3.**
Synthetic calibration (8 samples, 2K seq_len, S=32). Streaming
completions, batch=1, output_len=256, n_requests=4, concurrency=1,
token_budget=1024, recent=64, sink=4. v1.1 selection: two-stage
block-topk Triton (stage-1 inline Q_label gather, stage-2 merge) +
score-aware torch-on-CUDA union (capture-safe).

### Dense baselines

| context | decode tok/s | aggregate tok/s | TBT p50 (ms) | TBT p95 (ms) |
|---|---|---|---|---|
| 8K  | 197.29 | 169.02 | 5.07 | 5.15 |
| 32K | 171.70 |  93.75 | 5.83 | 5.88 |
| 64K | 147.58 |  49.46 | 6.78 | 6.86 |

### DS-on at 8K — `BLOCK_T × K_block` sweep

8 of 8 configs attempted; **6 rejected by `union_safe_threshold=4096`
during CUDA-graph capture** (server exit -9). With Llama-3.1-8B's
GQA H_kv_local=8, the union pass runs `H_kv_local × effective_budget +
recent + sink` candidates per row, which exceeds the 4096 threshold
for any `effective_budget > 503`. Only configs that cap below that
ran successfully:

| BLOCK_T | K_block | eff_budget | decode tok/s | TBT p50 (ms) | DS / dense |
|---|---|---|---|---|---|
| 512  | 32  | 544  | FAILED — union threshold | — | — |
| 512  | 64  | 1024 | FAILED — union threshold | — | — |
| 1024 | 32  | 288  | **50.37** | **19.84** | **0.26×** |
| 1024 | 64  | 576  | FAILED — union threshold | — | — |
| 1024 | 128 | 1024 | FAILED — union threshold | — | — |
| 2048 | 64  | 320  | 40.33 | 24.78 | 0.27× |
| 2048 | 128 | 640  | FAILED — union threshold | — | — |
| 2048 | 256 | 1024 | FAILED — union threshold | — | — |

**Best 8K config: BLOCK_T=1024, K_block=32 → eff_budget=288.**

### DS-on at 32K and 64K with the best 8K config — both fail

At 32K and 64K with `BLOCK_T=1024, K_block=32`, `effective_budget`
saturates back to `token_budget=1024` (since `num_blocks × K_block`
goes well above 1024), re-triggering the union threshold. The kernel
**hangs silently in CUDA-graph capture** (no traceback in server log)
instead of raising — a separate bug worth filing against the chunked-
union path. Both 32K and 64K runs killed by 480s timeout.

### Status: v1.1 ship-gate not met on 8B/H200

The v1.1 selection path is correct (109 unit tests green; first time
the bench is measuring DS-the-architecture, not the v1 placeholder
over `req_to_token.shape[1]`). What v1.1 found:

1. **Even at the smallest viable `effective_budget=288`, DS adds ~15 ms
   to TBT vs dense at 8K** (19.84 ms DS vs 5.07 ms dense). The two-
   stage Triton kernel cuts the v1 placeholder's ~13 ms overhead, but
   not enough to win on 8B/H200 at 8K. This was expected: dense FA3
   TBT on 8B/H200 is bandwidth-cheap (~1.7 ms KV-bandwidth ceiling at
   64K, see math below), so DS's bandwidth savings can't recover the
   selection overhead at this scale.

2. **The union-pass capacity limit blocks DS from running at the
   configurations where it would win.** With H_kv_local=8 (8B model
   without TP) and the v1.1 single-program union pass, `effective_
   budget × H_kv_local + recent + sink ≤ 4096` forces eff_budget ≤
   503 — well below `token_budget=1024` and far below what's needed
   to reach long-context wins. The v1.1.x chunked-union path (already
   referenced in the runtime warning) is the real unblocker for 8B.

3. **70B/TP=8 is the architecturally correct target.** With
   H_kv_local=1 (8 KV heads ÷ TP=8), the union threshold is not the
   bottleneck — `1 × 1024 + 68 = 1092 ≪ 4096`. The two-stage
   `(bs, 1, num_blocks)` grid was designed for exactly this case.
   No 70B run is included in v1.1's results (no TP=8 access during
   the v1.1 cycle); v1.2 will run it.

**Where DS bandwidth savings can structurally show up on this
hardware/model.** Back-of-envelope; counts both K and V (FA3 reads
both during decode attention):

```
8B at 64K, bf16:
  KV cache size = 32 layers × 8 KV heads × 128 dim × 2 (K+V) × 2 B × 64K tok
                = 8 GB
  H200 HBM      = 4.8 TB/s
  KV read floor ≈ 8 GB / 4.8 TB/s ≈ 1.7 ms

  Dense TBT measured = 6.7 ms.
  → KV bandwidth is ~25% of dense TBT at this scale; the rest
    is kernel launch + Q·K compute + softmax + V GEMM. DS targets
    the bandwidth slice; the compute slice it can't help with.

  DS upper-bound saving ≈ 1.7 ms × (1 - budget/seq_len).
  At budget=1024, seq_len=64K: ≈ 1.7 × (1 - 1024/65536) ≈ 1.67 ms.
  That's the perf-path budget DS has to fit under to start winning.
```

DS-the-architecture is built to win where dense attention is more
strongly bandwidth-bound. That's longer context, larger model, or
both:

```
70B at 64K (per H200, TP=8), bf16:
  KV cache size per rank = 80 × 1 × 128 × 2 (K+V) × 2 B × 64K
                         = 2.5 GB per rank
  KV read floor          ≈ 2.5 GB / 4.8 TB/s ≈ 0.5 ms per rank
  But 80 layers compose serially; typical total dense TBT ~15–25 ms,
  with KV bandwidth a much larger share of that than at 8B/64K.
  → meaningful headroom for DS to win once selection overhead drops
    below the bandwidth slice.

70B at 128K, TP=8: KV cache per rank = 5 GB → KV-bound regime is
clearly entered. This is the real ship-gate workload.
```

**Caveat on these numbers.** The above is a rough KV-bandwidth ceiling,
not a prediction. Actual dense TBT depends on FA3's specific access
pattern (K then V), kernel-launch overhead per layer, softmax, and
non-attention work in the layer. The DS selection adds its own
bandwidth cost too — reading K_label per step. The right way to
settle this is measurement on a workload where DS-the-architecture is
not capacity-blocked from running, which on Llama-3.1-8B requires the
v1.1.x chunked-union path.

**v1.2 priorities to actually ship the speedup.**
1. **Chunked-union path.** Lift the union-pass capacity limit so
   `effective_budget = token_budget` is reachable on 8B/H_kv=8.
   Without this, every long-context win the architecture predicts
   for 8B is unmeasurable.
2. **70B/H200 TP=8 bench.** With H_kv_local=1 the union threshold is
   not the bottleneck — this is where the two-stage `(bs, 1,
   num_blocks)` grid was designed to win. v1.1's lack of TP=8 access
   is a missing measurement, not a known loss.
3. **Real calibration** on a held-out long-context retrieval set for
   the accuracy probe (NIAH).
4. **Capture-time deadlock** when union threshold is exceeded: the
   guard fires cleanly outside CUDA-graph capture but hangs silently
   inside it. File a kernel bug; the chunked-union path obviates the
   need but the underlying behavior is still wrong.

**v1.1 ship status.** Architecture correct and measured for the first
time. 109 unit tests green including stage-1/stage-2/union parity,
score-aware drops, and CUDA-graph zero-allocation replay. v1.1
selection adds ~15 ms TBT at the working configurations on 8B/H200/8K
(vs the v1 placeholder's ~13 ms — comparable, but now we know which
of those two numbers represents DS-the-architecture). **Ships off-by-
default** pending v1.2's chunked union and 70B run.

## v1.1 limitations (auto-set when --enable-double-sparsity)

- `disable_piecewise_cuda_graph = True`: the v1.1 selection runs Triton
  kernels (opaque to compile already) plus a torch-on-CUDA union pass,
  but is not yet wrapped as a registered split op
  (`register_custom_op + register_split_op`). v1.2 wraps it.
- `disable_cuda_graph = False`: full CUDA graphs work for configs that
  satisfy `H_kv_local × effective_budget + recent + sink ≤
  union_safe_threshold (4096)`. Configs that violate this hang in
  capture rather than raising — track the chunked-union path as the
  fix.
- The FA3 metadata adaptor is capture-safe, pinned by
  `test_double_sparsity_adaptor.py::TestCudaGraphCaptureReplay`. The
  union pass is also capture-safe (verified by
  `test_double_sparsity_union.py::TestUnionCudaGraphCaptureReplay`).

## Memory budget reference

For Llama-3.1-70B at S=32, num_kv_heads=8 (TP=8 → 1 KV head per rank),
page_size=1, bf16 K_label:

```
K_label per token per rank = 80 layers × 1 × 32 × 2B = 5 KB
KV per token per rank      = 80 layers × 1 × 128 × 2 × 2B = 40 KB
                                               ↑ K + V
→ K_label is 12.5% overhead per rank.
For 200K tokens: K_label = 1 GB; KV = 8 GB.
```
