# DSV4-NPU Fused Compressor Validation

Runbook for validating `SGLANG_DSV4_NPU_FUSED_COMPRESSOR=1`
against the per-request Python fallback on Ascend NPU.

## Prerequisites

* Ascend 910 A3 with `custom_ops` wheel exposing
  `torch.ops.custom.compressor` (CANN ops-transformer
  `experimental/attention/compressor`).
* DeepSeek-V4-Flash-W8A8 weights at `$MODEL_PATH`.
* The launch script in `/home/t00937989/run_dsv4-flash*.sh` (or your own).

## Phase A — Eager-mode sanity (3 prompts)

Run twice; baseline first, fused second. Compare answers verbatim.

```bash
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR=0   # baseline
./bench_sanity.sh > /tmp/sanity_baseline.log 2>&1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR=1   # fused
./bench_sanity.sh > /tmp/sanity_fused.log 2>&1

diff -u /tmp/sanity_baseline.log /tmp/sanity_fused.log
```

Expected: identical text for the three reference prompts
(Paris capital / 56 multiplication / Janet apples — see
`bench_sanity.sh`).

## Phase B — GSM8K 5-shot 50q

Reuse the existing `benchmark/gsm8k/bench_sglang.py`. From the
sglang repo root:

```bash
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR=0
python benchmark/gsm8k/bench_sglang.py --num-questions 50 --num-shots 5 \
    > /tmp/gsm8k_baseline.log

export SGLANG_DSV4_NPU_FUSED_COMPRESSOR=1
python benchmark/gsm8k/bench_sglang.py --num-questions 50 --num-shots 5 \
    > /tmp/gsm8k_fused.log

diff -u /tmp/gsm8k_baseline.log /tmp/gsm8k_fused.log
grep -E '^Accuracy|^Total latency' /tmp/gsm8k_{baseline,fused}.log
```

Success criteria (matches a5eded6 acceptance):
* Baseline matches `47/50 (acc 0.9400)` ± per-question ULP flips.
* Fused accuracy delta within ±2 questions of baseline.
* Per-question wall time ≤ baseline (any speedup is welcome;
  regressions ≥ +10% are a fail).

## Phase C — Graph mode

Repeat Phase A + B with whatever cuda-graph flags the launch script
already sets (our backend is `cuda_graph_runner.py` driven; no extra
toggles). If Phase A passes in eager but fails in graph mode, the
most likely culprits are:

1. Stale `positions_cmp_padding` / `start_pos` content from the
   previous replay step — bisect by adding `fm.<field>.fill_(0)`
   in `init_forward_metadata_replay_cuda_graph` before the
   `_compute_compress_locs` copy.
2. `state_cache_3d` reshape failing on a non-multiple page_size
   (shouldn't happen with the lcm padding in `CompressStatePool`,
   but check `compress_state.kv_score.shape` matches
   `[k * swa_page_size, 2*coff*head_dim]`).
3. `_fused_caches_built` flag stale across capture/replay — the
   weight views are stable in our case but try setting the flag
   to False once in `init_forward_metadata_capture_cuda_graph`
   if accuracy drifts only in graph mode.

## Bisecting accuracy regressions

Toggle in order of suspicion:

```bash
# 1. Skip hadamard (set Compressor.rotate=False in model construction
#    or in apply_ape_hotfix — temporary debug only).
# 2. Force eager mode (--disable-cuda-graph in launch flags).
# 3. Compare fused state_cache writes vs fallback by dumping
#    pool.get_attention_compress_state_cache(layer_id) before/after
#    a single forward in both modes and diffing.
```

Reference for the op signature: `/tmp/ops-transformer/experimental/attention/compressor/README.md`.

Reference for `dsv4_release`'s working implementation:
`/home/t00937989/sglang-khalil-dsv4/python/sglang/srt/layers/attention/nsa/nsa_indexer.py:396`.
