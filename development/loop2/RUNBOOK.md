# Loop 2 — Standalone Double Sparsity Operator Runbook

This runbook is the operator-phase handoff. Phases 1–5 produce the
calibration, benchmark, and hardware-fixture artifacts the loop-1
DEC-2 (radix-cache default) decision depends on. Phase 0 is a preflight
that **must** pass before any of Phases 1–5 runs.

The actual calibration run, benchmark runs, and the M3-B hardware fixture
are operator work and are explicitly out of scope for the SGLang patch
series — only the scripts, gates, and infrastructure to run them live in
this repo.

---

## Phase 0 — Preflight (REQUIRED before Phase 1)

The launcher (`development/serve_double_sparsity.sh`) bakes in a specific
node profile: single H200 (compute capability 9.x), TP world size 8,
`fp8_e4m3` KV cache, page size 64, `top_k=2048`, NSA `flashmla_kv`
backend. Synthetic CI cannot validate any of these against a real node;
the preflight does.

**Invariants checked:**

1. `--attention-backend` resolves to `nsa` with `nsa_decode_backend=flashmla_kv`.
2. `--kv-cache-dtype == fp8_e4m3`.
3. `--page-size == 64`.
4. The DS config's `top_k == 2048`.
5. `--tp-size == 8` AND `torch.cuda.device_count() >= 8`.
6. CUDA arch detected on `cuda:0` has `compute_capability == (9, *)` (H200 / H100 / B200 class).
7. The channel-mask file referenced by the DS config exists, loads without raising, and passes the value-domain validation (no NaN / Inf / all-zero rows).
8. The selector successfully transitions out of placeholder mode via `bind_runtime_data` for every DS-enabled attention layer (i.e. boot reaches `IS_PLACEHOLDER == False`).

**Operator command (concrete):**

```bash
bash development/loop2/preflight.sh \
    --backend flashmla_kv \
    --dtype fp8_e4m3 \
    --page-size 64 \
    --top-k 2048 \
    --tp-size 8 \
    --cuda-arch-major 9
```

Exit code 0 means safe to proceed. Any non-zero exit aborts the
operator pass; do not start the server.

> If `development/loop2/preflight.sh` is not yet present, the same
> invariants are checked inline by the validator (`validate_double_sparsity`)
> at server boot — but you want the failure to surface BEFORE the
> launcher allocates the model weights. The preflight script is the
> documented preferred path.

---

## Phase 1 — Calibrate the channel mask

```bash
python -m sglang.srt.layers.attention.double_sparsity.calibrate \
    --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2/ \
    --output /models/dsv32-fp8-channel-mask.safetensors \
    --dtype fp8_e4m3 \
    --page-size 64 \
    --label-dim 16
```

Outputs a calibrated `channel_mask.safetensors` containing
`channel_selection` and `channel_weights`, plus a content hash. The
loader (`load_channel_mask`) verifies the hash AND now (R1) rejects NaN /
+Inf / -Inf / all-zero per-row weights with
`DoubleSparsityChannelMaskCorrupt` so corruption surfaces at boot rather
than as opaque NaNs in `compute_page_scores`.

---

## Phase 2 — Boot DS and baseline NSA

```bash
# DS server (uses the calibrated mask):
CHANNEL_MASK_PATH=/models/dsv32-fp8-channel-mask.safetensors \
bash development/serve_double_sparsity.sh

# Native NSA baseline (no DS):
bash development/serve_native_nsa.sh
```

Both scripts launch SGLang against `DeepSeek-V3.2-FP8` on 8-way TP. The DS
launcher no longer exports any `SGLANG_DS_ALLOW_*` env override — boot
must succeed with the real channel mask and `bind_runtime_data` wiring.

Verify boot health:

* DS launcher: server logs show `"double_sparsity: bind_runtime_data
  completed for layer=N, tp_rank=R, world_size=W"` for every DS-enabled
  attention layer.
* `/metrics` exposes `sglang_double_sparsity_channel_mask_valid 1`.
* No production code path reads `SGLANG_DS_ALLOW_NO_ADAPTER` or
  `SGLANG_DS_ALLOW_PLACEHOLDER`. Sanity check:
  `rg "SGLANG_DS_ALLOW_NO_ADAPTER|SGLANG_DS_ALLOW_PLACEHOLDER" python/sglang/srt development/serve_double_sparsity.sh`
  returns zero hits.

---

## Phase 3 — Benchmark DS and baseline at agreed concurrencies

```bash
# Both scripts hit the matching server with bench_serving at concurrencies {16, 32, 64}.
bash development/benchmark.sh           # DS server
bash development/benchmark_baseline.sh  # native NSA server
```

Outputs go to `development/results/` with `--workload` and `--column`
labels embedded in the filenames. Each row contains TPS, P50/P99 TTFT,
P50/P99 ITL, and the DS-only fields `sparsity_rate`, `selected_pages`,
`dense_fallback`.

---

## Phase 4 — Side-by-side SLO + quality report

```bash
python development/benchmark_compare.py \
    --ds-results development/results/double_sparsity_*.json \
    --baseline-results development/results/native_nsa_*.json \
    --out development/results/loop2_comparison.md
```

The comparator enforces the `{GPU id, TP size, page size, radix-cache,
concurrency}` invariants across columns and asserts the SLO floors
(30 tok/s P50, 22 s P99 TTFT). A `dense_fallback > 0` finding or
`selected_pages == total_pages` triggers the no-op detector and fails
the comparison.

---

## Phase 5 — Hardware M3-B fixture + DEC-2 decision

```bash
python -m sglang.srt.layers.attention.double_sparsity.page_signature_write \
    --m3b-fixture-hardware-run \
    --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2/ \
    --channel-mask /models/dsv32-fp8-channel-mask.safetensors
```

This is the **hardware** M3-B page-stability fixture — the same shape
the synthetic CI hook exercises (see
`test_ds_m3b_synthetic_ci_hook` in `test_double_sparsity_unit.py`), but
run against real V3.2 weights and a real KV cache so the
cold/warm prefix runs are bit-stable on real hardware.

When the hardware run passes, set
`server_args._double_sparsity_radix_fixture_passed = True` in the
deployed launch config, then flip the loop-1 DEC-2 default (radix cache
allowed under DS). Loop 2 does NOT flip DEC-2 automatically — the flip
is an explicit operator decision after this run completes.

---

## Anchor checks for the operator

After each phase, run the corresponding regression sweep:

| Phase | Anchor check |
|-------|--------------|
| 0 | `bash development/loop2/preflight.sh --check-only` returns exit 0. |
| 1 | `load_channel_mask` on the freshly produced file does not raise. |
| 2 | Both launchers boot, `/metrics` shows healthy DS state, and the env-gate `rg` returns zero hits. |
| 3 | Benchmark JSONs include the new DS fields (`sparsity_rate`, `selected_pages`, `dense_fallback`). |
| 4 | `benchmark_compare.py` exits 0 and the SLO floors hold. |
| 5 | Hardware M3-B run cold/warm signatures match. |
