# Table-free Double Sparsity — serving, calibration, and the conc-64 perf eval

Table-free Double Sparsity (DS) is a query-signature attention sparsifier for
GLM-5.1-FP8 (it reuses the DeepSeek MLA path through `glm4_moe`). It selects a
top-K of KV positions per decode step from an offline **channel mask** (`S_h`,
`w_c`) and serves them through the FlashMLA sparse path. This document records
how to calibrate the mask, serve with DS, and reproduce the concurrency-64
performance reference.

> **Performance posture.** The validated candidate lands ≈26.9 decode TPS /
> ≈25.1 s P99 TTFT at concurrency 64 — it does **not** meet a 30-TPS floor there
> (neither does native DSA). The eval below is a **regression gate** (parity vs.
> that reference), not an SLO gate.

## 1. Channel-mask calibration

The mask is model- and quant-specific. Regenerate it with the shipped
calibrator (GLM-native recipe: `--dtype fp8_e4m3 --label-dim 32`):

```bash
python -m sglang.srt.layers.attention.double_sparsity.calibrate \
  --model <GLM-5.1-FP8 path> \
  --dtype fp8_e4m3 --kv-cache-dtype fp8_e4m3 --tp 8 \
  --label-dim 32 --page-size 64 --num-samples 256 --block-size 512 --seed 42 \
  --dataset <calibration corpus, one document per line> \
  --output <mask>.safetensors
```

Calibration is the **only** step that may set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (a separate, short-lived
process — it must never be set for serving, where it breaks custom all-reduce
IPC at TP=8).

### Provenance of the mask used for this run

| Field | Value |
|-------|-------|
| Base commit (`<BASE>`) | `105e095e005d02a178fb6c5a23bd22ba644c90e4` |
| Corpus | Pile-val, 300 documents, one per line (deterministic; `seed 42`) |
| Recipe | `--dtype fp8_e4m3 --label-dim 32 --page-size 64 --num-samples 256 --block-size 512 --seed 42` |
| Mask path | `<calibrated GLM-5.1-FP8 channel mask>.safetensors` (large external artifact, referenced by path) |
| Mask shape | `channel_selection / channel_weights = [78 layers, 64 heads, 32 label_dim]`, page_size 64, dtype fp8_e4m3 |
| Mask content SHA-256 | `35155ac46ad79fa82e531138434ff35708e2d8c2932889323a21a455342a9b00` |
| Mask file SHA-256 | `5c89c516428f379c983461ceb58fb366c0d6cb12733b3f957d98edb5406f7b21` |

The corpus and the calibrated mask are large external artifacts referenced by
path (not committed); the SHA-256 above pins the exact mask the loader accepts.

## 2. Serving with Double Sparsity

```bash
python -m sglang.launch_server \
  --model-path <GLM-5.1-FP8 path> \
  --tp-size 8 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.8 \
  --max-running-requests 64 --cuda-graph-max-bs 64 --page-size 64 \
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule --disable-piecewise-cuda-graph \
  --enable-double-sparsity \
  --double-sparsity-config '{"top_k": 2048, "page_size": 64, "channel_mask_path": "<mask>.safetensors", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0}' \
  --random-seed 42 --trust-remote-code
```

CUDA graphs and radix cache are **on** (radix needs no fixture artifact or
override — DS + radix just works). To confirm DS is genuinely active, check the
startup log: it loads the channel mask (`Loaded channel mask file ...
content_sha256=...`) and selects the DSA attention backend, and the decode
throughput tracks the selector-width graph ladder (§3) rather than the dense
~18.8 TPS floor.

> This branch is off latest `origin/main`, which requires `sglang-kernel >=
> 0.4.4` (its flash-attention path uses the 0.4.4 `only_qv` kernel). Install it
> before serving (`pip install sglang-kernel==0.4.4`). The Double Sparsity
> kernels themselves are unchanged from the validated candidate.

## 3. The conc-64 performance eval

A thin wrapper over stock `bench_serving` pins the workload and derives the
gated metrics:

```bash
python benchmarks/bench_double_sparsity.py \
  --model <GLM-5.1-FP8 path> --host 127.0.0.1 --port 30000 \
  --num-prompts 256 --seed 42 --evidence-dir <dir>
```

Workload: `generated-shared-prefix`, gsp 2253 / 1843 (ISL ≈4096, system prompt
~55% of each input), OSL 512, range-ratio 1.0, max-concurrency 64, one trial.
The wrapper pins **one** shared-prefix group (`--gsp-num-groups 1
--gsp-prompts-per-group <num_prompts>`) so all `--num-prompts` requests share the
one system prompt — the stock default would otherwise be 64 groups × 16 and
ignore `--num-prompts`. It reports **p50 decode TPS** = median of
`(output_tokens − 1) / Σ(inter-token latencies)` and **P99 TTFT**, fails closed
unless exactly `--num-prompts` requests completed, and passes iff p50 decode TPS
≥ 24.2 **and** P99 TTFT ≤ 30.1 s.

### Measured (this run, conc-64, 1 prefix group, 256 prompts completed, seed 42, GLM-5.1-FP8, 8×H200)

| Metric | loop-11b ref | Band | Native DSA (same base, context only) | **DS (this branch)** |
|--------|-----------|------|------|------|
| p50 decode TPS | 26.9 | ≥ 24.2 | 26.06 | **35.05** ✅ |
| P99 TTFT | 25.1 s | ≤ 30.1 s | 46.50 s (not in band) | **22.90 s** ✅ |

The **accepted** result is the DS column, measured on the corrected single-group
workload: `actual_completed=256`, `gsp_num_groups=1`, `request_shape_ok=true`,
`parity=true` — the eval writes these fields to `verdict.json` under its
`--evidence-dir`. The
native-DSA column is **same-base context only**: it was a separate earlier run
made *before* the wrapper pinned the GSP grouping, so it is NOT a corrected-shape
measurement and is not a pass/fail baseline. Its 46.50 s P99 TTFT is shown only
to illustrate that the high conc-64 TTFT on this base is not DS-specific (DS, at
22.90 s, is well inside the band).

DS meets the loop-11b parity band. The decode result depends on the
selector-width graph ladder
(`selector_width_buckets`, default `[5120]`): the captured graph scores only the
covering width (5120) instead of the full `req_to_token` width (~202k here), so
without it DS decode collapses to ~18.8 TPS. The CUDA-graph runner keys decode
graphs by `(batch size, selector width)` for this reason.
