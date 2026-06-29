# Table-free Double Sparsity — serving, calibration, and the conc-64 perf eval

Table-free Double Sparsity (DS) is a query-signature attention sparsifier for
GLM-5.1-FP8 (it reuses the DeepSeek MLA path through `glm4_moe`). It selects a
top-K of KV positions per decode step from an offline **channel mask** (`S_h`,
`w_c`) and serves them through the FlashMLA sparse path. This document records
how to calibrate the mask, serve with DS, and reproduce the concurrency-64
performance reference.

> **Performance posture.** At concurrency 64 the raw-dot selector lands ≈25.6
> decode TPS / ≈24.4 s P99 TTFT (the `cosine` accuracy default is ~4 TPS slower);
> neither meets a 30-TPS floor, and neither does native DSA on this base. The
> eval below is a **regression gate** (parity vs. that reference), not an SLO gate.

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
  --double-sparsity-config '{"top_k": 2048, "page_size": 64, "channel_mask_path": "<mask>.safetensors"}' \
  --random-seed 42 --trust-remote-code
```

`channel_mask_path` is the only required config field; `top_k`/`page_size`
default to `2048`/`64`. The scorer defaults to `cosine` with current-slot
inclusion (the accuracy-preferred selector); pass `"scorer_norm": "off"` for the
raw channel-dot ablation.

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

### Measured (conc-64, 1 prefix group, 256/256 completed, seed 42, GLM-5.1-FP8, 8×H200)

| Selector | p50 decode TPS (≥ 24.2) | P99 TTFT (≤ 30.1 s) | Parity gate |
|----------|------|------|------|
| `scorer_norm:"off"` (raw-dot) | **25.60** | **24.38 s** | ✅ PASS |
| `scorer_norm:"cosine"` (default) | 21.1 | 32.8 s | accuracy-preferred (below gate) |

The parity band (`reference 26.9 TPS / 25.1 s`) was calibrated against the
raw-dot selector, which meets it (`parity=true`, `request_shape_ok=true` in
`verdict.json`). The `cosine` default does more per-step work (the in-kernel
per-head norm division) and trades ~4 TPS for the accuracy it restores
(GSM8K §4); it is the recommended default, with `off` available for the
speed-parity operating point.

The decode result depends on the selector-width graph ladder
(`selector_width_buckets`, default `[5120]`): the captured graph scores only the
covering width (5120) instead of the full `req_to_token` width (~202k here), so
without it DS decode collapses to ~18.8 TPS. The CUDA-graph runner keys decode
graphs by `(batch size, selector width)` for this reason.

## 4. Accuracy (few-shot GSM8K)

GLM-5.1-FP8, 8×H200, 200 questions, chat path, default (`cosine`) selector:
**GSM8K 0.950 (190/200), 0% invalid** — in band with the dense/native references
(GSM8K carries 1–5% run-to-run variance per the contribution guide).
