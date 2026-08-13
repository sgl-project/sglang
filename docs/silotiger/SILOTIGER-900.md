# SILOTIGER-900: MiniMax-M3 EP1 native MXFP8 MoE

## Scope

This commit consolidates the accepted native AITER/FlyDSL MXFP8 MoE support and
routes the shipping TP4/EP1 contract through the paired AITER implementation.
The runtime key is:

```text
model_dim=6144, inter_dim=768, experts=128, topk=4
```

EP1 does not carry an expert mask or append a fake top-k slot. AITER must look
up the raw `128/4` key. The paired AITER commit supplies exhaustive fused-quant
stage-one/stage-two tunes for every padded token bucket.

The dense SILOTIGER-722 path is independent of expert parallelism and remains
unchanged.

## Paired AITER requirement

Build this commit with the matching AITER v0.1.19.post2 SILOTIGER-900 commit.
That source must contain:

- explicit fake-slot normalization for standard EP callers;
- `minimax_m3_ep1_mxfp8_{untuned,tuned}_fmoe.csv`;
- fused FP8/E8M0 stage-one kernels selected for the EP1 rows;
- FlyDSL 0.3.0.

## Runtime contract

```bash
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_AR=1

python -m sglang.launch_server \
  --model-path <MiniMax-M3-MXFP8-snapshot> \
  --quantization mxfp8 \
  --dtype bfloat16 \
  --tp 4 \
  --ep-size 1 \
  --attention-backend aiter \
  --moe-runner-backend aiter
```

Use an explicit backend and restart the server when changing AITER revisions.

## Validation

```bash
pytest -q test/registered/unit/layers/moe/test_aiter_runner.py
pytest -q test/registered/unit/test_model_overrides.py -k mxfp8
```

The serving log must select exact `6144/768/128/4` rows for covered buckets and
must not emit `no tuned config` for them. The paired AITER rows are tagged
`ep1_paired_validated_20260813`: all 18 passed seven alternating paired
production replays. An exhaustive eight-GPU retune improved the geomean by only
0.726%, below the 1% noise threshold, and regressed five buckets, so it was not
adopted. Deployment still requires the full TP4/EP1 concurrency sweep and
fixed-seed GSM8K.
