# Muse Spark MI355X ROCm cookbook

## Scope

This path is independent of NVIDIA hardware. The failed mixed-model experiment
combined the fork's CUDA/Hopper v12 model contracts with the ROCm overlay. The
ROCm kernels themselves are validated: Plan B aiter 3D attention and FlyDSL MoE
work together.

The fork therefore keeps CUDA on its existing v12 implementation and dispatches
HIP to `sglang.srt.rocm_muse_spark`.

## Optimization ladder

- **A — baseline:** `rocm/quark-mxfp4`, Triton MoE + Triton attention.
- **B — kernel fusion:** `rocm/decode-fusion`, Triton MoE + Triton attention.
  Adds cross-layer FFN boundary, fused Q/K RMSNorm, tuned/split-K MoE, router
  GEMV, and pooler/unpooler aiter tuned GEMM.
- **C — Plan B attention:** the B fusion stack with page-16 CK prefill and
  aiter unified-attention 3D decode.
- **D — best:** the C attention stack with FlyDSL expert GEMM/finalize.

The checked-in HIP implementation contains the complete D code. Select B/C/D
at runtime:

```bash
# B
ATTN=triton MUSE_MOE_BACKEND=triton scripts/serve_muse_spark_rocm.sh

# C
ATTN=aiter PAGE_SIZE=16 MUSE_MOE_BACKEND=triton \
  scripts/serve_muse_spark_rocm.sh

# D (recommended; these are the script defaults)
scripts/serve_muse_spark_rocm.sh
```

Exact historical A requires the `rocm/quark-mxfp4` source revision; merely
disabling environment switches in D is not identical because B also changed
kernel source and tile selection.

## One-time aiter setup

Validated revisions:

- aiter: `4ad99832823dde2315b361cbd3b54b1c5c12acd5`
- composable-kernel: `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`

Images built from `docker/muse_spark/muse-spark-rocm.Dockerfile` already carry
these patches, so this step is only for containers built from the plain ROCm
image:

```bash
cd /sgl-workspace/sglang-muse-spark
AITER_ROOT=/sgl-workspace/aiter \
  3rdparty/amd/muse_spark/apply_aiter_patches.sh
```

This applies the page-scatter fix, Plan B 3D selector, and FlyDSL MoE kernels
idempotently. Do not stack Plan A patch 02 with Plan B patch 03.

## Recommended production launch

```bash
cd /sgl-workspace/sglang-muse-spark
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL=/data/Muse-Spark-1.2-816B-A42B-Quark-MXFP4 \
PORT=30000 \
scripts/serve_muse_spark_rocm.sh
```

Defaults are TP8/EP8, context 32768, page size 16, Plan B aiter attention,
FlyDSL MoE, and Plan C disabled. The script prepends this checkout's `python`
directory to `PYTHONPATH`, so it does not accidentally run
`/sgl-workspace/sglang`.

Confirm runtime identity:

```bash
cd /tmp
PYTHONPATH=/sgl-workspace/sglang-muse-spark/python python3 - <<'PY'
from sglang.srt.models.muse_spark import MuseSpark1xForCausalLM
print(MuseSpark1xForCausalLM.__module__)
PY
```

Expected: `sglang.srt.rocm_muse_spark.model`.

## Correctness smoke

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/data/Muse-Spark-1.2-816B-A42B-Quark-MXFP4",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 512,
    "temperature": 0
  }'
```

Expected final answer contains `Paris`.

Full acceptance:

- GSM8K: 1319 questions, 5-shot, max 2048 tokens.
- Required accuracy: at least 0.955, invalid answers: 0.
- Serving: IL8192/OL1024, random range ratio 0.8, ignore EOS.
- Four warmups and four measured requests per concurrency slot.

The host benchmark helper supports that protocol:

```bash
CONCURRENCIES="4 8 16 32" \
PROMPT_MULT=4 WARMUP_MULT=4 PROFILE=0 \
RESULT_DIR=/home/yichiche/port-validation/my_run \
TAG=D PORT=30000 \
/home/yichiche/run_spark_client.sh
```

## Matched A versus D result

Run date: 2026-09-10. Both used the same checkpoint, container, serving script,
TP8/EP8, request seed, and 4x warmup/4x measured protocol.

- c4: A `198.74` vs D `304.11` output tok/s (**+53.0%**);
  median TPOT `18.58` vs `12.14 ms` (**34.7% lower**).
- c8: A `339.45` vs D `475.68` output tok/s (**+40.1%**);
  median TPOT `21.65` vs `15.62 ms` (**27.9% lower**).
- c16: A `512.56` vs D `713.80` output tok/s (**+39.3%**);
  median TPOT `28.90` vs `20.78 ms` (**28.1% lower**).
- Both A and D returned the correct `Paris` smoke response.
- D full GSM8K: accuracy `0.955`, invalid `0`.

Raw results are under
`/home/yichiche/port-validation/ad_cookbook_20260910`.

## Experimental Plan C SWA decode

Plan C in the patch naming is the optional FlyDSL short-window attention
kernel, not optimization-ladder item C above. It remains disabled by default.

```bash
ENABLE_PLAN_C=1 AITER_ROOT=/sgl-workspace/aiter \
  3rdparty/amd/muse_spark/apply_aiter_patches.sh

SGLANG_MUSE_FLYDSL_DECODE=1 scripts/serve_muse_spark_rocm.sh
```

It passed a correctness smoke without a GPU fault, but it is limited to small
decode batches and has not beaten Plan B consistently at end-to-end level.
