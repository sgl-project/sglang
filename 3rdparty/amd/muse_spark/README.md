# Muse Spark 1.2 on MI355X

This directory carries the aiter-side patches and measured evidence for the
ROCm implementation under `sglang.srt.rocm_muse_spark`. On HIP,
`sglang.srt.models.muse_spark` dispatches to that implementation; CUDA keeps
the repository's existing FlashInfer/JIT model unchanged.

The HIP boundary is intentional. A surgical Quark/FlyDSL splice into the
fork's v12 model loaded successfully but generated repetitive invalid text
with both Triton and FlyDSL experts. The same checkpoint and aiter stack
generated correctly through the isolated ROCm implementation. Do not merge
the two model paths until an A/B logits test identifies and validates the
v12 numerical difference; startup and operator parity alone did not catch it.

Validated revisions:

- SGLang baseline: `dc2843801d02f9130feff04635803353561eb604`
- aiter: `4ad99832823dde2315b361cbd3b54b1c5c12acd5`
- composable-kernel: `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`

The aiter revision is the `AITER_COMMIT_DEFAULT` pinned by
`docker/rocm.Dockerfile`; composable-kernel follows from it as a submodule.

`docker/muse_spark/muse-spark-rocm.Dockerfile` applies these patches at build
time, so an image built from it needs no runtime patching. Apply them by hand
only when working in a container built from the plain ROCm image:

```bash
AITER_ROOT=/sgl-workspace/aiter \
  3rdparty/amd/muse_spark/apply_aiter_patches.sh

# Also install the experimental Plan C short-window decode kernel:
ENABLE_PLAN_C=1 AITER_ROOT=/sgl-workspace/aiter \
  3rdparty/amd/muse_spark/apply_aiter_patches.sh
```

`MUSE_MOE_BACKEND=triton` is the default on HIP. Set it to `flydsl` for the
optimized FlyDSL experts, or `native` for a correctness fallback. Plan C
short-window attention remains opt-in with `SGLANG_MUSE_FLYDSL_DECODE=1`; its
aiter kernel is under `patches/experimental`.

Launch the validated TP8/EP8 Plan B configuration:

```bash
MODEL=/data/Muse-Spark-1.2-816B-A42B-Quark-MXFP4 \
MUSE_MOE_BACKEND=flydsl \
scripts/serve_muse_spark_rocm.sh
```

The BF16 checkpoint lives at `/data/Muse-Spark-1.2-816B-A42B-open` and
runs through the same script via `MODEL`.

Acceptance gates:

1. Operator parity for BF16 and Quark MXFP4, including HIP graph replay.
2. `few_shot_gsm8k`: 1319 questions, 5-shot, 2048 max tokens; accuracy at
   least 0.955 and zero invalid answers.
3. Canonical serving workload: IL8192/OL1024, TP8/EP8, random range ratio 0.8,
   ignore EOS, four warmup and four measured requests per concurrency slot.
4. Compare against `muse_moe_flydsl_20260910_results.json`; require completed
   request counts and no more than 3% regression in throughput or TPOT.

Local branch validation (`jacky-v0.5.19-rocm720-mi35x-muse-spark-day0-test`,
2026-09-10):

- GSM8K: accuracy `0.955`, invalid `0`, 1319/1319 completed.
- Final `amd-support` rebase: Plan B/FlyDSL and opt-in Plan C both generated
  `Paris` correctly; matched c4 IL8192/OL1024 reached `303.82` output tok/s
  and `12.16 ms` median TPOT.
- IL8192/OL1024 c4: `303.94` output tok/s, `12.17 ms` median TPOT.
- IL8192/OL1024 c8: `474.98` output tok/s, `15.65 ms` median TPOT.
- IL8192/OL1024 c16: `712.28` output tok/s, `20.86 ms` median TPOT.
- IL8192/OL1024 c32: `1008.08` output tok/s, `29.88 ms` median TPOT.
- c4/c8/c16 throughput deltas versus the recorded FlyDSL run are `-0.66%`,
  `-1.77%`, and `-0.19%`, all inside the 3% gate.

See `muse_spark_aiter_attention_repro.md` and
`muse_moe_flydsl_perf_plan.md` for exact commands and known limitations.
