# Experimental MiniMax-M3 FlyDSL attention on ROCm

This opt-in integration uses the AITER paged-attention kernel and work planner
from [AITER #4332](https://github.com/ROCm/aiter/pull/4332) and
[#5546](https://github.com/ROCm/aiter/pull/5546). It calls AITER directly and
does not modify the AITER kernel.

**Experimental: five numerical checks fail the current tolerance, and small
batches regress in serving performance. Keep this disabled by default.**

## Routing

| Stage | Implementation |
| --- | --- |
| Main KV | Write FP8 directly to SHUFFLE 5D |
| Index KV and selection | Existing NHD cache, indexer, and top-k reuse policy |
| Dense decode | FlyDSL paged attention, optional GPU work planner |
| Sparse decode | Selected blocks become page tables; static FlyDSL splits |
| Sparse prefill | Causal query rows in chunks of 1024; static FlyDSL splits |
| Dense prefill | Gather/dequantize used KV, then existing AITER linear prefill |

The sparse adapter folds KV heads into the page axis with views of the cache,
so each head can have its own selection without copying the whole cache. A
partially filled page is placed last so AITER's final-page mask remains valid.
Graph decode owns persistent int32 length buffers; refresh from SGLang's int64
inputs is captured once per forward. Dense plans are allocated for captured
batch sizes and refreshed on replay. Eager execution uses static splits.

## Dependencies and launch

The measured stack used SGLang `fc9e1c8d296216ff1e216dfbe7286ef392448d28`
plus this integration, AITER `94dca7bc674650b6de2609eeae2bcdd58e970bf3`,
FlyDSL `0.3.4.1`, PyTorch `2.10.0+rocm7.2.4.git3d3aa833`, and
Triton `3.7.0+amd.rocm7.2.0.git89002410` on gfx950. Install the patched
SGLang source and matching AITER/FlyDSL in a compatible ROCm environment.
Missing AITER kernel/compiler support raises an error when opted in.

The Dockerfile accepts `AITER_COMMIT` and `AITER_APPLY_DEFAULT_BACKPORTS=0`
for this newer pin, which already includes the default backports. The default
image dependencies and backport behavior remain unchanged. Example build:

```bash
docker build -f docker/rocm.Dockerfile \
  --build-arg BRANCH_TYPE=local --build-arg BUILD_TYPE=srt \
  --build-arg GPU_ARCH=gfx950-rocm724 \
  --build-arg AITER_COMMIT=94dca7bc674650b6de2609eeae2bcdd58e970bf3 \
  --build-arg AITER_APPLY_DEFAULT_BACKPORTS=0 \
  -t sglang-minimax-flydsl:experimental .
```

This Docker build has not been validated. The measurements used source and
dependency overlays in an existing ROCm container.

```bash
MODEL_PATH=/path/to/MiniMax-M3-MXFP8 MODE=planned \
  bash test/manual/minimax_m3/flydsl/launch.sh
```

`MODE=baseline` disables the feature; `MODE=static` enables FlyDSL without the
dense planner; `MODE=planned` enables both. The wrapper sets
`SGLANG_MINIMAX_FLYDSL_DECODE` and `SGLANG_MINIMAX_FLYDSL_PLAN`, selects AITER,
FP8 KV, page 16, TP4, and disables radix cache. It also disables installed
general plugins so the comparison uses native SGLang. It binds localhost.

The implementation accepts gfx942/gfx950 E4M3 KV, but this pilot tested gfx950
only. It rejects speculative decoding, HiSparse/HiCache, PD disaggregation,
DP/context parallelism, two/single batch overlap, HND KV, and the separate dense
sparse-decode option. Those paths require their own cache/mask qualification.

## Reproduce adapter checks and sparse timing

Run from the SGLang repository root in the pinned GPU environment:

```bash
python3 test/manual/minimax_m3/flydsl/validate_adapter.py \
  --adapter python/sglang/srt/layers/attention/minimax_sparse_ops/flydsl_decode.py \
  --output adapter-results.json

python3 test/manual/minimax_m3/flydsl/benchmark_sparse.py \
  --adapter python/sglang/srt/layers/attention/minimax_sparse_ops/flydsl_decode.py \
  --baseline-kernels python/sglang/kernels/ops/attention/minimax_sparse \
  --output sparse-results.json
```

The validator returns nonzero on failed numerical checks. Do not loosen its
tolerance to obtain a passing result. These manual scripts are not registered
in CI because the feature requires the newer dependency stack and qualification
is incomplete.

## Reproduce the serving pilot

Use `MiniMaxAI/MiniMax-M3-MXFP8` revision
`c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`. The harness expects a work directory
containing `model/`, `sglang/`, and `launch.sh`, and runs on GPUs 0–3:

```bash
work=$(mktemp -d)
ln -s "$(pwd)" "$work/sglang"
ln -s /path/to/the/pinned/checkpoint "$work/model"
cp test/manual/minimax_m3/flydsl/launch.sh "$work/launch.sh"
python3 test/manual/minimax_m3/flydsl/serve-comparison.py --work "$work"
python3 test/manual/minimax_m3/flydsl/summarize-serving.py "$work/serving-results"
```

Use an isolated GPU allocation. The harness starts a localhost server and
terminates only its own process group. It runs the existing MiniMax GSM8K
chat/thinking helper before each mode's benchmarks. Results retain per-request
lengths, errors, latency details, and server configuration. The summarizer
rejects failed requests, invalid metrics, and mismatched request lengths.

## Measured results (2026-09-27)

Four MI355X/gfx950 GPUs, TP4, MXFP8 weights, BF16 activations, FP8 KV, page 16,
context limit 65536, chunked prefill 8192, maximum running requests 32. Decode
graph buckets: 1/2/4/8/12/16/24/32. No speculative decoding; NCCL all-reduce with
custom all-reduce disabled. All modes reported effective memory fraction 0.68
after SGLang scaled the requested 0.80 by 0.85.

Two repetitions per mode/workload/concurrency, 8–40 requests per point, fixed
seed 20260927, forced output lengths, and warmup requests equal to concurrency.
All **60 measurements / 1,272 timed requests succeeded**. Input/output length
arrays match across modes. Loading and startup are excluded from timed results.

Planned FlyDSL versus baseline, using medians of run-level metrics:

| Workload | Concurrency | Output throughput (tok/s) | Throughput change | Median TPOT change |
| --- | --- | --- | --- | --- |
| 8K input / 256 output | 1 | 74.43 → 70.99 | -4.6% | 6.2% slower |
| 8K input / 256 output | 2 | 136.01 → 129.29 | -4.9% | 6.6% slower |
| 8K input / 256 output | 10 | 312.54 → 317.80 | +1.7% | 0.4% lower |
| 8K input / 256 output | 15 | 379.14 → 388.80 | +2.5% | 0.9% lower |
| 8K input / 256 output | 20 | 430.83 → 447.45 | +3.9% | 2.3% lower |
| Mixed 1K–32K input / 8–256 output | 20 | 128.79 → 138.72 | +7.7% | 7.8% lower |

At concurrency 20, median TTFT falls 3574.75 → 3325.88 ms for fixed 8K and
2165.59 → 1973.47 ms for mixed lengths. Full metrics and repeat ranges are in
[serving-summary.json](results/serving-summary.json).

Limitations of this pilot:

- Mixed-length concurrency 2 is unstable: baseline throughput ranges
  46.88–62.39 tok/s. Its aggregate +13.7% is not reliable evidence of improvement.
- The first static sweep is slower than the second. Shared persistent JIT
  caches and fixed mode order confound attribution to the planner. Comparing
  second repetitions, static/planned throughput is within 0.3% at concurrency
  10–20. This run does not establish an additional planner benefit.
- Short traces and two repetitions cannot establish production tail latency
  or statistical significance for small gains. Warm all traces and vary mode
  order in follow-up measurements.
- This pilot uses MXFP8, no speculative decoding, and NCCL all-reduce. Results
  do not predict other weight formats, speculation modes, or communication paths.

The isolated sparse kernel (including page-table construction) has 21.8% lower
latency at batch 20, but 8–10% higher latency at batches 1–2. It excludes indexer,
KV writes, other model layers, and communication; it repeatedly uses warm
tensors. See [sparse timing](results/sparse-results-v2.json).

## Accuracy and remaining qualification

GSM8K chat/thinking, first 64 questions, temperature zero, max 4096 output
tokens: baseline **64/64**, static **63/64**, planned **64/64**, no invalid
answers. This smoke check does not establish accuracy equivalence. The upstream
helper did not retain per-answer outputs, so the static difference needs a
larger paired evaluation.

The GPU adapter validator passes **43/48** checks, including dense decode,
changed-length graph replay, sparse decode, and 12 exact comparisons against
independent CPU-built page tables. Five short sparse-prefill cases exceed
`atol=0.01, rtol=0.08` against dequantized FP32 attention. In those cases, adapter
output exactly matches direct AITER using independent metadata. This points to
the kernel's FP8 arithmetic, but does not resolve acceptable model accuracy.
The TP4/page-16/one-KV-head group passes its six numerical checks. See
[individual checks](results/adapter-results-v2.json).

Before enabling generally, resolve the numerical failures, run full paired
accuracy evaluation, investigate small-batch regressions, and qualify the
intended production communication and speculation configuration.
