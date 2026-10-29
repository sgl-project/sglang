# DeepSeek V4.1 small-batch dense split-K experiment

`SGLANG_OPT_DSV41_SMALL_BATCH_SPLIT_K=1` opts the target model's WQKV-A and
shared-expert gate/up projections into FlashInfer CuTe DSL split-K=4 on B300.
The default is off. Eligible calls have BF16 output, 1 <= M <= 8 and local
(N, K) equal to (1792, 5120) or (1152, 5120). Other shapes, backends, devices,
output types, and draft models keep their existing dispatch. Deterministic
inference and batch-invariant mode bypass the opt-in.

Split-K changes FP32 reduction order. This option does not promise bitwise
equivalence to the original GEMM, unchanged generated text, or model-quality
parity. It is intended for controlled performance and accuracy evaluation.

## Validation

With a compatible SGLang environment and a B300 available:

```bash
python test/registered/kernels/ops/gemm/test_dsv41_mxfp8_split_k.py
```

The test covers both shapes at M=1/2/4/6/8 against an FP32 GEMM of dequantized
operands, prequantized input parity, shape/dtype/hardware fallbacks, both
consistency controls, model opt-in wiring, bias and leading dimensions,
CUDA Graph replay with changed input, and the torch.compile custom-op boundary.

To compare serving, start otherwise identical servers with the variable set
to `0` and `1`, using the same GPUs and prompts. For example:

```bash
SGLANG_OPT_DSV41_SMALL_BATCH_SPLIT_K=1 python -m sglang.launch_server \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --tp 4 --ep-size 4 --fp8-gemm-backend flashinfer_cutedsl \
  --cuda-graph-bs-decode 1 --random-seed 42
```

For DSPARK, add `--speculative-algorithm DSPARK
--speculative-dspark-block-size 5`. Keep the draft configuration and acceptance
measurement unchanged. Measure serving without the profiler, then separately
profile to verify which target projections use the split-K kernel. A 40-layer
target with unfused shared experts should have 80 such calls per eligible
forward. Repeat/interleave runs before attributing small differences to GEMMs.

## Initial results: pinned-tactic baseline

These measurements used SGLang `bf9a666c7495e85f651849f154cffd0f94adb39d`,
FlashInfer 0.6.18, PyTorch 2.13.0+cu130 and four B300 GPUs (TP4/EP4).
The experimental launcher selected the same two target projections and tactic.
They are **not a speed measurement of the current PR against current main**:
newer main autotunes small-M MXFP8 calls and may already select split-K.

The serving comparison used twelve natural SPEED-Bench prompts (mean input
1074 tokens), two repetitions per configuration, concurrency 1, greedy decoding
and exactly 512 output tokens. There were 96 successful timed requests across
plain/DSPARK and baseline/optimized, with cache flushes before requests and
profiling excluded from timing.

| Metric | Baseline | Split-K=4 | Change |
| --- | ---: | ---: | ---: |
| Plain median TPOT | 3.58360 ms | 3.53936 ms | -1.23% |
| Plain decode throughput | 279.05 tok/s | 282.54 tok/s | +1.25% |
| Plain mean full-request latency | 2069.83 ms | 2049.14 ms | -1.00% |
| DSPARK mean full-request latency | 1144.46 ms | 1132.45 ms | -1.05%, inconclusive |
| DSPARK weighted accept length | 2.90634 | 2.93200 | +0.88% |

Plain trial median TPOT was 3.58422/3.58306 ms before and 3.53848/3.54034 ms
after. DSPARK's paired per-prompt TPOT improvement had a descriptive bootstrap
95% interval of [-2.52%, 4.58%], including zero. Its pooled median TPOT fell
6.73%, but generation/acceptance variation prevents attributing that number
to kernel speed. The sequential experiment does not eliminate system drift.

In separate serving traces, WQKV decreased from 7.390 to 5.842 us/call and
shared FC1 from 6.974 to 5.110 us/call. Shared-expert execution overlaps routed
experts, so the summed kernel savings overstate the wall-clock benefit. All
optimized target graphs had 40 split-K WQKV plus 40 split-K shared-FC1 calls;
baseline and draft graphs had none.

The preceding isolated sweep found M=1 WQKV 6.31 -> 4.60 us and shared FC1
6.59 -> 4.11 us, using rotating buffers beyond L2 capacity. Across the full
60-case sweep, relative L2 error against a dequantized FP32 reference was
below 1%, and the largest relative L2 difference against the baseline BF16
output was 4.71e-5. This is kernel-level evidence, not model-quality validation.

Before considering a default change, repeat the full serving comparison on
current main with its normal autotuning, and run checkpoint-level accuracy
and acceptance evaluations. High-concurrency benefit and the separate
end-to-end contribution of each projection have not been established.
