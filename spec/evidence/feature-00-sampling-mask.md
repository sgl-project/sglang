Status: implemented

# Sampling-mask verification

Validation on NVIDIA H200, Python 3.12, PyTorch 2.13.0, sglang-kernel
0.4.6.post1, FlashInfer 0.6.18, and Transformers 5.12.1. Server integration
uses the repository pins TVM-FFI 0.1.11 and CUTLASS DSL 4.6.2.

## Unit and GPU capture regression tests

From the repository root, with the checkout installed in `.venv`:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/python" .venv/bin/python -m pytest -q \
  test/registered/unit/sampling/test_sampling_batch_info.py \
  test/registered/unit/managers/test_batch_result_processor_hidden_states.py \
  test/registered/unit/managers/test_batch_result_processor_mamba_boundary.py \
  test/registered/unit/managers/test_generation_auxiliary_output.py \
  test/registered/unit/disaggregation/test_disaggregation_wire.py \
  test/registered/unit/server_args/test_server_args.py \
  test/registered/sampling/test_sampling_mask.py::TestSamplingMaskCapture \
  test/registered/sampling/test_sampling_mask.py::TestSamplingMaskPacking \
  --disable-warnings --tb=short
```

Result: **311 passed, 38 subtests passed**, 15 warnings, in **19.35 seconds**.
This includes real CUDA sampling and asynchronous copying, cutoff ties, min-p,
mixed opt-in rows, overflow and invalid support, synchronized-token logprobs,
batch filtering/merging, abort cleanup, pipeline payload reconstruction, and
disaggregation metadata transport.

### Pipeline decode without sampling metadata

The [AMD two-GPU CI job](https://github.com/sgl-project/sglang/actions/runs/33957402761/job/101283699801)
on `76bac722f43640f004b7d1083e175940891ad227` exposed an absent-logits
dereference in decode result processing. Pipeline results may omit logits when
no sampling metadata is requested. Prefill and decode now use the shared
materialization helper's existing absent-output guard.

The checked-in `TestDecodeWithoutLogits` regression reproduces the same
`AttributeError` before the fix and verifies that the received token is committed
and streamed after the fix. Run the regression and neighboring result-processing
coverage from the repository root:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/python" .venv/bin/python -m pytest -q \
  test/registered/unit/managers/test_batch_result_processor_hidden_states.py \
  test/registered/unit/managers/test_batch_result_processor_mamba_boundary.py \
  test/registered/unit/managers/test_generation_auxiliary_output.py \
  test/registered/sampling/test_sampling_mask.py::TestSamplingMaskPacking \
  --disable-warnings --tb=short
```

Result on 2026-09-05: **40 passed, 6 subtests passed**, 15 warnings, in
**11.90 seconds**. The isolated checkout excludes the local sampler edit.

## Server integration matrix

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/python" \
  SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 \
  .venv/bin/python spec/evidence/run_sampling_mask.py
```

The checked-in runner uses public `Qwen/Qwen2.5-0.5B-Instruct` weights. It runs
FlashInfer and PyTorch with overlap enabled and disabled, then seeded sampling
parity with mask capture enabled and disabled. Prefill graph capture
is configured up to 32 tokens and decode capture up to batch size 8, subject to
the server's mode-compatibility rules. KV capacity is 4096 tokens.

Multi-process pipeline and prefill/decode deployments are covered here by
transport unit tests, not live deployment tests.

| Backend / test | Overlap | Passed |
| --- | --- | ---: |
| FlashInfer | Enabled | 6 |
| PyTorch | Enabled | 6 |
| FlashInfer | Disabled | 6 |
| PyTorch | Disabled | 6 |
| Seeded parity, PyTorch with Triton matrix multiplication | Enabled | 1 |

Result: **25 passed**. The seeded-parity test passed in **31.562 seconds** and
preserved both token IDs and text. The command uses Triton matrix multiplication
for deterministic inference because the installed DeepGEMM binary is incompatible
with the local PyTorch build. DeepGEMM deterministic inference was not validated
in this environment.

## Qwen3-8B decode throughput

The [benchmark runner](bench_sampling_mask.py) compares upstream
`756d0e0a851c0aee706670affa90c3c47e317d15` with overlap disabled against PR
`cb30f7c6a969b5d19633d5f51c387426f4c00b9a` with overlap enabled. Both run in
detached worktrees, excluding local edits, sequentially on GPU 0: NVIDIA H200
(143771 MiB, 700 W power limit), driver 595.71.05. Dependencies match those
listed above. This measures the combined implementation and scheduling change.

Model: `Qwen/Qwen3-8B`, BF16, TP1, pinned to Hugging Face revision
`b968826d9c46dd6066d109eabc6255188de91218`. Both sampling and attention use
FlashInfer. Each request has 128 synthetic input tokens, generated with seed 42,
and requests 512 output tokens with `temperature=1`, `top_k=4096`, `top_p=1`,
`min_p=0`, `ignore_eos=true`, and `return_sampling_mask=true`. The PR uses
`--sampling-mask-max-tokens=8192` to accommodate cutoff ties; the baseline has no
equivalent ordinary-serving cap.

Common server settings: TP1, static memory fraction 0.7, KV capacity 65536 tokens,
maximum 64 running requests, chunked prefill size 8192, decode CUDA graph maximum
batch size 64, prefill CUDA graph maximum 128 tokens, prefix caching disabled,
and decode logging every 16 steps. The runner records complete launch commands
and resolved server settings in `results.json`.

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python spec/evidence/bench_sampling_mask.py \
  --model-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --gpu 0 --output-dir /tmp/sglang-mask-benchmark
```

Use a new output directory when repeating the command. The runner writes the
prompts, server logs, environment, and per-run interval rates there. It runs one
warmup and three measured trials for each batch size and revision, flushing the
cache between requests. Requests are non-streaming and return complete masks.

The metric is scheduler-reported steady decode tokens/s. Each trial drops its
first two and final 16-step log intervals, then retains only full-batch intervals
with CUDA graphs active. Equal token counts per interval make their harmonic
mean the trial throughput. Prefill, startup, warmup, and final HTTP response
serialization are outside this metric; sampling-mask capture, copying, and
decode result processing remain included. Every completed request must contain
512 aligned tokens, masks, and finite selected-token logprobs. Failed requests
are recorded as failures and excluded from successful throughput samples.

### Benchmark results

Measured 2026-09-05 with Python 3.12.3. All **18 measured trials
and six warmups passed**, with no overflow or invalid-result failures. Every
request returned 512 tokens with aligned masks and logprobs. Observed mask sizes
were **4096–4396 tokens**. The prescribed trimming retained 28 intervals per
baseline trial and 29 per PR trial.

Values below are median decode tokens/s, with the three-run range in parentheses.
Change is `(after / before - 1) * 100`.

| Batch size | Before: overlap off | After: overlap on | Change |
| ---: | ---: | ---: | ---: |
| 1 | 155.47 (154.90–155.56) | 182.37 (181.99–182.40) | +17.30% |
| 16 | 1659.03 (1650.52–1665.77) | 2216.61 (2216.46–2219.79) | +33.61% |
| 64 | 3179.15 (3174.73–3201.97) | 4826.58 (4637.89–4837.76) | +51.82% |

Per-trial throughput, before taking the median:

| Revision | Batch size | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: | ---: |
| before | 1 | 154.898033 | 155.470442 | 155.561325 |
| before | 16 | 1650.515506 | 1665.771059 | 1659.034310 |
| before | 64 | 3179.151680 | 3201.969839 | 3174.725147 |
| after | 1 | 182.367802 | 182.395034 | 181.985833 |
| after | 16 | 2216.612904 | 2219.787022 | 2216.463615 |
| after | 64 | 4837.764115 | 4826.584321 | 4637.890961 |

### Server log excerpts

First retained interval from run 1 of each configuration:

```text
before: [2026-09-05 09:05:42] Decode batch, #running-req: 1, #token: 176, token usage: 0.00, cuda graph: True, gen throughput (token/s): 157.77, #queue-req: 0
before: [2026-09-05 09:06:02] Decode batch, #running-req: 16, #token: 2816, token usage: 0.04, cuda graph: True, gen throughput (token/s): 2146.29, #queue-req: 0
before: [2026-09-05 09:06:53] Decode batch, #running-req: 64, #token: 11264, token usage: 0.17, cuda graph: True, gen throughput (token/s): 4098.22, #queue-req: 0
after: [2026-09-05 09:08:39] Decode batch, #running-req: 1, #token: 177, token usage: 0.00, cuda graph: True, gen throughput (token/s): 182.89, #queue-req: 0
after: [2026-09-05 09:08:56] Decode batch, #running-req: 16, #token: 2832, token usage: 0.04, cuda graph: True, gen throughput (token/s): 2345.61, #queue-req: 0
after: [2026-09-05 09:09:42] Decode batch, #running-req: 64, #token: 11328, token usage: 0.17, cuda graph: True, gen throughput (token/s): 7739.52, #queue-req: 0
```

<details>
<summary>All retained interval rates (tokens/s), keyed by revision/batch/run</summary>

```json
{
  "before/1/1": [157.77, 154.73, 157.75, 157.54, 155.12, 157.76, 157.33, 154.18, 155.69, 155.87, 153.21, 156.37, 156.38, 149.53, 155.63, 155.65, 152.21, 142.29, 156.31, 153.38, 156.35, 156.44, 153.54, 156.33, 155.48, 156.3, 153.47, 156.32],
  "before/1/2": [157.97, 154.75, 157.77, 157.77, 154.91, 157.62, 157.96, 154.89, 157.85, 157.87, 154.94, 157.02, 156.23, 153.61, 156.29, 156.37, 153.46, 144.07, 156.27, 153.41, 156.1, 156.15, 153.34, 156.21, 156.16, 155.87, 153.54, 156.11],
  "before/1/3": [158.06, 154.81, 157.79, 157.76, 154.95, 158.08, 157.13, 154.71, 158.0, 157.9, 154.77, 157.98, 157.74, 154.82, 156.27, 156.21, 153.29, 143.88, 155.97, 153.24, 156.35, 156.3, 153.2, 156.22, 156.24, 156.29, 153.12, 156.07],
  "before/16/1": [2146.29, 1473.81, 1934.03, 1923.85, 1421.96, 1914.77, 1685.25, 1551.04, 1897.38, 1704.34, 1582.84, 1691.74, 1901.67, 1563.91, 937.5, 1901.45, 1535.43, 1690.74, 1897.72, 1378.68, 1889.55, 1894.07, 1368.64, 1877.71, 1673.94, 1868.5, 1482.88, 1659.47],
  "before/16/2": [2145.1, 1473.17, 1929.98, 1939.05, 1410.88, 1910.35, 1696.09, 1546.28, 1897.54, 1696.19, 1566.18, 1079.11, 1887.33, 1561.46, 1684.89, 1902.7, 1571.01, 1684.63, 1896.26, 1421.53, 1893.98, 1875.62, 1371.86, 1868.85, 1674.03, 1878.39, 1483.06, 1665.54],
  "before/16/3": [2106.52, 1369.57, 1914.79, 1895.36, 1406.49, 1926.45, 1722.71, 1544.45, 1907.85, 1174.28, 1526.28, 1680.31, 1899.23, 1534.48, 1676.65, 1886.42, 1516.02, 1682.83, 1885.32, 1403.42, 1886.35, 1883.04, 1359.13, 1871.7, 1675.49, 1879.87, 1479.94, 1663.37],
  "before/64/1": [4098.22, 2547.52, 4056.5, 3838.4, 1834.48, 4090.31, 3870.13, 2766.19, 3893.37, 4025.47, 2645.66, 4149.31, 2043.76, 2757.63, 3822.21, 4086.96, 2592.77, 4068.26, 3843.57, 2738.97, 2164.52, 3960.57, 2615.8, 4064.36, 3761.17, 4030.6, 2473.83, 3936.64],
  "before/64/2": [4206.03, 2761.82, 4197.0, 2578.09, 2804.36, 4175.1, 3885.59, 2894.09, 3961.45, 4173.44, 1776.39, 4138.33, 3937.4, 2886.22, 3861.72, 4102.02, 2702.83, 4127.34, 2137.31, 2850.6, 3905.52, 4050.4, 2711.27, 4009.15, 3790.61, 4037.7, 1664.06, 3917.49],
  "before/64/3": [4199.4, 2194.61, 4152.42, 3908.44, 2799.59, 4200.53, 3920.81, 2873.75, 2172.59, 4062.53, 2672.72, 4134.85, 3854.53, 2725.97, 3837.06, 4065.8, 1703.35, 4054.3, 3843.29, 2677.73, 3872.03, 4079.39, 2593.05, 3996.62, 2112.78, 3999.68, 2521.32, 4015.27],
  "after/1/1": [182.89, 183.05, 182.84, 182.93, 182.75, 182.84, 182.71, 182.72, 182.54, 182.66, 182.53, 182.58, 182.71, 182.75, 182.55, 182.47, 182.66, 175.02, 182.42, 182.6, 182.55, 182.56, 182.55, 182.58, 182.43, 182.62, 182.45, 182.55, 182.48],
  "after/1/2": [183.16, 182.74, 183.01, 182.87, 182.76, 182.84, 182.75, 182.78, 182.69, 182.73, 182.68, 182.73, 182.66, 182.54, 182.56, 182.63, 182.65, 182.67, 173.89, 182.64, 182.66, 182.65, 182.66, 182.66, 182.67, 182.65, 182.63, 182.67, 182.66],
  "after/1/3": [182.59, 182.66, 182.55, 182.37, 182.53, 182.52, 182.01, 182.66, 182.46, 182.31, 182.39, 182.14, 182.24, 182.25, 182.36, 182.2, 182.2, 182.11, 173.85, 182.17, 182.1, 182.27, 181.88, 182.36, 182.16, 182.15, 182.27, 182.12, 182.11],
  "after/16/1": [2345.61, 2049.38, 2704.39, 2340.27, 2053.88, 2322.46, 2657.94, 2046.08, 2303.26, 2653.73, 2037.05, 2297.0, 2649.79, 1782.95, 2647.43, 2639.35, 1016.0, 2631.37, 2279.43, 2033.68, 2622.81, 2274.3, 2002.1, 2613.35, 2248.06, 2584.82, 1803.69, 2587.8, 2585.47],
  "after/16/2": [2339.88, 2068.93, 2704.41, 2334.97, 2065.72, 2326.29, 2659.86, 2056.58, 2301.12, 2659.14, 2066.3, 2292.41, 2654.79, 1798.09, 2645.54, 2644.09, 1012.11, 2636.05, 2278.49, 2051.72, 2625.7, 2272.73, 2011.39, 2615.53, 2251.73, 2589.03, 1778.33, 2587.41, 2588.1],
  "after/16/3": [2346.82, 2053.52, 2714.04, 2336.6, 2051.15, 2323.03, 2661.97, 2021.34, 2297.26, 2658.64, 2051.22, 2297.47, 2653.16, 1775.68, 2643.19, 2641.9, 1012.84, 2634.41, 2281.81, 2042.4, 2625.41, 2273.16, 2002.97, 2617.83, 2245.72, 2590.48, 1803.97, 2589.29, 2589.47],
  "after/64/1": [7739.52, 3623.71, 7445.26, 6499.98, 3779.47, 6534.82, 7496.62, 2061.65, 7543.85, 6580.2, 3840.49, 5305.14, 7352.67, 3592.65, 7324.84, 2766.45, 3854.37, 6530.29, 7287.08, 3560.19, 7269.11, 6383.37, 3752.2, 2748.22, 7087.53, 6264.87, 3802.41, 6280.85, 7084.76],
  "after/64/2": [7741.19, 3575.12, 7280.65, 6534.68, 3704.07, 2923.01, 7433.17, 3475.52, 7525.31, 6271.11, 3885.03, 6535.13, 7481.56, 2043.26, 7330.31, 6408.97, 3611.97, 6518.81, 7302.87, 3553.15, 7261.93, 2766.72, 3793.29, 6403.36, 7089.2, 6255.61, 3470.44, 6308.35, 7075.51],
  "after/64/3": [7752.11, 3484.63, 7297.15, 3547.16, 3761.3, 6474.12, 7548.12, 3548.92, 7629.94, 6261.79, 3636.99, 2741.17, 7327.67, 3357.1, 7332.86, 6405.2, 3830.49, 6488.83, 7028.0, 1914.56, 7174.74, 6363.39, 3430.73, 6264.26, 7081.0, 6240.76, 3480.36, 2715.62, 7066.19]
}
```

For each array, `len(rates) / sum(1 / rate for rate in rates)` reproduces the
corresponding trial throughput.

</details>

Artifact SHA-256 hashes from this run:

```text
05c374bde5d89da662a1fba02221b5d4f3da45bea4ce64175469720930741fc2  bench_sampling_mask.py
b84e7e3f8d6385dba794211e8688ff577d2d341a9ded2c3830b8bd9a7dd531dd  before.log
c1316974baae3356c768041bcd97f8a49b1abd48f04160095882983e1dbd4331  after.log
```
