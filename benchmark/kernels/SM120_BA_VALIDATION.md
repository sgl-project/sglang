# SM120 BF16 BA projection: validation and reproduction

This opt-in experiment targets contiguous BF16 `in_proj_ba` projections on
SM120, with N=48, K=5120 and M=2..8. M=1 and unsupported cases use the existing
fallback. It is disabled by default. It is not a general GEMM replacement.

## Executable operator checks

Use a Linux SGLang development environment with PyTorch, Triton and an RTX 5090.
From the repository root:

```bash
PYTHONPATH=python python test/registered/unit/layers/quantization/test_sm120_ba_dispatch.py -v
PYTHONPATH=python python test/registered/kernel/gemm/test_sm120_ba_gemm.py -v
PYTHONPATH=python python benchmark/kernels/benchmark_sm120_ba_gemm.py --device 0 --repeats 20
```

The GPU test is registered in `base-b-test-1-gpu-small`; this repository maps
`1-gpu-small` to `1-gpu-5090`. A non-SM120 machine skips the numerical GPU tests.
The CPU dispatch suite is `base-a-test-cpu`. A single-GPU runner cannot exercise
the optional second-device test; that case was covered in the local two-GPU run.
The microbenchmark measures graph-replayed operator latency, not serving TPS.

## Recorded serving experiment

The September 10 measurements below used source commit
`cd2a21c7bc4e3846f68b6f3ab171f2d070c3e590`, imported into the existing Linux
dependency environment. Later test-registration/documentation corrections do
not change the kernel or dispatcher; they are not a new GPU measurement.

The branch was subsequently rebased onto upstream
`ad7f57c9ea796e64b1c244597a552d8a01ee60de` to satisfy the CI base-commit gate.
The kernel and `unquant.py` are unchanged across that rebase, but other upstream
code changed. The recorded full-model results therefore do not constitute a
new full-model validation of that newer base.

Hardware: TP2 RTX 5090. Tested checkpoint: Qwen3.8-27B NVFP4 RTX5090 export,
with BF16 BA projections. Relevant settings: FP8 E4M3 KV, BF16 SSM, page size
256, chunked prefill 4096, static memory fraction 0.85, max-running 8, Mamba
cache size 32, decode graph max batch 8, single-batch overlap, no speculative
decoding. KV capacity was 1,028,608 logical tokens in both arms.

### Protocol to reproduce the comparison

1. Use an isolated server and freeze the checkpoint, tokenizer, dependency
   versions and all launch arguments. Save their hashes/version manifest.
2. Prepare 16 fixed WikiText-103 token-ID prompts of exactly 50,000 tokens.
   The recorded corpus was `Salesforce/wikitext`, `wikitext-103-raw-v1`.
   Submit token IDs to `/generate`, not chat messages, to avoid template drift.
3. Use six fresh server boots in A1/B1/B2/A2/A3/B3 order. A disables
   `SGLANG_ENABLE_SM120_BA_GEMM`; B sets it to `1`. This flag is the only
   treatment. Do not enable profiling or numerical hooks for timed runs.
4. In each boot, test client concurrency 1, 4 and 8, with 16 requests per group.
   Before each group, flush the isolated server's cache, then generate one
   token from the first 45,056 prompt tokens. Exclude this warmup from timing.
5. Use temperature 0, `max_new_tokens=1000`, `ignore_eos=true`, streaming.
   Record every request, including failures, actual output counts and cached
   tokens. Do not replace slow runs. Record server running/queue peaks too.
6. TPS = sum of output tokens / group wall time. TTFT starts when each client
   request is admitted by its concurrency semaphore and ends at the first
   stream event reporting an output token. Average the three per-arm TPS
   results; report mean per-run TTFT quantiles explicitly, not pooled quantiles.

Recorded token-ID SHA256 (concatenate each prompt in order, each ID as unsigned
32-bit little-endian):
`3162a9eb3c5161c06503cfa6dc6ad3801d0d20dbe0dbeb4fed4267903de73016`.
Natural shared prefix: 45,063 tokens; warmed prefix: 45,056; measured hit:
90.112%. The hash identifies the archived input, but this document does not
redistribute that token corpus or the private full-model diagnostic harness.
A newly prepared corpus is a replication workload, not the identical archived
input unless the hash matches. Only the operator checks above are standalone
executable reproductions in this change.

### All recorded per-boot aggregate TPS

| Client concurrency | A1 | B1 | A2 | B2 | A3 | B3 | A mean | B mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 116.15 | 116.26 | 116.69 | 116.25 | 116.28 | 116.09 | 116.37 | 116.20 |
| 4 | 327.39 | 350.37 | 334.52 | 347.92 | 327.41 | 350.47 | 329.77 | 349.59 |
| 8 | 397.75 | 415.29 | 400.97 | 412.59 | 390.88 | 415.44 | 396.53 | 414.44 |

All 288 timed requests succeeded. Changes were -0.15%, +6.01%, +4.52%.
At client concurrency 8, sampled server running peak was 7, not sustained
eight-way decode. Three pairs do not establish a universal speedup.

| Client concurrency | A TTFT P50/P90 (s) | B TTFT P50/P90 (s) |
| --- | ---: | ---: |
| 1 | 0.733 / 0.754 | 0.733 / 0.755 |
| 4 | 2.091 / 2.609 | 2.146 / 2.619 |
| 8 | 3.394 / 10.494 | 3.377 / 10.198 |

Concurrency-4 TTFT did not improve. Values are means of per-run quantiles.

## Quality evidence and limits

- Same-input checks across 48 BA layers and both ranks compared 110,592,000
  active output elements with zero differences. This includes repeated inputs
  and graph padding; it is not that many independent samples.
- Controlled native-batch admission passed three A/B pairs: 12,096 full-vocabulary
  teacher-forced positions and 200 free-generation entries per pair matched.
  Entries include repeated questions across concurrency groups.
- A separate A/B/A trajectory audit matched all 8,000 logits positions while
  verifying the same scheduler metadata on both ranks. Instrumented timing
  is not used as performance evidence.
- Independent HTTP free-generation checks had no correctness flips in three
  pairs, but token equality was 196/200, 200/200, 200/200. Baseline repeats
  also varied; this is not universal losslessness.
- **Independent HTTP strict logits checks failed**, including baseline repeats.
  Candidate/baseline mean KL was 0.0380393, P99 KL 0.359714; baseline repeat
  was 0.0227127 and 0.307314. Gates of 0.001 and 0.01 were not relaxed.
  Controlled admission does not retroactively make this protocol pass.
- A two-request, two-rank fixed-input replay isolated BF16 recurrent-state
  rounding at different prefill boundaries as one source. FP32 replay state
  removed those tested local differences; it does not establish the cause of
  every model-level difference. No FP32-state change is included here.

Remaining review gates: upstream CI, actual eight-way decode performance,
broader checkpoints/workloads, dependency-version portability and same-shape
alternative-kernel comparisons. No claim is made for multimodal, 97k/512k,
other GPU architectures, or default enablement.
