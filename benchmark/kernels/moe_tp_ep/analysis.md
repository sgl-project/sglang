# MoE TP versus TP+EP: corrected A3 measurements

Measured on September 16, 2026 (UTC+8), for [Ascend/sglang #1098](https://github.com/Ascend/sglang/issues/1098).

## Result and scope

On two compute devices within one A3 physical card, pure TP had lower median MoE
block latency at every measured global token count. EP/TP latency ratios were
1.46–1.89 for Prefill and 1.16–1.52 for Decode. These are observations on this
single topology and workload. They do not locate a crossover at larger device
counts or establish a statistically significant population-level difference.

The previous equal-per-rank measurements compared TP processing N global tokens
with EP processing 2N. Those ratios and the claim that all differences were due
to communication are withdrawn. All tables below use the corrected workload.

## Reproduction identity

- The device is Ascend910_9382: physical NPU 1, chips 0/1, physical IDs 2/3,
  connected by SIO. This is one A3 card with two compute devices, not two cards.
- The model is `Qwen/Qwen3-30B-A3B`, BF16, with 48 layers, hidden size 2048,
  128 experts, top-8 routing and expert intermediate size 768. Module tests use
  layer 0 checkpoint weights; serving uses the full model.
- The executed SGLang source archive is
  `5891f1c557bd7a047aeea6d62ab916544ffe9cd5`, with the benchmark files identified
  by the per-run SHA256 records. Runtime code is unchanged from `1895cabfa8`.
  `qwen3_moe.py` SHA256 is
  `07ab423bf32b47951df6522ffa397b10421385607e22010759f99cd970ae6bb4`.
  Python imports SGLang from the task's source archive. The installed distribution
  metadata still reports `0.5.14.dev172+gf308abc05`; that is not the executed source
  identity. The submitted timing script matches the measured SHA256
  `b0f20f707088d35dbb8af2b2d2627f230f740ec0a7a15ab74222985799c83eb7`.
- The environment uses Python 3.11.15, PyTorch 2.10.0+cpu, torch-npu 2.10.0,
  CANN compiler/OPP 9.0.0, driver 25.5.1, and DeepEP
  `1.0.0+e145d906.cann.9.0.0.b250`. See the
  [release and package checksums](results/a3-2026-09-16/revision-environment.json)
  and [installation/environment commands](README.md#environment).
- The cloud provider exposes a temporary container, hostname `4311e380b494`.
  Its creation image name and digest are unavailable. The component and source
  records fix the measured software identity, but are not a pinned base image.

The [full environment snapshot](results/a3-2026-09-16/environment-full.json)
records installed package versions and SHA256 values for all 16 unique weight
shards (61,066,575,648 bytes), the model configuration and tokenizer files.

DeepEP is installed in a task-specific directory. Both its Python package and
native extension resolve there. The old system DeepEP package remains untouched.
An initial missing-TBE import and a subsequent mixed old/new native extension
failure were corrected before measurement; neither attempt contributes samples.

The first 13 of 28 correctness artifacts record an earlier benchmark-helper hash,
`d0579f7038c7e6505aaedf1a620ba337b0c7ca62b820e453b362107d45a4d0b6`.
The [recorded change](results/a3-2026-09-16/correctness-metadata-change.json)
reorders imports and records an unavailable Git diff as null instead of an empty
string's hash. Input generation, weight loading, forward execution and output
checking are unchanged. All 28 artifacts identify the same source archive and
use identical versions of `check_equivalence.py` and `workload.py`. All formal
performance runs use the final helper hash listed above.

## Workload and statistics

Rank 0 creates one seeded BF16 global input (CPU seed 20260922). Pure TP receives
identical complete inputs on both ranks. TP+EP receives disjoint contiguous
shards whose lengths sum to N, with no module padding. Every paired result has
matching input and logical checkpoint-weight hashes.

The full MoE block runs eagerly under `torch.inference_mode()`. Device Events
cover routing, expert computation and reduction/combine. Input distribution,
output gathering, pre-iteration barriers and final synchronization are outside
the device Event interval. The timer saves both rank times and uses the maximum
rank time in each iteration. Module measurements exclude attention, scheduling
and layout conversions at the attention/MoE boundary.

Each point uses 10 warmups and 30 samples, in three independent process launches.
Configuration order is TP/EP, EP/TP, then TP/EP. The table reports the median of
three run medians; spread is `(max(run medians)-min(run medians))/median`.
Small Decode cases have up to 14.8% TP run-to-run spread. Three process repeats
and these descriptive ranges do not provide a significance test.

At representative Prefill N=8192, host elapsed medians are 3.111 ms (TP) and
4.744 ms (EP), versus device Event medians 3.005 and 4.633 ms. At Decode N=512,
host medians are 1.009 and 1.321 ms, versus Event medians 0.903 and 1.219 ms.
The host values include synchronization and launch overhead and support the
same observed ordering at these points.

The [formal JSON files](results/a3-2026-09-16/formal/) retain every per-rank sample,
input hash, timing-script hash, setting and process repeat ID. Recompute the
[module summary](results/a3-2026-09-16/module-summary.json) with `summarize.py`.

## Prefill module latency

| Global N | TP ms | TP spread | TP+EP ms | EP spread | EP/TP latency |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.9459 | 3.3% | 1.7380 | 3.5% | 1.837 |
| 512 | 0.9443 | 3.4% | 1.7820 | 1.6% | 1.887 |
| 2048 | 1.3031 | 1.8% | 2.3601 | 0.9% | 1.811 |
| 4096 | 1.8345 | 0.9% | 3.1428 | 1.4% | 1.713 |
| 8192 | 3.0054 | 0.3% | 4.6330 | 0.7% | 1.542 |
| 16384 | 5.1781 | 0.2% | 7.5692 | 0.2% | 1.462 |
| 32768 | 9.3456 | 0.2% | 13.6090 | 0.5% | 1.456 |

## Decode module latency

| Global N | TP ms | TP spread | TP+EP ms | EP spread | EP/TP latency |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.9338 | 8.2% | 1.0812 | 3.3% | 1.158 |
| 8 | 0.9262 | 13.9% | 1.0942 | 0.6% | 1.181 |
| 32 | 0.9253 | 14.8% | 1.1777 | 3.7% | 1.273 |
| 64 | 0.8897 | 12.3% | 1.1655 | 3.1% | 1.310 |
| 128 | 0.8713 | 8.5% | 1.1550 | 3.3% | 1.326 |
| 256 | 0.8783 | 6.4% | 1.1386 | 5.9% | 1.296 |
| 512 | 0.9032 | 6.4% | 1.2195 | 1.4% | 1.350 |
| 1001 | 0.9358 | 5.9% | 1.4191 | 2.0% | 1.516 |

## Full-output correctness

All 14 TP/EP pairs passed the pre-existing criterion
`max(abs(TP-EP)) / (max(abs(TP)) + 1e-9) <= 0.02`.
Every EP rank's output is gathered in global token order, and both complete TP
replicas are checked. All TP replica differences were exactly zero. This is a
BF16 block consistency check; it is not a full-model quality evaluation or proof
that every discrepancy is solely rounding error.

| Global N | EP source rows | Seed | Biased experts | Prefill relative error | Decode relative error |
| --- | --- | --- | --- | --- | --- |
| 1001 | [501, 500] | 20260922 | 0 | 0.006410 | 0.006410 |
| 128 | [64, 64] | 20260922 | 0 | 0.003846 | 0.003846 |
| 1 | [1, 0] | 133 | 0 | 0.001524 | 0.001524 |
| 1 | [1, 0] | 20260922 | 0 | 0.001078 | 0.001078 |
| 4 | [2, 2] | 20260922 | 0 | 0.001374 | 0.001374 |
| 777 | [389, 388] | 20260922 | 0 | 0.006410 | 0.006410 |
| 777 | [389, 388] | 20260922 | 8 | 0.003788 | 0.003788 |

Global N=1 gives an actual empty source rank in both DeepEP phases. Odd N=777 and
1001 exercise uneven splits. The biased-router test scales the first 8 router
rows by 50 and the remaining rows by 0.01. Actual routing distributions are
recorded separately rather than inferred from these scale factors.

The CPU regression suite uses real two-process Gloo collectives and verifies that
corrupting EP rank 1, corrupting TP replica 1 or changing an input hash fails the
comparison. Its three tests passed locally. The registered CPU wrapper also
passed in the A3 container's software environment. Full comparison metadata is
in [correctness-summary.json](results/a3-2026-09-16/correctness-summary.json).

## Observed expert loads

Actual Top-K IDs matched between TP and reconstructed EP in both profiled cases.
The normal seeded input is not a uniform expert-load distribution.

| Phase / global N | Active experts | Maximum tokens per expert | Assignments to experts 0–63 | Assignments to experts 64–127 |
| --- | --- | --- | --- | --- |
| Prefill / 8192 | 128 | 1444 | 30231 | 35305 |
| Decode / 512 | 128 | 93 | 1893 | 2203 |
| Either phase / 1, seed 133 | 8 | 1 | 0 | 8 |

Each row sums to `N * 8` expert assignments. With seed 133, EP source rows are
`[1, 0]` while expert-rank assignments are `[0, 8]`: one rank has no source
input, and the other has no assigned expert work. Both phases passed complete
output comparison. [Routing JSON](results/a3-2026-09-16/routing/) preserves every
expert's count and every source rank's distribution.

## Profiling interpretation

Both ranks were profiled for 10 forwards after an unprofiled measurement at
Prefill N=8192 and Decode N=512. The table contains kernel-duration sums per
iteration, in ms. HCCL operation spans are kept separately from their enclosed
AICPU kernels. Their durations must not be added to the kernel sums. The parser
uses interval unions to expose overlap and reads actual names when the profiler
emits `Type=N/A`.

| Configuration | Rank | Matmul | Dispatch/combine family | All-reduce kernel | Routing/other | Kernel interval union |
| --- | --- | --- | --- | --- | --- | --- |
| deepep_decode | 0 | 0.370 | 0.304 | 0.000 | 0.052 | 0.726 |
| deepep_decode | 1 | 0.386 | 0.242 | 0.000 | 0.050 | 0.678 |
| deepep_prefill | 0 | 1.282 | 3.279 | 0.000 | 0.266 | 4.827 |
| deepep_prefill | 1 | 1.426 | 2.096 | 0.000 | 0.294 | 3.815 |
| none_decode | 0 | 0.371 | 0.000 | 0.147 | 0.147 | 0.648 |
| none_decode | 1 | 0.375 | 0.000 | 0.157 | 0.146 | 0.673 |
| none_prefill | 0 | 1.554 | 0.000 | 0.185 | 0.880 | 2.435 |
| none_prefill | 1 | 1.571 | 0.000 | 0.188 | 0.862 | 2.434 |

Prefill traces contain `CamMoeDispatchNormal`, `CamMoeCombineNormal`,
`DispatchLayout` and `NotifyDispatch`. Decode traces contain
`MoeLowLatencyDispatchV2` and `MoeLowLatencyCombineV2`. Both contain two grouped
matmul calls per forward. TP traces contain `allreduceAicpuKernel`.

Dispatch/combine operators include packing, communication, synchronization and
reduction work, so this category is not a measurement of communication alone.
The large inter-rank difference in Prefill `NotifyDispatch` also shows sensitivity
to rank arrival times under profiling. TP matmul sums are 1.55–1.57 ms versus EP
1.28–1.43 ms at Prefill N=8192; compute is not identical. At Decode N=512 the
matmul sums are approximately 0.37–0.39 ms in both configurations.

TP holds 128 experts with half-width intermediate projections (384); EP holds
64 full-width experts (768). For the same global routing, total arithmetic is
similar, but grouped-matmul shapes, routing and data movement differ. This helps
interpret the observed times without equating equal arithmetic with equal time.

Profiler duration sums exclude gaps and may overlap. The observed HCCL spans and
kernel times also vary under instrumentation; they are not substitutes for the
unprofiled latency sweep and do not quantitatively explain its whole gap. The compact [kernel CSV files](results/a3-2026-09-16/profile-kernels/)
preserve the profiler fields with line endings normalized to LF and
allow the profile summary to be recomputed with `summarize_profile.py`. See
[all-rank operator details](results/a3-2026-09-16/profile-summary.json).

## Mapping request length to actual MoE input

A separate service run installed `observe_tokens:make_hook` on
`model.layers.0.mlp` for both ranks. Each input length used four requests at
concurrency four, four distinct warmups and eight generated tokens, seed
20260922. These short runs observe layout semantics; their latency is excluded
from the performance results below. They are not a measurement of the batch-size
frequency in the 64-output-token performance runs. Startup forwards have a null
request-length label and are excluded from this table.

| Request input length | Observed phase | Global valid N | TP rows on each rank | EP rows on each rank |
| --- | --- | --- | --- | --- |
| 1024 | Prefill | 1024, 3072 | 1024, 3072 | 512, 1536 |
| 8192 | Prefill | 8192 | 8192 | 4096 |
| 16384 | Prefill | 8192 per chunk | 8192 | 4096 |
| 1024 or 8192 | Decode | 4 | 4 | 2 |
| 16384 | Decode | 1, 3 | 1, 3 | 1, 2 |

For an unchunked forward, N is the sum of the input tokens admitted by that
scheduler batch. In the observed 1024-token case the scheduler admits one and
then three requests, so the forwards contain 1024 and 3072 tokens; concurrency
four does not guarantee one 4096-token forward. The 8192-token chunk limit splits
a 16384-token request into multiple forwards. Increasing request length can thus
increase the number of MoE calls without increasing their maximum N.

During ordinary Decode each active request contributes one valid token. Longer
context increases attention work and influences scheduling, but does not multiply
MoE valid input rows by context length. In these short observation runs, some
requests finish before the remaining long prompts enter Decode, producing N=1
and N=3.

Serving EP pads odd global counts: global N=1 has one physical row on each rank,
and N=3 has two on each rank. The runtime `batch_size` field can also include
padding (2 or 4 here), so it is not the active request count in those cases. The
isolated module benchmark instead splits only valid tokens, e.g. `[1, 0]` and
`[2, 1]`, and excludes the attention/MoE layout conversions. This difference is
why its latency ratios must not be treated as predictions of serving speed.
The [rank-local layout records](results/a3-2026-09-16/layout/) retain global valid
counts and physical row counts separately.

## End-to-end serving

The full 48-layer model runs with TP=2, EP=1 or 2, the Ascend attention backend,
BF16, CUDA/NPU graph execution disabled, radix cache disabled, chunked Prefill
size 8192, maximum concurrent requests 4 and static memory fraction 0.8.
DeepEP uses automatic normal/low-latency mode selection. Each configuration has
three independently started services, ordered TP/EP, EP/TP, then TP/EP.
Each input length sends 16 requests at concurrency 4 and generates exactly 64
tokens per request. Four distinct warmup requests use the same length and
concurrency before measurement. Paired runs have identical prompt token hashes.

All 18 measured files contain 16 complete requests and 1024 generated tokens.
Throughput is total generated tokens divided by measured wall time. TTFT starts
after the client acquires a concurrency slot. The client computes TPOT as
`(response_end_time - first_token_time) / (output_tokens - 1)`. The end timestamp
is taken after the response stream closes, so this measurement includes stream
completion overhead and interference from other requests, including their
Prefill. It is not isolated Decode kernel latency.
The table reports the median of three run-level medians (or throughputs), not
a pooled request distribution. Warmup traffic is excluded.

| Input tokens | TP output tokens/s | EP output tokens/s | TP TTFT ms | EP TTFT ms | TP TPOT ms | EP TPOT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 48.11 | 34.22 | 256.4 | 534.9 | 81.17 | 111.60 |
| 8192 | 40.24 | 26.29 | 889.2 | 2244.9 | 86.59 | 116.21 |
| 16384 | 23.31 | 15.51 | 2614.7 | 4511.1 | 92.83 | 125.04 |

Pure TP has higher observed throughput and lower TTFT/TPOT medians at these
three lengths. TP throughput run-to-run relative ranges are 16.2%, 9.1% and 6.3%;
EP ranges are 3.1%, 2.0% and 1.4%. These three-repeat descriptive measurements
do not establish statistical significance. Longer inputs also change attention,
scheduling and the number of chunked Prefill forwards. The isolated MoE ratio
cannot quantify their separate effects or predict total serving speed.

The first cold service required about 15 minutes to read the model from shared
NFS. Model hashing warmed the shared filesystem cache; later startup loads were
faster. Startup and hashing are excluded from request timing, and the hashing
job completed before formal runs. This is a warm-service benchmark, not a cold
startup benchmark. Service readiness waiting was extended before adoption of
the original service; no incomplete request result is retained as a valid run.

[Individual request results](results/a3-2026-09-16/e2e/),
[exact server launch commands](results/a3-2026-09-16/server-commands/) and
[all run values and variation](results/a3-2026-09-16/serving-summary.json) are
included. Recompute the summary with `summarize_serving.py` as described in the
[reproduction instructions](README.md).

## Remaining scope

The results validate one Qwen3 MoE model on two compute devices of one A3 card.
They do not cover larger EP sizes, cross-card/cross-node communication,
quantization, graph-mode optimization or full-model quality. Maintainer review, CI and task acceptance are separate from these measurements.
