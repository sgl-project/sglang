# DeepSeek V4.1 shared expert and router fusion

This opt-in AMD optimization combines the shared MLP and router into two GPU
launches for small DSpark verification batches. The current public-main port
is undergoing qualification. Historical results below do not qualify this new
revision or establish incremental benefit over other pending upstream PRs.

## Motivation and implementation

The shared MLP and router consume the same hidden states independently. Stage
one combines native MXFP8 shared gate/up workgroups with BF16 router split-K
workgroups. Stage two combines shared activation/down with the ordinary router
scoring and top-k. There are no inter-workgroup spin barriers. Routed expert
GEMMs, sorting, attention and collectives are unchanged.

The shared down kernel retains BF16 activation rounding and the per-32 MXFP8
grid, but changes the FP32 reduction order. Outputs are not promised bit-exact.
An exact BF16 representation of the quantized down weight costs 5.625 MiB per
eligible layer per rank. It is prepared before graph capture and invalidated
when the source weight or scale changes. Preparing all 40 eligible target
layers retains 225 MiB per GPU. One public C2 pair observed warm tokenizer
readiness at 90.39s native versus 96.52s fused, with target-verify graph setup
18.03s versus 24.34s. Compilation/cache effects and co-location confound this
startup comparison; retained tensor storage is not process peak memory.

## Supported scope

- gfx950 (MI355X tested), DeepSeek-V4.1-Flash, TP4/EP1.
- Target verification only, M=6 or M=12, hidden size 5120, shared width 576.
- Native MXFP8 shared weights, BF16 router, 384 experts and top-6 sqrtsoftplus.
- Hash/draft/prefill, other shapes or layouts, EP, EPLB, simulated routing,
  shared-expert TP1 and a shared expert merged into routed GEMMs fall back.

Enable with `SGLANG_DSV41_SHARED_ROUTER_FUSION=1`; default is off. M is the
kernel row count, not the client concurrency. Small tail batches can engage
even during a high-concurrency workload.

## Numerical tests and microbenchmark

See [Reproduction commands](REPRODUCE.md) for portable server and client commands,
dataset hashes, evaluation protocol and untimed device-trace collection.

Install SGLang and its ROCm dependencies from source according to the upstream
[contribution guide](https://docs.sglang.io/docs/developer_guide/contribution_guide).
Run on a gfx950 GPU from this repository root:

```bash
python3 test/registered/amd/test_shared_router_gfx950.py -v
python3 benchmark/shared_router/bench.py --output shared-router-micro.json
python3 benchmark/shared_router/checkpoint_test.py \
  --model /path/to/DeepSeek-V4.1-Flash --output checkpoint-numerics.json
```

The CI-registered tests cover exact native shared projection, router IDs and
weights, shared-output tolerances, nonuniform power-of-two block scales, native
weight-layout decoding, alternating graph shapes, zero padded rows, dispatch
fallbacks and derived-weight invalidation. The microbenchmark measures the
complete shared/router sequence under graph replay, not the entire MoE layer
or end-to-end serving. It currently uses hot-cache synthetic operands.

The optional checkpoint test loads only the shared/router tensors from layers
6, 18 and 36. It covers all four TP shards at M6/M12, exact down-weight decoding,
top-k agreement and changed-input graph replay. Activations remain synthetic;
this is not a captured-input test or a substitute for GSM8K. Tensor hashes and
derived-weight memory/setup measurements are included in its JSON output.

## Current public-environment C2 evidence

Pinned public main is `f03a183719c9e7fb2bd528ed5c7e3b627d9df92d`.
Both arms use the same source, public image and public AITER from REPRODUCE.md,
with only horizontal fusion off/on. Three measured repetitions follow warmup;
the table reports the median of the per-repetition P50 statistics.

| Common source | Native P50 TPOT (ms) | Fused P50 TPOT (ms) | Change | Output tokens/s change | Full GSM8K correct, native/fused |
| --- | ---: | ---: | ---: | ---: | --- |
| Main + this patch, `76d9a6c2a6` | 2.994574 | 2.923689 | -2.367% | +2.254% | 1284/1287 of 1319 |
| Same + #42055, `56f8583e02` | 2.965578 | 2.927802 | -1.274% | +1.139% | 1288/1286 of 1319 |

Both accuracy pairs had zero request errors and passed the local 0.5 pp loss
tolerance. Accuracy uses real acceptance, while timing fixes synthetic AL3.51.
Four-rank untimed traces confirm both fused stages in graph replay. These are
candidate-first sequential pairs; reverse-order and all-concurrency checks
remain in progress. Do not add the two gains or claim statistical significance.
The preparation tree changes only formatting and reproduction support relative
to the main measured runtime: Python AST is identical, C++/Triton unchanged.

## Historical evidence

On the earlier isolated source `aca7cf81d6`, compared with serving prerequisite
`938a02853a`, TP4/EP1 DSpark block 5, 8K/1K range 0.8, synthetic acceptance length
3.51 (`match-expected`), three measured repetitions after warmup:

| Concurrency | P50 TPOT baseline to candidate in ms | Change | Output throughput change |
|---:|---|---:|---:|
| 1 | 2.3269 to 2.2559 | -3.05% | +2.70% |
| 2 | 3.0783 to 3.0070 | -2.32% | +2.25% |
| 4 | 4.1297 to 4.1342 | +0.11% | -0.04% |
| 8 | 5.7265 to 5.7602 | +0.59% | -0.40% |
| 16 | 8.8082 to 8.7977 | -0.12% | +0.83% |
| 32 | 14.5803 to 14.6086 | +0.19% | -0.69% |

Values are medians of repetition statistics, not pooled percentiles. Paired
servers ran sequentially on one allocation; other jobs could share the node.
Do not claim universal no-regression or statistical significance from one pair.

Accuracy was evaluated separately with real acceptance and all 1319 GSM8K
questions. Baseline/candidate correct counts were 1287/1285, 1286/1281,
1283/1282, 1284/1284, 1283/1285 and 1286/1285 at C1/2/4/8/16/32. All passed the
local 0.5-percentage-point loss tolerance; that tolerance is not an upstream
standard or proof of numerical equivalence. Synthetic outputs are never used
as accuracy evidence. Current-main all-concurrency accuracy and reverse-order
confirmation remain pending.

## Prior art and qualification still required

- [AITER 5321](https://github.com/ROCm/aiter/pull/5321) and
  [AITER 4504](https://github.com/ROCm/aiter/pull/4504) explore Kimi shared/router
  projection fusion. This draft does not claim invention of horizontal fusion.
- [SGLang 42055](https://github.com/sgl-project/sglang/pull/42055) optimizes the
  same activation-quantization boundaries. The C2 incremental result is above;
  remaining concurrency and reverse-order checks are still required.
- [SGLang 42011](https://github.com/sgl-project/sglang/pull/42011) optimizes an
  alternative shared expert folded into routed MoE; it is not simply additive.

Before submission: complete all-concurrency and reverse-order qualification,
review the folded-expert alternative, and attach raw evidence. Public full-model
C2 reproduction and per-rank startup/memory logs are available; process peak
memory remains unmeasured. Real-checkpoint cases and registered fallback tests
are in place; their scope is described above.
