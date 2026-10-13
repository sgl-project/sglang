# DSV4.1 PDMux upstream performance ports

These ports target H100/H20-3e (SM90), TP8/DP8/EP8, DSPARK, HiCache, `dsv4` attention
and `flashinfer_mxfp4` MoE. They retain the existing PDMux lane submission and
communication order. The first three kernel ports are opt-in; the #41098
GEMM tuning tables are selected automatically for matching H20-3e shapes.

## PR #42365: fused prefill indexer scoring

Source: [sgl-project/sglang#42365](https://github.com/sgl-project/sglang/pull/42365),
head `71bbd5402845f6651080339814c266d505dd9bab`.

Fuses QK, ReLU, head weighting/reduction and the causal mask. The port preserves
CPU request geometry and uses the calling stream, without a helper stream or
shared mutable scratch. Dispatch requires SM90, BF16 operands and head dim 128;
other inputs keep the existing scorer. Head reduction rounding can change
cutoff ties, so top-k overlap alone does not establish model accuracy.

Add before the existing serve command:

```bash
export SGLANG_OPT_USE_FUSED_DSV41_INDEXER_SCORES=1
```

Set it to `0` to restore existing scoring. `SGLANG_DSV41_TORCH_PREFILL_INDEXER=1`
also forces the Torch prefill oracle. No new serve CLI argument is needed.

GPU kernel checks:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/kernels/ops/attention/dsv4/test_fp4_indexer_prefill.py \
  test/registered/unit/layers/test_dsv41_prefill_submission.py
```

## PR #41294: fused low-ratio compressor on Hopper

Source: [sgl-project/sglang#41294](https://github.com/sgl-project/sglang/pull/41294),
head `8ab1eefaed6d3eed8f18a3ad6ed5e36c02e7663d`.

Enables the existing c1/c2 fused paths on SM90, including split prefill and
equal-block DSPARK verify. CPU, unsupported CUDA devices and existing SM100/HIP
admission retain their previous behavior. No new graph tiers are introduced.

```bash
export SGLANG_OPT_USE_FUSED_DSV41_COMPRESS_SM90=1
```

Set to `0` and **restart** to roll back: the flag fixes the c2 packed weight
layout before loading and graph capture, and is not a runtime toggle.

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/test_dsv41_compressor_dispatch.py \
  test/registered/attention/unittests/dsv4/test_dsv41_fused_compress.py \
  test/registered/kernels/ops/attention/dsv4/test_c2_prefill_ring.py
```

GPU acceptance includes odd/even pair starts, unfinished pairs crossing chunk
boundaries, padded rows, shared KV source layers and prefill-to-decode migration.

## PR #40860: compact SM90 candidate indexers

Source: [sgl-project/sglang#40860](https://github.com/sgl-project/sglang/pull/40860),
head `f983aca2d2f04f87bb504235728156441888d6f1`.

Source layers retain full scoring (including #42365 when enabled). Consumers
score compact candidate blocks in prefill and decode/verify. The port adapts
the current `Selection` interface and CPU request geometry, and keeps candidate
metadata on the owning forward batch. Compact scan lengths remain distinct
from causal K lengths; output mapping sorts logical positions and clears padding.

```bash
export SGLANG_OPT_USE_DSV41_SM90_CANDIDATES=1
```

Set before server startup; use `0` and restart to restore the dense candidate
path and recapture graphs. Prefill requires H32/D128 BF16 and 8-position blocks;
unsupported shapes and the Torch prefill oracle retain the existing scorer.
The upstream SM90 graph-variant expansion is deliberately excluded: this port
does not alter captured batch sizes, DSPARK block widths or lane graph selection.

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/kernels/ops/attention/test_candidate_bf16_mqa.py \
  test/registered/kernels/ops/attention/test_dsv41_triton_candidate_indexer.py \
  test/registered/kernels/ops/attention/test_fp4_indexer.py::test_hopper_indexer_backends_replay \
  test/registered/unit/layers/test_dsv41_candidate_blocks.py \
  test/registered/unit/layers/test_dsv41_candidate_graph_variants.py
```

Tests include request/tail coordinate mapping, empty candidates, causal holes,
changed-input CUDA graph replay, independent streams sharing a read-only cache,
and the existing dense path. Full-model PDMux GPU validation is still required.

These three environment variables default to `0`. For the user's H20-3e
workload, keep #42365 + #40860 enabled and #41294 disabled when evaluating
#41098: the reported total-throughput gain was 21.1% for that pair, whereas
enabling all three regressed 18.0% relative to the original baseline.

## PR #41098: H20-3e block-FP8 GEMM tuning tables

Source: [sgl-project/sglang#41098](https://github.com/sgl-project/sglang/pull/41098),
head `c6bf06b645c58582ac52d8bf314e3e6ca60c6120`.

Adds six byte-identical upstream JSON files under
`python/sglang/kernels/ops/gemm/configs/`. Upstream used the old
`kernels/ops/quantization/configs/` directory; the current branch's
`get_w8a8_block_fp8_configs` reads from `gemm/configs/`, so applying the old
paths unchanged would silently leave the default configuration active.

The files cover `(N, K)` = `(1792, 5120)`, `(25600, 6144)`, `(4096, 1280)`,
`(5120, 1024)`, `(5120, 288)` and `(576, 5120)`, all with quantization block
`[32, 32]`. Each file contains 14 token-count entries, from M=1 to M=8192;
the existing runtime picks the nearest M. These are GEMM tile choices, not
DSPARK step sizes or CUDA Graph batch-size tiers.

### What the optimization changes

For `A[M, K] @ B[N, K].T`, M is the number of token rows processed by that
individual GEMM, not the full conversation length. A 130k cached conversation
can therefore still execute small-M decode/verify GEMMs. DP8 also distributes
requests across ranks, so global concurrency is not the local M.

The default kernel uses one generic tile layout. These tables choose tile
sizes, pipeline stages and existing Hopper options separately for each shape
and token count:

- `SWAP_AB` computes the transposed product and transposes the result back.
  This changes the internal tensor-core tile orientation for small-M work.
- `SPLIT_K` lets multiple GPU programs compute different parts of the K sum,
  then reduces their FP32 partial results. This can expose more parallel work
  when the output has too few tiles to occupy the GPU, at the cost of scratch
  memory and an extra reduction kernel.

The branch already implements both options. Scratch is allocated per call on
the calling stream; the port only adds tuning data. The tables were not tuned
for the user's 62-SM prefill stream competing with full-device decode, so
their end-to-end effect still needs measurement with PDMux running.

### Activation and verification

No new serve argument or environment variable is required. Install this
revision and restart the server to reload configs and capture graphs. Lookup
requires the exact device name `NVIDIA H20-3e`, a matching `(N, K)` and block
`[32, 32]`; plain `NVIDIA H20` and unmatched shapes keep their existing paths.
The existing `kernels/**/*` package-data rule includes these JSON files.

When a matching Triton FP8 operation executes, startup/warmup logs should show
`Using configuration from .../gemm/configs/N=...,device_name=NVIDIA_H20-3e,...`
for `W8A8 Block FP8 kernel`. Confirm the deployed installation's path in that
message. Shapes that never execute need not produce a lookup message.
The cached-weight BF16 fast path can bypass these tables for eligible M>=64
operations; installing a table does not force all linear layers onto FP8.

Keep the existing A+C baseline for the comparison:

```bash
export SGLANG_OPT_USE_FUSED_DSV41_INDEXER_SCORES=1
export SGLANG_OPT_USE_DSV41_SM90_CANDIDATES=1
export SGLANG_OPT_USE_FUSED_DSV41_COMPRESS_SM90=0
```

Compare the parent revision and this port with identical checkpoints,
cache warmup/hit rates, request sequence, concurrency and PDMux YAML (62
prefill SMs, `max_split_forward_layers: 4`). For the user's 130k-history /
400-output multi-turn workload, measure total throughput after warmup and
repeat runs; most history is cached, so do not treat all 130k tokens as new
prefill work. Also verify output quality and DSPARK acceptance: split-K changes
the accumulation order and can change low BF16 bits. Upstream's eager/graph
checks cover M=6/12/24 on six shapes, but do not validate this PDMux workload.
To undo the port, revert its commit and restart the server.

Local validation (2026-10-13, CPU-only): all six files match the pinned
upstream Git blob hashes. The production loader function, executed with
mocked device identification, loads all 84 entries unchanged from the new
paths. All entries match the selected kernel's parameter names (46 Hopper,
38 generic); JSON structure, scale-block compatibility, split-K constraints,
device/shape/block fallbacks, compile bypass and package-data inclusion passed.
This checks configuration integration, not GPU numerical accuracy, graph
replay, PDMux liveness or throughput.

## Combination acceptance

Run each optimization alone and then together, with the same checkpoint,
DSPARK block, SM allocation and workload. Include zero-work DP ranks, graph
padding, short/long prompts, cold/warm HiCache and split continuations. Confirm
greedy output/quality, peak memory, TTFT, ITL and throughput, and sample a trace
to verify the lanes still overlap.

Repeat the historical chunk1024 / `max_split_forward_layers: 2` hang case at
c10 for at least 30 minutes before c24. The user's current YAML requests 62
prefill SMs (the driver may round) and `max_split_forward_layers: 4`; this is
not that historical configuration. CPU tests do not verify CUDA numerics,
multi-rank liveness or throughput.

## Local validation (2026-10-11)

- PDMux, DP, DSPARK, HiCache, graph policy and port regression selection:
  470 passed, 309 subtests passed, 91 skipped on the CPU-only macOS host.
- Environment and kernel inventory checks: 77 passed. One subprocess import
  check (`test_amx_backend_probe_is_lazy[srt/utils/common.py]`) fails because
  Triton is not installed; importing the identical file from baseline
  `a64f33f089` reproduces that failure. This check is not claimed as passed.
- Ruff, isort, Python AST, registered-test lint and whitespace checks passed.
- CUDA kernel tests, full-model accuracy, multi-rank liveness and throughput
  remain unverified. In particular, these ports do not establish a hang fix.
