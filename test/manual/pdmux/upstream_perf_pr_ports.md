# DSV4.1 PDMux upstream performance ports

These ports target H100/SM90, TP8/DP8/EP8, DSPARK, HiCache, `dsv4` attention
and `flashinfer_mxfp4` MoE. They retain the existing PDMux lane submission and
communication order. New SM90 paths are opt-in pending GPU acceptance.

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

To enable all three ports, set all three environment variables above to `1`
before the unchanged serve command. All default to `0` on this branch.

## Combination acceptance

Run each optimization alone and then together, with the same checkpoint,
DSPARK block, SM allocation and workload. Include zero-work DP ranks, graph
padding, short/long prompts, cold/warm HiCache and split continuations. Confirm
greedy output/quality, peak memory, TTFT, ITL and throughput, and sample a trace
to verify the lanes still overlap.

Repeat the historical chunk1024 / `max_split_forward_layers: 2` hang case at
c10 for at least 30 minutes before c24. The user's current YAML requests 62
prefill SMs (the driver may round); its omitted layer cap and chunk size are
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
