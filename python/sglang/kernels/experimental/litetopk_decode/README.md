# Experimental LiteTopK decode on B200

This opt-in path computes FP8 decode scores and a coarse histogram in one
DeepGEMM kernel, then selects exact FP32 TopK and maps the result to physical KV
slots. SGLang's default dispatch and dependency pins are unchanged. Unsupported
inputs are rejected by this experimental API; there is no automatic serving hook.

A call launches **three kernel nodes**: one score/histogram producer, followed by
two device-guarded selector nodes on joined streams. Only the selected small- or
large-batch selector performs selection. The scorer still writes dense logits;
the histogram narrows the selector's candidate work, with exact FP32 refinement
and fallback when the certificate is insufficient.

The selector implementation is directly readable in
[selectors/selector.py](selectors/selector.py). The short
[fused_source.py](selectors/fused_source.py) derives the small/large variants;
[vendor/gvr2_topk_decode.py](vendor/gvr2_topk_decode.py) contains only their required
NVIDIA-licensed helpers and base-class methods. Unused GVR kernels, compiler/cache
entry points and the superseded standalone producer are omitted.
The two builds specialize unused generic branches and share one exact radix
implementation for whole rows, staged candidates and boundary keys. Their
qualified binaries remain byte-identical to the original measured selectors.

## Build

Prepare official SGL DeepGEMM main
`1853080cf74589228fcd5dc333197c2ca6df49cf` with the five-file companion patch
(+236 / -15), then build it into a private directory using
[deepgemm/README.md](deepgemm/README.md). The patch and builder are included because
DeepGEMM is a separate repository. Nothing is globally installed. The default
output is `build/fused/deepgemm`.

Then build the selectors from the SGLang repository root:

```sh
CUDA_VISIBLE_DEVICES= MAX_JOBS=1 CUTE_DSL_ARCH=sm_100a \
  python3 python/sglang/kernels/experimental/litetopk_decode/build_fused.py
```

`--check-sources` checks the pinned source hashes without compilation;
`--output-dir` selects an alternative output directory. The default selector
output is `build/fused/{small,large}/select.so` plus `manifest.json`. The manifest
records source, dependency and rebuilt library hashes. Changed toolchains require
fresh qualification; source similarity alone is not a binary-equivalence claim.

## Explicit API

```python
from sglang.kernels.experimental.litetopk_decode.fused import FusedDecodePlan

# q: float8_e4m3fn [B,1,32,128]; weights: float32 [B,32]
# cache: native packed uint8 [pages,64,1,132]
# lengths: int32 [B]; table: int32 [B,ceil(max_context_len/64)]
plan = FusedDecodePlan(q, cache, weights, max_context_len=1048576)
schedule = plan.metadata(lengths)
physical_topk = plan(table, lengths, schedule)  # int32 [B,2048], unsorted
```

Pass `deepgemm_package` and `selector_dir` to use custom build directories. The
private loader verifies artifact hashes and imports under a separate module
alias, preserving the existing `deep_gemm` module and function bindings.

Scope: 148-SM B200, Q1/H32/D128, page64, K2048, contiguous operands, actual B equal
to any integer in 1..128, maximum context length 1..1048576. Native preconditions
for valid device lengths, page IDs and schedule contents apply. This wrapper uses
no indices tensor; the low-level companion API has a wider contract.

Create and warm the plan before CUDA graph capture. Update inputs in place and
rebuild metadata whenever lengths change; copy metadata into the captured schedule
buffer before graph replay. Keep the plan and captured inputs alive, and serialize
calls/replays sharing a plan. Output is overwritten on subsequent calls. Captured
storage remains owned until plan destruction; use a new plan when discarding old
captures. Select the plan's CUDA device before construction and calls; use one
process per device.

Each actual B uses a matching plan/graph, with the same two selector artifacts.
This API does not change the device-side active batch size inside an existing
graph. Allocations, input updates and metadata construction are outside the
previously measured score+TopK graph latency.

## Validation

[smoke_fused.py](smoke_fused.py) checks eager and graph calls, fixed-address input
updates, zero/short/boundary lengths and recovery. It compares all live FP32 score
bits to both the private original scorer and the separately resolved default
scorer, validates exact TopK and physical IDs, checks all histogram bins and
workspace restoration, and verifies original module/function bindings.

```sh
python3 python/sglang/kernels/experimental/litetopk_decode/smoke_fused.py \
  --all-batches --length 4097 --max-context-len 4097 --output smoke.json
```

Run in an environment with the original `deep_gemm` available for the isolation
check, after both private builds. Keep the JSON receipt and manifests. The prior
40-shape latency matrix measures a single-layer score+TopK graph with shared
physical KV pages, not serving throughput; it is distinct from these correctness
checks. Historical B1/B2/B4/B8/B16 standalone-producer receipts do not qualify this
fusion path. The DeepGEMM companion README describes its separate kernel tests.
