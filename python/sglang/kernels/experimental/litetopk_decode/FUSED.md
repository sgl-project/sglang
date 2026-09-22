# Opt-in DeepGEMM main score + histogram + exact TopK

This addition targets `Heisenberg-Yin/sglang` branch `litetop-decode` at
`948945d84b398c9874942d0a9ad200bf44bfe9cd`. The original standalone producer,
selector, build entry, dispatch constants and SGLang dependency pins are unchanged.
Importing the original package does not import or enable this path.

The producer is official SGL DeepGEMM main
`1853080cf74589228fcd5dc333197c2ca6df49cf` with the companion five-file
score/histogram patch (+236 / -15). DeepGEMM lives in another repository, so
this SGLang patch carries the [companion patch](deepgemm/opt_in_histogram.patch)
and an isolated builder instead of copying or replacing its source tree.

The selector uses the measured guarded small/large pair with the same concurrent
CUDA event dependencies. Its source is reproduced from the existing licensed
selector/vendor files by [fused_source.py](selectors/fused_source.py). Both
generated sources must match the frozen SHA256 values before compilation;
the builder records the actual rebuilt library hashes separately. This retains
the exact FP32 fallback and physical KV mapping used by the measured chain.

## Build

First prepare the pinned external DeepGEMM checkout and build the private package
as described in [deepgemm/README.md](deepgemm/README.md). The default output is
`build/fused/deepgemm`; no package is globally installed. Then, from the SGLang
repository root, build the two selectors with the GPU hidden:

```sh
CUDA_VISIBLE_DEVICES= MAX_JOBS=1 CUTE_DSL_ARCH=sm_100a \
  python3 python/sglang/kernels/experimental/litetopk_decode/build_fused.py
```

The default output is `build/fused/{small,large}/select.so` and
`build/fused/manifest.json`. Both builders accept an explicit `--output-dir`.
The original `build.py --all` remains available for the original path.
`build_fused.py --check-sources` validates source reproduction without compiling.
Compilation uses the CUDA 13/PyTorch/CuTe/TVM-FFI environment recorded by the build
manifests. Changed compiler or dependency versions require fresh qualification.

## Use before CUDA graph capture

```python
from sglang.kernels.experimental.litetopk_decode.fused import FusedDecodePlan

# q: float8_e4m3fn [B,1,32,128]; weights: float32 [B,32]
# cache: native packed uint8 [pages,64,1,132]
# lengths: int32 [B]; table: int32 [B,ceil(max_context_len/64)]
plan = FusedDecodePlan(q, cache, weights, max_context_len=1048576)
schedule = plan.metadata(lengths)
physical_topk = plan(table, lengths, schedule)  # int32 [B,2048]
```

For custom output directories, pass `deepgemm_package=<directory containing the
DeepGEMM manifest>` and `selector_dir=<directory containing the selector manifest>`
to the constructor. The private loader validates the manifest and artifact hashes,
then imports under a private module alias. It does not replace
`sys.modules["deep_gemm"]`, change `sys.path` or alter the installed
`sgl-deep-gemm==0.2.0` package. Default SGLang dispatch remains opt-in at the caller;
this patch does not install a model-specific serving hook.

Supported composition: 148-SM B200, Q=1, H32/D128, page64, K2048, contiguous
operands, actual batch size **any integer from 1 to 128**, and maximum length
1..1048576. The existing DeepGEMM valid device-value preconditions for lengths,
page IDs and schedule apply. The low-level companion producer supports additional
page sizes and an indices path; this wrapper intentionally exposes only the
qualified page64/no-indices composition. Output order is not a sorted-order API.

A plan owns its Q/cache/weights references and its output, histogram, diagnostics
and workspace. Update input contents in place. Rebuild schedule metadata whenever
lengths change; for an existing graph, copy it into the captured schedule tensor
before replay. Warm the JIT and the call before capture. Keep the plan and captured
inputs alive while graphs exist, and serialize calls/replays that share a plan.
Captured buffers remain owned until plan destruction; use a fresh plan when
discarding and rebuilding captures to release the old graph's retained storage.
Select the plan's CUDA device before construction, metadata creation and calls.
Each private DeepGEMM instance is bound to one device; use one process per device.
The output is overwritten on subsequent calls. Allocation/input updates/schedule
construction are not part of the previously reported kernel latency.

Each plan and graph retains its actual B. A different B uses a matching plan and
capture; no per-B selector compilation or power-of-two-only dispatch is needed.
This wrapper does not expose a device-side active-batch scalar that changes B
inside one already captured graph. The internal selector scalar is fixed to B.

## Qualification and performance provenance

[smoke_fused.py](smoke_fused.py) exercises this packaged interface with eager and
CUDA graph calls, fixed-address input changes, short/zero/boundary lengths and
recovery. It compares live score bits to both the private old scorer and the
already-installed default scorer, validates exact TopK values and physical IDs,
and checks histogram/workspace restoration. It also checks that the default
module and function bindings remain unchanged. Run it after both builds; retain
its JSON receipt with the build manifests.

The earlier 40-shape latency matrix used this producer/selector computation at
B=1/2/4/8/16/32/64/128 and L=131072/262144/524288/786432/1048321. Its results
are from the frozen standalone main-fusion benchmark, not a new end-to-end run
of this SGLang branch. Each value is the median of two run medians, for a
single-layer score+TopK graph with shared physical KV pages. Two vLLM reference
points failed exact TopK qualification and remain NA. Repackaging alone does
not establish serving-engine throughput or new full-matrix latency numbers.

The separate DeepGEMM companion's qualification and unchanged-default-path
evidence are described in its README. Those historical results and this new
wrapper's smoke receipt are different evidence and should remain separately
identified. The source files and default dependency versions of the original
SGLang route are unchanged by this addition.
