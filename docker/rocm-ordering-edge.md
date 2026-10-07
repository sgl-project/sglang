# Optional ROCm device-resident ordering-edge runtime build

The device-resident ordering-edge feature from
[ROCm/rocm-systems#11212](https://github.com/ROCm/rocm-systems/pull/11212) is built directly by
`rocm.Dockerfile` as an **opt-in** variant, mirroring what vLLM did in
[vllm-project/vllm#55099](https://github.com/vllm-project/vllm/pull/55099). It is integrated into
the main Dockerfile (rather than a standalone image-layering recipe) so the daily image pipeline
can build it without a separate step.

## Why

On ROCm, cross-stream / hipgraph dependency polls default to host-visible memory and cross PCIe.
With `GPU_MAX_HW_QUEUES > 4` this shows up as a low-concurrency decode regression. The #11212
series adds `hsa_amd_signal_create_v2` + `HSA_AMD_SIGNAL_CREATE_DEVICE_MEM_VALUE_WORD` (ROCr) and
the CLR side that names a device-resident ordering-edge value word on cross-queue barrier deps, so
those polls come from device-local VRAM instead.

## What it does

When enabled, `rocm.Dockerfile` rebuilds ROCr (`libhsa-runtime64`) and CLR/HIP (`libamdhip64`) from
the exact `rocm-systems` commit the ROCm-10.0 base was built from, with the #11212 series
cherry-picked on top (`docker/ordering_edge_11212_on_rocm10.patch`), then swaps those two shared
libraries into the image and sets `ROCPROFILER_QUEUE_INTERPOSITION=0`.

Starting from the image's own base commit keeps ROCr at its native version (ABI-compatible with the
image's `rocminfo`/`aiter` arch detection). `ROCM_KPACK_ENABLED=ON` is required so the rebuilt
`libamdhip64` remains compatible with the image's kpack device-code archives.

## Build

The feature is gated by the `ORDERING_EDGE_SRC` build ARG, which selects the stage that supplies
the patched libs. It defaults to `ordering_edge_none` (an empty no-op), so a normal build produces
an unchanged image. Set it to `ordering_edge_build` to compile and swap in the patched runtimes.
Only meaningful for the `*-rocm1000` flavors.

```
docker build \
  --build-arg GPU_ARCH=gfx950-rocm1000 \
  --build-arg ORDERING_EDGE_SRC=ordering_edge_build \
  -t <image>-edge -f rocm.Dockerfile .
```

`ROCM_RUNTIME_COMMIT` defaults to the rocm-systems commit the current ROCm-10.0 base was built
from. If you retarget the base to a different ROCm SDK, **retarget `ROCM_RUNTIME_COMMIT` together**
(a mismatched commit risks an ABI break).

## Runtime revert (no rebuild)

```
DEBUG_CLR_DISABLE_ORDERING_EDGE=1
```
