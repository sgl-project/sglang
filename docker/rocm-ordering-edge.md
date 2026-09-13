# Optional ROCm device-resident ordering-edge runtime build

`rocm-ordering-edge.Dockerfile` is an **optional** build layered on top of an existing SGLang
ROCm image. It brings the device-resident ordering-edge feature from
[ROCm/rocm-systems#11212](https://github.com/ROCm/rocm-systems/pull/11212) into the runtime,
mirroring what vLLM did in [vllm-project/vllm#55099](https://github.com/vllm-project/vllm/pull/55099).

## Why

On ROCm, cross-stream / hipgraph dependency polls default to host-visible memory and cross PCIe.
With `GPU_MAX_HW_QUEUES > 4` this shows up as a low-concurrency decode regression. The #11212
series adds `hsa_amd_signal_create_v2` + `HSA_AMD_SIGNAL_CREATE_DEVICE_MEM_VALUE_WORD` (ROCr) and
the CLR side that names a device-resident ordering-edge value word on cross-queue barrier deps, so
those polls come from device-local VRAM instead.

## What it does

Rebuilds ROCr (`libhsa-runtime64`) and CLR/HIP (`libamdhip64`) from the exact `rocm-systems`
commit the base image was built from, with the #11212 series cherry-picked on top, then swaps
those two shared libraries into the image and sets `ROCPROFILER_QUEUE_INTERPOSITION=0`.

Starting from the image's own base commit keeps ROCr at its native version (ABI-compatible with the
image's `rocminfo`/`aiter` arch detection). `ROCM_KPACK_ENABLED=ON` is required so the rebuilt
`libamdhip64` remains compatible with the image's kpack device-code archives.

## Build

The build ARGs must match the base image. Defaults target the validated
`...-20260909` base; **retarget both `BASE_IMAGE` and `ROCM_RUNTIME_COMMIT` together** for any
other base (a mismatched commit risks an ABI break):

```
DOCKER_BUILDKIT=0 docker build -f docker/rocm-ordering-edge.Dockerfile \
  --build-arg BASE_IMAGE=<sglang-rocm base image> \
  --build-arg ROCM_RUNTIME_COMMIT=<rocm-systems commit that base was built from> \
  -t <sglang-rocm base image>-edge docker
```

## Runtime revert (no rebuild)

```
DEBUG_CLR_DISABLE_ORDERING_EDGE=1
```
