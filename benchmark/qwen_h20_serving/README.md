# Qwen serving fixes qualified on H20

The [experiment report](https://github.com/leviking98z-rgb/sglang/issues/2)
records throughput tuning, latency tradeoffs, and the separate memory and
correctness fixes found while deploying Qwen3.8-Flash-Next. The performance
experiments use explicitly pinned source/model revisions. Their layout and MTP
gains are not attributed to the fixes below.

## Optional allocation settings

```bash
# Register one contiguous PLE host table in bounded CUDA chunks.
# The default 0 retains torch's pinned allocator.
export SGLANG_QWEN4_PLE_HOST_REGISTER_CHUNK_MB=256

# Run GPU image transforms in groups of four and concatenate their features
# on CPU. The default 0 retains request-sized preprocessing.
export SGLANG_QWEN4_IMAGE_PREPROCESS_CHUNK_SIZE=4
```

The PLE option uses a prefaulted private mapping on Linux, excludes it from fork,
and verifies identical host/device addresses for the existing gather kernel.
Registrations remain alive through tensor storage views and are released after
device synchronization. Partial registration failure rolls back completed
ranges. Failed unregistration retains the mapping until process exit.

Image chunks apply only to the Qwen4 model's Qwen2-VL image processor with
ordinary CPU feature transport. GPU-resident features and hashing before CPU
transfer are rejected. The processor retains the original path for small
batches and CPU inputs. Multi-image GPU preprocessing preserves pixel values,
image grids and order, then returns CPU tensors to the existing transport path.

For a synthetic 47-page fixture, four-image chunks reduced peak PyTorch GPU
allocation from 6.43 GiB to 0.80 GiB with bitwise-identical features. Processing
time increased from 0.790 s to 1.184 s. This is a memory tradeoff, not a speedup.

## Correctness fixes

- Empty grouped Gemma RMSNorm input returns after tensor validation instead of
  launching a zero-block CUDA grid. Nonempty kernel code is unchanged.
- GPU JPEG decoding explicitly requests RGB, matching the processor's expected
  channel count for grayscale JPEG sources.

## Focused regression commands

Run on a CUDA node with the normal source-install dependencies:

```bash
python -m pytest -q \
  test/registered/kernels/ops/layernorm/test_grouped_gemma_rmsnorm.py \
  test/registered/kernels/ops/embeddings/test_qwen4_ple_registered.py \
  test/registered/unit/multimodal/test_qwen_vl_image_chunks.py \
  test/registered/unit/utils/test_common.py
```

These cover empty/nonempty BF16/FP16 RMSNorm and CUDA Graph replay, registration
rollback and boundary gathers through parameter views, mixed-shape image feature
parity, CPU fallback, installation validation, and grayscale/RGB GPU JPEG decode.
The allocation options are disabled by default and do not change global MTP,
parallelism, concurrency or feature-transport settings.
