# Cuda Graph for Multi-Modal Encoder in SGLang

## Motivation

In multimodal reasoning services, the visual encoder (ViT / Vision Transformer) typically has a few characteristic traits:

Many layers, fragmented operators: Each layer includes LN, QKV projections, attention, MLP, residual connections, etc., resulting in extremely frequent kernel launches.

Server-side “small batch / low latency” is common: The batch size is very small (sometimes it looks like 1 after “flattening” the batch), so kernel launch overhead accounts for a large portion of end-to-end latency.

Input token count (number of patches) varies frequently: Different image/video resolutions and different batch composition lead to different sequence lengths
S — and this is precisely the biggest obstacle for CUDA Graph (unstable shapes).

The value of CUDA Graph: It captures a long sequence of GPU kernels with fixed shapes and fixed memory addresses into a graph; later, for the same shapes, it can replay the graph directly, dramatically reducing launch overhead and making GPU scheduling more compact.

This led us to seek a CUDA Graph enabled feature for ViT in order to improve ViT performance.

## Design and Restrictions

The new CUDA Graph enabled ViT logic is built on ViTCudaGraphRunner. This runner captures the "blocks + merger + deepstack merger (optional)" part of a vision transformer into a CUDA graph and replays it for identical shapes. See the following design consideration and restrictions for more details.

### Dynamic inputs to fit static constraints of CUDA Graph

Variable sequence length S is very common in ViT. While CUDA Graph requires fixed shapes. The solution is to build a graph cache by S(e.g., graph_key = S). The first time create a new S, and then capture a graph; afterwards, replay it.

If there are many distinct S values, we need to increase VRAM usage which is graph-private memory pools for many graphs.

### Stable addresses

Everything "parameter-like" becomes a static buffer:

- block_input / block_ws / block_output
- cu_full_len / cu_window_len and their kk variants
- sin_cos_ws

In this way to solve the underlying requirement: during replay, not allowed to swap tensors, can only modify tensor contents.

### Attention backend arguments
Attention backend arguments are fixed inside the graph:

TritonAttn expects [cu_seqlens, cu_seqlens_kk, max_len]
FA3 expects [cu_seqlens, max_len]

max_len is frozen as an int constant.
cu_seqlens is cached into a dict during create_graph(), and its contents are not updated during subsequent replays.

For the same graph_key = S, you not only require the input shape to match, but also require the segmentation pattern in cu_seqlens (and window seqlens) to be identical. Otherwise, attention will segment the sequence incorrectly.

### Rotary buffer management
The feature reallocates a larger sin_cos_ws when seq_len increases.
The max_content_len is used to make sure the maximum size of the allocated rotary buffer.


## Command Example
You can enable CUDA Graph for ViT by setting env variable `SGLANG_VIT_ENABLE_CUDA_GRAPH=1`, for example:
```
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-VL-8B-Instruct
```
Or you can run CUDA Graph for ViT together with Piecewise CUDA Graph feature by both setting env variable `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` and setting `--enable-piecewise-cuda-graph`, for example:
```
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --piecewise-cuda-graph-max-tokens 4096 \
  --enable-piecewise-cuda-graph \
  --piecewise-cuda-graph-compiler eager
```

## Known supported models
- Qwen2.5-VL (https://github.com/sgl-project/sglang/pull/14422)
- Qwen3-VL (https://github.com/sgl-project/sglang/pull/15320)
