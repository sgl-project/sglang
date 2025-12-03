# DP for Multi-Modal Encoder in SGLang

A typical VLM architecture involves two main components: an multi-modal encoder and a text decoder.

Most VLMs utilize a Vision Transformer (ViT) as their multi-modal encoder, it is responsible for processing visual data, extracting features (objects, colors, textures, etc.), and transforming them into a format that can be understood by the model.

The text deocoder is based on LLM. It processes textual data and generates output based on the encoded visual features.

However, since the size of ViT is very small compared to language decoders,
there is relatively little gain from TP. On the other hand, TP incurs significant communication
overhead because of all-reduce being performed after every layer.

Placing the ViT in data parallel while keeping the LLM in tensor parallel consistently lowers TTFT and boosts end-to-end throughput. In this hybrid layout, the vision front-end becomes parallel and lightweight, while scarce interconnect bandwidth and collective ops are reserved for the LLM.

Data parallelism replicates the entire model across multiple GPU sets and processes different batches of requests in parallel.

## Pros and Cons for DP Multi-Modal Encoder

- Unfavorable compute/communication ratio for small ViTs
ViTs used in multimodal stacks are typically modest in parameter count and activation sizes. TP introduces per-layer all-reduce collectives (attention/MLP) whose latency and synchronization overhead outweigh the speedup of splitting relatively small GEMMs. With DP, each GPU runs a full ViT locally—no inference-time collectives—so latency is dominated by compute, not wire time.

- Graph-capture gaps amplify TP overhead
In production, the vision path often has dynamic shapes (pre/post-processing, variable resolution, patching) that break CUDA Graphs and limit torch.compile fusion. Without capture, we need to pay extra kernel-launch and framework overhead; TP then multiplies that cost with additional NCCL synchronizations. Keeping ViT in DP avoids layering collective latency on top of non-captured kernels.

- Better interconnect hygiene for the true bottleneck (the LLM)
The LLM’s prefill and decode phases benefit materially from TP on fast links. Offloading ViT to DP eliminates “chatty” small collectives on the same fabric, reducing congestion and jitter for the LLM’s large, bandwidth-hungry all-reduces.

- Shorter and steadier critical path → lower TTFT
TTFT ≈ T(image encode via ViT) + T(LLM prefill) + T(softmax/sample)
DP has several advantages:
(a) batch and prefetch ViT encodes independently,
(b) overlap them with other requests’ LLM decodes on separate streams,
(c) hand off compact visual embeddings to the TP LLM with minimal queuing.

- For vision encoders that use hardware-unoptimized Conv3D operations,
batch-level DP can provide another 40% improvement compared to regular TP.

- Nevertheless, since the weights of the multi-modal encoder are replicated across each TP rank,
there will be a minor increase in memory consumption and may cause OOM if you can barely fit the model already.

## Command Example
You can enable batch-level DP by setting `mm-enable-dp-encoder`, for example:
```
SGLANG_MM_FEATURE_CACHE_MB=4096 \
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_VLM_CACHE_SIZE_MB=512 \
python3 -m sglang.launch_server --host 127.0.0.1 \
    --mem-fraction-static 0.7 \
    --port 30000 \
    --trust-remote-code \
    --dtype auto \
    --max-running-requests 4 \
    --chunked-prefill-size 8192 \
    --attention-backend flashinfer \
    --tp 4 \
    --enable-multimodal \
    --chat-template internvl-2-5 \
    --model OpenGVLab/InternVL2_5-8B \
    --disable-radix-cache \
    --mm-enable-dp-encoder
```
!!! important
    Batch-level multi-modal DP is not to be confused with API request-level DP
    (which is instead controlled by `data_parallel_size`).

## Known supported models
- Qwen2.5-VL (<https://github.com/sgl-project/sglang/pull/13126>)
- Qwen3-VL (<https://github.com/sgl-project/sglang/pull/13724>)
- InternVL (<https://github.com/sgl-project/sglang/pull/13925>)
