# Pipeline Parallelism for Long Context

## Why Pipeline Parallelism?

With chunked prefill, pipeline parallelism has the potential to reduce the TTFT of long-context inputs. For each request, its input tokens can be partitioned into multiple chunks, each no longer than the chunked prefill size. Different chunks of the same request can be processed simultaneously by different nodes, thus parallelizing the processing and reducing TTFT.

Also, pipeline parallelism only requires cross-node communication at the boundaries of each pipeline stage, which can achieve better computation-communication overlap compared to a large TP. Therefore, it is also a promising parallelization strategy for improving throughput.

## Implementation Refactoring based on Async Communication
SGLang has supported Pipeline Parallelism (#5724) for some time and made it compatible with the PD Disaggregation feature (#8846), but the implementation was not perfect and had significant room for performance improvements.

To reduce PP bubbles, SGLang now utilizes asynchronous sends for communication between PP stages. This approach was first proposed in #7979 and has been redesigned and included in #11852.

## Guidance about Dynamic Chunking

### Why Dynamic Chunking
Chunked prefill with a fixed size can cause bubbles in the pipeline, especially when the pp size is large. The main reason behind this phenomenon is that the model has a non-uniform running time, even though each chunk size is identical (brought by the Transformer structure). The larger the prefix sequence length, the longer the running time of the chunk. And these bubbles will be propagated to the next stage, and will significantly degrade the scale efficiency of larger pp ranks.

To address this issue, we introduce a dynamic chunking mechanism and use a quadratic function to fit this condition: Runtime(Prefix Sequence Length + Next Chunk Size) - Runtime(Prefix Sequence Length) = Runtime(Initial Chunk Size). Based on this method, we can dynamically reduce the chunk size to minimize the bubbles caused by the stage misalignment.

### Chunked Prefill Size and Smoothing Factor

When `--enable-dynamic-chunking` is enabled, each chunk size of a sequence is determined dynamically based on the quadratic model that predicts the next chunk size based on the estimated runtime of the initial chunk length. In this case, we use `--chunked-prefill-size` to set up the initial chunk size. When switching to the dynamic chunking mode, the initial chunk size (`--chunked-prefill-size`) should be set to a larger value comparable to the original chunked prefill size, so that there won't be too many chunks.

**`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`** is a parameter that controls the smoothing factor for the dynamic chunking algorithm, defaulting to 0.75. It determines how much the chunk size can change during the prefill phase. A larger value means a more aggressive chunk size change, which may lead to better performance but also to greater chunk size changes (the chunk size at the end may become very small, which could lead to performance degradation) and more total chunks. When it is set to 1, the chunk size will be adjusted strictly based on the aforementioned quadratic model that predicts the next chunk size. A smaller value means a more conservative chunk size change, which may lead to smaller chunk size changes and fewer total chunks. When it is set to 0, the chunk size will not be adjusted dynamically, so it is identical to the traditional way with a fixed chunked prefill size.

## Best Practice for Pipeline Parallelism

### Tuning the Chunked Prefill Size
Optimizing the chunked prefill size is crucial for balancing pipeline efficiency and resource utilization. The ideal size depends on factors including model architecture, hardware configuration, and typical input lengths. We recommend starting with a small chunk size, such as 4K, and gradually increasing it until you find the optimal size for your specific use case. Alternatively, you can analyze the hardware capacity and determine the optimal chunk size based on the roofline model.

### Enable Dynamic Chunking and Adjust Smoothing Factor (Experimental feature)
SGLang also offers a dynamic chunking solution that could further improve performance. This feature is currently an experimental feature that requires a certain amount of tuning experimentation and may not be suitable for all workloads. In addition, fine-tuning the smoothing factor can help optimize performance for specific workloads and model characteristics.

### Case Study on NVIDIA H20

When evaluating pipeline parallelism with fixed chunked prefill sizes from 2K to 16K, experiment results show that a 4K chunk size delivered optimal prefill TTFT performance for the DeepSeek-V3.1, and a 6K chunk size delivered optimal prefill TTFT performance for the Qwen3-235B-A22B-Thinking-2507-FP8.

When enabling dynamic chunking, we first scale the optimal fixed chunked prefill size by a factor of 3 as the initial chunk size. Through experimentation, we found that a multiplier of 2-3 provides an appropriate balance—avoiding excessive initial pipeline bubbles while ensuring that subsequent chunks don't become too small as context length increases. With the default dynamic chunking smoothing factor of 0.75, we performed parameter tuning and determined that a value of 0.65 works optimally with the 12K initial chunk size for the DeepSeek-V3.1, while a value of 0.8 works optimally with the 18K initial chunk size for the Qwen3-235B-A22B-Thinking-2507-FP8.

#### DeepSeek-V3.1 with 128K Input Token Length
```bash
# prefill node 0 (fixed chunked prefill size)
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr 192.168.0.137:62001 \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 4096
```

```bash
# prefill node 0 (with dynamic chunking)
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr 192.168.0.137:62001 \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 12288 --enable-dynamic-chunking
```

#### Qwen3-235B-A22B-Thinking-2507-FP8 with 128K Input Token Length
```bash
# prefill node 0 (fixed chunked prefill size)
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 --trust-remote-code \
  --nnodes 2 --node-rank 0 --tp 4 --pp-size 2 \
  --port 30000 --dist-init-addr 192.168.0.137:62001 \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 6144
```

```bash
# prefill node 0 (with dynamic chunking)
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.8
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 --trust-remote-code \
  --nnodes 2 --node-rank 0 --tp 4 --pp-size 2 \
  --port 30000 --dist-init-addr 192.168.0.137:62001 \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 18432 --enable-dynamic-chunking
```

Note: `--disable-radix-cache` is enabled only for reproducible benchmarking purposes. It is not recommended to use it in production.

## Best Practice for Pipeline Parallelism with PD Disaggregation
To be added. Stay tuned for the latest updates on Pipeline Parallelism with PD Disaggregation.
