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


## Best Practice for Pipeline Parallelism with PD Disaggregation
To be added. Stay tuned for the latest updates on Pipeline Parallelism with PD Disaggregation.
