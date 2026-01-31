# Pipeline Parallelism for Long Context

## Why Pipeline Parallelism?

As Large Language Models (LLMs) scale toward trillion-parameter architectures and "infinite" context windows, the underlying serving infrastructure must evolve toward more granular, cross-node parallelization strategies. While KV cache techniques effectively mitigate redundant computation, they cannot circumvent the prohibitive Time to First Token (TTFT) inherent in ultra-long sequences with extremely large initial Input Token Length (ITL). Although Tensor Parallelism (TP) remains the conventional approach for intra-node scaling, it frequently encounters communication bottlenecks during multi-node deployments. On the other hand, pipeline parallelism only requires cross-node communication at the boundaries of each pipeline stage, which can achieve better computation-communication overlap compared to a large TP. Therefore, it is also a promising parallelization strategy for improving throughput.

Detailed analysis can be found in this [blog](https://lmsys.org/blog/2026-01-15-chunked-pipeline/).

## Implementation Refactoring based on Async Communication
With Dynamic Chunked Prefill, pipeline parallelism has the potential to reduce the TTFT of long-context inputs. For each request, its input tokens can be partitioned into multiple chunks, each no longer than the chunked prefill size. Different chunks of the same request can be processed simultaneously by different nodes, thus parallelizing the processing and reducing TTFT. SGLang has supported Pipeline Parallelism (#5724) for some time and made it compatible with the PD Disaggregation feature (#8846), but the implementation was not perfect and had significant room for performance improvements.

To eliminate this performance hazard, SGLang implements a Micro-batching Event Loop with non-blocking asynchronous peer-to-peer (P2P) communication to overlap GPU computation with CPU metadata processing and PP communication. This ensures that while one micro-batch is being computed on the GPU, the next one is already being prepared and moved into position effectively, ensuring the pipeline remains as saturated as possible. This approach was first proposed in #7979 and has been redesigned and included in #11852.

The key mechanisms of the implementation include:

* **Decoupled Sync/Async Logic in the Event Loop:** The scheduler uses `async_send` in `_pp_send_pyobj_to_next_stage`. Instead of waiting for a transfer to complete, it returns a `P2PWork` handle. The actual synchronization (`P2PWork.work.wait()`) is deferred until `_pp_commit_comm_work` is called, allowing the CPU to perform other work—like scheduling the next batch or processing metadata—while data is in flight.
* **Multi-Stream Execution:** In addition to the main `default_stream`, which serves as the synchronization stream, SGLang utilizes dedicated `forward_stream` and `copy_stream` to execute forward pass GPU computation and Data-to-Host (D2H) memory transfers separately for better overlapping. While `_pp_launch_batch` is executing the current micro-batch on the GPU for the current stage, the CPU processes the previous micro-batch's results using `_pp_process_batch_result`.

## Guidance about Dynamic Chunking

### Why Dynamic Chunking
Chunked prefill with a fixed size can cause bubbles in the pipeline, especially when the pp size is large. The main reason behind this phenomenon is that the model has a non-uniform running time, even though each chunk size is identical (brought by the Transformer structure). The larger the prefix sequence length, the longer the running time of the chunk. And these bubbles will be propagated to the next stage, and will significantly degrade the scale efficiency of larger pp ranks.

To address this issue, SGLang introduces a dynamic chunking mechanism to predict the optimal size for the next chunk such that it satisfies this condition:

Runtime(L + Next Chunk Size) - Runtime(L) = Runtime(Initial Chunk Size)

where ***L*** denotes the Prefix Sequence Length. By profiling a series of requests with different ITLs, we model the cumulative runtime as a quadratic function of sequence length. Using this model, we solve the optimal next chunk size for any given prefix length ***L***. Since the computation complexity of the Attention mechanism scales with ***L***, the next chunk size will be progressively reduced as ***L*** grows to maintain an aligned chunk execution time across pipeline stages.

Based on this method, the scheduler can predict and dynamically reduce the chunk size during runtime to minimize the bubbles caused by the stage misalignment. To be noticed, the scheduler does not use the raw predicted value. To facilitate efficient KVCache memory management and ensure affinity with hardware execution efficiency, the value is aligned downward to the nearest multiple of max(`--page-size`, 64).


### Chunked Prefill Size and Smoothing Factor

When `--enable-dynamic-chunking` is enabled, each chunk size of a sequence is determined dynamically based on the quadratic model that predicts the next chunk size based on the estimated runtime of the initial chunk length. In this case, we use `--chunked-prefill-size` to set up the initial chunk size. When switching to the dynamic chunking mode, the initial chunk size (`--chunked-prefill-size`) should be set to a larger value comparable to the original chunked prefill size, so that there won't be too many chunks.

**`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`** is an environmental variable that controls the smoothing factor for the dynamic chunking algorithm, defaulting to 0.75. It determines how much the chunk size can change during the prefill phase. A larger value means a more aggressive chunk size change, which may lead to better performance but also to greater chunk size changes (the chunk size at the end may become very small, which could lead to performance degradation) and more total chunks. When it is set to 1, the chunk size will be adjusted strictly based on the aforementioned quadratic model that predicts the next chunk size. A smaller value means a more conservative chunk size change, which may lead to smaller chunk size changes and fewer total chunks. When it is set to 0, the chunk size will not be adjusted dynamically, so it is identical to the traditional way with a fixed chunked prefill size.

Due to the variation in hardware, models, and target workloads, a static configuration is seldom optimal across all scenarios. Consequently, achieving peak performance necessitates a degree of hyperparameter tuning when switching to the dynamic chunking mode.

**Tuning Guidance for Dynamic Chunked Prefill**

* **Step 1 \- Iterate to find the optimal fixed chunked prefill size for the targeted PP size**: Different PP sizes for targeted ITL may have different optimal chunked prefill sizes. Therefore, users should iterate to obtain the baseline according to the available resources for scaling.
* **Step 2 \- Initial Chunk Size Selection for Dynamic Chunking**: Set the initial size to 2× or 3× the optimal fixed chunked prefill size. This reduces the total number of chunks and prevents "tail chunks" from underutilizing hardware. To maintain efficiency for extremely large Input Token Lengths (ITL), the dynamic predictor automatically ensures subsequent chunks are at least 1/4 of this initial size. In addition, it is recommended to use a larger initial chunk size (e.g., 4× the optimal fixed chunked prefill size) for such cases as well.
* **Step 3 \- Smooth Factor Adjustment**: This factor controls how strictly the chunk size adjusts the prediction given by the quadratic performance fitting model.
  * 1.0: Follows the model strictly.
  * **0.6 – 0.85 (Recommended)**: Typical range for the best balance between dynamic scaling and hardware stability. Through experiments, we find that a range between 0.6 and 0.85 typically yields the best performance for dynamic chunking.
  * 0: Disables dynamic adjustment, reverting to traditional fixed-size chunking.
* **Another small optimization tip:** Put the larger partition in the higher PP rank when the layers are not evenly divisible across ranks. It can increase the GPU utilization when a larger PP rank is waiting for the previous stage’s result, hence reducing the bubbles on higher PP ranks. If we take DeepSeek-V3.1 as an example, `SGLANG_PP_LAYER_PARTITION=15,15,15,16` usually performs better than `16,15,15,15`.

## Best Practice for Long Context

### Tuning the Chunked Prefill Size
Optimizing the chunked prefill size is crucial for balancing pipeline efficiency and resource utilization. The ideal size depends on factors including model architecture, hardware configuration, and typical input lengths. We recommend starting with a small chunk size, such as 4K, and gradually increasing it until you find the optimal size for your specific use case (Different targeted ITL and PP Sizes may have different optimal chunked prefill sizes. Therefore, users should iterate to obtain the baseline according to the available resources for scaling). Alternatively, you can analyze the hardware capacity and determine the optimal chunk size based on the roofline model.

### Enable Dynamic Chunking and Adjust Smoothing Factor for Ultra-long ITL
SGLang also offers a dynamic chunking solution that could further improve performance. This feature is currently an experimental feature that requires a certain amount of tuning experimentation and may not be suitable for all workloads. In addition, fine-tuning the smoothing factor can help optimize performance for specific workloads and model characteristics.

### Case Study on NVIDIA H20

When evaluating pipeline parallelism with fixed chunked prefill sizes from 2K to 16K, experiment results show that a 4K chunk size delivered optimal prefill TTFT performance for the DeepSeek-V3.1, and a 6K chunk size delivered optimal prefill TTFT performance for the Qwen3-235B-A22B-FP8.

When enabling dynamic chunking, we first scale the optimal fixed chunked prefill size by a factor of 3 as the initial chunk size. Through experimentation, we found that a multiplier of 2-3 provides an appropriate balance—avoiding excessive initial pipeline bubbles while ensuring that subsequent chunks don't become too small as context length increases. With the default dynamic chunking smoothing factor of 0.75, we performed parameter tuning and determined that a value of 0.65 works optimally with the 12K initial chunk size for the DeepSeek-V3.1, while a value of 0.8 works optimally with the 18K initial chunk size for the Qwen3-235B-A22B-FP8.

#### DeepSeek-V3.1 with 128K Input Token Length
```bash
# prefill node 0 (fixed chunked prefill size)
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
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
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 12288 --enable-dynamic-chunking
```

#### Qwen3-235B-A22B-FP8 with 128K Input Token Length
```bash
# prefill node 0 (fixed chunked prefill size)
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 6144
```

```bash
# prefill node 0 (with dynamic chunking)
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.8
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 18432 --enable-dynamic-chunking
```

Note: `--disable-radix-cache` is enabled only for reproducible benchmarking purposes. It is not recommended to use it in production.

## Best Practice for Pipeline Parallelism with PD Disaggregation
To be added. Stay tuned for the latest updates on Pipeline Parallelism with PD Disaggregation.
