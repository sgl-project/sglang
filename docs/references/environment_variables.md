# Environment Variables

SGLang supports various environment variables that can be used to configure its runtime behavior. This document provides a comprehensive list and aims to stay updated over time.

*Note: SGLang uses two prefixes for environment variables: `SGL_` and `SGLANG_`. This is likely due to historical reasons. While both are currently supported for different settings, future versions might consolidate them.*

## General Configuration

| Environment Variable                      | Description                                                                                                                      | Default Value                |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| `SGLANG_USE_MODELSCOPE`                   | Enable using models from ModelScope                                                                                              | `false`                      |
| `SGLANG_HOST_IP`                          | Host IP address for the server                                                                                                   | `0.0.0.0`                    |
| `SGLANG_PORT`                             | Port for the server                                                                                                              | auto-detected                |
| `SGLANG_LOGGING_CONFIG_PATH`              | Custom logging configuration path                                                                                                | Not set                      |
| `SGLANG_LOG_REQUEST_HEADERS`              | Comma-separated list of additional HTTP headers to log when `--log-requests` is enabled. Appends to the default `x-smg-routing-key`. | Not set                      |
| `SGLANG_HEALTH_CHECK_TIMEOUT`             | Timeout for health check in seconds                                                                                              | `20`                         |
| `SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL` | The interval of passes to collect the metric of selected count of physical experts on each layer and GPU rank. 0 means disabled. | `0`                          |
| `SGLANG_FORWARD_UNKNOWN_TOOLS`            | Forward unknown tool calls to clients instead of dropping them                                                                   | `false` (drop unknown tools) |
| `SGLANG_REQ_WAITING_TIMEOUT`              | Timeout (in seconds) for requests waiting in the queue before being scheduled                                                    | `-1`                         |
| `SGLANG_REQ_RUNNING_TIMEOUT`              | Timeout (in seconds) for requests running in the decode batch                                                                    | `-1`                         |
| `SGLANG_CACHE_DIR`                        | Cache directory for model weights and other data | `~/.cache/sglang` |
| `SGLANG_PREFETCH_BLOCK_SIZE_MB`           | Block size (in MB) for sequential checkpoint prefetch reads that warm the OS page cache before workers load weights via mmap | `16` |

## Performance Tuning

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_ENABLE_TORCH_INFERENCE_MODE` | Control whether to use torch.inference_mode | `false` |
| `SGLANG_ENABLE_TORCH_COMPILE` | Enable torch.compile | `false` |
| `SGLANG_SET_CPU_AFFINITY` | Enable CPU affinity setting (often set to `1` in Docker builds) | `false` |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | Allows the scheduler to overwrite longer context length requests (often set to `1` in Docker builds) | `false` |
| `SGLANG_IS_FLASHINFER_AVAILABLE` | Control FlashInfer availability check | `true` |
| `SGLANG_SKIP_P2P_CHECK` | Skip P2P (peer-to-peer) access check | `false` |
| `SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD` | Sets the threshold for enabling chunked prefix caching | `8192` |
| `SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION` | Enable RoPE fusion in Fused Multi-Layer Attention | `1` |
| `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` | Disable overlap schedule for consecutive prefill batches | `false` |
| `SGLANG_SCHEDULER_MAX_RECV_PER_POLL` | Set the maximum number of requests per poll, with a negative value indicating no limit | `-1` |
| `SGLANG_DISABLE_FA4_WARMUP` | Disable Flash Attention 4 warmup passes (set to `1`, `true`, `yes`, or `on` to disable) | `false` |
| `SGLANG_DATA_PARALLEL_BUDGET_INTERVAL` | Interval for DPBudget updates | `1` |
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT` | Default weight value for scheduler recv skipper counter (used when forward mode doesn't match specific modes). Only active when `--scheduler-recv-interval > 1`. The counter accumulates weights and triggers request polling when reaching the interval threshold. | `1000` |
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE` | Weight increment for decode forward mode in scheduler recv skipper. Works with `--scheduler-recv-interval` to control polling frequency during decode phase. | `1` |
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY` | Weight increment for target verify forward mode in scheduler recv skipper. Works with `--scheduler-recv-interval` to control polling frequency during verification phase. | `1` |
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE` | Weight increment when forward mode is None in scheduler recv skipper. Works with `--scheduler-recv-interval` to control polling frequency when no specific forward mode is active. | `1` |
| `SGLANG_MM_BUFFER_SIZE_MB` | Size of preallocated GPU buffer (in MB) for multi-modal feature hashing optimization. When set to a positive value, temporarily moves features to GPU for faster hash computation, then moves them back to CPU to save GPU memory. Larger features benefit more from GPU hashing. Set to `0` to disable. | `0` |
| `SGLANG_MM_PRECOMPUTE_HASH` | Enable precomputing of hash values for MultimodalDataItem | `false` |
| `SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH` | Enable NCCL for gathering when preparing mlp sync batch under overlap scheduler (without this flag gloo is used for gathering) | `false` |
| `SGLANG_SYMM_MEM_PREALLOC_GB_SIZE` | Size of preallocated GPU buffer (in GB) for NCCL symmetric memory pool to limit memory fragmentation. Only have an effect when server arg `--enable-symm-mem` is set. | `-1` |
| `SGLANG_CUSTOM_ALLREDUCE_ALGO` | The algorithm of custom all-reduce. Set to `oneshot` or `1stage` to force use one-shot. Set to `twoshot` or `2stage` to force use two-shot. | `` |
| `SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR` | Skip-softmax threshold scale factor for TRT-LLM prefill attention in flashinfer. `None` means standard attention. See https://arxiv.org/abs/2512.12087 | `None` |
| `SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR` | Skip-softmax threshold scale factor for TRT-LLM decode attention in flashinfer. `None` means standard attention. See https://arxiv.org/abs/2512.12087 | `None` |
| `SGLANG_USE_SGL_FA3_KERNEL`               | Use sgl-kernel implementation for FlashAttention v3 | `true` |


## DeepGEMM Configuration (Advanced Optimization)

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_ENABLE_JIT_DEEPGEMM` | Enable Just-In-Time compilation of DeepGEMM kernels (enabled by default on NVIDIA Hopper (SM90) and Blackwell (SM100) GPUs when the DeepGEMM package is installed; set to `"0"` to disable) | `"true"` |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` | Enable precompilation of DeepGEMM kernels | `"true"` |
| `SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS` | Number of workers for parallel DeepGEMM kernel compilation | `4` |
| `SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE` | Indicator flag used during the DeepGEMM precompile script | `"false"` |
| `SGLANG_DG_CACHE_DIR` | Directory for caching compiled DeepGEMM kernels | `~/.cache/deep_gemm` |
| `SGLANG_DG_USE_NVRTC` | Use NVRTC (instead of Triton) for JIT compilation (Experimental) | `"false"` |
| `SGLANG_USE_DEEPGEMM_BMM` | Use DeepGEMM for Batched Matrix Multiplication (BMM) operations | `"false"` |
| `SGLANG_JIT_DEEPGEMM_FAST_WARMUP` | Precompile less kernels during warmup, which reduces the warmup time from 30min to less than 3min. Might cause performance degradation during runtime. | `"false"` |

## DeepEP Configuration

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_DEEPEP_BF16_DISPATCH` | Use Bfloat16 for dispatch | `"false"` |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | The maximum number of dispatched tokens on each GPU | `"128"` |
| `SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | The maximum number of dispatched tokens on each GPU for --moe-a2a-backend=flashinfer | `"1024"` |
| `SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS` | Number of SMs used for DeepEP combine when single batch overlap is enabled | `"32"` |
| `SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO` | Run shared experts on an alternate stream when single batch overlap is enabled on GB200. When not setting this flag, shared experts and down gemm will be overlapped with DeepEP combine together. | `"false"` |

## MORI Configuration

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_MORI_DISPATCH_DTYPE` | Override MoRI-EP dispatch quantization type. `auto` uses auto-detection from weight dtype; `bf16`/`fp8`/`fp4` forces the specified type for all layers | `"auto"` |
| `SGLANG_MORI_FP8_COMB` | Use FP8 for combine | `"false"` |
| `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | Maximum number of dispatch tokens per rank for MORI-EP buffer allocation | `4096` |
| `SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD` | Threshold for switching between `InterNodeV1` and `InterNodeV1LL` kernel types. `InterNodeV1LL` is used if `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` is less than or equal to this threshold; otherwise, `InterNodeV1` is used. | `256` |
| `SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS` | This argument devives `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` which indicates customized amount of tokens preallocated for a rank, valid range from 1 to world_size*SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK, by default `0` means maximum. Setting a smaller value will reduce memory footprint but too small value could cause buffer overflow. | `0` |
| `SGLANG_MORI_MOE_MAX_INPUT_TOKENS` | Truncate the dispatch buffer to this many rows before MoE computation, reducing kernel overhead on padding tokens. The value must be >= the actual number of received tokens (`totalRecvTokenNum`); setting it too small causes incorrect results. `0` disables truncation (use full buffer). | `0` |
| `SGLANG_MORI_QP_PER_TRANSFER` | Number of RDMA Queue Pairs (QPs) used per transfer operation | `1` |
| `SGLANG_MORI_POST_BATCH_SIZE` | Number of RDMA work requests posted in a single batch to each QP | `-1` |
| `SGLANG_MORI_NUM_WORKERS` | Number of worker threads in the RDMA executor thread pool | `1` |

## NSA Backend Configuration (For DeepSeek V3.2)

<!-- # Environment variable to control mtp precomputing of metadata for multi-step speculative decoding -->

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_NSA_FUSE_TOPK` | Fuse the operation of picking topk logits and picking topk indices from page table  | `true` |
| `SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA` | Precompute metadata that can be shared among different draft steps when MTP is enabled | `true` |
| `SGLANG_USE_FUSED_METADATA_COPY` | Control whether to use fused metadata copy kernel for cuda graph replay  | `true` |
| `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` | When the maximum kv len in current prefill batch exceeds this value, the sparse mla kernel will be applied, else it falls back to dense MHA implementation. Default to the index topk of model (2048 for DeepSeek V3.2) | `2048` |


## Memory Management

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_DEBUG_MEMORY_POOL` | Enable memory pool debugging | `false` |
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | Clip max new tokens estimation for memory planning | `4096` |
| `SGLANG_DETOKENIZER_MAX_STATES` | Maximum states for detokenizer | Default value based on system |
| `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK` | Enable checks for memory imbalance across Tensor Parallel ranks | `true` |
| `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` | Configure the custom memory pool type for Mooncake. Supports `NVLINK`, `BAREX`, `INTRA_NODE_NVLINK`. If set to `true`, it defaults to `NVLINK`. | `None` |

## Model-Specific Options

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_USE_AITER` | Use AITER optimize implementation | `false` |
| `SGLANG_ROCM_USE_MULTI_STREAM` | Allocate alt CUDA/HIP stream on ROCm/AITER to overlap shared and routed experts in DeepseekV2 MoE. Requires the HIP env `GPU_MAX_HW_QUEUES>=5` (default `4`, the cap on HSA/ROCr HW queues HIP creates) so the alt stream gets its own queue instead of serializing with the main stream. Best paired with `--deepep-mode low_latency` so Mori's AsyncLL kernel offloads dispatch/combine to copy engines and frees CUs. | `false` |
| `SGLANG_MOE_PADDING` | Enable MoE padding (sets padding size to 128 if value is `1`, often set to `1` in Docker builds) | `false` |
| `SGLANG_CUTLASS_MOE` (deprecated) | Use Cutlass FP8 MoE kernel on Blackwell GPUs (deprecated, use --moe-runner-backend=cutlass) | `false` |

## Quantization

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_INT4_WEIGHT` | Enable INT4 weight quantization | `false` |
| `SGLANG_FORCE_FP8_MARLIN` | Force using FP8 MARLIN kernels even if other FP8 kernels are available | `false` |
| `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN` | Quantize q_b_proj from BF16 to FP8 when launching DeepSeek NVFP4 checkpoint | `false` |
| `SGLANG_MOE_NVFP4_DISPATCH` | Use nvfp4 for moe dispatch (on flashinfer_cutlass or flashinfer_cutedsl moe runner backend) | `"false"` |
| `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE` | Quantize moe of nextn layer from BF16 to FP8 when launching DeepSeek NVFP4 checkpoint | `false` |
| `SGLANG_QUANT_ALLOW_DOWNCASTING` | Allow weight dtype downcasting during loading (e.g., fp32 → fp16). By default, SGLang rejects this kind of downcasting when using quantization. | `false` |
| `SGLANG_FP8_IGNORED_LAYERS` | A comma-separated list of layer names to ignore during FP8 quantization. For example: `model.layers.0,model.layers.1.,qkv_proj`. | `""` |


## Distributed Computing

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_BLOCK_NONZERO_RANK_CHILDREN` | Control blocking of non-zero rank children processes | `1` |
| `SGLANG_IS_FIRST_RANK_ON_NODE` | Indicates if the current process is the first rank on its node | `"true"` |
| `SGLANG_PP_LAYER_PARTITION` | Pipeline parallel layer partition specification | Not set |
| `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` | Set one visible device per process for distributed computing | `false` |

## PD Disaggregation — Staging Buffer (Heterogeneous TP)

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_DISAGG_STAGING_BUFFER` | Enable GPU staging buffer for heterogeneous TP KV transfer. Required when prefill and decode use different TP/attention-TP sizes. Only for non-MLA models (e.g. GQA, MHA). | `false` |
| `SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB` | Prefill-side per-worker staging buffer size in MB. Used for gathering KV head slices before bulk RDMA transfer. | `64` |
| `SGLANG_DISAGG_STAGING_POOL_SIZE_MB` | Decode-side ring buffer pool total size in MB. Shared buffer receiving RDMA data from all prefill ranks. Larger values support higher concurrency. | `4096` |
| `SGLANG_STAGING_USE_TORCH` | Force using PyTorch gather/scatter fallback instead of Triton fused kernels for staging operations. Useful for debugging. | `false` |

## Testing & Debugging (Internal/CI)

*These variables are primarily used for internal testing, continuous integration, or debugging.*

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_IS_IN_CI` | Indicates if running in CI environment | `false` |
| `SGLANG_IS_IN_CI_AMD` | Indicates running in AMD CI environment | `false` |
| `SGLANG_TEST_RETRACT` | Enable retract decode testing | `false` |
| `SGLANG_TEST_RETRACT_NO_PREFILL_BS` | When SGLANG_TEST_RETRACT is enabled, no prefill is performed if the batch size exceeds SGLANG_TEST_RETRACT_NO_PREFILL_BS. | `2 ** 31`     |
| `SGLANG_RECORD_STEP_TIME` | Record step time for profiling | `false` |
| `SGLANG_TEST_REQUEST_TIME_STATS` | Test request time statistics | `false` |
| `SGLANG_DEBUG_SYMM_MEM` | Enable debug checks that verify tensors passed to NCCL communication ops are allocated in the symmetric memory pool. Logs warnings (rank 0 only) with stack traces for any tensor not in the pool. | `false` |
| `SGLANG_KERNEL_API_LOGLEVEL` | Controls crash-debug kernel API logging. `0` disables logging, `1` logs API names, `3` logs tensor metadata, `5` adds tensor statistics, and `10` also writes pre-call dump snapshots. | `0` |
| `SGLANG_KERNEL_API_LOGDEST` | Destination for crash-debug kernel API logs. Use `stdout`, `stderr`, or a file path. `%i` is replaced with the process PID. | `stdout` |
| `SGLANG_KERNEL_API_DUMP_DIR` | Output directory for level-10 kernel API input/output dumps. `%i` is replaced with the process PID. | `sglang_kernel_api_dumps` |
| `SGLANG_KERNEL_API_DUMP_INCLUDE` | Comma-separated wildcard patterns for kernel API names to include in level-10 dumps. | Not set |
| `SGLANG_KERNEL_API_DUMP_EXCLUDE` | Comma-separated wildcard patterns for kernel API names to exclude from level-10 dumps. | Not set |

## Profiling & Benchmarking

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_TORCH_PROFILER_DIR` | Directory for PyTorch profiler output | `/tmp` |
| `SGLANG_PROFILE_WITH_STACK` | Set `with_stack` option (bool) for PyTorch profiler (capture stack trace) | `true` |
| `SGLANG_PROFILE_RECORD_SHAPES` | Set `record_shapes` option (bool) for PyTorch profiler (record shapes) | `true` |
| `SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS` | Config BatchSpanProcessor.schedule_delay_millis if tracing is enabled | `500` |
| `SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE` | Config BatchSpanProcessor.max_export_batch_size if tracing is enabled | `64` |

## Storage & Caching

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_WAIT_WEIGHTS_READY_TIMEOUT` | Timeout period for waiting on weights | `120` |
| `SGLANG_DISABLE_OUTLINES_DISK_CACHE` | Disable Outlines disk cache | `false` |
| `SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE` | Use SGLang's custom Triton kernel cache implementation for lower overheads (automatically enabled on CUDA) | `false` |
| `SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE` | Decode-side incremental KV cache offload stride. Rounded down to a multiple of `--page-size` (min is `--page-size`). If unset/invalid/<=0, it falls back to `--page-size`. | Not set (uses `--page-size`) |


## Function Calling / Tool Use

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_TOOL_STRICT_LEVEL` | Controls the strictness level of tool call parsing and validation. <br>**Level 0**: Off - No strict validation <br>**Level 1**: Function strict - Enables structural tag constraints for all tools (even if none have `strict=True` set) <br>**Level 2**: Parameter strict - Enforces strict parameter validation for all tools, treating them as if they all have `strict=True` set | `0` |
