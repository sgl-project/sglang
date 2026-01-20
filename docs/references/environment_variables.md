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
| `SGLANG_DISABLE_REQUEST_LOGGING`          | Disable request logging                                                                                                          | `false`                      |
| `SGLANG_HEALTH_CHECK_TIMEOUT`             | Timeout for health check in seconds                                                                                              | `20`                         |
| `SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL` | The interval of passes to collect the metric of selected count of physical experts on each layer and GPU rank. 0 means disabled. | `0`                          |
| `SGLANG_FORWARD_UNKNOWN_TOOLS`            | Forward unknown tool calls to clients instead of dropping them                                                                   | `false` (drop unknown tools) |
| `SGLANG_QUEUED_TIMEOUT_MS`                | Timeout (in ms) for requests in the waiting queue                                                                                | `-1` |

## Performance Tuning

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_ENABLE_TORCH_INFERENCE_MODE` | Control whether to use torch.inference_mode | `false` |
| `SGLANG_ENABLE_TORCH_COMPILE` | Enable torch.compile | `true` |
| `SGLANG_SET_CPU_AFFINITY` | Enable CPU affinity setting (often set to `1` in Docker builds) | `0` |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | Allows the scheduler to overwrite longer context length requests (often set to `1` in Docker builds) | `0` |
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
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_VERIFY` | Weight increment for target verify forward mode in scheduler recv skipper. Works with `--scheduler-recv-interval` to control polling frequency during verification phase. | `1` |
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE` | Weight increment when forward mode is None in scheduler recv skipper. Works with `--scheduler-recv-interval` to control polling frequency when no specific forward mode is active. | `1` |
| `SGLANG_MM_BUFFER_SIZE_MB` | Size of preallocated GPU buffer (in MB) for multi-modal feature hashing optimization. When set to a positive value, temporarily moves features to GPU for faster hash computation, then moves them back to CPU to save GPU memory. Larger features benefit more from GPU hashing. Set to `0` to disable. | `0` |
| `SGLANG_MM_PRECOMPUTE_HASH` | Enable precomputing of hash values for MultimodalDataItem | `false` |


## DeepGEMM Configuration (Advanced Optimization)

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_ENABLE_JIT_DEEPGEMM` | Enable Just-In-Time compilation of DeepGEMM kernels (enabled by default on NVIDIA Hopper (SM90) and Blackwell (SM100) GPUs when the DeepGEMM package is installed; set to `"0"` to disable) | `"true"` |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` | Enable precompilation of DeepGEMM kernels | `"true"` |
| `SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS` | Number of workers for parallel DeepGEMM kernel compilation | `4` |
| `SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE` | Indicator flag used during the DeepGEMM precompile script | `"false"` |
| `SGLANG_DG_CACHE_DIR` | Directory for caching compiled DeepGEMM kernels | `~/.cache/deep_gemm` |
| `SGL_DG_USE_NVRTC` | Use NVRTC (instead of Triton) for JIT compilation (Experimental) | `"0"` |
| `SGL_USE_DEEPGEMM_BMM` | Use DeepGEMM for Batched Matrix Multiplication (BMM) operations | `"false"` |

## DeepEP Configuration

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_DEEPEP_BF16_DISPATCH` | Use Bfloat16 for dispatch | `"false"` |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | The maximum number of dispatched tokens on each GPU | `"128"` |
| `SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS` | Number of SMs used for DeepEP combine when single batch overlap is enabled | `"32"` |

## NSA Backend Configuration (For DeepSeek V3.2)

<!-- # Environment variable to control mtp precomputing of metadata for multi-step speculative decoding -->

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_NSA_FUSE_TOPK` | Fuse the operation of picking topk logits and picking topk indices from page table  | `true` |
| `SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA` | Precompute metadata that can be shared among different draft steps when MTP is enabled | `true` |


## Memory Management

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_DEBUG_MEMORY_POOL` | Enable memory pool debugging | `false` |
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | Clip max new tokens estimation for memory planning | `4096` |
| `SGLANG_DETOKENIZER_MAX_STATES` | Maximum states for detokenizer | Default value based on system |
| `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK` | Enable checks for memory imbalance across Tensor Parallel ranks | `true` |

## Model-Specific Options

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_USE_AITER` | Use AITER optimize implementation | `false` |
| `SGLANG_MOE_PADDING` | Enable MoE padding (sets padding size to 128 if value is `1`, often set to `1` in Docker builds) | `0` |
| `SGLANG_CUTLASS_MOE` (deprecated) | Use Cutlass FP8 MoE kernel on Blackwell GPUs (deprecated, use --moe-runner-backend=cutlass) | `false` |

## Quantization

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_INT4_WEIGHT` | Enable INT4 weight quantization | `false` |
| `SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2` | Apply per token group quantization kernel with fused silu and mul and masked m | `false` |
| `SGLANG_FORCE_FP8_MARLIN` | Force using FP8 MARLIN kernels even if other FP8 kernels are available | `false` |
| `SGLANG_FLASHINFER_FP4_GEMM_BACKEND` (deprecated) | Select backend for `mm_fp4` on Blackwell GPUs. **DEPRECATED**: Please use `--fp4-gemm-backend` instead. | `` |
| `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN` | Quantize q_b_proj from BF16 to FP8 when launching DeepSeek NVFP4 checkpoint | `false` |
| `SGLANG_MOE_NVFP4_DISPATCH` | Use nvfp4 for moe dispatch (on flashinfer_cutlass or flashinfer_cutedsl moe runner backend) | `"false"` |
| `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE` | Quantize moe of nextn layer from BF16 to FP8 when launching DeepSeek NVFP4 checkpoint | `false` |
| `SGLANG_ENABLE_FLASHINFER_FP8_GEMM` (deprecated) | Use flashinfer kernels when running blockwise fp8 GEMM on Blackwell GPUs. **DEPRECATED**: Please use `--fp8-gemm-backend=flashinfer_trtllm` instead. | `false` |
| `SGLANG_SUPPORT_CUTLASS_BLOCK_FP8` (deprecated) | Use Cutlass kernels when running blockwise fp8 GEMM on Hopper or Blackwell GPUs. **DEPRECATED**: Please use `--fp8-gemm-backend=cutlass` instead. | `false` |


## Distributed Computing

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_BLOCK_NONZERO_RANK_CHILDREN` | Control blocking of non-zero rank children processes | `1` |
| `SGLANG_IS_FIRST_RANK_ON_NODE` | Indicates if the current process is the first rank on its node | `"true"` |
| `SGLANG_PP_LAYER_PARTITION` | Pipeline parallel layer partition specification | Not set |
| `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` | Set one visible device per process for distributed computing | `false` |

## Testing & Debugging (Internal/CI)

*These variables are primarily used for internal testing, continuous integration, or debugging.*

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_IS_IN_CI` | Indicates if running in CI environment | `false` |
| `SGLANG_IS_IN_CI_AMD` | Indicates running in AMD CI environment | `0` |
| `SGLANG_TEST_RETRACT` | Enable retract decode testing | `false` |
| `SGLANG_TEST_RETRACT_NO_PREFILL_BS` | When SGLANG_TEST_RETRACT is enabled, no prefill is performed if the batch size exceeds SGLANG_TEST_RETRACT_NO_PREFILL_BS. | `2 ** 31`     |
| `SGLANG_RECORD_STEP_TIME` | Record step time for profiling | `false` |
| `SGLANG_TEST_REQUEST_TIME_STATS` | Test request time statistics | `false` |

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
| `SGLANG_DISABLE_OUTLINES_DISK_CACHE` | Disable Outlines disk cache | `true` |
| `SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE` | Use SGLang's custom Triton kernel cache implementation for lower overheads (automatically enabled on CUDA) | `false` |

## Function Calling / Tool Use

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_TOOL_STRICT_LEVEL` | Controls the strictness level of tool call parsing and validation. <br>**Level 0**: Off - No strict validation <br>**Level 1**: Function strict - Enables structural tag constraints for all tools (even if none have `strict=True` set) <br>**Level 2**: Parameter strict - Enforces strict parameter validation for all tools, treating them as if they all have `strict=True` set | `0` |
