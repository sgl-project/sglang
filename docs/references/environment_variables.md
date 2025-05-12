# Environment Variables

SGLang supports various environment variables that can be used to configure its behavior. This document provides a comprehensive list of all supported environment variables.

## General Configuration

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_USE_MODELSCOPE` | Enable using models from ModelScope | `false` |
| `SGLANG_HOST_IP` | Host IP address for the server | `0.0.0.0` |
| `SGLANG_PORT` | Port for the server | auto-detected |
| `SGLANG_LOGGING_CONFIG_PATH` | Custom logging configuration path | Not set |
| `SGLANG_DISABLE_REQUEST_LOGGING` | Disable request logging | `false` |
| `SGLANG_HEALTH_CHECK_TIMEOUT` | Timeout for health check in seconds | `20` |

## Performance Tuning

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_ENABLE_TORCH_INFERENCE_MODE` | Control whether to use torch.inference_mode | `false` |
| `SGLANG_ENABLE_TORCH_COMPILE` | Enable torch.compile | `true` |
| `SGLANG_SET_CPU_AFFINITY` | Enable CPU affinity setting | `false` |
| `SGLANG_IS_FLASHINFER_AVAILABLE` | Control FlashInfer availability check | `true` |
| `SGLANG_SKIP_P2P_CHECK` | Skip P2P (peer-to-peer) access check | `false` |

## Memory Management

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_DEBUG_MEMORY_POOL` | Enable memory pool debugging | `false` |
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | Clip max new tokens estimation | Not set |
| `SGLANG_DETOKENIZER_MAX_STATES` | Maximum states for detokenizer | Default value based on system |

## Model-Specific Options

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_AITER_MOE` | Use AITER MOE implementation | `false` |
| `SGLANG_INT4_WEIGHT` | Enable INT4 weight quantization | `false` |
| `SGLANG_MOE_PADDING` | Enable MOE padding (128 if set to 1) | `0` |
| `SGLANG_FORCE_FP8_MARLIN` | Force using FP8 MARLIN | `false` |
| `SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION` | Enable RoPE fusion in MLA | `1` |
| `SGLANG_PP_LAYER_PARTITION` | Pipeline parallel layer partition | Not set |

## Distributed Computing

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_BLOCK_NONZERO_RANK_CHILDREN` | Control blocking of non-zero rank children | `1` |

## Testing & Debugging (CI)

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_IS_IN_CI` | Indicates if running in CI environment | `false` |
| `SGLANG_AMD_CI` | Indicates running in AMD CI environment | `0` |
| `SGLANG_TEST_RETRACT` | Enable retract decode testing | `false` |
| `SGLANG_RECORD_STEP_TIME` | Record step time for profiling | `false` |
| `SGLANG_TEST_REQUEST_TIME_STATS` | Test request time statistics | `false` |
| `SGLANG_CI_SMALL_KV_SIZE` | Use small KV cache size in CI | Not set |

## Profiling & Benchmarking

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_TORCH_PROFILER_DIR` | Directory for PyTorch profiler output | `/tmp` |

## Storage & Caching

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SGLANG_DISABLE_OUTLINES_DISK_CACHE` | Disable Outlines disk cache | `true` |
