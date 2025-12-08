# Deprecated Features, Server Arguments and Environment Variables

This document lists all deprecated server arguments and environment variables in SGLang, along with their replacements and removal timelines.

:::{warning}
**For Developers**: Please be careful when modifying deprecated features. Always check this document before removing or changing deprecated items, and ensure proper migration paths are documented. Deprecated items may still be in use by existing code or configurations.
:::

## Deprecated Server Arguments

### MoE Runner Backend Arguments
These arguments are deprecated in favor of `--moe-runner-backend`:

| Deprecated Argument | Replacement | Notes |
|---------------------|-------------|-------|
| `--enable-ep-moe` | `--ep-size` (set to same value as `--tp-size`) | Use `--ep-size` instead |
| `--enable-deepep-moe` | `--moe-a2a-backend=deepep` | Use `--moe-a2a-backend` instead |
| `--enable-flashinfer-cutlass-moe` | `--moe-runner-backend=flashinfer_cutlass` | Use `--moe-runner-backend` instead |
| `--enable-flashinfer-cutedsl-moe` | `--moe-runner-backend=flashinfer_cutedsl` | Use `--moe-runner-backend` instead |
| `--enable-flashinfer-trtllm-moe` | `--moe-runner-backend=flashinfer_trtllm` | Use `--moe-runner-backend` instead |
| `--enable-triton-kernel-moe` | `--moe-runner-backend=triton_kernel` | Use `--moe-runner-backend` instead |
| `--enable-flashinfer-mxfp4-moe` | `--moe-runner-backend=flashinfer_mxfp4` | Use `--moe-runner-backend` instead |

### Tool Call Parser Arguments
| Deprecated Value | Replacement | Notes |
|------------------|-------------|-------|
| `tool_call_parser=qwen25` | `tool_call_parser=qwen` | Use `qwen` instead |
| `tool_call_parser=glm45` | `tool_call_parser=glm` | Use `glm` instead |

### Other Deprecated Arguments
| Deprecated Argument | Replacement | Notes |
|---------------------|-------------|-------|
| `--nccl-init-addr` | `--dist-init-addr` | For backward compatibility, will be removed in the future. Use `--dist-init-addr` instead |
| `--enable-lora` | `--lora-paths` | Automatically set to `True` when `--lora-paths` is provided for backward compatibility. No need to set `--enable-lora` explicitly |
| `--modelopt-quant` | `--quantization` | Legacy flag, being replaced by unified quantization flags. Use `--quantization` instead |

## Deprecated Environment Variables

### Environment Variables Deprecated in Favor of CLI Flags

These environment variables are deprecated and will be **completely removed in v0.5.7**. Use the corresponding CLI flags instead:

| Deprecated Env Var | Replacement CLI Flag | Removal Version |
|-------------------|----------------------|-----------------|
| `SGLANG_ENABLE_FLASHINFER_FP8_GEMM` | `--fp8-gemm-backend=flashinfer_trtllm` | v0.5.7 |
| `SGLANG_ENABLE_FLASHINFER_GEMM` | `--fp8-gemm-backend=flashinfer_trtllm` | v0.5.7 |
| `SGLANG_SUPPORT_CUTLASS_BLOCK_FP8` | `--fp8-gemm-backend=cutlass` | v0.5.7 |
| `SGLANG_CUTLASS_MOE` | `--moe-runner-backend=cutlass` or `--speculative-moe-runner-backend=cutlass` | v0.5.7 |

### Environment Variables with Renamed Equivalents

These environment variables are deprecated in favor of new names (old names are automatically converted):

| Deprecated Env Var | New Env Var | Notes |
|-------------------|-------------|-------|
| `SGLANG_LOG_GC` | `SGLANG_GC_LOG` | Old name is automatically converted |
| `SGLANG_ENABLE_FLASHINFER_FP8_GEMM` | `SGLANG_ENABLE_FLASHINFER_GEMM` | Old name is automatically converted (but both are deprecated in favor of CLI flag) |
| `SGLANG_MOE_NVFP4_DISPATCH` | `SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH` | Old name is automatically converted |
| `SGL_*` (any variable) | `SGLANG_*` (same variable) | All `SGL_` prefixed variables are deprecated in favor of `SGLANG_` prefix |

### Other Deprecated Environment Variables

| Deprecated Env Var | Notes |
|-------------------|-------|
| `USE_VLLM_CUTLASS_W8A8_FP8_KERNEL` | vLLM dependency, marked as deprecated (can be removed safely) |
| `USE_TRITON_W8A8_FP8_KERNEL` | SGLang FP8 quantization flag, marked as deprecated (can be removed safely). Used to force Triton kernels instead of CUTLASS for W8A8 FP8 operations |
| `SGLANG_ENABLE_DETERMINISTIC_INFERENCE` | Set automatically when `rl_on_policy_target` is used. TODO: remove this environment variable as a whole |

## Deprecated HTTP Endpoints

| Deprecated Endpoint | Replacement | Notes |
|---------------------|-------------|-------|
| `/get_model_info` | `/model_info` | Will be removed in a future version |
| `/get_weight_version` | `/weight_version` | Will be removed in a future version |
| `/get_server_info` | `/server_info` | Will be removed in a future version |

## Deprecated API Fields

| Deprecated Field | Replacement | Notes |
|------------------|-------------|-------|
| `max_tokens` (in OpenAI API) | `max_completion_tokens` | Use `max_completion_tokens` instead |

## Deprecated LoRA Backends

| Deprecated Backend | Replacement | Notes |
|-------------------|-------------|-------|
| `flashinfer` | `triton` | FlashInfer LoRA backend has been deprecated, use `triton` instead |

## Deprecated Load Balancing Methods

| Deprecated Method | Replacement | Notes |
|------------------|-------------|-------|
| `minimum_tokens` | `round_robin` | The 'minimum_tokens' load balancing method is deprecated and will be introduced later. Falls back to 'round_robin' |

## Deprecated Internal Features

| Feature | Notes |
|---------|-------|
| `forward_deepgemm_contiguous` | Deprecated, will assert if used |
| `forward_deepgemm_masked` | Deprecated, will assert if used |
| `head_first` (in FLA chunk) | Deprecated and will be removed in a future version |
| `kv_scale` (in checkpoint format) | Deprecated in favor of separate `k_scale` and `v_scale` tensors |
| `batch_get` and `batch_set` (in HiCache storage) | TODO: Deprecate - internal API methods |
| `_generic_page_get` and `_generic_page_set` (in cache controller) | TODO: Deprecate - internal API methods |

## Priority Rules

**Important**: When both server arguments and environment variables are set for the same configuration:

- **Server arguments take priority** over environment variables
- If `--fp8-gemm-backend` is explicitly set (not "auto"), it will override `SGLANG_ENABLE_FLASHINFER_FP8_GEMM` and `SGLANG_SUPPORT_CUTLASS_BLOCK_FP8`
- A warning will be logged when environment variables are overridden by server arguments

## Migration Guide

1. **For FP8 GEMM backend**: Replace environment variables with `--fp8-gemm-backend` CLI flag
2. **For MoE runner backend**: Replace `--enable-*-moe` flags with `--moe-runner-backend=<backend>`
3. **For environment variable prefixes**: Update all `SGL_` prefixed variables to `SGLANG_`
4. **For tool call parsers**: Update `qwen25` → `qwen` and `glm45` → `glm`
5. **For LoRA backend**: Replace `flashinfer` LoRA backend with `triton`
6. **For load balancing**: Replace `minimum_tokens` method with `round_robin` (or wait for future implementation)
7. **For distributed init**: Replace `--nccl-init-addr` with `--dist-init-addr`
8. **For LoRA**: Remove `--enable-lora` flag and just use `--lora-paths` (it's automatically enabled)
9. **For ModelOpt quantization**: Replace `--modelopt-quant` with `--quantization` flag

:::{seealso}
- [Server Arguments](../advanced_features/server_arguments.md)
- [Environment Variables](environment_variables.md)
:::

## Deprecation Policy

:::{important}
Deprecated features, server arguments, and environment variables will only be removed in major version releases (e.g., v0.5.6 → v0.6.7). This ensures users have sufficient time to migrate to the recommended alternatives and provides a stable upgrade path between major versions.
:::
