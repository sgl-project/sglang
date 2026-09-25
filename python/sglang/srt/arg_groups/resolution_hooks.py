"""Plugin overrides for named resolution steps.

Overrides receive ``(server_args, previous)`` and may call ``previous`` to
wrap the existing step. Only whitelisted names are accepted. Registration
changes the implementation at that step's existing pipeline position.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List

# Names must match run_hook call sites in pipeline.py. An override applies
# to every invocation of that name, including repeated steps.
_OVERRIDABLE_HOOKS: FrozenSet[str] = frozenset(
    {
        "handle_mega_moe",
        "handle_return_hidden_states_mode",
        "handle_media_url_security",
        "handle_hicache_ratio_default",
        "handle_offload_compatibility",
        "validate_prefill_decode_interval",
        "default_unset_prefill_decode_interval",
        "validate_response_store",
        "validate_sampling_mask_max_tokens",
        "validate_prefill_cp_platform",
        "handle_hardware_runtime_validation",
        "handle_model_source_paths",
        "handle_multimodal",
        "handle_ssl_validation",
        "handle_asr_validation",
        "handle_deprecated_args",
        "handle_prefill_delayer_env_compat",
        "handle_missing_default_values",
        "handle_expert_pack",
        "handle_pd_disaggregation",
        "validate_prefill_only_disable_kv_cache_args",
        "handle_decode_context_parallelism",
        "apply_inkling_prefill_cuda_graph_default",
        "apply_muse_glimmer_prefill_cuda_graph_max_bs_default",
        "handle_dwdp",
        "handle_cuda_graph_config",
        "apply_glm5_chunked_prefill_default",
        "handle_hpu_backends",
        "handle_cpu_backends",
        "handle_npu_backends",
        "handle_mps_backends",
        "handle_xpu_backends",
        "handle_symm_mem_device_support",
        "handle_platform_defaults",
        "handle_gpu_memory_settings",
        "handle_model_specific_adjustments",
        "handle_deterministic_inference",
        "handle_nvfp4_prefill_kv_dequant_dtype",
        "handle_attention_backend_compatibility",
        "disable_prefill_cuda_graph_for_deepseek_trtllm_mla",
        "handle_mamba_backend",
        "handle_int8_mamba_checkpoint",
        "handle_linear_attn_backend",
        "apply_glm5_prefill_cuda_graph_policy",
        "handle_kv4_compatibility",
        "handle_mxfp8_kv_cache_compatibility",
        "handle_amd_specifics",
        "handle_nccl_pre_warm",
        "handle_grammar_backend",
        "handle_multi_item_scoring",
        "handle_prefill_only_disable_kv_cache",
        "handle_hicache",
        "handle_data_parallelism",
        "handle_load_balance_method",
        "handle_context_parallelism",
        "handle_moe_kernel_config",
        "handle_a2a_moe",
        "handle_eplb_and_dispatch",
        "handle_expert_distribution_metrics",
        "handle_shared_experts_tp",
        "handle_elastic_ep",
        "validate_experimental_sgl_marlin",
        "handle_speculative_decoding",
        "handle_layernorm_sp",
        "validate_cutedsl_a2a_token_budget",
        "validate_mega_moe_token_budget_for_model",
        "handle_load_format",
        "handle_encoder_disaggregation",
        "handle_tokenizer_batching",
        "handle_environment_variables",
        "handle_cache_compatibility",
        "handle_page_major_kv_layout",
        "handle_unified_memory_pool",
        "handle_dllm_inference",
        "handle_crash_dump_env",
        "handle_debug_utils",
        "handle_other_validations",
        "handle_model_capability_adjustments",
        "validate_deepep_v2_speculative_draft",
        "validate_deepep_v2_dispatch_token_budget",
    }
)

# Step name -> wrappers in registration order.
_HOOKS: Dict[str, List[Callable[[Any, Callable[[Any], None]], None]]] = {}


def register_resolution_hook(name: str):
    """Register ``fn(server_args, previous)`` for a pipeline step.

    ``previous(server_args)`` calls the previously registered wrapper, or the
    built-in step for the first registration. Omitting it replaces the step.
    The last registration runs outermost, so plugin import order matters.
    """
    if name not in _OVERRIDABLE_HOOKS:
        raise ValueError(
            f"{name!r} is not an overridable resolution hook; the "
            f"overridable set is {sorted(_OVERRIDABLE_HOOKS)}. A new entry "
            "needs a matching `run_hook(...)` call at the step's site in "
            "pipeline.py, not just a name here."
        )

    def decorator(fn):
        _HOOKS.setdefault(name, []).append(fn)
        return fn

    return decorator


def run_hook(builtin: Callable[[Any], None], server_args: Any) -> None:
    """Run registered wrappers for ``builtin.__name__``, or the built-in step.

    ``builtin`` must be a named function. Wrappers receive the same server
    arguments as the built-in step.
    """
    step = builtin
    for fn in _HOOKS.get(builtin.__name__, ()):
        step = _bind(fn, step)
    step(server_args)


def _bind(fn, previous):
    def wrapped(server_args):
        fn(server_args, previous)

    return wrapped
