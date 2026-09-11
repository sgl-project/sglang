# SPDX-License-Identifier: Apache-2.0
"""The resolution pipeline: the ordered dispatcher every publishing entry runs.

``ServerArgs.resolve_once`` is the only caller. It lives here rather than on the
record because a step decides *about* the record; none of them is a member of
it.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.arg_groups.arg_utils import record_fields
from sglang.srt.arg_groups.overrides import (
    _page_size_default,
    _pipeline_parallel_overlap_disable,
    _sampling_backend_default,
    resolving_view,
    run_post_process_pass,
)
from sglang.srt.arg_groups.resolution_hooks import run_hook


def run_resolution_pipeline(server_args: Any) -> None:
    """
    Orchestrates the handling of various server arguments, ensuring proper configuration and validation.

    Dispatcher style principles:
    1. Keep this function as an ordered dispatcher. Each step should be a
       named call into an ``arg_groups`` family; put imports, conditionals,
       mutations, and raises inside the family instead of inline here.
    2. Keep the dummy-model boundary as early as correctness allows. Only
       model-independent bootstrap, API/network/protocol validation, and
       errors that should fire for dummy models should run before it.
    3. Order handlers by dependency domains, not by historical insertion:
       internal/bootstrap, API/network/protocol, model source/path
       resolution, hardware/platform, model-specific adjustment,
       parallelism, kernel/attention backend, cuda graph, memory/cache,
       and advanced/debug features.
    4. Hide narrow integrations behind general handler names. The
       dispatcher should say what phase is being handled, not expose a
       vendor-, hook-, or feature-specific implementation detail.
    5. Give each handler one clear contract: what state it expects, what it
       may mutate, and whether it validates only. Long ordering comments
       belong in the helper or signal that the helper should be split.
    6. Call each step through ``run_hook(handle_x, server_args, ...)``, not
       ``handle_x(server_args, ...)`` directly -- see
       ``arg_groups/resolution_hooks.py``. This is every step's fixed
       position in the pipeline either way; registering an override changes
       what runs here, never when.
    """

    # What the caller asked for, before any handler runs; this plus the
    # stash is the resolution result the projection reads.
    server_args._raw_input = {
        field.name: getattr(server_args, field.name)
        for field in record_fields(type(server_args))
    }

    # Preserve launcher-stage declarations made before Engine starts. They are
    # part of the same resolution result as the declarations accumulated below.
    server_args._resolved_overrides = list(
        getattr(server_args, "_resolved_overrides", ())
    )

    cfg = resolving_view(server_args)

    from sglang.srt.arg_groups.mega_moe_hook import handle_mega_moe

    run_hook(handle_mega_moe, server_args)
    from sglang.srt.arg_groups.serving_hook import (
        handle_asr_validation,
        handle_crash_dump_env,
        handle_debug_utils,
        handle_deprecated_args,
        handle_environment_variables,
        handle_grammar_backend,
        handle_load_balance_method,
        handle_media_url_security,
        handle_missing_default_values,
        handle_multimodal,
        handle_other_validations,
        handle_prefill_delayer_env_compat,
        handle_return_hidden_states_mode,
        handle_ssl_validation,
        handle_tokenizer_batching,
    )

    run_hook(handle_return_hidden_states_mode, server_args)
    run_hook(handle_media_url_security, server_args)
    from sglang.srt.arg_groups.hicache_hook import (
        handle_hicache,
        handle_hicache_ratio_default,
    )

    run_hook(handle_hicache_ratio_default, server_args)
    from sglang.srt.arg_groups.memory_hook import handle_offload_compatibility

    run_hook(handle_offload_compatibility, server_args)
    from sglang.srt.arg_groups.validation_hook import (
        default_unset_prefill_decode_interval,
        validate_experimental_sgl_marlin,
        validate_prefill_decode_interval,
        validate_sampling_mask_max_tokens,
    )

    run_hook(validate_prefill_decode_interval, server_args)
    run_hook(validate_sampling_mask_max_tokens, server_args)

    # Reject an explicitly enabled but incompatible hardware runtime before
    # model path resolution, downloads, or the dummy-model short circuit.
    from sglang.srt.arg_groups.parallel_hook import validate_prefill_cp_platform
    from sglang.srt.arg_groups.platform_hook import (
        handle_hardware_runtime_validation,
    )

    run_hook(validate_prefill_cp_platform, server_args)
    run_hook(handle_hardware_runtime_validation)
    if cfg.model_path.lower() in ["none", "dummy"]:
        return

    from sglang.srt.arg_groups.model_path_hook import (
        handle_load_format,
        handle_model_source_paths,
    )

    run_hook(handle_model_source_paths, server_args)

    # Validate mm_process_config.
    run_hook(handle_multimodal, server_args)
    # Validate SSL arguments early.
    run_hook(handle_ssl_validation, server_args)
    # Validate transcription/ASR-specific server args.
    run_hook(handle_asr_validation, server_args)

    # Handle deprecated arguments.
    run_hook(handle_deprecated_args, server_args)

    # Handle deprecated environment variables for prefill delayer.
    run_hook(handle_prefill_delayer_env_compat, server_args)

    # Set missing default values.
    run_hook(handle_missing_default_values, server_args)

    # expert_pack may replace a raw GGUF input with its generated local
    # model metadata before any model-specific handler calls model_config_of.
    # It also establishes eager-only invariants before CUDA graph parsing.
    from sglang.srt.arg_groups.expert_pack_hook import handle_expert_pack

    run_hook(handle_expert_pack, server_args)

    # Validate PD disaggregation flags before CUDA graph config.
    from sglang.srt.arg_groups.pd_disaggregation_hook import (
        handle_encoder_disaggregation,
        handle_pd_disaggregation,
    )

    run_hook(handle_pd_disaggregation, server_args)

    from sglang.srt.arg_groups.kv_cache_hook import (
        handle_cache_compatibility,
        handle_kv4_compatibility,
        handle_mxfp8_kv_cache_compatibility,
        handle_page_major_kv_layout,
        handle_prefill_only_disable_kv_cache,
        handle_unified_memory_pool,
        validate_prefill_only_disable_kv_cache_args,
    )
    from sglang.srt.arg_groups.parallel_hook import (
        handle_context_parallelism,
        handle_data_parallelism,
        handle_decode_context_parallelism,
        handle_dwdp,
        handle_elastic_ep,
        handle_eplb_and_dispatch,
        handle_expert_distribution_metrics,
    )

    run_hook(validate_prefill_only_disable_kv_cache_args, server_args)
    run_hook(handle_decode_context_parallelism, server_args)

    # Model-arch prefill CUDA-graph default must land before cuda-graph
    # resolution (the declarative registry materializes too late to affect
    # it). Inkling opts into full-graph prefill capture here.
    from sglang.srt.arg_groups.cuda_graph_hook import (
        apply_glm5_chunked_prefill_default,
        apply_glm5_prefill_cuda_graph_policy,
        apply_inkling_prefill_cuda_graph_default,
        apply_muse_glimmer_prefill_cuda_graph_max_bs_default,
        disable_prefill_cuda_graph_for_deepseek_trtllm_mla,
        handle_cuda_graph_config,
    )

    run_hook(apply_inkling_prefill_cuda_graph_default, server_args)
    run_hook(apply_muse_glimmer_prefill_cuda_graph_max_bs_default, server_args)

    # must run before _handle_cuda_graph_config and _handle_data_parallelism
    run_hook(handle_dwdp, server_args)

    run_hook(handle_cuda_graph_config, server_args)
    # Requires the parsed backend and explicit-input locks, and must precede
    # handle_gpu_memory_settings so the chunk size feeds memory budgeting.
    run_hook(apply_glm5_chunked_prefill_default, server_args)

    # Handle device-specific backends.
    from sglang.srt.arg_groups.platform_hook import (
        handle_amd_specifics,
        handle_cpu_backends,
        handle_hpu_backends,
        handle_mps_backends,
        handle_nccl_pre_warm,
        handle_npu_backends,
        handle_platform_defaults,
        handle_symm_mem_device_support,
        handle_xpu_backends,
    )

    run_hook(handle_hpu_backends, server_args)
    run_hook(handle_cpu_backends, server_args)
    run_hook(handle_npu_backends, server_args)
    run_hook(handle_mps_backends, server_args)
    run_hook(handle_xpu_backends, server_args)
    # Must precede handle_gpu_memory_settings: its symm-mem prealloc default
    # keys off enable_symm_mem.
    run_hook(handle_symm_mem_device_support, server_args)

    run_hook(handle_platform_defaults, server_args)

    # Handle memory-related, chunked prefill, and CUDA graph batch size configurations.
    from sglang.srt.arg_groups.memory_hook import handle_gpu_memory_settings

    run_hook(handle_gpu_memory_settings, server_args)

    # Apply model-specific adjustments.
    from sglang.srt.arg_groups.model_hook import (
        handle_model_capability_adjustments,
        handle_model_specific_adjustments,
    )

    run_hook(handle_model_specific_adjustments, server_args)
    run_hook(default_unset_prefill_decode_interval, server_args)
    # After the model overrides: Qwen4-Exp declares the PLE offload default there.
    run_hook(handle_offload_compatibility, server_args)

    # Set kernel backends.
    run_post_process_pass(server_args, _sampling_backend_default)
    # Must run before _handle_attention_backend_compatibility so the
    # deterministic backend is set before auto-detection fills it in.
    from sglang.srt.arg_groups.attention_hook import (
        handle_attention_backend_compatibility,
        handle_deterministic_inference,
        handle_linear_attn_backend,
        handle_multi_item_scoring,
    )

    run_hook(handle_deterministic_inference, server_args)
    run_hook(handle_attention_backend_compatibility, server_args)
    # Must run after the attention backend is resolved so the trtllm_mla
    # default (auto-selected for DeepseekV3ForCausalLM on sm100) is visible.
    run_hook(disable_prefill_cuda_graph_for_deepseek_trtllm_mla, server_args)
    from sglang.srt.arg_groups.mamba_hook import (
        handle_int8_mamba_checkpoint,
        handle_mamba_backend,
    )

    run_hook(handle_mamba_backend, server_args)
    run_hook(handle_int8_mamba_checkpoint, server_args)
    run_hook(handle_linear_attn_backend, server_args)
    run_hook(apply_glm5_prefill_cuda_graph_policy, server_args)
    run_hook(handle_kv4_compatibility, server_args)
    run_hook(handle_mxfp8_kv_cache_compatibility, server_args)
    run_post_process_pass(server_args, _page_size_default)
    run_hook(handle_amd_specifics, server_args)
    run_hook(handle_nccl_pre_warm, server_args)
    run_hook(handle_grammar_backend, server_args)

    # Handle multi-item scoring constraints. Must run after the above so
    # the final attention backend and chunked_prefill_size are in effect.
    run_hook(handle_multi_item_scoring, server_args)

    # Backend-dependent half of --prefill-only-disable-kv-cache validation.
    # Must stay after _handle_attention_backend_compatibility() (above) and
    # _handle_multi_item_scoring() so the resolved prefill backend is final;
    # the flag/precondition half runs earlier in
    # _validate_prefill_only_disable_kv_cache_args().
    run_hook(handle_prefill_only_disable_kv_cache, server_args)

    # Handle Hicache settings.
    run_hook(handle_hicache, server_args)

    # Handle data parallelism.
    run_hook(handle_data_parallelism, server_args)

    # Normalize load balancing defaults.
    run_hook(handle_load_balance_method, server_args)

    # Handle context parallelism.
    run_hook(handle_context_parallelism, server_args)

    # Handle MoE configurations.
    from sglang.srt.arg_groups.moe_hook import (
        handle_a2a_moe,
        handle_moe_kernel_config,
        validate_cutedsl_a2a_token_budget,
        validate_deepep_v2_dispatch_token_budget,
        validate_deepep_v2_speculative_draft,
    )

    run_hook(handle_moe_kernel_config, server_args)
    run_hook(handle_a2a_moe, server_args)
    run_hook(handle_eplb_and_dispatch, server_args)
    run_hook(handle_expert_distribution_metrics, server_args)
    run_hook(handle_elastic_ep, server_args)
    run_hook(validate_experimental_sgl_marlin, server_args)

    # Handle pipeline parallelism.
    run_post_process_pass(server_args, _pipeline_parallel_overlap_disable)

    # Handle speculative decoding logic.

    from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding

    run_hook(handle_speculative_decoding, server_args)

    # After the speculative hook so speculative_algorithm is final.
    from sglang.srt.arg_groups.layernorm_sp_hook import handle_layernorm_sp

    run_hook(handle_layernorm_sp, server_args)

    # Validate the CuteDSL A2A token budget now that num_tokens_per_req is final.
    run_hook(validate_cutedsl_a2a_token_budget, server_args)

    # Handle model loading format.
    run_hook(handle_load_format, server_args)

    # Handle Encoder disaggregation.
    run_hook(handle_encoder_disaggregation, server_args)

    # Validate tokenizer settings.
    run_hook(handle_tokenizer_batching, server_args)

    # Propagate environment variables.
    run_hook(handle_environment_variables, server_args)

    # Validate cache settings.
    run_hook(handle_cache_compatibility, server_args)

    run_hook(handle_page_major_kv_layout, server_args)

    run_hook(handle_unified_memory_pool, server_args)

    # Handle diffusion LLM inference.
    from sglang.srt.arg_groups.dllm_hook import handle_dllm_inference

    run_hook(handle_dllm_inference, server_args)

    # Handle crash dump environment variables (must run before CUDA init).
    run_hook(handle_crash_dump_env, server_args)

    # Handle debug utilities.
    run_hook(handle_debug_utils, server_args)

    # Handle any other necessary validations.
    run_hook(handle_other_validations, server_args)

    # Model-capability adjustments that legacy code applied at model-load
    # time; last declarations of the resolution, mirroring that order.
    run_hook(handle_model_capability_adjustments, server_args)

    # Validate after all batch-size declarations are visible.
    run_hook(validate_deepep_v2_speculative_draft, server_args)
    run_hook(validate_deepep_v2_dispatch_token_budget, server_args)

    server_args._resolution_finished = True
