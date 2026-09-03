# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for per-model and per-capability adjustments."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    _deepseek_moe_quant_resolution,
    _deepseek_spec_moe_resolution,
    _dsa_kv_cache_dtype_default,
    _dsa_split_backend_resolution,
    _enforce_disable_allreduce_fusion,
    _flashinfer_allreduce_fusion_auto_enable,
    _hrm_text_attention_force,
    _mamba_radix_cache_resolution,
    _sparse_head_overlap_disable,
    attention_backends_of,
    collect_model_override_declarations,
    declare_resolution,
    mamba_cache_chunk_size,
    mamba_extra_buffer_of,
    model_config_of,
    resolved_view,
    resolving_view,
    run_post_process_pass,
    use_mla_backend,
    validate_declarations,
)
from sglang.srt.configs.embedding_model_spec import BCGPrefillPolicy
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_spec_by_arch
from sglang.srt.connector import ConnectorType
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import (
    get_quantization_config,
    is_mps,
    parse_connector_type,
)

logger = logging.getLogger(__name__)


def handle_model_specific_adjustments(server_args: Any):

    cfg = resolving_view(server_args)
    from sglang.srt.configs.model_config import (
        get_mimo_v2_fused_qkv_expected_tp_size,
        is_deepseek_dsa,
    )

    if cfg.enable_deterministic_inference:
        declare_resolution(
            server_args,
            "_handle_model_specific_adjustments",
            enforce_disable_flashinfer_allreduce_fusion=True,
        )

    declare_resolution(
        server_args,
        "_handle_model_specific_adjustments",
        uses_mamba_radix_cache=False,
    )
    if parse_connector_type(cfg.model_path) == ConnectorType.INSTANCE:
        # No model overrides for an instance connector: no hf_config to
        # key them on.
        return

    model_config = model_config_of(server_args)
    hf_config = model_config.hf_config
    model_arch = hf_config.architectures[0]

    if model_arch == "InternS2MobiusForConditionalGeneration":
        unsupported = []
        if cfg.pp_size != 1:
            unsupported.append("pipeline parallelism (--pp-size must be 1)")
        if cfg.ep_size != 1:
            unsupported.append("expert parallelism (--ep-size must be 1)")
        if unsupported:
            raise ValueError(
                "Intern-S2-Mobius does not support: " + "; ".join(unsupported) + "."
            )

    if cfg.enable_dsa_cache_layer_split and not is_deepseek_dsa(hf_config):
        raise ValueError(
            "--enable-dsa-cache-layer-split is only supported for DSA "
            "(DeepSeek Sparse Attention) models."
        )

    if cfg.enable_cp_decode_attn_tp:
        from sglang.srt.layers.cp.cp_decode_attn_tp import (
            CP_DECODE_ATTN_TP_SUPPORTED_ARCHS,
        )

        if model_arch not in CP_DECODE_ATTN_TP_SUPPORTED_ARCHS:
            raise ValueError(
                "--enable-cp-decode-attn-tp is only supported for models "
                "whose attention linears are replicated across CP ranks "
                f"(attn_tp_size=1). Got {model_arch}; supported: "
                f"{sorted(CP_DECODE_ATTN_TP_SUPPORTED_ARCHS)}."
            )

    _hybrid_spec = get_linear_attn_spec_by_arch(model_arch)
    if _hybrid_spec is not None and _hybrid_spec.uses_mamba_radix_cache:
        handle_mamba_radix_cache(server_args, model_arch)

    # Collect the declarative model overrides (registry) on the
    # pristine config and stash them for publish-time flags resolution;
    # server_args is never mutated — mid-resolution readers see the
    # declared values through resolved_view, runtime readers through the
    # flags tier.

    model_overrides = collect_model_override_declarations(
        model_arch, server_args, hf_config
    )
    validate_declarations(server_args, model_overrides)
    server_args._resolved_overrides.extend(model_overrides)

    if model_arch in (
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
    ):
        from sglang.srt.arg_groups.kimi_k3_hook import (
            apply_kimi_k3_linear_attn_defaults,
            apply_kimi_k3_spec_backend_defaults,
        )

        apply_kimi_k3_linear_attn_defaults(server_args)
        apply_kimi_k3_spec_backend_defaults(server_args)

    if model_arch in [
        "DeepseekV4ForCausalLM",
    ]:
        from sglang.srt.arg_groups.deepseek_v4_hook import (
            apply_deepseek_v4_defaults,
        )

        apply_deepseek_v4_defaults(server_args, model_arch)

    if model_arch in [
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "KimiK25ForConditionalGeneration",
        "MistralLarge3ForCausalLM",
        "PixtralForConditionalGeneration",
        "GlmMoeDsaForCausalLM",
        "LongcatFlashForCausalLM",
        "Dots3NoteForCausalLM",
    ]:
        # Set attention backend for DeepSeek
        if is_deepseek_dsa(hf_config):  # DeepSeek 3.2/GLM 5
            if envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.is_set():
                logger.warning(
                    f"Dense attention kv len threshold is manually set to {envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()} for DSA. Caution: This may cause performance regression if the threshold is larger than the index topk of model."
                )
            else:
                # When threshold is not manually set, set it to the index topk of model
                from sglang.srt.configs.model_config import get_dsa_index_topk

                envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.set(
                    get_dsa_index_topk(hf_config)
                )
                logger.warning(
                    f"Set dense attention kv len threshold to model index_topk={envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()} for DeepSeek with DSA."
                )
            # The "dsa" attention fill moved to the override registry
            # (arg_groups/overrides.py: _deepseek_family_overrides).

            index_topk_freq = getattr(hf_config, "index_topk_freq", 1) or 1
            index_topk_pattern = getattr(hf_config, "index_topk_pattern", None)
            if cfg.enable_two_batch_overlap and (
                index_topk_freq > 1
                or (index_topk_pattern is not None and "S" in index_topk_pattern)
            ):
                raise ValueError(
                    "--enable-two-batch-overlap is not supported with DSA "
                    "index-topk sharing (index_topk_freq > 1 or an "
                    "index_topk_pattern containing shared layers): the TBO op "
                    "path does not propagate topk indices across layers, so "
                    "shared layers would run sparse attention without indices."
                )

            if (
                not get_platform().is_npu and not get_platform().is_xpu
            ):  # CUDA or ROCm GPU
                if cfg.enable_prefill_cp:
                    # The DSA CP field declarations moved to the override
                    # registry (arg_groups/overrides.py:
                    # _deepseek_family_overrides).
                    declare_resolution(
                        server_args,
                        "_handle_model_specific_adjustments",
                        cuda_graph_config=with_phase(
                            cfg.cuda_graph_config,
                            Phase.PREFILL,
                            backend=Backend.DISABLED,
                        ),
                    )
                else:
                    # Pure TP and partial DP Attention mode is active for DSA, logging a warning
                    if cfg.dp_size < cfg.tp_size:
                        logger.warning(
                            f"DSA with TP mode is active, dp_size={cfg.dp_size}, tp_size={cfg.tp_size}, "
                            f"attn_tp_size={cfg.tp_size}, attention weights will be sharded across {cfg.tp_size} ranks."
                        )

                # The DSA page-size selection moved to the override registry
                # (arg_groups/overrides.py: _deepseek_family_overrides).

                import torch

                major, _ = torch.cuda.get_device_capability()

                run_post_process_pass(server_args, _dsa_kv_cache_dtype_default)
                run_post_process_pass(server_args, _dsa_split_backend_resolution)

            if cfg.enable_prefill_cp:
                assert (
                    cfg.disaggregation_mode != "decode"
                ), "CP is only supported for prefill when PD disaggregation, please remove --enable-prefill-cp."
            if (
                cfg.enable_dsa_cache_layer_split
                and cfg.disaggregation_mode != "prefill"
            ):
                if cfg.disaggregation_mode == "decode":
                    raise ValueError(
                        "--enable-dsa-cache-layer-split is not supported on "
                        "decode workers. This flag is a prefill-CP "
                        "optimization; decode receives full cache shards "
                        "through PD transfer."
                    )
                raise ValueError(
                    "--enable-dsa-cache-layer-split is only supported on PD "
                    "prefill workers. Non-PD workers also run decode and "
                    "require ordinary local decode cache semantics."
                )
            if cfg.enable_dsa_cache_layer_split and (
                not cfg.enable_prefill_cp or cfg.cp_strategy != "interleave"
            ):
                raise ValueError(
                    "--enable-dsa-cache-layer-split requires "
                    "--enable-prefill-cp and --cp-strategy interleave "
                    "(or legacy --enable-nsa-prefill-context-parallel with "
                    "--nsa-prefill-cp-mode round-robin-split)."
                )
            # Layer split relies on the mooncake all-CP-rank KV/indexer
            # transfer path. mori/nixl support is a temporary limitation
            # and will be added later by the community.
            if (
                cfg.enable_dsa_cache_layer_split
                and cfg.disaggregation_transfer_backend != "mooncake"
            ):
                raise ValueError(
                    "--enable-dsa-cache-layer-split currently only supports "
                    "the mooncake transfer backend (mooncake / mooncake_tcp). "
                    f"Got --disaggregation-transfer-backend "
                    f"{cfg.disaggregation_transfer_backend!r}. mori/nixl "
                    "support will be added later by the community."
                )
            if cfg.enable_dsa_cache_layer_split and cfg.pp_size > 1:
                raise ValueError(
                    "--enable-dsa-cache-layer-split is not supported with "
                    "pipeline parallelism (pp_size > 1) yet. It requires "
                    "prefill context parallelism, and CP + PP has not been "
                    "validated for this feature."
                )

        else:
            # DeepSeek V3/R1/V3.1
            if cfg.cuda_graph_config.prefill.backend != Backend.DISABLED:
                logger.info("Piecewise CUDA graph is enabled, use MLA for prefill.")

            # The sm100 trtllm_mla fill moved to the override registry
            # (arg_groups/overrides.py: _deepseek_family_overrides).

            # MLA prefill CP auto-config: the field declarations moved to
            # the override registry (arg_groups/overrides.py:
            # _deepseek_family_overrides).
            if cfg.enable_prefill_cp and use_mla_backend(server_args):
                declare_resolution(
                    server_args,
                    "_handle_model_specific_adjustments",
                    cuda_graph_config=with_phase(
                        cfg.cuda_graph_config,
                        Phase.PREFILL,
                        backend=Backend.DISABLED,
                    ),
                )

        # Set moe backend for DeepSeek: the sm100 quant/moe resolution
        # moved to the resolution pipeline (arg_groups/overrides.py:
        # _deepseek_moe_quant_resolution -- a slot pass, because the DSA
        # kv-cache-dtype default above must read the pristine
        # quantization). The HIP arm (fusion log + spec_moe writes, the
        # latter awaiting the speculative-hook migration) stays below.

        run_post_process_pass(server_args, _deepseek_moe_quant_resolution)
        if get_platform().is_hip:
            if is_deepseek_dsa(hf_config):
                # The fused top-k v2 kernel (topk_transform_paged_v2) is a
                # CUDA/Hopper-only path: its JIT source includes
                # <cooperative_groups.h> and uses cg::this_cluster()
                # (thread-block clusters), neither of which exists on ROCm,
                # so it fails to JIT-compile on gfx9xx during CUDA-graph
                # capture. DeepSeek-V4 already disables it on HIP; mirror that
                # here for the rest of the DSA family (DeepSeek-V3.2 /
                # GLM-5.x) that shares the same decode top-k path.
                envs.SGLANG_OPT_USE_TOPK_V2.set(False)
            if model_arch == "GlmMoeDsaForCausalLM":
                # Open the fused top-k v2 kernel for the GLM-5.x DSA
                # family on ROCm: it shares this decode top-k path, and
                # the kernel's ROCm build compiles the streaming levels
                # on gfx9xx. Order is load-bearing: the blanket disable
                # above `set`s the variable unconditionally, so this has
                # to follow it.
                envs.SGLANG_OPT_USE_TOPK_V2.set(True)
            if not resolved_view(server_args).enable_dp_attention and cfg.nnodes == 1:
                # TODO (Hubert): Put this back later
                # server_args.enable_aiter_allreduce_fusion = True
                logger.info("Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM")

            # The fp4-checkpoint draft spec-MoE resolution moved to the
            # resolution pipeline (arg_groups/overrides.py:
            # _deepseek_spec_moe_resolution), invoked here at its legacy
            # slot.

            run_post_process_pass(server_args, _deepseek_spec_moe_resolution)

    elif model_arch in [
        "DeepseekV4ForCausalLM",
    ]:
        from sglang.srt.arg_groups.deepseek_v4_hook import (
            validate_deepseek_v4_cp,
            validate_deepseek_v4_mega_moe_token_budget,
        )

        validate_deepseek_v4_cp(server_args)
        validate_deepseek_v4_mega_moe_token_budget(server_args)

        if get_platform().is_sm120:
            # SM120 lacks tcgen05/TMEM: disable features that depend on
            # DeepGEMM or require >99KB SMEM (topk_v2).
            envs.SGLANG_OPT_FP8_WO_A_GEMM.set(False)
            envs.SGLANG_OPT_USE_TOPK_V2.set(False)
            if not envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.is_set():
                envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.set(False)
            if not envs.SGLANG_OPT_FUSE_MHC_POST_PRE.is_set():
                envs.SGLANG_OPT_FUSE_MHC_POST_PRE.set(True)
            if not envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.is_set():
                envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.set(False)
            # Out of the box the indexer runs the TileLang kernel (works on
            # stock DeepGEMM); both knobs stay env-overridable so a DeepGEMM
            # build with SM120 attention support can opt into
            # fp8_paged_mqa_logits by setting them to 0.
            if not envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.is_set():
                envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(True)
            if not envs.SGLANG_OPT_USE_TILELANG_INDEXER.is_set():
                envs.SGLANG_OPT_USE_TILELANG_INDEXER.set(True)
        elif get_platform().is_hip:
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.set(False)
            envs.SGLANG_OPT_FP8_WO_A_GEMM.set(False)
            envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.set(False)
            envs.SGLANG_OPT_USE_TOPK_V2.set(True)
            envs.SGLANG_OPT_USE_AITER_INDEXER.set(True)
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.set(False)
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.set(False)
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(True)
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.set(False)
            envs.SGLANG_EAGER_INPUT_NO_COPY.set(True)

    elif model_arch in ["GptOssForCausalLM"]:
        # Attention backend selection + XPU dtype validation moved to the
        # override registry (arg_groups/overrides.py: _gpt_oss_overrides).
        # Exempt MLX only: none of these backends exist on MPS, and MLX runs
        # attention inside its own runner, so attention_backend is still
        # unset here.  Plain macOS stays on the list -- torch_native has
        # neither sliding window nor attention sinks.
        if not (is_mps() and use_mlx()):
            supported_backends = [
                "triton",
                "trtllm_mha",
                "fa3",
                "fa4",
                "ascend",
                "intel_amx",
                "intel_xpu",
                "aiter",
            ]
            prefill_attn_backend, decode_attn_backend = attention_backends_of(
                resolved_view(server_args)
            )
            assert (
                prefill_attn_backend in supported_backends
                and decode_attn_backend in supported_backends
            ), (
                f"GptOssForCausalLM requires one of {supported_backends} attention backend, but got the following backends\n"
                f"- Prefill: {prefill_attn_backend}\n"
                f"- Decode: {decode_attn_backend}\n"
            )

        quant_method = get_quantization_config(hf_config)
        is_mxfp4_quant_format = quant_method == "mxfp4"
        if (
            not resolved_view(server_args).enable_dp_attention
            and cfg.nnodes == 1
            and get_platform().is_hip
        ):
            # TODO (Hubert): Put this back later
            # server_args.enable_aiter_allreduce_fusion = True
            logger.info("Enable Aiter AllReduce Fusion for GptOssForCausalLM")
        quantization_config = getattr(hf_config, "quantization_config", None)
        is_mxfp4_quant_format = (
            quantization_config is not None
            and quantization_config.get("quant_method") == "mxfp4"
        )
        # The mxfp4 dtype override moved to the override registry
        # (arg_groups/overrides.py: _gpt_oss_overrides).

        # The moe_runner_backend selection moved to the override registry
        # (arg_groups/overrides.py: _gpt_oss_overrides).

        if resolved_view(server_args).moe_runner_backend == "triton_kernel":
            assert (
                resolved_view(server_args).ep_size == 1
            ), "Triton kernel MoE is only supported when ep_size == 1"

    elif model_arch in ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM"):
        if model_arch == "MiMoV2ForCausalLM" and not cfg.encoder_only:
            expected_attn_tp_size = get_mimo_v2_fused_qkv_expected_tp_size(hf_config)
            view = resolved_view(server_args)
            attn_dp_size = cfg.dp_size if view.enable_dp_attention else 1
            effective_attn_tp_size = cfg.tp_size // attn_dp_size // view.attn_cp_size
            if (
                expected_attn_tp_size is not None
                and expected_attn_tp_size % effective_attn_tp_size != 0
            ):
                raise ValueError(
                    "MiMoV2ForCausalLM requires effective attention TP "
                    f"size {expected_attn_tp_size} because its fused "
                    "qkv_proj weights are "
                    f"TP={expected_attn_tp_size}-interleaved; got "
                    f"{effective_attn_tp_size} "
                    f"(tp_size={cfg.tp_size}, dp_size={cfg.dp_size}, "
                    f"enable_dp_attention={view.enable_dp_attention}, "
                    f"attn_cp_size={view.attn_cp_size}). "
                    "Set --tp, --dp, --enable-dp-attention, and "
                    "--attention-context-parallel-size so the effective "
                    f"attention TP size is {expected_attn_tp_size}."
                )

        # enable_multi_layer_eagle for EAGLE moved to the override registry
        # (arg_groups/overrides.py: _mimo_v2_overrides).

        # MiMoV2 hierarchical cache runs on the unified radix tree, which
        # is the default tree cache now. MiMoV2 has head_dim != v_head_dim,
        # so the host KV pool uses asymmetric K/V allocation. Both
        # kernel/page_first and direct/page_first_direct have split K/V
        # transfer paths.
    elif (
        "Step3p5ForCausalLM" in model_arch
        or "Step3p7ForConditionalGeneration" in model_arch
    ):
        # Attention backend selection + EAGLE multi-layer +
        # hierarchical-cache SWA writes moved to the override registry
        # (arg_groups/overrides.py: _step3p_overrides).
        pass
    elif (
        model_arch in ("Llama4ForConditionalGeneration", "Llama4ForCausalLM")
        and cfg.device != "cpu"
    ):
        # Attention backend auto-select moved to the override registry
        # (arg_groups/overrides.py: _llama4_overrides).
        attention_backend = resolved_view(server_args).attention_backend
        assert attention_backend in {
            "fa3",
            "aiter",
            "triton",
            "ascend",
            "trtllm_mha",
            "intel_xpu",
        }, f"fa3, aiter, triton, ascend, trtllm_mha or intel_xpu is required for Llama4 model but got {attention_backend}"
        # The moe_runner_backend selection moved to the override registry
        # (arg_groups/overrides.py: _llama4_overrides).
    # Gemma2/Gemma3 (disable_hybrid_swa_memory) moved to the override registry
    # (arg_groups/overrides.py: _gemma2_gemma3_overrides).
    elif model_arch in (
        "Gemma4ForConditionalGeneration",
        "Gemma4ForCausalLM",
        "Gemma4UnifiedForConditionalGeneration",
    ):
        # Default attention backend selection moved to the override registry
        # (arg_groups/overrides.py: _gemma4_overrides).
        prefill_backend, decode_backend = attention_backends_of(
            resolved_view(server_args)
        )
        accepted_backends = (
            "trtllm_mha",
            "triton",
            "ascend",
            "intel_xpu",
            "intel_amx",
        )
        assert (
            prefill_backend in accepted_backends and decode_backend in accepted_backends
        ), (
            "Gemma4 only supports trtllm_mha, triton, ascend, intel_xpu, or intel_amx "
            f"attention backend, got prefill={prefill_backend}, decode={decode_backend}"
        )

        # The quantization/moe_runner_backend resolution moved to the override
        # registry (arg_groups/overrides.py: _gemma4_overrides).
    elif model_arch == "MossVLForConditionalGeneration":
        # The prefill attention backend default + validation moved to the
        # override registry (arg_groups/overrides.py: _moss_vl_overrides).
        pass
    elif model_arch in ["Exaone4ForCausalLM", "ExaoneMoEForCausalLM"]:
        if hf_config.sliding_window_pattern is not None:
            # disable_hybrid_swa_memory moved to the override registry
            # (arg_groups/overrides.py: _exaone_overrides).
            # https://docs.sglang.ai/advanced_features/attention_backend.html
            accepted_backends = ["fa3", "triton", "trtllm_mha"]
            attention_backend = resolved_view(server_args).attention_backend
            assert (
                attention_backend in accepted_backends
            ), f"One of the attention backends in {accepted_backends} is required for {model_arch}, but got {attention_backend}"
    elif model_arch in ["Olmo2ForCausalLM"]:
        # disable_hybrid_swa_memory + attention backend selection moved to
        # the override registry (arg_groups/overrides.py: _olmo2_overrides).

        # Flashinfer appears to degrade performance when sliding window attention
        # is used for the Olmo2 architecture. Olmo2 does not use sliding window attention
        # but Olmo3 does.
        attention_backend = resolved_view(server_args).attention_backend
        assert (
            attention_backend != "flashinfer"
        ), "FlashInfer backend can significantly degrade the performance of Olmo3 models."

        logger.info(f"Using {attention_backend} as attention backend for {model_arch}.")
    elif model_arch in [
        "Qwen3MoeForCausalLM",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3NextForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
    ]:
        # The quantization/moe_runner_backend resolution moved to the
        # override registry (arg_groups/overrides.py:
        # _qwen3_moe_family_overrides); the hybrid sub-family's attention
        # backend + page size defaults to _qwen3_5_hybrid_overrides.
        pass

    elif model_arch in ["Glm4MoeForCausalLM"]:
        # The quantization/moe_runner_backend/enable_tf32_matmul resolution
        # moved to the override registry (arg_groups/overrides.py:
        # _glm4_moe_overrides).
        pass

    elif model_arch in ["Lfm2ForCausalLM", "Lfm2MoeForCausalLM"]:
        # Attention backend selection moved to the override registry
        # (arg_groups/overrides.py: _lfm2_overrides).
        assert resolved_view(server_args).attention_backend != "triton", (
            f"{model_arch} does not support triton attention backend, "
            "as the first layer might not be an attention layer"
        )

    # MiniMaxM2ForCausalLM (enable_tf32_matmul) moved to the override registry
    # (arg_groups/overrides.py: _minimax_m2_overrides).

    # Qwen3VL aiter unified-attention page_size moved to the override registry
    # (arg_groups/overrides.py: _qwen3vl_overrides).

    # Hybrid-mamba radix cache handling for the per-arch branch call sites
    # dissolved above: the resolution pass self-guards on the arch union
    # (and the Granite layer_types probe), so one call covers them all.
    # Hybrid-spec archs already resolved at the pre-dispatch call above;
    # for them this re-invocation is an idempotent no-op plus validation.
    # Kept ahead of the sparse-head pass: the legacy per-branch calls
    # resolved before that tail write of disable_overlap_schedule.
    handle_mamba_radix_cache(server_args, model_arch)

    run_post_process_pass(server_args, _sparse_head_overlap_disable)

    # The FlashInfer AllReduce Fusion auto-enable and the enforce-disable
    # terminal moved to the resolution pipeline (arg_groups/overrides.py:
    # _flashinfer_allreduce_fusion_auto_enable /
    # _enforce_disable_allreduce_fusion), invoked here at their legacy
    # slots.

    run_post_process_pass(server_args, _flashinfer_allreduce_fusion_auto_enable)
    run_post_process_pass(server_args, _enforce_disable_allreduce_fusion)


def handle_model_capability_adjustments(server_args: Any):
    from sglang.srt.arg_groups.cuda_graph_hook import (
        generate_prefill_cuda_graph_batch_sizes,
    )
    from sglang.srt.arg_groups.kv_cache_hook import (
        validate_prefill_only_disable_kv_cache_args,
    )

    cfg = resolving_view(server_args)
    if parse_connector_type(cfg.model_path) == ConnectorType.INSTANCE:
        return

    model_config = model_config_of(server_args)
    hf_config = model_config.hf_config

    # HRM-Text needs bidirectional prompt attention (prefill), which only
    # the Triton backend honors at the kernel level. Radix/prefix reuse is
    # also unsafe: the recurrent forward writes direction-dependent KV
    # across many slots.
    is_hrm_text = getattr(
        hf_config, "model_type", None
    ) == "hrm_text" or "HrmTextForCausalLM" in getattr(hf_config, "architectures", [])
    # prefix_lm defaults to True upstream; defaulting False would skip the
    # bidirectional-attention forcing and silently produce junk output.
    if is_hrm_text and getattr(hf_config, "prefix_lm", True):
        run_post_process_pass(server_args, _hrm_text_attention_force)
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            chunked_prefill_size=-1,
        )
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            disable_radix_cache=True,
        )
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            disable_cuda_graph=True,
        )
        # cuda_graph_config was already parsed from the legacy boolean, so
        # flipping the boolean alone would not stop graph capture.
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
            ),
        )
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
            ),
        )
        logger.warning(
            "HRM-Text (prefix_lm) detected: forcing --attention-backend "
            "triton, --chunked-prefill-size -1, --disable-radix-cache, and "
            "--disable-cuda-graph for correctness of the bidirectional "
            "prompt attention."
        )

    # EmbeddingGemma is a Gemma3TextModel with bidirectional prompt
    # attention. Prefix reuse and split prefills would reuse K/V states
    # whose values depend on later prompt tokens, so both are invalid.
    # Breakable CUDA Graph captures one complete prefill and is the graph
    # mode validated for this encoder-style attention.
    # Native encoder architectures declare a pooling-only task and do not
    # need the legacy --is-embedding intent flag. Decoder checkpoints still
    # require that explicit opt-in because their architecture alone does
    # not distinguish embedding from generation serving.
    #
    # ``_handle_model_capability_adjustments`` is also exercised directly
    # by a few focused tests that use a small ModelConfig stand-in. Keep
    # the old predicate as a compatibility fallback while production
    # ModelConfig instances use the central capability contract.
    embedding_model_spec = getattr(model_config, "embedding_model_spec", None)
    if (
        embedding_model_spec is not None
        and embedding_model_spec.auto_enable_embedding
        and not cfg.is_embedding
    ):
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            is_embedding=True,
        )
        logger.info(
            "Embedding architecture detected: enabling embedding mode automatically."
        )

    is_embedding_gemma = (
        embedding_model_spec is not None
        and embedding_model_spec.bcg_prefill_policy == BCGPrefillPolicy.FULL_ENCODER
    )
    if embedding_model_spec is None:
        is_embedding_gemma = getattr(model_config, "is_embedding_gemma", False)
    if is_embedding_gemma:
        # This is an encoder-only model even though its HF architecture is
        # named Gemma3TextModel. Marking it as embedding mode enables the
        # FlashAttention raw-K/V fast path, which does not write or read
        # the paged KV cache during its single prefill forward.
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            is_embedding=True,
        )
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            disable_radix_cache=True,
        )
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            chunked_prefill_size=-1,
        )
        # Submit a list-valued embeddings request atomically so BCG can
        # replay its full prefill batch instead of starting item zero
        # while the remaining texts are still being tokenized.
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            enable_tokenizer_batch_encode=True,
        )
        requested_prefill_backend = (
            cfg.prefill_attention_backend or cfg.attention_backend
        )
        if (
            get_platform().is_cuda
            and (get_platform().is_sm90 or get_platform().is_sm100)
            and requested_prefill_backend in (None, "fa3", "fa4")
        ):
            # Hopper/Blackwell's default FA backend can consume raw K/V
            # tensors for a single embedding prefill. Enable its no-KV
            # pool path before memory-pool sizing; an explicit non-FA
            # backend retains the existing paged-KV behavior.
            declare_resolution(
                server_args,
                "_handle_model_capability_adjustments",
                prefill_only_disable_kv_cache=True,
            )
            validate_prefill_only_disable_kv_cache_args(server_args)
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
            ),
        )
        if (
            get_platform().is_cuda
            and cfg.cuda_graph_config.prefill.backend != Backend.DISABLED
        ):
            declare_resolution(
                server_args,
                "_handle_model_capability_adjustments",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.BREAKABLE
                ),
            )
            # CUDA-graph sizing has already run by this point and derives
            # its generic maximum from the 8K chunked-prefill default.
            # On the Hopper/Blackwell FA raw-K/V path, raise the unlocked
            # default to a full eight-way 2K embedding batch; callers can
            # still override this for larger aggregate prefills.
            prefill_config = cfg.cuda_graph_config.prefill
            # Unit-level capability tests may invoke this hook without
            # running the full CUDA-graph configuration parser, which is
            # where this internal lock set is normally initialized.
            # Treat that minimal construction as having no user-locked
            # graph settings.
            cuda_graph_config_locked = getattr(
                server_args, "_cuda_graph_config_locked", set()
            )
            if (Phase.PREFILL, "max_bs") not in cuda_graph_config_locked:
                sizing = {
                    "max_bs": max(
                        prefill_config.max_bs or 0,
                        model_config.context_len,
                        16384,
                    )
                }
                if (Phase.PREFILL, "bs") not in cuda_graph_config_locked:
                    sizing["bs"] = generate_prefill_cuda_graph_batch_sizes(
                        sizing["max_bs"]
                    )
                declare_resolution(
                    server_args,
                    "_handle_model_capability_adjustments",
                    cuda_graph_config=with_phase(
                        cfg.cuda_graph_config, Phase.PREFILL, **sizing
                    ),
                )
        elif not get_platform().is_cuda:
            # BCG is CUDA-only. Other graph backends do not support this
            # encoder-style prefill, so retain the eager Triton path.
            declare_resolution(
                server_args,
                "_handle_model_capability_adjustments",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
                ),
            )
        logger.info(
            "EmbeddingGemma detected: disabling radix cache and chunked "
            "prefill; using breakable CUDA graph for CUDA prefill."
        )

    if (
        model_config.is_multimodal
        and not model_config.is_multimodal_chunked_prefill_supported
    ):
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            chunked_prefill_size=-1,
        )
        logger.info(
            f"Automatically turn off --chunked-prefill-size as it is not supported for "
            f"{hf_config.model_type}"
        )


def handle_mamba_radix_cache(server_args: Any, model_arch: str):
    # Resolution moved to the resolution pipeline (arg_groups/overrides.py:
    # _mamba_radix_cache_resolution), invoked here at each legacy call
    # slot; this handler keeps the validation.
    from sglang.srt.arg_groups.mamba_hook import (
        validate_mamba_extra_buffer,
        validate_mamba_no_buffer,
    )

    run_post_process_pass(server_args, _mamba_radix_cache_resolution)
    view = resolved_view(server_args)
    if not view.uses_mamba_radix_cache:
        return

    if mamba_extra_buffer_of(view):
        validate_mamba_extra_buffer(
            view,
            model_arch,
            mamba_cache_chunk_size_of=lambda: mamba_cache_chunk_size(server_args),
        )
    else:
        validate_mamba_no_buffer(view, model_arch)


def handle_language_model_only(server_args: Any):

    cfg = resolving_view(server_args)
    if not cfg.language_model_only:
        return
    for flag, name in (
        (cfg.encoder_only, "--encoder-only"),
        (cfg.language_only, "--language-only"),
        (cfg.enable_prefix_mm_cache, "--enable-prefix-mm-cache"),
        (
            cfg.enable_broadcast_mm_inputs_process,
            "--enable-broadcast-mm-inputs-process",
        ),
        (cfg.mm_enable_dp_encoder, "--mm-enable-dp-encoder"),
    ):
        if flag:
            raise ValueError(f"--language-model-only cannot be combined with {name}")
    if cfg.disaggregation_mode != "null":
        raise ValueError(
            "--language-model-only is incompatible with --disaggregation-mode "
            "prefill/decode"
        )
    architectures = model_config_of(server_args).hf_config.architectures
    if not any(
        a in server_args.LANGUAGE_MODEL_ONLY_ARCHITECTURES for a in architectures
    ):
        raise ValueError(
            f"--language-model-only does not support {architectures}. "
            f"Supported: {list(server_args.LANGUAGE_MODEL_ONLY_ARCHITECTURES)}."
        )
