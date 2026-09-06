"""Config-time override declarations for deepseek_v2.

Architectures: DeepseekV32ForCausalLM, DeepseekV3ForCausalLM, Dots3NoteForCausalLM, Glm5NextForConditionalGeneration, GlmMoeDsaForCausalLM, HYV4ForCausalLM, HYV4ForCausalLMNextN, KimiK25ForConditionalGeneration, LongcatFlashForCausalLM, LongcatFlashForCausalLMNextN, MistralLarge3ForCausalLM, PixtralForConditionalGeneration.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
    use_mla_backend,
)
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for(
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "KimiK25ForConditionalGeneration",
    "MistralLarge3ForCausalLM",
    "PixtralForConditionalGeneration",
    "GlmMoeDsaForCausalLM",
    "Glm5NextForConditionalGeneration",
    "HYV4ForCausalLM",
    "HYV4ForCausalLMNextN",
    "LongcatFlashForCausalLM",
    "LongcatFlashForCausalLMNextN",
    "Dots3NoteForCausalLM",
)
def _deepseek_family_overrides(server_args: Any, hf_config: Any) -> dict:
    """Order-safe declarations of the DeepSeek/DSA branch. The CP parallel
    writes (enable_dp_attention/ep_size/moe_a2a_backend have post-monolith
    writers), the kv-cache/split-backend defaults, the quant/moe block (read
    before it by _set_default_dsa_kv_cache_dtype) and the env writes stay in
    the branch."""
    cfg = resolving_view(server_args)
    from sglang.srt.configs.model_config import (
        is_deepseek_dsa,
        unwrap_modelopt_quantization_config,
    )

    model_arch = (getattr(hf_config, "architectures", None) or [None])[0]
    if model_arch in ("HYV4ForCausalLM", "HYV4ForCausalLMNextN"):
        if cfg.enable_prefill_cp:
            raise ValueError(
                "--enable-prefill-cp is not supported for HYV4 because its "
                "attention path does not implement DSA context-parallel metadata "
                f"and sharding. Got architecture={model_arch!r} and "
                f"enable_prefill_cp={cfg.enable_prefill_cp!r}."
            )
        dcp_size = getattr(cfg, "dcp_size", 1)
        if dcp_size > 1:
            raise ValueError(
                "--dcp-size > 1 is not supported for HYV4 because decode context "
                "parallelism gathers query heads across DCP ranks but does not "
                "provide single-owner semantics for learnable attention sinks. "
                f"Got architecture={model_arch!r} and dcp_size={dcp_size!r}."
            )

    overrides: Dict[str, Any] = {}

    if model_arch in ("HYV4ForCausalLM", "HYV4ForCausalLMNextN"):
        quant_cfg = getattr(hf_config, "quantization_config", None) or {}
        quant_algo = unwrap_modelopt_quantization_config(quant_cfg).get(
            "quant_algo", ""
        )
        if str(quant_algo).upper() == "MXFP8":
            from sglang.srt.layers import deep_gemm_wrapper

            # auto would otherwise select an unqualified FP8/MoE path for HYV4 MXFP8.
            if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
                if cfg.moe_runner_backend == "auto":
                    overrides["moe_runner_backend"] = "deep_gemm"
                if cfg.fp8_gemm_runner_backend == "auto":
                    overrides["fp8_gemm_runner_backend"] = "deep_gemm"
                if overrides:
                    logger.info(
                        "HYV4 MXFP8: defaulting MoE/FP8 GEMM backends to deep_gemm."
                    )

    if is_deepseek_dsa(hf_config):  # DeepSeek 3.2/GLM 5
        # Set attention backend for DeepSeek
        if is_attention_backend_not_set(cfg):
            overrides["attention_backend"] = "dsa"
            logger.info("Use dsa attention backend for DeepSeek with DSA.")
        if not get_platform().is_npu and not get_platform().is_xpu:  # CUDA or ROCm GPU
            if cfg.enable_prefill_cp:
                logger.warning(
                    "Context parallel feature is still under experiment. It has only been verified on Hopper platform."
                )
                overrides["enable_dp_attention"] = True
                overrides["moe_dense_tp_size"] = 1
                if cfg.cp_strategy == "zigzag":
                    overrides["moe_a2a_backend"] = "deepep"
                    overrides["ep_size"] = cfg.tp_size
                    logger.warning(
                        "zigzag DSA CP requires moe_dense_tp_size=1, "
                        "moe_a2a_backend=deepep, ep_size=tp_size, batch_size=1."
                    )
                else:
                    assert cfg.dp_size == 1, (
                        "interleave DSA CP does not support DP attention."
                    )
                assert cfg.tp_size <= 8, (
                    "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
                )
                # Note(kpham-sgl): Keep attn_tp_size == 1 under DSA CP.
                # DSACPLayerCommunicator does not all-reduce attention-TP
                # partial o_proj outputs before replicated dense FFNs.
                attn_cp_size = cfg.tp_size // cfg.dp_size
                overrides["attn_cp_size"] = attn_cp_size
                logger.warning(
                    "Enabled DSA context parallel: "
                    f"strategy={cfg.cp_strategy}, dp_size={cfg.dp_size}, "
                    f"moe_dense_tp_size={overrides['moe_dense_tp_size']}, "
                    f"ep_size={overrides.get('ep_size', cfg.ep_size)}, tp_size={cfg.tp_size}, "
                    f"attn_cp_size={attn_cp_size}, "
                    f"kv_cache_dtype={cfg.kv_cache_dtype}, "
                    f"moe_a2a_backend={overrides.get('moe_a2a_backend', cfg.moe_a2a_backend)}, "
                    f"cuda_graph_config[prefill].backend=disabled"
                )

            # Deferred import to avoid a circular import at module-load
            # time (dsa.utils imports the runtime-context accessors).
            from sglang.srt.layers.attention.dsa.utils import (
                aiter_can_use_preshuffle_paged_mqa,
            )

            if get_platform().is_hip and not aiter_can_use_preshuffle_paged_mqa():
                # Legacy ROCm DSA path: aiter's gluon paged-MQA kernel is
                # unavailable (Triton<3.5 and AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS
                # not set, or SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1 / SGLANG_USE_AITER=0).
                overrides["page_size"] = 1
                logger.warning(
                    "Setting page size to 1 for DeepSeek DSA on ROCm "
                    "(aiter preshuffle paged-MQA path unavailable: "
                    "needs Triton>=3.5.0 or AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1)."
                )
            else:
                overrides["page_size"] = 64
                logger.warning("Setting page size to 64 for DeepSeek DSA.")
    else:
        # DeepSeek V3/R1/V3.1
        if get_platform().is_sm100:
            if (
                cfg.attention_backend is None
                and cfg.prefill_attention_backend is None
                and cfg.decode_attention_backend is None
            ):
                overrides["attention_backend"] = "trtllm_mla"
                logger.info(
                    "Use trtllm_mla as attention backend on sm100 for DeepseekV3ForCausalLM"
                )
        # MLA prefill CP auto-config. Mirrors the NSA CP block above
        # (minus the in-seq/round-robin mode split, which MLA CP does not support)
        if cfg.enable_prefill_cp and use_mla_backend(server_args):
            logger.warning(
                "MLA prefill context parallel is still experimental. "
                "Verified on Hopper with the fa3 backend."
            )
            overrides["enable_dp_attention"] = True
            # TODO(kpham-sgl) Supports moe_dense_tp_size != 1.
            overrides["moe_dense_tp_size"] = 1
            overrides["moe_a2a_backend"] = "deepep"
            overrides["ep_size"] = cfg.tp_size
            logger.warning(
                "For MLA CP, we have the following restrictions: moe_dense_tp_size == 1, moe_a2a_backend == deepep, ep_size == tp_size, batch_size == 1"
            )
            # FIXME(kpham-sgl): Keep attn_tp_size == 1 under MLA CP.
            # DSACPLayerCommunicator does not all-reduce attention-TP
            # partial o_proj outputs before replicated dense FFNs.
            attn_cp_size = cfg.tp_size // cfg.dp_size
            overrides["attn_cp_size"] = attn_cp_size
            logger.warning(
                f"Enable Context Parallel opt for MLA, "
                f"Setting dp_size == {cfg.dp_size} and "
                f"attn_cp_size == {attn_cp_size}, "
                f"moe_dense_tp_size == {overrides['moe_dense_tp_size']}, "
                f"ep_size == {overrides['ep_size']}, "
                f"tp_size == {cfg.tp_size}, "
                f"moe_a2a_backend {overrides['moe_a2a_backend']}, "
                f"cuda_graph_config[prefill].backend=disabled"
            )
    return overrides
