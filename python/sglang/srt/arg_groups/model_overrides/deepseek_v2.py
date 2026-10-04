"""Config-time override declarations for deepseek_v2."""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    context_parallel_attn_dp_size,
    is_attention_backend_not_set,
    resolving_view,
    use_mla_backend,
)
from sglang.srt.runtime_context import derive_attn_tp_size, get_platform

logger = logging.getLogger(__name__)


def _dsa_dcp_overrides(cfg: Any, hf_config: Any) -> dict:
    """Gate the CUDA RoPE DSA path before cache or speculative setup."""
    if (
        cfg.dcp_size <= 1
        or not get_platform().is_cuda
        or hf_config.architectures[0] == "Glm5NextForConditionalGeneration"
        or getattr(hf_config, "use_mla_nope", False)
    ):
        return {}

    import torch

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise ValueError(
            "CUDA RoPE DSA DCP requires SM100 or SM103 with TRT-LLM sparse "
            f"attention; got SM{capability[0]}{capability[1]}."
        )
    if cfg.enable_hisparse:
        raise ValueError("RoPE DSA DCP does not support --enable-hisparse.")
    if cfg.enable_prefill_cp or cfg.attn_cp_size > 1:
        raise ValueError(
            "RoPE DSA DCP cannot be combined with prefill context parallelism."
        )
    if (
        cfg.enable_hierarchical_cache
        or cfg.hicache_storage_backend is not None
        or cfg.enable_unified_cache_external_linker
    ):
        raise ValueError("RoPE DSA DCP does not support HiCache or L3 cache storage.")
    if cfg.disaggregation_mode != "null":
        raise ValueError("RoPE DSA DCP does not support PD disaggregation.")
    if cfg.enable_unified_memory:
        raise ValueError(
            "RoPE DSA DCP does not support --enable-unified-memory: the dsa "
            "backend does not translate unified-pool virtual KV addresses."
        )
    if cfg.speculative_algorithm is not None:
        if cfg.speculative_algorithm.upper() not in ("EAGLE", "NEXTN"):
            raise ValueError(
                "RoPE DSA DCP speculative decoding supports only EAGLE/MTP "
                "with a single draft chain."
            )
        if (
            cfg.speculative_eagle_topk != 1
            or cfg.speculative_num_steps is None
            or cfg.speculative_num_steps < 1
            or cfg.speculative_num_draft_tokens != cfg.speculative_num_steps + 1
            or cfg.speculative_adaptive
            or cfg.enable_multi_layer_eagle
            or cfg.speculative_draft_model_path not in (None, cfg.model_path)
            or cfg.speculative_draft_attention_backend not in (None, "dsa")
            or cfg.speculative_draft_kv_cache_dtype is not None
        ):
            raise ValueError(
                "RoPE DSA DCP EAGLE requires --speculative-eagle-topk 1, "
                "explicit positive --speculative-num-steps and "
                "--speculative-num-draft-tokens equal to steps + 1, using the "
                "checkpoint's own MTP draft. Adaptive and multi-layer EAGLE "
                "are unsupported; use the dsa draft backend and inherit the "
                "target KV dtype. Branching trees need cross-rank KV relocation."
            )
    if cfg.dcp_replicate_q_proj:
        raise ValueError(
            "RoPE DSA DCP does not support --dcp-replicate-q-proj; use the "
            "ordinary Q all-gather. Quantized Q projections cannot use this "
            "optimization."
        )
    attn_tp_size = derive_attn_tp_size(
        tp_size=cfg.tp_size,
        attn_cp_size=cfg.attn_cp_size,
        attn_dp_size=cfg.attn_dp_size,
    )
    if attn_tp_size < cfg.dcp_size or attn_tp_size % cfg.dcp_size:
        raise ValueError(
            f"RoPE DSA DCP requires attention TP size ({attn_tp_size}) to be "
            f"divisible by --dcp-size ({cfg.dcp_size}); each DCP group must "
            "lie inside one attention TP group."
        )
    if cfg.kv_cache_dtype not in ("auto", "fp8_e4m3", "bf16", "bfloat16"):
        raise ValueError("RoPE DSA DCP supports only fp8_e4m3 or bfloat16 KV cache.")
    for field in (
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
    ):
        if getattr(cfg, field) not in (None, "dsa"):
            raise ValueError(
                "RoPE DSA DCP requires the dsa attention backend for both phases; "
                f"got --{field.replace('_', '-')} {getattr(cfg, field)!r}."
            )
    overrides = {}
    for field in ("dsa_prefill_backend", "dsa_decode_backend"):
        backend = getattr(cfg, field)
        if backend not in (None, "trtllm"):
            raise ValueError(
                "CUDA RoPE DSA DCP requires trtllm for both sparse-attention phases; "
                f"got --{field.replace('_', '-')} {backend!r}."
            )
        if backend is None:
            # BF16 otherwise defaults to flashmla_sparse prefill, which has
            # no CUDA DCP path. Keep the ordinary non-DCP default unchanged.
            overrides[field] = "trtllm"
    return overrides


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
    """Declare DeepSeek/DSA defaults; ordered CP, KV-cache, and MoE passes run in model_hook."""
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
        overrides.update(_dsa_dcp_overrides(cfg, hf_config))
        # Set attention backend for DeepSeek
        if is_attention_backend_not_set(cfg):
            overrides["attention_backend"] = "dsa"
            logger.info("Use dsa attention backend for DeepSeek with DSA.")
        if not get_platform().is_npu and not get_platform().is_xpu:  # CUDA or ROCm GPU
            if cfg.enable_prefill_cp:
                logger.warning(
                    "Context parallel feature is still under experiment. It has only been verified on Hopper platform."
                )
                attn_dp_size = context_parallel_attn_dp_size(
                    cfg, "DSA context parallelism"
                )
                overrides["attn_dp_size"] = attn_dp_size
                overrides["dp_size"] = 1
                if cfg.cp_strategy == "zigzag":
                    overrides["moe_dense_tp_size"] = 1
                    overrides["moe_a2a_backend"] = "deepep"
                    overrides["ep_size"] = cfg.tp_size
                    logger.warning(
                        "zigzag DSA CP requires moe_dense_tp_size=1, "
                        "moe_a2a_backend=deepep, ep_size=tp_size, batch_size=1."
                    )
                assert cfg.tp_size <= 8, (
                    "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
                )
                # Interleave can shard attention heads within each CP rank.
                # Keep an explicit CP width; default to attention TP1 as before.
                # Zigzag still requires attention TP1.
                attn_cp_size = (
                    cfg.attn_cp_size
                    if cfg.cp_strategy == "interleave" and cfg.attn_cp_size > 1
                    else cfg.tp_size // attn_dp_size
                )
                overrides["attn_cp_size"] = attn_cp_size
                logger.warning(
                    "Enabled DSA context parallel: "
                    f"strategy={cfg.cp_strategy}, attn_dp_size={attn_dp_size}, "
                    f"moe_dense_tp_size={overrides.get('moe_dense_tp_size', cfg.moe_dense_tp_size)}, "
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
        elif get_platform().is_xpu:
            overrides["page_size"] = 128
            logger.warning("Setting page size to 128 for DeepSeek DSA on XPU.")
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
            attn_dp_size = context_parallel_attn_dp_size(cfg, "MLA context parallelism")
            overrides["attn_dp_size"] = attn_dp_size
            overrides["dp_size"] = 1
            # TODO(kpham-sgl) Supports moe_dense_tp_size != 1.
            overrides["moe_dense_tp_size"] = 1
            overrides["moe_a2a_backend"] = "deepep"
            overrides["ep_size"] = cfg.tp_size
            logger.warning(
                "For MLA CP, we have the following restrictions: moe_dense_tp_size == 1, moe_a2a_backend == deepep, ep_size == tp_size, batch_size == 1"
            )
            # FIXME(kpham-sgl): Keep attn_tp_size == 1 under MLA CP.
            # The DSA / MLA CP gather and reduce-scatter
            # (the dsa_cp_* helpers in adapters/context_parallel.py) assume it.
            attn_cp_size = cfg.tp_size // attn_dp_size
            overrides["attn_cp_size"] = attn_cp_size
            logger.warning(
                f"Enable Context Parallel opt for MLA, "
                f"Setting attn_dp_size == {attn_dp_size} and "
                f"attn_cp_size == {attn_cp_size}, "
                f"moe_dense_tp_size == {overrides['moe_dense_tp_size']}, "
                f"ep_size == {overrides['ep_size']}, "
                f"tp_size == {cfg.tp_size}, "
                f"moe_a2a_backend {overrides['moe_a2a_backend']}, "
                f"cuda_graph_config[prefill].backend=disabled"
            )
    return overrides
