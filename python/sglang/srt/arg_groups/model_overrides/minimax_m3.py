"""Config-time override declarations for minimax_m3.

Architectures: MiniMaxM3SparseForCausalLM, MiniMaxM3SparseForConditionalGeneration.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import get_quantization_config

logger = logging.getLogger(__name__)


@_register_for("MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration")
def _minimax_m3_overrides(server_args: Any, hf_config: Any) -> dict:

    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}

    quant_method = get_quantization_config(hf_config)
    quant_resolved = cfg.quantization
    if (
        quant_resolved is None
        and not server_args._quantization_explicitly_unset
        and quant_method is not None
    ):
        overrides["quantization"] = quant_method
        quant_resolved = quant_method

    if get_platform().is_hip:
        if is_attention_backend_not_set(cfg):
            overrides["attention_backend"] = "triton"
        if cfg.moe_runner_backend == "auto" and quant_resolved == "mxfp8":
            overrides["moe_runner_backend"] = "triton"
        if not envs.USE_ROCM_AITER_ROPE_BACKEND.is_set():
            envs.USE_ROCM_AITER_ROPE_BACKEND.set("0")
        aiter_fusion_resolved = cfg.enable_aiter_allreduce_fusion
        if cfg.ep_size > 1 and cfg.moe_a2a_backend == "none" and aiter_fusion_resolved:
            logger.warning(
                "Disable --enable-aiter-allreduce-fusion for MiniMax-M3 "
                "standard EP on ROCm because the deferred fused all-reduce "
                "corrupts sparse MoE partial outputs."
            )
            overrides["enable_aiter_allreduce_fusion"] = False
            aiter_fusion_resolved = False
        # By default MiniMax-M3 on ROCm keeps NCCL all-reduce (custom AR off)
        # whenever aiter all-reduce fusion is not used. Opting in via
        # SGLANG_M3_ALLOW_CUSTOM_AR keeps custom all-reduce enabled so the
        # quick-reduce path (ROCM_QUICK_REDUCE_QUANTIZATION=INT4/INT6/INT8) can
        # accelerate the large prefill all-reduce.
        if not aiter_fusion_resolved and not envs.SGLANG_M3_ALLOW_CUSTOM_AR.get():
            overrides["disable_custom_all_reduce"] = True
    elif get_platform().is_sm100:
        if is_attention_backend_not_set(cfg):
            if (
                cfg.kv_cache_dtype == "fp8_e4m3"
                and not envs.SGLANG_DISABLE_M3_FP8_ATTN_GEMM.get()
            ):
                # fp8 attention GEMMs activate whenever possible
                # (m3_fp8_attn_gemm_enabled); only trtllm_mha serves the dense
                # fp8-q path, so prefer it over fa4 for fp8 KV. The
                # SGLANG_DISABLE_M3_FP8_ATTN_GEMM kill switch keeps the fa4
                # default (pre-fp8 behavior).
                overrides["attention_backend"] = "trtllm_mha"
            else:
                overrides["attention_backend"] = "fa4"
        backend_resolved = overrides.get("attention_backend", cfg.attention_backend)
        page_resolved = cfg.page_size
        # fa4 (fmha_sm100) and trtllm_mha both allow the page_size == 128
        # sparse block MSA needs (trtllm_mha via trtllm-gen's dynamic
        # tokens-per-page kernels).
        if page_resolved is None and backend_resolved in ("fa4", "trtllm_mha"):
            overrides["page_size"] = 128
            page_resolved = 128
        if cfg.moe_runner_backend == "auto" and quant_resolved == "mxfp8":
            overrides["moe_runner_backend"] = "deep_gemm"
        elif cfg.moe_runner_backend == "auto" and quant_resolved == "modelopt_mixed":
            overrides["moe_runner_backend"] = "flashinfer_trtllm_routed"
        logger.info(
            "MiniMax-M3 on SM100: attention_backend="
            f"{overrides.get('attention_backend', cfg.attention_backend)}, page_size={page_resolved}, "
            f"moe_runner_backend={overrides.get('moe_runner_backend', cfg.moe_runner_backend)}."
        )
    elif get_platform().is_sm90:
        if is_attention_backend_not_set(cfg):
            overrides["attention_backend"] = "fa3"
        page_resolved = cfg.page_size
        if (
            page_resolved is None
            and overrides.get("attention_backend", cfg.attention_backend) == "fa3"
        ):
            overrides["page_size"] = 128
            page_resolved = 128
        logger.info(
            "MiniMax-M3 on Hopper: attention_backend="
            f"{overrides.get('attention_backend', cfg.attention_backend)}, page_size={page_resolved} "
            "(MSA is SM100-only; sparse attention runs on the Triton path)."
        )

    # fp8 attention GEMMs have no opt-in flag: m3_fp8_attn_gemm_enabled
    # (server_args.py) derives the mode from kv_cache_dtype (fp8_e4m3) +
    # attention_backend (trtllm_mha) + SM100 at runtime. Surface the
    # resolution here: warn on fp8_e5m2 (fmha_sm100's variant lookup would
    # silently dispatch the e4m3 kernel, so e5m2 stays on the widening Triton
    # path), log when the fp8 GEMM mode is active, and log when the
    # SGLANG_DISABLE_M3_FP8_ATTN_GEMM kill switch suppresses it.
    if cfg.kv_cache_dtype == "fp8_e5m2":
        logger.warning(
            "MiniMax-M3 with kv_cache_dtype fp8_e5m2: fp8 attention GEMMs stay "
            "DISABLED (fmha_sm100's variant lookup would silently dispatch the "
            "e4m3 kernel for e5m2); sparse attention runs on the widening "
            "Triton path. Use --kv-cache-dtype fp8_e4m3 for fp8 attention GEMMs."
        )
    elif (
        cfg.kv_cache_dtype == "fp8_e4m3"
        and overrides.get("attention_backend", cfg.attention_backend) == "trtllm_mha"
        and get_platform().is_sm100
    ):
        if envs.SGLANG_DISABLE_M3_FP8_ATTN_GEMM.get():
            logger.info(
                "MiniMax-M3 fp8 attention GEMMs DISABLED by "
                "SGLANG_DISABLE_M3_FP8_ATTN_GEMM: bf16 indexer + widening "
                "Triton sparse path, bf16 q; dense layers keep trtllm_mha's "
                "fp8 KV cache."
            )
        else:
            logger.info(
                "MiniMax-M3 fp8 attention GEMMs active (kv_cache_dtype fp8_e4m3 + "
                "trtllm_mha on SM100): fp8 main/index KV, fp8-cast q, fp8 "
                "sparse/MSA kernels. Set SGLANG_DISABLE_M3_FP8_ATTN_GEMM=1 to "
                "force the pre-fp8 numerics."
            )

    moe_runner_resolved = overrides.get("moe_runner_backend", cfg.moe_runner_backend)
    if quant_resolved is None and moe_runner_resolved in ("auto", "deep_gemm"):
        if moe_runner_resolved == "deep_gemm":
            logger.warning(
                "MiniMax-M3: the deep_gemm MoE runner produces corrupted output "
                "on bf16 full weights; overriding --moe-runner-backend to 'triton'."
            )
        overrides["moe_runner_backend"] = "triton"

    return overrides
