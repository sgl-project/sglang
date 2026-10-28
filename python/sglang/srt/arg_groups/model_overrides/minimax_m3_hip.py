"""ROCm part of the minimax_m3 override declarations: the gfx950 defaults of the M3.1
NVFP4 and Quark checkpoints, the NVFP4 KV format and the TP guards. minimax_m3 calls it on
HIP; it is pure Python and never imports aiter. aiter's tuned rows for these checkpoints
are installed by the model (models/minimax_common/amd/aiter_intree_tune)."""

import logging
import os
from typing import Any, Optional

from sglang.srt.arg_groups.model_override_base import is_attention_backend_not_set
from sglang.srt.environ import envs
from sglang.srt.runtime_context import derive_attn_tp_size, get_platform

logger = logging.getLogger(__name__)


def uses_nvfp4_kv(
    quantization: Optional[str], *, speculative_algorithm: Optional[str]
) -> bool:
    """Whether gfx950 keeps the M3.1 KV cache in the packed NVFP4 (MSA-v1) pool."""
    platform = get_platform()
    if not (
        quantization == "minimax_m3_nvfp4" and platform.is_hip and platform.is_gfx95
    ):
        return False
    kv_format = envs.SGLANG_MINIMAX_M3_AMD_KV_FORMAT.get().strip().lower()
    if kv_format not in ("auto", "fp8"):
        raise ValueError(
            f"SGLANG_MINIMAX_M3_AMD_KV_FORMAT must be auto or fp8, got {kv_format!r}"
        )
    # only the DSPARK verify indexer reads an NVFP4 index K
    return kv_format == "auto" and speculative_algorithm == "DSPARK"


def validate_nvfp4(cfg: Any) -> None:
    """The M3.1 NVFP4 checks of ROCm."""
    algorithm = cfg.speculative_algorithm
    if algorithm == "DFLASH" or (algorithm == "DSPARK" and not get_platform().is_gfx95):
        supported = "Blackwell or AMD gfx950" if algorithm == "DSPARK" else "Blackwell"
        raise ValueError(f"M3.1 NVFP4 {algorithm} requires {supported}")
    # TP8 with attention DP ran away on whitespace in 12% of GSM8K answers on MI350X
    if cfg.tp_size > 4:
        raise ValueError("M3.1 NVFP4 on ROCm is validated at --tp-size 4 or less")


def nvfp4_overrides(cfg: Any, overrides: dict) -> dict:
    """The gfx950 serving stack of M3.1 NVFP4: aiter MXFP4 experts and the Triton attention
    and DSpark draft backends."""
    if (
        cfg.speculative_algorithm == "DSPARK"
        and cfg.speculative_draft_attention_backend is None
    ):
        # trtllm_mha is CUDA-only; triton serves the dense MHA draft here
        overrides["speculative_draft_attention_backend"] = "triton"
    overrides.update(page_size=128)
    if cfg.moe_runner_backend == "auto":
        # name the runner MiniMaxM3Nvfp4MoE builds: the dispatcher keeps global
        # expert ids (EP) only for aiter
        overrides["moe_runner_backend"] = "aiter"
    if is_attention_backend_not_set(cfg):
        overrides["attention_backend"] = "triton"
    # the DSpark draft's plain Fp8Config converts its MXFP8 projections to block-FP8 like
    # the target's MiniMaxM3Nvfp4Config, instead of running Triton dot_scaled
    if not envs.SGLANG_FORCE_MXFP8_BLOCK_CONVERT_DENSE.is_set():
        envs.SGLANG_FORCE_MXFP8_BLOCK_CONVERT_DENSE.set(True)
    _m31_gfx950_all_reduce(cfg)
    return overrides


def rocm_overrides(
    cfg: Any, hf_config: Any, quantization: Optional[str], overrides: dict
) -> bool:
    """The other MiniMax-M3 checkpoints on ROCm: the M3.1 Quark export's gfx950 defaults.
    Returns whether custom all-reduce stays on: on gfx950, quark_mxfp4 and the M3.1 Quark
    export are measured faster with it at equal accuracy; MXFP8 is not."""
    gfx950 = get_platform().is_gfx95
    m31_quark = gfx950 and quantization == "quark" and is_m31_quark_export(hf_config)
    if m31_quark:
        _m31_quark_gfx950_overrides(cfg, overrides)
    keeps_custom_all_reduce = m31_quark or (gfx950 and quantization == "quark_mxfp4")
    if keeps_custom_all_reduce:
        _require_attention_tp_within_kv_heads(cfg, hf_config)
    return keeps_custom_all_reduce


def is_m31_quark_export(hf_config: Any) -> bool:
    """A Quark export of M3.1: its attention, dense and shared-expert linears stay
    MXFP8 (fp8 per_block [1, 32], e8m0 scales) next to the MXFP4 routed experts,
    byte-identical to the NVFP4 checkpoint's. The M3 Quark export keeps those
    linears bf16 and lists no per-layer config."""
    quant_config = getattr(hf_config, "quantization_config", None)
    if quant_config is None:
        text_config = getattr(hf_config, "text_config", None)
        quant_config = getattr(text_config, "quantization_config", None)
    if not isinstance(quant_config, dict):
        return False
    if str(quant_config.get("quant_method", "")).lower() != "quark":
        return False
    for layer in (quant_config.get("layer_quant_config") or {}).values():
        weight = (layer or {}).get("weight") or {}
        if (
            weight.get("dtype") in ("fp8_e4m3", "fp8_e4m3fn")
            and weight.get("block_size") == [1, 32]
            and "e8m0" in str(weight.get("scale_type") or "")
        ):
            return True
    return False


def _m31_quark_gfx950_overrides(cfg: Any, overrides: dict) -> None:
    """The gfx950 defaults minimax_m3_nvfp4 resolves, for an M3.1 Quark export: same
    MXFP8 linears, same MXFP4 aiter experts. At TP4 on MI355X the Triton MXFP8
    GEMMs alone halve batch-1 decode."""
    # the aiter block-scale GEMMs instead of the Triton MXFP8 linear kernel
    if not envs.SGLANG_FORCE_MXFP8_BLOCK_CONVERT_DENSE.is_set():
        envs.SGLANG_FORCE_MXFP8_BLOCK_CONVERT_DENSE.set(True)
    if cfg.kv_cache_dtype == "auto":
        overrides["kv_cache_dtype"] = "fp8_e4m3"
    draft_kv = cfg.speculative_draft_kv_cache_dtype
    if cfg.speculative_algorithm == "DSPARK" and draft_kv in (None, "auto"):
        overrides["speculative_draft_kv_cache_dtype"] = "bf16"
    _m31_gfx950_all_reduce(cfg)


def _m31_gfx950_all_reduce(cfg: Any) -> None:
    """The quick all-reduce both M3.1 checkpoints take."""
    # INT4 halves the long-prefill all-reduce at TP4 with GSM8K unchanged; ROCm images
    # export INT8 for every model, so only another level the user set stays
    level = "ROCM_QUICK_REDUCE_QUANTIZATION"
    if (
        os.environ.get(level, "INT8") == "INT8"
        and envs.SGLANG_ENABLE_MINIMAX_M3_AMD_QUICK_REDUCE_INT4.get()
    ):
        os.environ[level] = "INT4"
        logger.info(
            "MiniMax-M3.1: quick all-reduce INT4 (from INT8 or unset); "
            "SGLANG_ENABLE_MINIMAX_M3_AMD_QUICK_REDUCE_INT4=0 keeps INT8"
        )
    if (
        cfg.tp_size != 4
        or os.environ.get(level) != "INT4"
        # the JIT build reduces bf16 cast to fp16 only, as QuickAllReduce sets it up
        or cfg.dtype not in ("auto", "bfloat16")
        or int(os.environ.get("ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", 1)) == 0
    ):
        return
    # the JIT build is faster at TP4; it matches sgl-kernel's bitwise at the level it runs,
    # which for INT4 is INT8 below 64 MiB
    if not envs.SGLANG_ROCM_QUICK_ALL_REDUCE_JIT.is_set():
        envs.SGLANG_ROCM_QUICK_ALL_REDUCE_JIT.set(True)
    if envs.SGLANG_ROCM_QUICK_ALL_REDUCE_JIT.get():
        # its INT8 beats aiter's custom AR from about 2 MiB, so custom AR stops at 2.4 MiB
        # (decode at 26+ requests, extends of 205+ tokens) instead of 64 MiB
        os.environ.setdefault("AITER_CUSTOM_AR_MAX_SIZE", "2516582")


def _require_attention_tp_within_kv_heads(cfg: Any, hf_config: Any) -> None:
    # on ROCm, past one KV / index head per rank the server returns NUL bytes
    kv_heads = getattr(hf_config, "text_config", hf_config).num_key_value_heads
    # the deprecated --dp-size/--enable-dp-attention spelling is already attn_dp_size here
    attn_tp = derive_attn_tp_size(
        tp_size=cfg.tp_size,
        attn_cp_size=cfg.attn_cp_size,
        attn_dp_size=cfg.attn_dp_size,
    )
    if attn_tp > kv_heads:
        raise ValueError(
            f"M3.1 attention TP is {attn_tp} but the model has {kv_heads} KV "
            "heads, which are not replicated: every request would return "
            f"garbage. Use --tp-size {kv_heads} or fewer, or divide the attention "
            f"TP with --attn-dp-size {cfg.tp_size // kv_heads}."
        )
