"""Config-time override declarations for deepseek_v4 / deepseek_v41.

Architectures: DeepseekV4ForCausalLM, DeepseekV41ForCausalLM.
HF remaps DeepseekV41ForCausalLM → DeepseekV4ForCausalLM; both names are
registered so overrides apply before or after that remap.

SM70 (V100) declarations: moe_runner_backend=marlin, dtype float16,
refuse BF16, keep page_size=256 (CSA2 pool / dsv4 backend hardcode). Engram
host-table residency is the SM70-conditional env default in environ.py, not
a ServerArgs field.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    model_config_of,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import is_flashinfer_available

logger = logging.getLogger(__name__)

# The compressed-KV pool's page; its C128 sub-pool hardcodes 256-token pages.
DSV4_PAGE_SIZE = 256
# Used when --swa-full-tokens-ratio is at its default and cap mode is unusable.
DSV4_DEFAULT_SWA_FULL_TOKENS_RATIO = 0.1
_DSV4_NPU_PAGE_SIZE = 128


# The dsv4 attention + indexer-K pools address the global KV in 256-token
# pages: DeepseekV4AttnBackend and PagedIndexerMetadata assert page_size==256.
# The SM70 TileLang default of 16 applies only to tilelang_fa_v100 /
# flash_attn_v100, not dsv4. Indexer-K pages are a separate pool size (64 on
# SM70 via dsv41_index_page_size).
def _is_sm70_v100() -> bool:
    """Volta serving path: SM70 CUDA, and not a Hopper/Blackwell override.

    Tests set is_sm90/is_sm100/is_sm120 independently of the live GPU, so an
    SM70 default must not fire when those facts are True.
    """
    platform = get_platform()
    if not platform.is_cuda or platform.is_hip:
        return False
    if platform.is_sm90 or platform.is_sm100 or platform.is_sm120:
        return False
    return platform.is_sm70


@_register_for("DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM")
def _deepseek_v4_overrides(server_args: Any, hf_config: Any) -> dict:
    """Attention, page and MoE defaults; the rest lives in deepseek_v4_hook."""
    cfg = resolving_view(server_args)

    model_arch = hf_config.architectures[0]
    overrides: Dict[str, Any] = {"attention_backend": "dsv4"}

    # MXFP8 serves this checkpoint's 32-wide ue8m0 blocks on SM100/SM103;
    # explicit backend choices, including Triton, take precedence.
    quant = getattr(hf_config, "quantization_config", None) or {}
    if (
        getattr(hf_config, "model_type", None) == "deepseek_v41"
        and cfg.device == "cuda"
        and not get_platform().is_hip
        and get_platform().is_sm100
        and cfg.fp8_gemm_runner_backend == "auto"
        and quant.get("quant_method") == "fp8"
        and quant.get("weight_block_size") == [32, 32]
        and quant.get("scale_fmt") == "ue8m0"
        and is_flashinfer_available()
    ):
        overrides["fp8_gemm_runner_backend"] = "flashinfer_cutedsl"
        logger.info("Use flashinfer_cutedsl for DeepSeek-V4.1 MXFP8 dense GEMMs.")

    # Left unset, the pool configurator sizes the SWA pool from the request cap.
    if (
        cfg.swa_full_tokens_ratio is None
        and getattr(hf_config, "model_type", None) != "deepseek_v41"
    ):
        overrides["swa_full_tokens_ratio"] = DSV4_DEFAULT_SWA_FULL_TOKENS_RATIO
        logger.info(
            f"Setting swa_full_tokens_ratio to "
            f"{DSV4_DEFAULT_SWA_FULL_TOKENS_RATIO} for {model_arch}."
        )

    page_size = DSV4_PAGE_SIZE
    if cfg.device == "npu":
        # NPU keeps the device-aware "dsv4" backend (the registry routes it to
        # the Ascend V4 subclass); only the pool geometry / dtype differ.
        # set_default_server_args() pins all three backends to "ascend" for
        # generic NPU models; override that here so V4 stays consistently on
        # dsv4.
        page_size = _DSV4_NPU_PAGE_SIZE
        overrides["prefill_attention_backend"] = "dsv4"
        overrides["decode_attention_backend"] = "dsv4"
    overrides["page_size"] = page_size
    logger.info(
        f"Use dsv4 attention backend for {model_arch}, setting page_size to {page_size}."
    )

    if cfg.device == "cuda" and _is_sm70_v100():
        if cfg.swa_full_tokens_ratio is None:
            overrides["swa_full_tokens_ratio"] = 0.1
            logger.info(f"Setting swa_full_tokens_ratio to 0.1 for {model_arch}.")
        if cfg.dtype == "bfloat16":
            raise ValueError(
                f"{model_arch} on SM70 does not support bfloat16 "
                "(Volta has no BF16 tensor cores). Use --dtype float16."
            )
        if cfg.dtype == "auto":
            overrides["dtype"] = "float16"
            logger.info(f"SM70: setting dtype=float16 for {model_arch}.")
        if cfg.moe_runner_backend == "flashinfer_mxfp4":
            raise ValueError(
                f"{model_arch} on SM70 cannot use moe_runner_backend="
                "flashinfer_mxfp4. Use --moe-runner-backend marlin."
            )
        if cfg.moe_runner_backend == "auto":
            overrides["moe_runner_backend"] = "marlin"
            logger.info(f"SM70: use marlin as MoE runner backend for {model_arch}.")
        return overrides

    if cfg.moe_runner_backend == "auto":
        model_config = model_config_of(server_args)
        # nvidia/DeepSeek-V4-Pro-NVFP4 uses the routed TRT-LLM runner.
        if model_config.nvfp4_moe_meta is not None:
            overrides["moe_runner_backend"] = "flashinfer_trtllm_routed"
            logger.info(
                "Use flashinfer_trtllm_routed as MoE runner backend for "
                f"{model_arch} hybrid FP8+NVFP4 checkpoint."
            )
        elif (
            cfg.device == "cuda"
            and not get_platform().is_hip
            and cfg.moe_a2a_backend == "none"
            and not envs.SGLANG_DSV4_FP4_DEQUANT.get()
            and model_config.is_fp4_experts
            and (
                get_platform().is_sm90
                or get_platform().is_sm100
                or get_platform().is_sm120
            )
        ):
            overrides["moe_runner_backend"] = "flashinfer_mxfp4"
            logger.info(f"Use flashinfer_mxfp4 as MoE runner backend for {model_arch}.")
    return overrides
