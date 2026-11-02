from __future__ import annotations

import json
import logging
import os
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolving_view,
)
from sglang.srt.connector import ConnectorType, parse_connector_type
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.hf_transformers_utils import get_config

logger = logging.getLogger(__name__)

MUSE_SPARK_ARCHITECTURES = ("MuseSparkForConditionalGeneration",)

MUSE_SPARK_CHAT_TEMPLATE_FILE = "chat_template.jinja"

# Graph-pool and KV-sizing switches the reference launcher (sglang_meta) sets.
MUSE_SPARK_ENV_DEFAULTS = (
    (envs.SGLANG_ENABLE_CUDA_GRAPH_DEDUP, True),
    (envs.SGLANG_ENABLE_GRAPH_POOL_BORROW, True),
    (envs.SGLANG_ENABLE_GRAPH_POOL_PRECARVE, True),
    (envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING, True),
    (envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS, False),
)


def _muse_spark_arch_of(hf_config: Any) -> str | None:
    architectures = getattr(hf_config, "architectures", None) or []
    arch = architectures[0] if architectures else ""
    return arch if arch in MUSE_SPARK_ARCHITECTURES else None


def apply_muse_spark_env_defaults(server_args: Any) -> None:
    """Muse Spark launch and environment defaults; explicit values take precedence.

    Runs before handle_gpu_memory_settings, whose mem-fraction heuristic reads
    the post-capture switch.
    """
    cfg = resolving_view(server_args)
    if parse_connector_type(cfg.model_path) == ConnectorType.INSTANCE:
        return
    hf_config = get_config(
        cfg.model_path,
        trust_remote_code=cfg.trust_remote_code,
        revision=cfg.revision,
        model_config_parser=cfg.model_config_parser,
    )
    arch = _muse_spark_arch_of(hf_config)
    if arch is None:
        return

    _apply_muse_spark_launch_defaults(server_args)

    applied = []
    for field, value in MUSE_SPARK_ENV_DEFAULTS:
        if not field.is_set():
            field.set(value)
            applied.append(f"{field.name}={value}")
    # The serialized NVFP4 checkpoint carries per-token activation scales.
    if (
        cfg.quantization == "modelopt_fp4"
        and not envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.is_set()
    ):
        envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.set(True)
        applied.append("SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=True")
    if applied:
        logger.info(f"{arch} environment defaults: {' '.join(applied)}")


def _apply_muse_spark_launch_defaults(server_args: Any) -> None:
    """Keep the minimal recipe equivalent to explicit, hardware-tuned flags."""
    cfg = resolving_view(server_args)
    platform = get_platform()
    defaults = {}
    # ModelConfig handles both config.json and ModelOpt hf_quant_config.json,
    # including local paths whose directory names do not mention NVFP4.
    if cfg.quantization is None and not cfg._quantization_explicitly_unset:
        if model_config_of(server_args).quantization == "modelopt_fp4":
            defaults["quantization"] = "modelopt_fp4"
    quantization = defaults.get("quantization", cfg.quantization)
    if quantization == "modelopt_fp4" and cfg.moe_runner_backend == "auto":
        # TRT-LLM FP4 MoE supports data-center Blackwell (SM100/SM103),
        # not the SM120 consumer cards handled by the generic Cutlass default.
        if platform.is_sm100:
            defaults["moe_runner_backend"] = "flashinfer_trtllm"
        elif platform.is_sm90:
            defaults["moe_runner_backend"] = "marlin"
    if cfg.mem_fraction_static is None:
        if platform.is_blackwell:
            defaults["mem_fraction_static"] = 0.8
        elif platform.is_sm90:
            defaults["mem_fraction_static"] = 0.85
    if cfg.enable_megamoe_for_layernorm_sp is None:
        defaults["enable_megamoe_for_layernorm_sp"] = (
            platform.is_sm100
            and quantization == "modelopt_fp4"
            and cfg.disaggregation_mode != "decode"
            and cfg.enable_layernorm_sp
            and cfg.ep_size > 1
            and cfg.ep_size == cfg.tp_size
        )
    if cfg.speculative_algorithm == "DFLASH" and cfg.speculative_draft_model_path:
        draft_config = get_config(
            cfg.speculative_draft_model_path,
            trust_remote_code=cfg.trust_remote_code,
            revision=cfg.speculative_draft_model_revision,
            model_override_args=json.loads(cfg.json_model_override_args),
        )
        if "DFlash2DraftModel" in (getattr(draft_config, "architectures", None) or []):
            if cfg.speculative_draft_attention_backend is None and (
                platform.is_sm100 or platform.is_sm90
            ):
                defaults["speculative_draft_attention_backend"] = "fa4"
            if (
                cfg.speculative_num_draft_tokens is None
                and cfg.speculative_dflash_block_size is None
            ):
                defaults["speculative_num_draft_tokens"] = 8
    if defaults:
        declare_resolution(server_args, "muse_spark_launch_defaults", **defaults)
        logger.info("Muse Spark launch defaults: %s", defaults)


def muse_spark_chat_template_default(server_args: Any) -> str | None:
    """The checkpoint's Jinja chat template when --chat-template is not set.

    Reads the memoised model configuration only, like the architecture parser
    defaults it is declared next to.
    """
    cfg = resolving_view(server_args)
    if cfg.chat_template is not None:
        return None
    model_config = getattr(server_args, "_model_config", None)
    if model_config is None:
        return None
    arch = _muse_spark_arch_of(model_config.hf_config)
    if arch is None:
        return None
    template = _checkpoint_file(model_config, MUSE_SPARK_CHAT_TEMPLATE_FILE)
    if template is not None:
        logger.info(f"Defaulting --chat-template to '{template}' for {arch}.")
        return template
    logger.warning(
        f"{arch} checkpoint has no {MUSE_SPARK_CHAT_TEMPLATE_FILE}; "
        "--chat-template stays unset."
    )
    return None


def _checkpoint_file(model_config: Any, filename: str) -> str | None:
    model_path = getattr(model_config, "model_path", None)
    if not model_path:
        return None
    if os.path.isdir(model_path):
        candidate = os.path.join(model_path, filename)
        return candidate if os.path.isfile(candidate) else None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=model_path,
            filename=filename,
            revision=getattr(model_config, "revision", None),
        )
    except (OSError, ValueError, ImportError):
        return None
