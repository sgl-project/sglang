from __future__ import annotations

import logging
import os
from typing import Any

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.connector import ConnectorType, parse_connector_type
from sglang.srt.environ import envs
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
    """Muse Spark environment defaults; an explicit value in the environment wins.

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
