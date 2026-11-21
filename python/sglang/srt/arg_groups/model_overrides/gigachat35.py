"""Config-time override declarations for gigachat35."""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@_register_for(
    "GigaChat35ForCausalLM",
    "GigaChat35ForCausalLMNextN",
    "GigaChat35OmniForConditionalGeneration",
)
def _gigachat35_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {"disable_shared_experts_fusion": True}
    if cfg.speculative_algorithm == "EAGLE":
        logger.info(
            "Enable multi-layer EAGLE speculative decoding for GigaChat 3.5 model."
        )
        overrides["enable_multi_layer_eagle"] = True
    return overrides


@_register_for("GigaChat35OmniForConditionalGeneration")
def _gigachat35_omni_overrides(server_args: Any, hf_config: Any) -> dict:
    """Size the embedding cache to keep a whole video across prefill chunks."""
    if not envs.SGLANG_VLM_CACHE_SIZE_MB.is_set():
        # A 512-frame video is 541 MiB of embeddings; the 100 MiB default
        # refuses it, so every prefill chunk would re-run the encoders.
        envs.SGLANG_VLM_CACHE_SIZE_MB.set(2048)
    return {}
