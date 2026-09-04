"""Config-time override declarations for qwen3_vl.

Architectures: Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration.
"""

import logging
from typing import Any, Optional

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    model_config_of,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import get_device_memory_capacity, is_sm90_supported

logger = logging.getLogger(__name__)

_HOPPER_QWEN3_VL_TYPES = {"qwen3_vl", "qwen3_vl_moe"}


def large_hopper_qwen3_vl_model_type(server_args: Any, gpu_mem=None) -> Optional[str]:
    """Return the HF model_type on large Hopper Qwen3-VL, else None."""
    cfg = resolving_view(server_args)
    if gpu_mem is None:
        gpu_mem = get_device_memory_capacity(cfg.device)
    if not is_sm90_supported() or gpu_mem is None or gpu_mem < 60 * 1024:
        return None
    model_config = model_config_of(server_args)
    model_type = getattr(model_config.hf_config, "model_type", "")
    if model_config.is_multimodal and model_type in _HOPPER_QWEN3_VL_TYPES:
        return model_type
    return None


def expand_multimodal_decode_graph_to_running_limit(
    server_args: Any, decode_config: Any, gpu_mem
) -> None:
    """Keep profiled high-concurrency Qwen3-VL decode inside CUDA graph."""
    from sglang.srt.model_executor.cuda_graph_config import Phase

    cfg = resolving_view(server_args)
    locked = getattr(server_args, "_cuda_graph_config_locked", set())
    max_running_requests = cfg.max_running_requests
    if not (
        gpu_mem is not None
        and max_running_requests is not None
        and max_running_requests <= 512
        and decode_config.max_bs is not None
        and decode_config.max_bs < max_running_requests
        and (Phase.DECODE, "max_bs") not in locked
        and (Phase.DECODE, "bs") not in locked
    ):
        return
    if large_hopper_qwen3_vl_model_type(server_args, gpu_mem) is None:
        return

    logger.info(
        "Expanding multimodal decode CUDA graph max_bs from %d to "
        "max_running_requests=%d.",
        decode_config.max_bs,
        max_running_requests,
    )
    decode_config.max_bs = max_running_requests


@_register_for("Qwen3VLForConditionalGeneration")
def _qwen3vl_overrides(server_args: Any, hf_config: Any) -> dict:

    cfg = resolving_view(server_args)
    if (
        get_platform().is_hip
        and envs.SGLANG_USE_AITER_UNIFIED_ATTN.get()
        and cfg.page_size is None
    ):
        logger.info(
            "Setting page_size=16 for aiter unified attention on Qwen3VLForConditionalGeneration."
        )
        return {"page_size": 16}
    return {}


@_register_for(
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
)
def _qwen3vl_hopper_serving_overrides(server_args: Any, hf_config: Any) -> dict:
    """Select the profiled Qwen3-VL serving path on large Hopper GPUs."""
    model_type = large_hopper_qwen3_vl_model_type(server_args)
    if model_type is None:
        return {}

    cfg = resolving_view(server_args)
    if not envs.SGLANG_VLM_CACHE_SIZE_MB.is_set():
        # Repeated-image traffic can opt in to embedding retention. Disable
        # it by default for streaming traffic, where every image is used once.
        envs.SGLANG_VLM_CACHE_SIZE_MB.set(0)

    updates = {}
    preprocess_cache_size_mb = cfg.mm_preprocess_cache_size_mb
    if preprocess_cache_size_mb is None:
        preprocess_cache_size_mb = 0
        updates["mm_preprocess_cache_size_mb"] = 0
    cache_retention_enabled = (
        preprocess_cache_size_mb > 0 or envs.SGLANG_VLM_CACHE_SIZE_MB.get() > 0
    )
    if cfg.mm_feature_transport is None and not cache_retention_enabled:
        updates["mm_feature_transport"] = "cuda_ipc"
    if (
        not envs.SGLANG_MM_FEATURE_CACHE_MB.is_set()
        and cfg.max_running_requests is not None
        and cfg.max_running_requests >= 400
        and not cache_retention_enabled
        and (
            updates.get("mm_feature_transport", cfg.mm_feature_transport)
            in (None, "cuda_ipc")
        )
    ):
        # Keep a full high-concurrency wave GPU-resident instead of
        # falling back to CPU transport while the scheduler drains it.
        envs.SGLANG_MM_FEATURE_CACHE_MB.set(3 * 1024)
    if (
        cfg.radix_eviction_policy == "lru"
        and not cfg._radix_eviction_policy_explicitly_set
    ):
        updates["radix_eviction_policy"] = "priority"
    if cfg.prefill_decode_interval is None:
        updates["prefill_decode_interval"] = 22
    if cfg.attention_backend is None and cfg.decode_attention_backend is None:
        updates["decode_attention_backend"] = "flashinfer"

    if updates:
        logger.info(
            "Applying profiled %s serving defaults on a large Hopper GPU: %s",
            model_type,
            updates,
        )
    return updates
