from __future__ import annotations

import json
import logging
import math
from typing import Optional

import msgspec
import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_dsa import DeepSeekDSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    DSABackendAdaptor,
    FlashAttentionAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseConfig,
    SparseCoordinator,
)
from sglang.srt.runtime_context import get_memory, get_schedule

logger = logging.getLogger(__name__)

_global_sparse_coordinator: Optional[SparseCoordinator] = None

_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_dsa": lambda config, device, **kw: DeepSeekDSAAlgorithm(
        config, device, **kw
    ),
}


def _create_sparse_algorithm(
    config: SparseConfig,
    device: torch.device,
    **kwargs,
) -> BaseSparseAlgorithm:
    algorithm_name = config.algorithm.lower()
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)

    if factory is None:
        raise ValueError(f"Unknown sparse algorithm: {algorithm_name}")

    return factory(config, device, **kwargs)


def _create_backend_adaptor(
    backend: str,
    device: torch.device,
    sparse_algorithm: BaseSparseAlgorithm,
    req_to_token_pool,
):
    """Create backend adaptor."""
    if isinstance(sparse_algorithm, DeepSeekDSAAlgorithm):
        return DSABackendAdaptor(device, req_to_token_pool)

    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device)

    raise ValueError(f"Unknown attention backend: {backend}")


def _parse_sparse_config() -> SparseConfig:
    """Parse hierarchical sparse config from JSON string.

    Required fields with defaults: top_k (2048), device_buffer_size (2*top_k),
    host_to_device_ratio (2), swap_in_block_size (960).
    Optional fields (default None): algorithm, backend, min_sparse_prompt_len,
    page_size. All remaining fields go to sparse_extra_config.
    """
    extra_config_str = get_memory().hisparse_config
    if extra_config_str is not None:
        try:
            extra_config = json.loads(extra_config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse hisparse_config: {e}") from e
    else:
        extra_config = {}

    top_k = extra_config.pop("top_k", 2048)
    device_buffer_size = extra_config.pop("device_buffer_size", 2 * top_k)
    host_to_device_ratio = extra_config.pop("host_to_device_ratio", 2)
    swap_in_block_size = extra_config.pop("swap_in_block_size", 960)

    if device_buffer_size < top_k:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be no smaller than top_k ({top_k})"
        )
    if not isinstance(swap_in_block_size, int) or isinstance(swap_in_block_size, bool):
        raise ValueError(
            f"swap_in_block_size must be an integer, got {swap_in_block_size!r}"
        )
    if swap_in_block_size <= 0 or swap_in_block_size > 1024:
        raise ValueError(
            f"swap_in_block_size ({swap_in_block_size}) must be in the range [1, 1024]"
        )

    algorithm = extra_config.pop("algorithm", None)
    backend = extra_config.pop("backend", None)
    min_sparse_prompt_len = extra_config.pop("min_sparse_prompt_len", None)
    page_size = extra_config.pop("page_size", None)

    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        swap_in_block_size=swap_in_block_size,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        sparse_extra_config=extra_config,
    )


def parse_hisparse_config() -> SparseConfig:
    """The hisparse config as resolved, with defaults where none was given."""
    return _parse_sparse_config()


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    **kwargs,
) -> SparseCoordinator:
    config = _parse_sparse_config()
    algorithm = _create_sparse_algorithm(config, device, **kwargs)
    backend_adaptor = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool
    )

    coordinator = SparseCoordinator(
        config=config,
        algorithm=algorithm,
        backend_adaptor=backend_adaptor,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
    )
    register_sparse_coordinator(coordinator)
    return coordinator


def register_sparse_coordinator(coordinator: SparseCoordinator) -> None:
    global _global_sparse_coordinator
    _global_sparse_coordinator = coordinator


def get_sparse_coordinator() -> Optional[SparseCoordinator]:
    return _global_sparse_coordinator


class HiSparseHiCacheConfig(msgspec.Struct, frozen=True, kw_only=True):
    top_k: int
    buffer_tokens: int
    expansion_ratio: float
    block_size: int


def parse_hisparse_hicache_config(*, model_top_k: int = 2048) -> HiSparseHiCacheConfig:
    memory = get_memory()
    return resolve_hisparse_hicache_config(
        config=memory.hisparse_config,
        model_top_k=model_top_k,
        hicache_ratio=memory.hicache_ratio,
        page_size=get_schedule().page_size,
    )


def resolve_hisparse_hicache_config(
    *, config: str | None, model_top_k: int, hicache_ratio: float, page_size: int
) -> HiSparseHiCacheConfig:
    raw = json.loads(config or "{}")
    if not isinstance(raw, dict):
        raise ValueError("hisparse_config must be a JSON object")
    top_k = raw.get("top_k", model_top_k)
    if top_k != model_top_k:
        raise ValueError(
            "HiSparse HiCache backend top_k must match the model's index_topk"
        )
    buffer_tokens = raw.get("device_buffer_size", top_k)
    block_size = raw.get("swap_in_block_size", 960)
    for name, value in (
        ("top_k", top_k),
        ("device_buffer_size", buffer_tokens),
        ("swap_in_block_size", block_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"HiSparse HiCache backend {name} must be a positive integer"
            )
    if buffer_tokens < top_k or buffer_tokens > 32767:
        raise ValueError(
            "HiSparse HiCache backend device_buffer_size must be in [top_k, 32767]"
        )
    if buffer_tokens % page_size:
        raise ValueError(
            "HiSparse HiCache backend device_buffer_size must be page-aligned"
        )
    if block_size > 1024 or block_size % 32:
        raise ValueError(
            "HiSparse HiCache backend swap_in_block_size must be a warp multiple <= 1024"
        )
    expansion = raw.get("expansion_ratio", hicache_ratio)
    if isinstance(expansion, bool) or not isinstance(expansion, (int, float)):
        raise ValueError("HiSparse HiCache backend expansion_ratio must be numeric")
    if not math.isfinite(expansion) or expansion <= 0:
        raise ValueError(
            "HiSparse HiCache backend expansion_ratio must be finite and positive"
        )
    return HiSparseHiCacheConfig(
        top_k=top_k,
        buffer_tokens=buffer_tokens,
        expansion_ratio=float(expansion),
        block_size=block_size,
    )
