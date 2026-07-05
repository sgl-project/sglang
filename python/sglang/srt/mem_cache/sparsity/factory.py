import json
import logging
from typing import Optional

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

logger = logging.getLogger(__name__)

_global_sparse_coordinator: Optional[SparseCoordinator] = None

_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_dsa": lambda config, device, **kw: DeepSeekDSAAlgorithm(
        config, device, **kw
    ),
}

# Attention backends the experimental --enable-sparse-attention path supports.
_SUPPORTED_SPARSE_BACKENDS = {"fa3", "flashattention"}


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


def _parse_sparse_config(server_args) -> SparseConfig:
    """Parse hierarchical sparse config from JSON string.

    Required fields with defaults: top_k (2048), device_buffer_size (2*top_k),
    host_to_device_ratio (2), swap_in_block_size (960).
    Optional fields (default None): algorithm, backend, min_sparse_prompt_len,
    page_size. All remaining fields go to sparse_extra_config.
    """
    extra_config_str = server_args.hisparse_config
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


def parse_hisparse_config(server_args) -> SparseConfig:
    """Parse hisparse config from server_args, returning defaults if no config provided."""
    return _parse_sparse_config(server_args)


def parse_sparse_attention_config(server_args) -> SparseConfig:
    """Parse the experimental --sparse-attention-config JSON into a SparseConfig.

    Distinct from parse_hisparse_config (which reads --hisparse-config for the
    DeepSeek/DSA host-offload path). Fills sensible defaults so the coordinator is
    safe to construct for a non-MLA FlashAttention model. Recognized top-level
    keys: algorithm, backend, page_size, min_sparse_prompt_len, top_k,
    device_buffer_size, host_to_device_ratio. Any other keys (e.g. sparsity_ratio,
    num_recent_pages) -- or a nested "algorithm_config" dict -- become
    sparse_extra_config, consumed by the algorithm implementation.
    """
    raw = getattr(server_args, "sparse_attention_config", None)
    cfg = {}
    if raw:
        try:
            cfg = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse sparse_attention_config: {e}") from e

    algorithm = str(cfg.pop("algorithm", "quest")).lower()
    if algorithm not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown sparse-attention algorithm '{algorithm}'. "
            f"Available: {sorted(_ALGORITHM_REGISTRY)}"
        )

    backend = (
        cfg.pop("backend", None)
        or getattr(server_args, "attention_backend", None)
        or "fa3"
    )
    if backend not in _SUPPORTED_SPARSE_BACKENDS:
        raise ValueError(
            f"Sparse attention does not support attention backend '{backend}'. "
            f"Supported: {sorted(_SUPPORTED_SPARSE_BACKENDS)} "
            f"(run with --attention-backend fa3)."
        )

    page_size = (
        cfg.pop("page_size", None) or getattr(server_args, "page_size", None) or 1
    )
    min_sparse_prompt_len = int(cfg.pop("min_sparse_prompt_len", 0))
    if min_sparse_prompt_len < 0:
        raise ValueError(
            f"min_sparse_prompt_len must be >= 0, got {min_sparse_prompt_len}"
        )
    top_k = cfg.pop("top_k", 2048)
    device_buffer_size = cfg.pop("device_buffer_size", 2 * top_k)
    host_to_device_ratio = cfg.pop("host_to_device_ratio", 2)

    extra = dict(cfg.pop("algorithm_config", {}) or {})
    # remaining flat keys (e.g. sparsity_ratio, num_recent_pages) -> extra config
    extra.update(cfg)

    sparsity_ratio = extra.get("sparsity_ratio")
    if sparsity_ratio is not None and not 0.0 <= float(sparsity_ratio) < 1.0:
        raise ValueError(f"sparsity_ratio must be in [0, 1), got {sparsity_ratio}")

    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        sparse_extra_config=extra,
    )


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    server_args,
    config: Optional[SparseConfig] = None,
    **kwargs,
) -> SparseCoordinator:
    # `config` lets callers (e.g. the experimental --enable-sparse-attention path)
    # pass a SparseConfig parsed from their own flag. Falls back to the hisparse
    # config parser for backward compatibility.
    if config is None:
        config = _parse_sparse_config(server_args)
    if not config.page_size:
        config.page_size = getattr(server_args, "page_size", None) or 1
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
