import json
import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_nsa import DeepSeekNSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    FlashAttentionAdaptor,
    NSABackendAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseConfig,
    SparseCoordinator,
)
from sglang.srt.mem_cache.sparsity.core.sparse_kvcache_manager import (
    SparseKVCacheManager,
)

logger = logging.getLogger(__name__)

_global_sparse_coordinator: Optional[SparseCoordinator] = None

_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_nsa": lambda config, device, **kw: DeepSeekNSAAlgorithm(
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
    sparse_kv_cache_manager,
):
    """Create backend adaptor."""
    if isinstance(sparse_algorithm, DeepSeekNSAAlgorithm):
        return NSABackendAdaptor(device, req_to_token_pool, sparse_kv_cache_manager)

    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device, req_to_token_pool, sparse_kv_cache_manager)

    raise ValueError(f"Unknown attention backend: {backend}")


def _parse_sparse_config(server_args) -> SparseConfig:
    """Parse hierarchical sparse config"""
    # Parse extra config if provided
    extra_config_str = server_args.hierarchical_sparse_attention_extra_config
    if extra_config_str is not None:
        try:
            extra_config = json.loads(extra_config_str)

            # Extract algorithm and backend
            algorithm = extra_config.pop("algorithm", "quest")
            backend = extra_config.pop("backend", "flashattention")

            # Everything else goes to algorithm_extra_config
            sparse_extra_config = extra_config
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse hierarchical_sparse_attention_extra_config: {e}"
            )

    config = SparseConfig(
        algorithm=algorithm,
        backend=backend,
        page_size=server_args.page_size,
        sparse_extra_config=sparse_extra_config,
    )
    return config


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    token_to_kv_pool_allocator,
    tp_group,
    server_args,
    **kwargs,
) -> SparseCoordinator:
    config = _parse_sparse_config(server_args)
    algorithm = _create_sparse_algorithm(config, device, **kwargs)
    sparse_kv_cache_manager = SparseKVCacheManager(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tp_group=tp_group,
        server_args=server_args,
    )

    backend_adaptor = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool, sparse_kv_cache_manager
    )

    coordinator = SparseCoordinator(
        config=config,
        algorithm=algorithm,
        backend_adaptor=backend_adaptor,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        sparse_kv_cache_manager=sparse_kv_cache_manager,
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


def is_hierarchical_sparse_attention_enabled() -> bool:
    return _global_sparse_coordinator is not None
