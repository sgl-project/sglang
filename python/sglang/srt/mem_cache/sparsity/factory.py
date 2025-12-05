import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithm,
    PageMeanPoolingAlgorithm,
    SparseMode,
)
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_nsa import DeepSeekNSAAlgorithm
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
    "page_mean_pooling": lambda config, device, start_layer, end_layer, **kw: PageMeanPoolingAlgorithm(
        config, device, start_layer, end_layer, **kw
    ),
    "deepseek_nsa": lambda config, device, **kw: DeepSeekNSAAlgorithm(
        config, device, **kw
    ),
}


def _create_sparse_algorithm(
    config: SparseConfig,
    device: torch.device,
    start_layer: int,
    end_layer: int,
    **kwargs,
) -> BaseSparseAlgorithm:
    algorithm_name = config.algorithm.lower()
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)

    if factory is None:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    logger.info(f"Creating {algorithm_name} algorithm")
    return factory(
        config, device, start_layer=start_layer, end_layer=end_layer, **kwargs
    )


def _create_backend_adaptor(
    backend: str,
    device: torch.device,
    sparse_algorithm: BaseSparseAlgorithm,
    req_to_token_pool,
    decode_offload_manager,
):
    """Create backend adaptor."""
    sparse_mode = sparse_algorithm.get_sparse_mode()
    if isinstance(sparse_algorithm, DeepSeekNSAAlgorithm):
        logger.info("Creating NSA backend adaptor")
        return NSABackendAdaptor(
            device, sparse_mode, req_to_token_pool, decode_offload_manager
        )

    if backend in ["fa3", "flashattention"]:
        logger.info("Creating FlashAttention backend adaptor")
        return FlashAttentionAdaptor(device, sparse_mode)

    raise ValueError(f"Unknown backend: {backend}")


def _calculate_total_pages(token_to_kv_pool, start_layer: int, page_size: int) -> int:
    kv_cache_capacity = token_to_kv_pool.get_key_buffer(start_layer).shape[0]
    return kv_cache_capacity // page_size


def create_sparse_coordinator(
    device: torch.device,
    page_size: int,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    token_to_kv_pool_allocator,
    tp_group,
    server_args,
    **kwargs,
) -> SparseCoordinator:
    config = SparseConfig(page_size=page_size, algorithm="deepseek_nsa")
    algorithm = _create_sparse_algorithm(
        config, device, start_layer, end_layer, **kwargs
    )
    sparse_mode = algorithm.get_sparse_mode()

    if sparse_mode == SparseMode.TOKEN_WISE:
        assert page_size == 1, "TOKEN_WISE sparse requires page_size=1"

    sparse_kv_cache_manager = SparseKVCacheManager(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tp_group=tp_group,
        server_args=server_args,
    )

    backend_adaptor = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool, sparse_kv_cache_manager
    )
    total_num_pages = _calculate_total_pages(
        token_to_kv_pool, start_layer, config.page_size
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
        total_num_pages=total_num_pages,
    )
    sparse_kv_cache_manager.req_states = coordinator.states

    register_sparse_coordinator(coordinator)
    logger.info(
        f"SparseCoordinator created: algorithm={config.algorithm}, mode={sparse_mode.value}"
    )
    return coordinator


def register_sparse_coordinator(coordinator: SparseCoordinator) -> None:
    global _global_sparse_coordinator
    _global_sparse_coordinator = coordinator


def get_sparse_coordinator() -> Optional[SparseCoordinator]:
    return _global_sparse_coordinator
