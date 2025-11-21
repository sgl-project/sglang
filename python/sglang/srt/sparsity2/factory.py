import logging
from typing import Optional

import torch

from sglang.srt.sparsity2.algorithms.base_algorithm import (
    BaseSparseAlgorithm,
    FakeRandomSparseAlgorithm,
    SparseMode,
)
from sglang.srt.sparsity2.algorithms.page_mean_pooling import PageMeanPoolingAlgorithm
from sglang.srt.sparsity2.backend.backend_adaptor import (
    BackendAdaptor,
    FlashAttentionAdaptor,
)
from sglang.srt.sparsity2.core.sparse_coordinator import SparseConfig, SparseCoordinator

logger = logging.getLogger(__name__)

_global_sparse_coordinator: Optional[SparseCoordinator] = None


def create_sparse_algorithm(
    config: SparseConfig,
    device: torch.device,
    start_layer: int,
    end_layer: int,
    **kwargs,
) -> BaseSparseAlgorithm:
    """Create sparse algorithm based on config."""
    algorithm_name = config.algorithm.lower()

    if algorithm_name == "fake_random_sparse":
        return FakeRandomSparseAlgorithm(config, device, **kwargs)
    elif algorithm_name == "page_mean_pooling":
        return PageMeanPoolingAlgorithm(
            config, device, start_layer, end_layer, **kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def create_backend_adaptor(
    backend: str, device: torch.device
) -> Optional[BackendAdaptor]:
    """Create backend adaptor based on backend name."""
    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device)
    return None


def create_sparse_coordinator(
    device: torch.device,
    page_size: int,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    **kwargs,
) -> SparseCoordinator:
    """Create a sparse coordinator."""
    config = SparseConfig(page_size=page_size, algorithm="page_mean_pooling")

    sparse_algorithm = create_sparse_algorithm(
        config, device, start_layer, end_layer, **kwargs
    )

    if sparse_algorithm.get_sparse_mode() == SparseMode.TOKEN_WISE:
        assert page_size == 1, "TOKEN_WISE sparse requires page_size=1"

    kv_cache_capacity = token_to_kv_pool.get_key_buffer(start_layer).shape[0]
    total_num_pages = kv_cache_capacity // config.page_size
    backend_adaptor = create_backend_adaptor(config.backend, device)

    coordinator = SparseCoordinator(
        config=config,
        algorithm=sparse_algorithm,
        backend_adaptor=backend_adaptor,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
        total_num_pages=total_num_pages,
    )

    register_sparse_coordinator(coordinator)
    logger.info(
        f"Created SparseCoordinator with algorithm={config.algorithm}, backend={config.backend}"
    )
    return coordinator


def register_sparse_coordinator(coordinator: SparseCoordinator) -> None:
    """Register global sparse coordinator."""
    global _global_sparse_coordinator
    _global_sparse_coordinator = coordinator
    logger.info("Registered global sparse coordinator")


def get_sparse_coordinator() -> Optional[SparseCoordinator]:
    """Get global sparse coordinator."""
    return _global_sparse_coordinator
