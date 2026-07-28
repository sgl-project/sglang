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


def _parse_sparse_config(
    server_args,
    *,
    default_algorithm: Optional[str] = None,
    default_backend: Optional[str] = None,
    default_page_size: Optional[int] = None,
    default_min_sparse_prompt_len: Optional[int] = None,
) -> SparseConfig:
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
        if not isinstance(extra_config, dict):
            raise ValueError(
                f"hisparse_config must be a JSON object, got {type(extra_config).__name__}."
            )
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

    algorithm = extra_config.pop("algorithm", default_algorithm)
    backend = extra_config.pop("backend", default_backend)
    if isinstance(algorithm, str):
        algorithm = algorithm.strip().lower()
    if isinstance(backend, str):
        backend = backend.strip().lower()
    min_sparse_prompt_len = extra_config.pop(
        "min_sparse_prompt_len", default_min_sparse_prompt_len
    )
    page_size = extra_config.pop("page_size", default_page_size)

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


def parse_runtime_sparse_config(server_args) -> SparseConfig:
    """Parse and validate the runtime Quest configuration."""
    from sglang.srt.arg_groups.overrides import attention_backends_of

    prefill_backend, decode_backend = attention_backends_of(server_args)
    runtime_page_size = server_args.page_size
    config = _parse_sparse_config(
        server_args,
        default_algorithm="quest",
        default_backend=decode_backend or prefill_backend,
        default_page_size=runtime_page_size,
        default_min_sparse_prompt_len=0,
    )
    if not config.algorithm:
        raise ValueError("Sparse runtime config requires an algorithm.")
    if not config.backend:
        raise ValueError("Sparse runtime config requires an attention backend.")
    if not isinstance(config.page_size, int) or isinstance(config.page_size, bool):
        raise ValueError(
            f"Sparse runtime config page_size must be an integer, got {config.page_size!r}."
        )
    if config.page_size <= 0:
        raise ValueError(
            f"Sparse runtime config page_size must be positive, got {config.page_size}."
        )
    if runtime_page_size is not None and config.page_size != runtime_page_size:
        raise ValueError(
            "Sparse runtime config page_size must match the runtime KV cache "
            f"--page-size ({runtime_page_size}), got {config.page_size}."
        )
    if config.min_sparse_prompt_len is not None and (
        not isinstance(config.min_sparse_prompt_len, int)
        or isinstance(config.min_sparse_prompt_len, bool)
        or config.min_sparse_prompt_len < 0
    ):
        raise ValueError(
            "Sparse runtime config min_sparse_prompt_len must be a non-negative "
            f"integer, got {config.min_sparse_prompt_len!r}."
        )

    sparsity_ratio = config.sparse_extra_config.get("sparsity_ratio")
    if sparsity_ratio is not None and (
        not isinstance(sparsity_ratio, (int, float))
        or isinstance(sparsity_ratio, bool)
        or not 0 < sparsity_ratio <= 1
    ):
        raise ValueError(
            "Sparse runtime config sparsity_ratio must be in the range (0, 1], "
            f"got {sparsity_ratio!r}."
        )
    num_recent_pages = config.sparse_extra_config.get("num_recent_pages")
    if num_recent_pages is not None and (
        not isinstance(num_recent_pages, int)
        or isinstance(num_recent_pages, bool)
        or num_recent_pages <= 0
    ):
        raise ValueError(
            "Sparse runtime config num_recent_pages must be a positive integer, "
            f"got {num_recent_pages!r}."
        )
    enable_cuda_graph_retrieval = config.sparse_extra_config.get(
        "enable_cuda_graph_retrieval", False
    )
    if not isinstance(enable_cuda_graph_retrieval, bool):
        raise ValueError(
            "Sparse runtime config enable_cuda_graph_retrieval must be a boolean, "
            f"got {enable_cuda_graph_retrieval!r}."
        )

    return config


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    server_args,
    **kwargs,
) -> SparseCoordinator:
    config = parse_runtime_sparse_config(server_args)
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
