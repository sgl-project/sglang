import json
import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_nsa import DeepSeekNSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity import (
    DoubleSparsityAlgorithm,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    FlashAttentionAdaptor,
    NSABackendAdaptor,
)
from sglang.srt.mem_cache.sparsity.backend.ds_flash_attention_adaptor import (
    DSFlashAttentionAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseConfig,
    SparseCoordinator,
)

logger = logging.getLogger(__name__)

_global_sparse_coordinator: Optional[SparseCoordinator] = None

_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_nsa": lambda config, device, **kw: DeepSeekNSAAlgorithm(
        config, device, **kw
    ),
    "double_sparsity": lambda config, device, **kw: DoubleSparsityAlgorithm(
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
    if isinstance(sparse_algorithm, DeepSeekNSAAlgorithm):
        return NSABackendAdaptor(device, req_to_token_pool)

    if isinstance(sparse_algorithm, DoubleSparsityAlgorithm):
        if backend not in ("fa3", "flashattention"):
            raise ValueError(
                f"Double Sparsity v1 requires FA3, got backend={backend!r}"
            )
        return DSFlashAttentionAdaptor(
            device,
            max_selected_per_request=sparse_algorithm.runtime_config.max_selected_per_request,
        )

    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device)

    raise ValueError(f"Unknown attention backend: {backend}")


def _parse_sparse_config(server_args) -> SparseConfig:
    """Parse hierarchical sparse config from JSON string.

    Required fields with defaults: top_k (2048), device_buffer_size (2*top_k),
    host_to_device_ratio (2).
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

    if device_buffer_size < top_k:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be no smaller than top_k ({top_k})"
        )

    algorithm = extra_config.pop("algorithm", None)
    backend = extra_config.pop("backend", None)
    min_sparse_prompt_len = extra_config.pop("min_sparse_prompt_len", None)
    page_size = extra_config.pop("page_size", None)

    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        sparse_extra_config=extra_config,
    )


def parse_hisparse_config(server_args) -> SparseConfig:
    """Parse hisparse config from server_args, returning defaults if no config provided."""
    return _parse_sparse_config(server_args)


def parse_double_sparsity_config(server_args) -> SparseConfig:
    """Build a SparseConfig for the Double Sparsity algorithm from ServerArgs.

    Unlike hisparse_config (free-form JSON in --hisparse-config), Double Sparsity
    derives most fields from typed --double-sparsity-* flags so they can be
    validated at startup. The calibration JSON path is propagated through
    sparse_extra_config and consumed by DoubleSparsityAlgorithm.
    """
    if not getattr(server_args, "enable_double_sparsity", False):
        raise ValueError(
            "parse_double_sparsity_config called without --enable-double-sparsity"
        )

    extra: dict = {
        "config_path": server_args.double_sparsity_config,
        "heavy_channels": server_args.double_sparsity_heavy_channels,
        "token_budget": server_args.double_sparsity_token_budget,
        "recent_tokens": server_args.double_sparsity_recent_tokens,
        "sink_tokens": server_args.double_sparsity_sink_tokens,
        "max_selected_per_request": server_args.double_sparsity_max_selected_per_request,
        "gqa_reduction": server_args.double_sparsity_gqa_reduction,
        "klabel_dtype": server_args.double_sparsity_klabel_dtype,
    }

    return SparseConfig(
        top_k=server_args.double_sparsity_token_budget,
        device_buffer_size=2 * server_args.double_sparsity_token_budget,
        host_to_device_ratio=2,
        algorithm="double_sparsity",
        backend="fa3",
        page_size=1,
        min_sparse_prompt_len=server_args.double_sparsity_min_seq_len,
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
    """Build and register a SparseCoordinator.

    `config` is parsed from `server_args.hisparse_config` if not provided.
    Caller-supplied `config` lets distinct sparse algorithms (e.g. Double
    Sparsity) provide their own typed configuration without going through
    the hisparse JSON path.
    """
    if config is None:
        config = _parse_sparse_config(server_args)
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
