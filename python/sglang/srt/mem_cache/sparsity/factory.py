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


def hisparse_v2_expansion_ratio(server_args) -> float:
    """Expanded indexer region size as a multiple of the attention pool:
    expanded_tokens = ratio * max_total_num_tokens.

    Override via hisparse_config {"expansion_ratio": R}; default
    1 + hicache_ratio (admitted prefixes can span at most device + host
    capacity). DefaultPoolConfigurator (budget cell) and
    KVCacheConfigurator (index_buf_size) MUST both derive from this
    helper, or the capacity accounting splits.
    """
    cfg = _parse_sparse_config(server_args)
    ratio = cfg.sparse_extra_config.get("expansion_ratio")
    if ratio is not None:
        ratio = float(ratio)
        if ratio <= 0:
            raise ValueError(f"expansion_ratio must be > 0, got {ratio}")
        return ratio
    if getattr(server_args, "hicache_size", 0):
        logger.warning(
            "HiSparse V2: --hicache-size is set, so the default expanded "
            "indexer sizing (1 + hicache_ratio = %.1f) may not match the "
            "actual host capacity; set hisparse_config expansion_ratio "
            "explicitly.",
            1.0 + server_args.hicache_ratio,
        )
    return 1.0 + float(server_args.hicache_ratio)


def hisparse_v2_top_k(server_args, model_config) -> int:
    """The operative V2 decode top-k: the model's ``index_topk`` when
    present, else hisparse_config.top_k. A config top_k that diverges
    from the model's is rejected — the DSA indexer always selects
    index_topk positions, which governs all capacity sizing.
    """
    cfg = _parse_sparse_config(server_args)
    model_top_k = getattr(model_config.hf_text_config, "index_topk", None)
    if model_top_k is None:
        return cfg.top_k
    if int(model_top_k) != cfg.top_k:
        raise ValueError(
            f"HiSparse V2: hisparse_config top_k={cfg.top_k} differs from the "
            f"model's index_topk={int(model_top_k)}. The DSA indexer always "
            f"selects index_topk positions, which governs temp-slot sizing "
            f"and capacity accounting; remove top_k from hisparse_config or "
            f"set it to {int(model_top_k)}."
        )
    return int(model_top_k)


def hisparse_v2_device_buffer_tokens(server_args, model_config) -> int:
    """Per-request temp device-buffer size in tokens — V1's
    ``device_buffer_size`` knob, but defaulting to ``top_k`` (1x) for V2:
    a larger buffer trades hit rate for per-request device footprint and
    only pays off when host-DMA bandwidth is the bottleneck, so V2 grows
    it only when set explicitly (V1 keeps its 2*top_k parser default).

    This is the request's lifetime device floor; all capacity sizing
    MUST derive from this helper. The coordinator additionally validates
    page alignment and power-of-2.
    """
    _parse_sparse_config(server_args)  # config validation (clear errors)
    top_k = hisparse_v2_top_k(server_args, model_config)
    raw = json.loads(server_args.hisparse_config) if server_args.hisparse_config else {}
    explicit = raw.get("device_buffer_size")
    if explicit is None:
        return top_k
    return max(int(explicit), top_k)


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    server_args,
    **kwargs,
) -> SparseCoordinator:
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
