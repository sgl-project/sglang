import json
import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.hisparse_protocol import HiSparseCoordinator

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

# The two documented ways to run HiSparse, quoted back at the user whenever the
# cache configuration matches neither.
_PRIVATE_HOST_RECIPE = "--enable-hisparse --disable-radix-cache"
_HICACHE_RECIPE = (
    "--enable-hisparse --enable-hierarchical-cache --hicache-write-policy write_back"
)


class HiSparseBacking(str, Enum):
    """The logical KV pool that backs HiSparse's swapped-out attention KV.

    HiSparse keeps the indexer KV GPU-resident over an expanded region and holds
    only a small hot attention working set per request; everything else is
    fetched per decode step by top-k swap-in. What differs between deployments is
    *where a token's attention KV lives when it is not in that hot working set*:

    - `PRIVATE_HOST`: the coordinator owns a private pinned host pool and stages
      a request's whole prefix into it at admission, so the host copy is complete
      by construction and no eviction protocol is needed. The regular GPU pool is
      bypassed, and the radix cache must be off (there is nothing to share).
    - `HICACHE`: the radix tree plus the HiCache host tier own the KV. Attention
      KV stays in the regular GPU pool (full prefix reuse, hot KV kept in HBM)
      and is written back to host only under memory pressure, so host residency
      changes over time and eviction has to be coordinated.

    The backing is a startup decision resolved from the cache configuration
    rather than from a flag of its own -- `--enable-hisparse` is the only switch
    users write. It lives here, below `managers/`, because its consumers sit on
    both sides of that line: pool construction and the capacity model read it to
    size memory, and the coordinator factory reads it to pick an implementation.
    """

    PRIVATE_HOST = "private_host"
    HICACHE = "hicache"

    def __str__(self) -> str:
        return self.value


def resolve_hisparse_backing(
    *,
    enable_hisparse: bool,
    disable_radix_cache: bool,
    enable_hierarchical_cache: bool,
    hicache_write_policy: str,
) -> Optional[HiSparseBacking]:
    """Resolve the HiSparse backing, or None when HiSparse is off.

    Raises ValueError on a cache configuration that HiSparse cannot back, with
    both legal recipes in the message. Pure: callers include server-args
    validation, the KV cache / pool configurators and the coordinator factory,
    and they must all agree, so this is the single place the rule is written
    down. `hisparse_backing` below is the same rule for a caller holding a
    `server_args`.
    """
    if not enable_hisparse:
        return None

    if disable_radix_cache:
        # No tree, so nothing can evict KV to host on demand: the coordinator
        # stages each prefix into its own host pool instead.
        return HiSparseBacking.PRIVATE_HOST

    if not enable_hierarchical_cache:
        raise ValueError(
            "--enable-hisparse with the radix cache enabled requires "
            "--enable-hierarchical-cache: with a radix tree but no host tier, "
            "evicted attention KV has nowhere to go. Run either "
            f"'{_HICACHE_RECIPE}' (HiCache backs the evicted KV, prefixes are "
            f"shared) or '{_PRIVATE_HOST_RECIPE}' (private host pool, no prefix "
            "reuse)."
        )

    if hicache_write_policy != "write_back":
        raise ValueError(
            "--enable-hisparse with hierarchical cache requires "
            "--hicache-write-policy write_back (got "
            f"'{hicache_write_policy}'): HiSparse reads evicted attention KV "
            "back from the host tier, which only holds a copy of every evicted "
            f"page under write_back. Run '{_HICACHE_RECIPE}', or "
            f"'{_PRIVATE_HOST_RECIPE}' for the private-host backing."
        )

    return HiSparseBacking.HICACHE


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


def hisparse_backing(server_args) -> Optional[HiSparseBacking]:
    """Which logical KV pool backs HiSparse for this configuration (None if off).

    The one resolution entry point for code holding a `server_args`: startup
    validation, the pool / KV cache configurators and the coordinator factory
    all go through here so they cannot disagree. Runtime consumers read
    `coordinator.backing` instead of re-resolving.
    """
    return resolve_hisparse_backing(
        enable_hisparse=server_args.enable_hisparse,
        disable_radix_cache=server_args.disable_radix_cache,
        enable_hierarchical_cache=server_args.enable_hierarchical_cache,
        hicache_write_policy=server_args.hicache_write_policy,
    )


def _indexer_top_k(*, config, model_config) -> int:
    """`hisparse_indexer_top_k` for a caller that already parsed the config."""
    # HF configs are dynamic and only DSA checkpoints carry index_topk, so its
    # absence is a fact about the model, not a missing attribute on our own type.
    return getattr(model_config.hf_text_config, "index_topk", config.top_k)


def hisparse_indexer_top_k(*, server_args, model_config) -> int:
    """The operative indexer top-k, shared by every backing.

    A model that publishes its own `index_topk` wins over the config default: the
    DSA indexer selects that many positions whatever was asked for, and both
    backings size their swap-in geometry from this number, so they must agree on
    it.
    """
    return _indexer_top_k(
        config=_parse_sparse_config(server_args), model_config=model_config
    )


def hisparse_indexer_expansion_ratio(server_args) -> float:
    """Indexer region size as a multiple of the attention pool's token capacity.

    HiSparse keeps the indexer KV GPU-resident for the whole history while the
    attention KV of an evicted prefix lives elsewhere, so the region covers more
    tokens than the attention pool holds. How many more depends on the backing:

    - `PRIVATE_HOST`: `host_to_device_ratio`, the size of the private host pool
      and the only place a swapped-out token can be. The indexer index IS the
      logical token index, so the region is exactly the logical space.
    - `HICACHE`: `2 + hicache_ratio` -- one pool for the base region (a KV page id
      is its indexer page id) plus `1 + hicache_ratio` for the expanded region,
      which holds a copy of every admitted prefix and is bounded by the two tiers
      those prefixes live in. `1 + hicache_ratio` is the natural misreading, since
      that IS the two tiers; it leaves the expanded region one pool short and
      silently halves admission depth (measured: 3 admitted 30.8K-token requests
      per rank where 5 fit). `hisparse_config {"expansion_ratio": R}` overrides,
      and sets the TOTAL multiple.

    Returns 1.0 when HiSparse is off. The capacity model
    (`model_executor/pool_configurator.py`, indexer overhead cell) and the pool
    construction (`mem_cache/kv_cache_configurator.py`, `index_buf_size`) MUST
    both derive from this helper, or the reserved memory and the allocated
    buffer disagree and the pool overruns the budget it was sized against.
    """
    backing = hisparse_backing(server_args)
    if backing is None:
        return 1.0

    config = _parse_sparse_config(server_args)
    if backing is HiSparseBacking.PRIVATE_HOST:
        ratio = config.host_to_device_ratio
        # The private-host pool multiplies token counts by this, so a fraction
        # would produce a non-integral buffer size deep inside pool allocation.
        if int(ratio) != ratio:
            raise ValueError(
                "hisparse_config host_to_device_ratio must be an integer for the "
                f"private-host backing, got {ratio!r}."
            )
        return float(ratio)

    override = config.sparse_extra_config.get("expansion_ratio")
    if override is not None:
        ratio = float(override)
        if ratio <= 0:
            raise ValueError(f"expansion_ratio must be > 0, got {ratio}")
        return ratio
    if server_args.hicache_size:
        logger.warning(
            "HiSparse: --hicache-size is set, so the default indexer expansion "
            "(2 + hicache_ratio = %.1f) may not match the actual host capacity; "
            "set hisparse_config expansion_ratio explicitly.",
            2.0 + server_args.hicache_ratio,
        )
    return 2.0 + float(server_args.hicache_ratio)


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


def hisparse_indexer_regions(
    *,
    page_size: int,
    num_indexer_pages: int,
    device_pool_size: int,
) -> tuple[int, int]:
    """Split the HiCache backing's indexer buffer: (base_pages, expanded_pages).

    The buffer covers more tokens than the attention pool holds (see
    `hisparse_indexer_expansion_ratio`). Its base region holds one page per
    attention KV page, so a KV page id *is* its indexer page id -- no mapping
    table. The expanded region, which starts at `base_pages`, is handed out per
    request at admission: a page the tree evicts goes back to the allocator and
    the next request writes over it, base indexer page included, so an evicting
    request's indexer rows must be copied somewhere private first.

    `num_indexer_pages` is the buffer's own first-dimension length and
    `device_pool_size` the attention pool's token capacity. Both consumers -- the
    expanded-page allocator and the hybrid page table -- must derive page ids from
    here, or two page tables disagree on what a page id means and the indexer
    scores another request's keys.

    The private-host backing needs no split: its indexer index is the logical
    token index, nobody else can claim it, so nothing is copied or carved.
    """
    # Paged allocators hand out page ids from 1, so a token loc reaches
    # device_pool_size + page_size - 1 and the base region spans that many pages.
    base_pages = (device_pool_size + page_size) // page_size
    if num_indexer_pages <= base_pages:
        raise ValueError(
            f"indexer buffer has {num_indexer_pages} pages but the attention pool "
            f"alone needs {base_pages}; HiSparse on the HiCache backing needs an "
            "expanded region on top (raise hisparse_config expansion_ratio or "
            "--hicache-ratio)."
        )
    return base_pages, num_indexer_pages - base_pages


def create_hisparse_coordinator(
    *,
    server_args,
    model_config,
    req_to_token_pool,
    token_to_kv_pool_allocator,
    device: str,
    tp_group,
    pp_size: int,
    is_speculative: bool,
) -> Optional["HiSparseCoordinator"]:
    """The HiSparse coordinator for this configuration, or None when it is off.

    The one place that turns configuration into a coordinator, so the choice of
    logical KV pool is made once instead of at every call site. Callers get a
    `managers/hisparse_protocol.py` implementation and never need to know which
    backing they got.
    """
    backing = hisparse_backing(server_args)
    if backing is None:
        return None

    config = _parse_sparse_config(server_args)
    top_k = _indexer_top_k(config=config, model_config=model_config)

    if backing is HiSparseBacking.PRIVATE_HOST:
        # Imported here, not at module scope: the coordinator imports this module
        # back for HiSparseBacking, and it pulls in torch + the swap-in kernels
        # that a config-only caller (startup validation) has no use for.
        from sglang.srt.managers.hisparse_coordinator import (
            PrivateHostHiSparseCoordinator,
            resolve_shared_index_layers,
        )

        return PrivateHostHiSparseCoordinator(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            top_k=top_k,
            device_buffer_size=config.device_buffer_size,
            device=device,
            tp_group=tp_group,
            host_to_device_ratio=config.host_to_device_ratio,
            swap_in_block_size=config.swap_in_block_size,
            shared_index_layers=resolve_shared_index_layers(
                hf_text_config=model_config.hf_text_config,
                pp_size=pp_size,
                is_speculative=is_speculative,
            ),
        )

    # Imported here for the same reason as above: this module is also read by
    # startup validation, which must not pull in torch and the swap-in kernels.
    from sglang.srt.managers.hisparse_hicache_coordinator import (
        HiCacheHiSparseCoordinator,
    )

    return HiCacheHiSparseCoordinator(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        top_k=top_k,
        device_buffer_size=config.device_buffer_size,
        device=device,
        tp_group=tp_group,
        swap_in_block_size=config.swap_in_block_size,
    )
