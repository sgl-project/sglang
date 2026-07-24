"""FlexKV-backed RadixCache integration for sglang.

Two ways to select this backend at server launch:

1. ``--enable-flexkv`` (default chain in ``default_radix_cache_factory``)
2. ``--radix-cache-backend=flexkv`` (explicit registry path)

Importing this package registers the explicit name with the registry,
so the second form is available without further wiring.
"""

from __future__ import annotations

import logging

from sglang.srt.mem_cache.registry import register_radix_cache_backend

logger = logging.getLogger(__name__)


def _flexkv_factory(ctx):
    """Build a :class:`FlexKVRadixCache` from a ``TreeCacheBuildContext``.

    ``TreeCacheBuildContext`` carries TP rank/size and the TP group
    coordinator, but not PP/CP. We pick those up from the global
    accessors in :mod:`sglang.srt.distributed.parallel_state`; FlexKV
    needs them to fan out lookup/store decisions across the full TP × CP
    × PP topology.
    """
    from sglang.srt.distributed.parallel_state import (
        get_attn_cp_group,
        get_attn_tp_group,
        get_pp_group,
    )
    from sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache import (
        FlexKVRadixCache,
    )

    server_args = ctx.server_args

    # PP group is always available; attn TP / attn CP groups may share
    # the regular TP group when attn DP is off — that's fine, the
    # connector treats size-1 groups as no-ops.
    try:
        pp_group = get_pp_group()
    except (RuntimeError, AssertionError):
        pp_group = None
    try:
        attn_tp_group = get_attn_tp_group()
    except (RuntimeError, AssertionError):
        attn_tp_group = ctx.tp_group
    try:
        attn_cp_group = get_attn_cp_group()
    except (RuntimeError, AssertionError):
        attn_cp_group = None

    # PP / CP ranks: use the group's own rank_in_group view if available;
    # fall back to 0 for single-rank dims.
    pp_rank = pp_group.rank_in_group if pp_group is not None else 0
    attn_cp_rank = attn_cp_group.rank_in_group if attn_cp_group is not None else 0
    parallel_state = getattr(ctx.tp_worker, "ps", None)
    if server_args.enable_dp_attention:
        dp_rank = getattr(parallel_state, "attn_dp_rank", 0)
    else:
        dp_rank = getattr(parallel_state, "dp_rank", 0) or 0

    common_kwargs = dict(
        params=ctx.params,
        model_config=ctx.model_config,
        server_args=server_args,
        tp_rank=ctx.tp_rank,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
        attn_cp_rank=attn_cp_rank,
        tp_group=ctx.tp_group,
        pp_group=pp_group,
        attn_tp_group=attn_tp_group,
        attn_cp_group=attn_cp_group,
    )

    if ctx.is_hybrid_ssm:
        raise NotImplementedError("FlexKV does not support Mamba/SSM pools yet")

    if ctx.is_hybrid_swa:
        from sglang.srt.mem_cache.storage.flexkv.flexkv_hybrid_radix_cache import (
            FlexKVHybridRadixCache,
        )
        from sglang.srt.mem_cache.unified_cache_components import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        ctx.params.tree_components = (ComponentType.FULL, ComponentType.SWA)
        inner_cache = UnifiedRadixCache(ctx.params)
        return FlexKVHybridRadixCache(
            inner_cache=inner_cache,
            **common_kwargs,
        )

    return FlexKVRadixCache(
        tp_size=ctx.tp_size,
        **common_kwargs,
    )


try:
    register_radix_cache_backend("flexkv", _flexkv_factory)
except ValueError as exc:
    # The registry refuses duplicates. Importing this package twice
    # (e.g. via both --enable-flexkv and --radix-cache-backend=flexkv)
    # is fine — log and move on.
    logger.debug("flexkv backend already registered: %s", exc)
