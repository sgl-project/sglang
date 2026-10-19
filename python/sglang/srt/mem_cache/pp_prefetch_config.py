"""Read the PP cache opt-in at its consumers, without constructor plumbing.

The environment flag is a launch-time option, not a live policy switch. The
shared profile gate keeps unrelated TP/Decode/L2/buffer-only paths unchanged,
even when a deployment injects the same environment into every process.
"""

from sglang.srt.environ import envs

_CURRENT_BACKEND = object()


def pp_cache_admission_requested(
    *, config=None, storage_backend=_CURRENT_BACKEND
) -> bool:
    """One profile/flag predicate for ServerArgs and published runtime config.

    ServerArgs exposes the fields in one flat resolving view; runtime uses
    separate bags. An explicit backend (including None) forecasts a storage
    change without mutating either. Role compatibility is handled separately
    so ServerArgs can reject an explicitly requested incompatible profile.
    """
    if not envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.get():
        return False
    if config is None:
        from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel

        memory, parallel, disagg = get_memory(), get_parallel(), get_disagg()
    else:
        memory = parallel = disagg = config
    backend = (
        memory.hicache_storage_backend
        if storage_backend is _CURRENT_BACKEND
        else storage_backend
    )
    return (
        parallel.pp_size > 1
        and disagg.disaggregation_mode == "prefill"
        and memory.enable_hierarchical_cache
        and backend is not None
        and memory.hicache_host_memory_mode == "cache"
    )


def pp_cache_admission_enabled(*, storage_backend=_CURRENT_BACKEND) -> bool:
    """Select the runtime path, optionally under a proposed L3 attachment."""
    if not pp_cache_admission_requested(storage_backend=storage_backend):
        return False
    from sglang.srt.runtime_context import get_disagg

    # A native role-switching engine must not enable admission on Decode -> Prefill.
    # Static Prefill conflicts are rejected by ServerArgs using requested().
    return not get_disagg().enable_pd_role_switch


def requires_logical_dsa_pp_topk(*, model_config):
    """Allocator-local slots cannot be relayed into an independent PP cache.

    Only DSA backend construction calls this helper. Fresh-indexer boundaries
    transmit no top-k tensor, so keep their native fused path. If any boundary
    reuses an upstream indexer, every target stage uses request-local logical
    positions and performs its normal local page-table transform. No extra
    PP message or collective is required. Non-target profiles remain unchanged.
    """
    if not pp_cache_admission_enabled():
        return False

    from sglang.srt.configs.model_config import dsa_layer_skips_topk
    from sglang.srt.distributed.utils import get_pp_indices
    from sglang.srt.runtime_context import get_parallel

    pp_size = get_parallel().pp_size
    return any(
        dsa_layer_skips_topk(
            model_config,
            get_pp_indices(model_config.num_hidden_layers, rank, pp_size)[0],
        )
        for rank in range(1, pp_size)
    )
