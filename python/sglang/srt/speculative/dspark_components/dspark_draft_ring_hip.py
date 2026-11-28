"""ROCm per-request sliding-window KV ring wiring for GQA DSpark drafts."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.base_spec_worker import HiCacheDraftPlan
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
    draft_kv_ring_rows_per_req_from_config,
)
from sglang.srt.speculative.dflash_utils import (
    get_dflash_attention_sliding_window_size,
    get_dflash_layer_types,
    parse_dflash_draft_config,
)

logger = logging.getLogger(__name__)


def resolve_rocm_dspark_draft_kv_ring(
    server_args: Any, load_draft_hf_config: Callable[[Any], Any]
) -> None:
    """SGLANG_ENABLE_DSPARK_DRAFT_KV_RING=1 turns the ring on for this draft, or fails
    at startup naming why it cannot. Runs after the draft backend is final, which it
    reads; ``load_draft_hf_config(cfg)`` is called only once the flags allow a ring."""
    if (
        not current_platform.is_rocm()
        or not envs.SGLANG_ENABLE_DSPARK_DRAFT_KV_RING.get()
    ):
        return
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.arg_groups.overrides import declare_resolution

    cfg = resolving_view(server_args)
    reason = rocm_dspark_gqa_ring_unsupported_reason(
        lambda: load_draft_hf_config(cfg),
        draft_backend=cfg.speculative_draft_attention_backend,
        disaggregation_mode=cfg.disaggregation_mode,
        draft_window_size=cfg.speculative_draft_window_size,
        enable_unified_memory=cfg.enable_unified_memory,
    )
    if reason is not None:
        raise ValueError(f"SGLANG_ENABLE_DSPARK_DRAFT_KV_RING=1, but {reason}.")
    logger.info("DSpark draft KV ring: on.")
    declare_resolution(
        server_args,
        "resolve_rocm_dspark_draft_kv_ring",
        speculative_dflash_draft_kv_ring=True,
    )


def rocm_dspark_gqa_ring_unsupported_reason(
    load_draft_hf_config: Callable[[], Any],
    *,
    draft_backend: Optional[str],
    disaggregation_mode: str,
    draft_window_size: Optional[int],
    enable_unified_memory: bool = False,
) -> Optional[str]:
    """Return why a ROCm Triton GQA DSpark ring cannot be enabled, or None."""
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    if draft_backend != "triton":
        return "the DSpark draft KV ring runs only on ROCm's Triton draft backend"
    if enable_unified_memory:
        return "--enable-unified-memory places the draft KV in the target's pool"

    # Only the static verify path writes draft KV through the ring; the ragged
    # (compact, cap-accept) injector writes by pool ids.
    if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
        return "the DSpark draft KV ring needs SGLANG_RAGGED_VERIFY_MODE=static"
    if disaggregation_mode != "null":
        return "ROCm GQA DSpark draft KV ring does not support PD disaggregation"
    if draft_window_size is not None:
        return "--speculative-draft-window-size keeps the compact draft table"
    draft_hf_config = load_draft_hf_config()
    if parse_dflash_draft_config(draft_hf_config=draft_hf_config).is_mla:
        return "ROCm GQA DSpark ring does not serve MLA drafts"
    layer_types = get_dflash_layer_types(draft_hf_config)
    if not layer_types or any(t != "sliding_attention" for t in layer_types):
        return f"not every draft layer is sliding_attention ({layer_types})"
    window_size = get_dflash_attention_sliding_window_size(draft_hf_config)
    if window_size is None or int(window_size) <= 0:
        return "the draft has no positive sliding_window"
    return None


def build_dspark_draft_kv_ring(
    *,
    draft_hf_config,
    draft_pool,
    req_to_token_rows: int,
    device,
    log_rank_zero: bool,
) -> DFlashDraftKVRing:
    """Create the per-request KV ring sized for a GQA or MLA DSpark draft."""
    is_mla = parse_dflash_draft_config(draft_hf_config=draft_hf_config).is_mla
    parallel = get_parallel()
    ring = DFlashDraftKVRing(
        rows_per_req=draft_kv_ring_rows_per_req_from_config(draft_hf_config),
        dcp_size=parallel.attn_dcp_size if is_mla else 1,
        req_to_token_rows=req_to_token_rows,
        device=device,
        window_left=get_dflash_attention_sliding_window_size(draft_hf_config),
        dcp_rank=parallel.attn_dcp_rank if is_mla else 0,
        pool=draft_pool,
    )
    if log_rank_zero:
        logger.info(
            "DSPARK draft KV ring: %d rows per request (%d positions), "
            "%d req_to_token rows, draft pool %d rows.",
            ring.rows_per_req,
            ring.span,
            req_to_token_rows,
            draft_pool.size,
        )
    return ring


def bind_dspark_draft_kv_ring(
    *,
    ring: Optional[DFlashDraftKVRing],
    kv_injector,
    proposer,
) -> None:
    kv_injector.draft_kv_ring = ring
    proposer.draft_kv_ring = ring


def build_worker_draft_kv_ring(worker) -> None:
    """Build and bind the worker's draft ring once its draft pool exists."""
    if not get_spec().speculative_dflash_draft_kv_ring:
        return
    worker.draft_kv_ring = build_dspark_draft_kv_ring(
        draft_hf_config=worker.draft_model_runner.model_config.hf_config,
        draft_pool=worker.draft_model_runner.token_to_kv_pool,
        req_to_token_rows=worker.model_runner.req_to_token_pool.req_to_token.shape[0],
        device=worker.device,
        log_rank_zero=worker._target_tp_rank == 0,
    )
    bind_dspark_draft_kv_ring(
        ring=worker.draft_kv_ring,
        kv_injector=worker._kv_injector,
        proposer=worker._proposer,
    )


def bind_worker_draft_kv_ring_to_backend(worker) -> None:
    if worker.draft_kv_ring is not None:
        worker.draft_model_runner.attn_backend.set_draft_kv_ring(worker.draft_kv_ring)


def record_worker_draft_kv_ring_floors(worker, batch) -> None:
    if worker.draft_kv_ring is not None:
        worker.draft_kv_ring.record_floors(
            batch.reqs,
            batch.prefix_lens,
            worker.model_runner.req_to_token_pool.req_generation,
        )


def worker_hicache_draft_plan(worker) -> Optional[HiCacheDraftPlan]:
    """The ring's draft KV lives outside the HiCache pools, so the plan is empty; None
    keeps the base plan."""
    if worker.draft_kv_ring is not None:
        return HiCacheDraftPlan()
    return None


def bind_proposer_out_cache_loc(
    proposer,
    batch,
    draft_forward_batch,
    *,
    bs: int,
    query_token_num: int,
    draft_positions,
) -> None:
    if proposer.draft_kv_ring is not None:
        draft_forward_batch.out_cache_loc = proposer.draft_kv_ring.block_locs(
            batch.req_pool_indices[:bs],
            draft_positions,
            block_size=query_token_num,
        )


def prefill_inject_cache_loc(
    kv_injector,
    *,
    batch,
    positions,
    ctx_lens,
    draft_seq_lens,
    iteration_ids,
):
    if kv_injector.draft_kv_ring is None:
        return kv_injector.ids_for(iteration_ids)
    return kv_injector.draft_kv_ring.extend_locs(
        req_pool_indices=batch.req_pool_indices,
        positions=positions,
        extend_lens=ctx_lens.to(torch.int64),
        seq_ends=(draft_seq_lens + ctx_lens).to(torch.int64),
    )


def verify_inject_cache_loc(
    kv_injector, *, batch, verify_window, bs: int, verify_num_draft_tokens: int
):
    """Returns (cache_loc, iteration_ids to pass on): the ring takes no iteration ids."""
    if kv_injector.draft_kv_ring is None:
        return kv_injector.ids_for(
            verify_window.iteration_ids
        ), verify_window.iteration_ids
    cache_loc = kv_injector.draft_kv_ring.block_locs(
        batch.req_pool_indices[:bs],
        verify_window.positions_2d.reshape(-1),
        block_size=verify_num_draft_tokens,
    )
    return cache_loc, None
