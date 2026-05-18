from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.kv_cache_canary.host_state import BatchPlan
from sglang.srt.kv_cache_canary.pool_patch import PoolKind, install_swa_free_hook
from sglang.srt.kv_cache_canary.runner import CanaryRunner

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.memory_pool import KVCache
    from sglang.srt.mem_cache.req_to_token_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_GLOBAL_RUNNER_KEY = "_kv_cache_canary_runner"


def attach(
    *,
    pool: "KVCache",
    config: CanaryConfig,
    req_to_token_pool: "ReqToTokenPool",
    device: torch.device,
    pool_kind: PoolKind = PoolKind.FULL,
) -> Optional[CanaryRunner]:
    """Attach canary to ``pool`` and create a runner.

    Must be called AFTER ``init_memory_pool`` and BEFORE ``init_device_graphs``
    so that the canary kernel is captured into the CUDA graph and the shadow
    tensors are baked into the graph's pointer table.
    """
    if not config.enabled:
        return None
    if hasattr(pool, _GLOBAL_RUNNER_KEY):
        logger.warning("kv-canary: pool already has a runner attached; reusing it")
        return get_runner(pool)

    runner = CanaryRunner(
        config=config,
        pool=pool,
        num_req_slots=int(req_to_token_pool.size),
        device=device,
        pool_kind=pool_kind,
    )
    setattr(pool, _GLOBAL_RUNNER_KEY, runner)
    logger.info(
        "kv-canary: attached runner in mode=%s pool_kind=%s",
        config.mode.value,
        pool_kind.value,
    )
    return runner


def get_runner(pool: "KVCache") -> Optional[CanaryRunner]:
    return getattr(pool, _GLOBAL_RUNNER_KEY, None)


def install_req_to_token_pool_free_hook(
    *,
    runner: CanaryRunner,
    req_to_token_pool: "ReqToTokenPool",
) -> None:
    """Wrap ``ReqToTokenPool.free`` so canary state is dropped on req release.

    Without this, a reused ``req_pool_idx`` inherits the previous request's
    ``K_req`` / ``prev_hash_tail``, which would make the new req's first token
    look like a verify-then-mismatch.

    NOTE: there are bypass paths that ``append`` to ``free_slots`` directly
    without going through ``.free()`` (streaming session,
    disaggregation/decode ``ReqToTokenPool.free``). Those call sites must
    invoke :func:`release_req_pool_idx` to keep the canary state in sync;
    relying on hooking ``.free()`` alone is insufficient.
    """
    if getattr(req_to_token_pool, "_kv_canary_free_patched", False):
        return
    setattr(req_to_token_pool, "_kv_canary_runner_ref", runner)
    original_free = req_to_token_pool.free

    def patched_free(req) -> None:
        idx = req.req_pool_idx
        original_free(req)
        if idx is not None:
            runner.host_state.reset_request(int(idx))

    req_to_token_pool.free = patched_free
    setattr(req_to_token_pool, "_kv_canary_free_patched", True)


def release_req_pool_idx(
    req_to_token_pool: "ReqToTokenPool",
    req_pool_idx: Optional[int],
) -> None:
    """Notify canary that a ``req_pool_idx`` was released via a bypass path.

    Call sites that ``free_slots.append(idx)`` directly (streaming session,
    disaggregation decode pool) MUST invoke this so the canary host state for
    that ``req_pool_idx`` is dropped — otherwise the next request reusing the
    same index will see a stale ``K_req`` / ``prev_hash_tail`` and the very
    first verify entry will mismatch.

    Safe no-op when the canary is disabled (no runner attached) or
    ``req_pool_idx`` is ``None``.
    """
    if req_pool_idx is None:
        return
    runner = getattr(req_to_token_pool, "_kv_canary_runner_ref", None)
    if runner is None:
        return
    runner.host_state.reset_request(int(req_pool_idx))


def install_swa_eviction_hook(
    *,
    runner: CanaryRunner,
    pool: "BaseSWAKVPool",
) -> None:
    """Hook SWA window-slide eviction so host state forgets evicted slots.

    When SWA slides past a token the slot it occupied is freed back to the
    SWA sub-pool. The host's ``last_committed.slot_idx`` may still point at
    that now-free (or soon-to-be-reused) slot; if we tried to verify the
    next batch against it we'd be reading a stranger's data. The conservative
    fix is to drop ``last_committed`` for every tracked request on every
    eviction batch. The next forward writes fresh entries and re-anchors.
    """
    install_swa_free_hook(
        pool=pool,
        on_free=runner.host_state.reset_all_last_committed,
    )


def install_spec_allocator_free_hook(
    *,
    runner: CanaryRunner,
    model_runner: "ModelRunner",
) -> None:
    """Hook spec decoding's allocator.free so rejected slots reset chain state.

    Eagle/MTP verify accepts a prefix of drafts; the rest are 'rejected' and
    their slots are freed via ``token_to_kv_pool_allocator.free(...)``. The
    canary's host state (``last_committed.slot_idx`` and the prev-hash chain)
    must rewind so the next batch doesn't try to verify against a slot we
    just gave back. We can't enumerate which req_pool_idx owns each slot
    here, so we use the same conservative fallback as SWA: clear all
    last_committed on every free batch. ``K_req`` is preserved (correct
    by-position writes are still safe).

    Idempotent. Only patches when an allocator is present.
    """
    allocator = model_runner.token_to_kv_pool_allocator
    if allocator is None:
        return
    if getattr(allocator, "_kv_canary_free_patched", False):
        return
    if not hasattr(allocator, "free"):
        return
    original_free = allocator.free

    def patched_free(free_index: torch.Tensor) -> None:
        original_free(free_index)
        try:
            runner.host_state.reset_all_last_committed()
        except Exception:
            logger.exception("kv-canary: spec allocator.free hook failed")

    allocator.free = patched_free
    setattr(allocator, "_kv_canary_free_patched", True)


def export_pd_canary_snapshot(
    *,
    pool: "KVCache",
    req_pool_idx: int,
) -> tuple[int, int]:
    """Prefill side: return ``(k_req, prev_hash_tail)`` for MetadataBuffers.

    Returns ``(0, 0)`` if the pool has no canary attached or the request was
    never seen. The decode side treats ``(0, 0)`` as "fresh chain" — pure
    writes from the start, no verify entries. This keeps PD transport
    behaviour identical when the canary is disabled on one or both sides.
    """
    runner = get_runner(pool)
    if runner is None:
        return 0, 0
    snapshot = runner.host_state.export_pd_snapshot(req_pool_idx)
    if snapshot is None:
        return 0, 0
    return snapshot.k_req, snapshot.prev_hash_tail


def apply_pd_canary_snapshot(
    *,
    pool: "KVCache",
    req_pool_idx: int,
    k_req: int,
    prev_hash_tail: int,
) -> None:
    """Decode side: rebuild canary host state from PD-transported metadata.

    No-op if the pool has no canary attached. ``k_req == 0`` means the
    prefill side had no canary state (fresh chain); we leave host state
    untouched so the next plan_batch initializes from seed.
    """
    runner = get_runner(pool)
    if runner is None:
        return
    if k_req <= 0:
        return
    runner.host_state.import_pd_snapshot(
        req_pool_idx=req_pool_idx,
        k_req=k_req,
        prev_hash_tail=prev_hash_tail,
    )


_PERTURB_RNG_CACHE: dict = {}


def maybe_perturb_req_to_token(
    *,
    runner: Optional[CanaryRunner],
    req_to_token_pool: "ReqToTokenPool",
    rank: int,
    active_req_pool_indices: Optional[List[int]] = None,
    active_seq_lens: Optional[List[int]] = None,
) -> None:
    """Self-test helper: probabilistically swap slot pointers in ``req_to_token``.

    Per-rank stateful RNG (seeded deterministically at first use) so the
    perturbation sequence is reproducible AND advances every call instead of
    sampling the same first draw repeatedly.

    Active-row-aware: when ``active_req_pool_indices`` and ``active_seq_lens``
    are provided (the typical call path from ``_plan_from_forward_batch``),
    the perturbation only swaps within the in-use ``[0, seq_len)`` range of
    a randomly chosen active request — that's where the canary verify
    actually reads, so a swap there is observable. Without this targeting,
    swaps land in zero-padded / inactive columns and the perturb hit rate is
    effectively zero for any realistic table size.
    """
    if runner is None:
        return
    prob = runner.config.perturb_req_to_token_prob
    if prob <= 0.0:
        return

    rng = _PERTURB_RNG_CACHE.get(rank)
    if rng is None:
        rng = random.Random(
            _rng_seed_for_rank(runner.config.perturb_req_to_token_seed, rank)
        )
        _PERTURB_RNG_CACHE[rank] = rng
    if rng.random() >= prob:
        return
    table = req_to_token_pool.req_to_token
    if not isinstance(table, torch.Tensor) or table.numel() == 0:
        return
    rows, cols = int(table.shape[0]), int(table.shape[1])
    if rows <= 1 or cols <= 1:
        return

    if active_req_pool_indices and active_seq_lens:
        candidates: List[Tuple[int, int]] = [
            (int(idx), int(n))
            for idx, n in zip(active_req_pool_indices, active_seq_lens)
            if int(idx) > 0 and int(n) >= 2
        ]
        if not candidates:
            return
        r, active_cols = candidates[rng.randrange(len(candidates))]
        a = rng.randrange(active_cols)
        b = rng.randrange(active_cols)
    else:
        r = rng.randrange(1, rows)
        a = rng.randrange(cols)
        b = rng.randrange(cols)

    if a == b:
        return
    tmp = table[r, a].clone()
    table[r, a] = table[r, b]
    table[r, b] = tmp
    logger.warning(
        "kv-canary self-test: perturbed req_to_token_pool[%d] columns %d <-> %d (rank=%d)",
        r,
        a,
        b,
        rank,
    )


def _rng_seed_for_rank(base_seed: int, rank: int) -> int:
    return (
        (base_seed & 0xFFFFFFFF) * 0x9E3779B1 + rank * 0xBF58476D1CE4E5B9
    ) & 0xFFFFFFFFFFFFFFFF


def run_head(
    *,
    runner: Optional[CanaryRunner],
    forward_batch: "ForwardBatch",
) -> Optional[BatchPlan]:
    if runner is None:
        return None
    plan = _plan_from_forward_batch(runner=runner, forward_batch=forward_batch)
    if plan is None:
        return None
    runner.run_head(plan=plan)
    return plan


def run_tail(
    *,
    runner: Optional[CanaryRunner],
    forward_batch: "ForwardBatch",
    plan: Optional[BatchPlan],
) -> None:
    if runner is None or plan is None:
        return
    runner.run_tail(plan=plan)
    runner.host_state.commit_plan(plan)
    runner.end_of_forward()


def _plan_from_forward_batch(
    *,
    runner: CanaryRunner,
    forward_batch: "ForwardBatch",
) -> Optional[BatchPlan]:
    """Translate a ``ForwardBatch`` into the per-slot expectations the kernel needs."""

    if forward_batch.out_cache_loc is None or forward_batch.out_cache_loc.numel() == 0:
        return None

    req_pool_indices = forward_batch.req_pool_indices.detach().cpu().tolist()
    input_ids_list: List[int] = forward_batch.input_ids.detach().cpu().tolist()
    out_cache_loc_list: List[int] = forward_batch.out_cache_loc.detach().cpu().tolist()

    forward_mode = forward_batch.forward_mode
    if forward_mode is None:
        return None
    is_extend = forward_mode.is_extend() or forward_mode.is_mixed()
    if is_extend:
        if (
            forward_batch.extend_seq_lens is None
            or forward_batch.extend_prefix_lens is None
        ):
            return None
        seq_lens = forward_batch.extend_seq_lens.detach().cpu().tolist()
        prefix_lens = forward_batch.extend_prefix_lens.detach().cpu().tolist()
    elif forward_mode.is_decode() or forward_mode.is_target_verify():
        seq_lens = [1] * len(req_pool_indices)
        full_seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
        prefix_lens = [int(s) - 1 for s in full_seq_lens]
    else:
        return None

    # Padding-aware truncation: cuda-graph padded batches set padding rows'
    # req_pool_indices to 0 (the dedicated padding row) and pad input_ids on
    # the tail. ``num_token_non_padded`` (when present) tells us how many
    # tokens are real.
    num_real_tokens = _num_real_tokens(forward_batch, len(input_ids_list))
    if sum(seq_lens) != num_real_tokens:
        return None
    if len(out_cache_loc_list) != num_real_tokens:
        return None

    cursor = 0
    filtered_req_pool_indices: List[int] = []
    filtered_seq_lens: List[int] = []
    filtered_prefix_lens: List[int] = []
    tokens_per_req: List[List[int]] = []
    write_slots_per_req: List[List[int]] = []
    for req_pool_idx, n, prefix_len in zip(req_pool_indices, seq_lens, prefix_lens):
        next_cursor = cursor + n
        # Padding row in ReqToTokenPool is index 0 (cuda-graph padded batches
        # set padding rows' req_pool_indices to 0). Skipping it avoids
        # corrupting the host state with synthetic warmup writes.
        if int(req_pool_idx) == 0:
            cursor = next_cursor
            continue
        tokens_per_req.append(input_ids_list[cursor:next_cursor])
        write_slots_per_req.append(out_cache_loc_list[cursor:next_cursor])
        filtered_req_pool_indices.append(int(req_pool_idx))
        filtered_seq_lens.append(int(n))
        filtered_prefix_lens.append(int(prefix_len))
        cursor = next_cursor

    if not filtered_req_pool_indices:
        return None

    return runner.host_state.plan_batch(
        req_pool_indices=filtered_req_pool_indices,
        req_token_counts=filtered_seq_lens,
        req_start_positions=filtered_prefix_lens,
        input_tokens_per_req=tokens_per_req,
        write_slot_indices_per_req=write_slots_per_req,
    )


def _num_real_tokens(forward_batch: "ForwardBatch", total_input_len: int) -> int:
    if hasattr(forward_batch, "num_token_non_padded_cpu"):
        value = forward_batch.num_token_non_padded_cpu
        if value is not None:
            try:
                return int(value)
            except TypeError:
                pass
    return total_input_len
