from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.kv_cache_canary.runner import CanaryRunner
from sglang.srt.kv_cache_canary.sweep import compute_alive_owned_slots
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_PERTURB_RNG_CACHE: Dict[int, random.Random] = {}
_REAL_PERTURB_RNG_CACHE: Dict[int, random.Random] = {}


def maybe_perturb_hook(
    *,
    runner: Optional[CanaryRunner],
    model_runner: ModelRunner,
    forward_batch: ForwardBatch,
) -> None:
    """Shared eager + replay self-test perturb hook."""
    active_indices, active_seq_lens = _extract_active_rows(forward_batch)
    maybe_perturb_req_to_token(
        runner=runner,
        req_to_token_pool=model_runner.req_to_token_pool,
        rank=model_runner.tp_rank,
        active_req_pool_indices=active_indices,
        active_seq_lens=active_seq_lens,
    )


def maybe_perturb_req_to_token(
    *,
    runner: Optional[CanaryRunner],
    req_to_token_pool: ReqToTokenPool,
    rank: int,
    active_req_pool_indices: Optional[List[int]] = None,
    active_seq_lens: Optional[List[int]] = None,
) -> None:
    """Self-test helper: probabilistically swap slot pointers in ``req_to_token``.

    Per-rank stateful RNG (seeded deterministically at first use) so the
    perturbation sequence is reproducible AND advances every call instead of
    sampling the same first draw repeatedly.

    Active-row-aware: when ``active_req_pool_indices`` and ``active_seq_lens``
    are provided, the perturbation only swaps within the in-use ``[0,
    seq_len)`` range of a randomly chosen active request — that's where the
    canary verify actually reads, so a swap there is observable.
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


def _extract_active_rows(
    forward_batch: Optional[ForwardBatch],
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Pull (req_pool_indices, seq_lens) lists for active-row-aware perturb."""
    if forward_batch is None:
        return None, None
    if forward_batch.req_pool_indices is None or forward_batch.seq_lens is None:
        return None, None
    indices = forward_batch.req_pool_indices.detach().cpu().tolist()
    seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
    return indices, seq_lens


def _rng_seed_for_rank(base_seed: int, rank: int) -> int:
    return (
        (base_seed & 0xFFFFFFFF) * 0x9E3779B1 + rank * 0xBF58476D1CE4E5B9
    ) & 0xFFFFFFFFFFFFFFFF


def maybe_perturb_real_kv_bytes(
    *,
    runner: Optional[CanaryRunner],
    req_to_token_pool: Optional[ReqToTokenPool],
    forward_batch: Optional[ForwardBatch],
) -> None:
    """Self-test helper: probabilistically flip one byte of real KV at a slot
    owned by an alive req in the current batch but NOT in this step's
    per-step verify list.

    Targeting alive-but-not-verified-this-step slots is what proves the
    periodic sweep's independent detection value: the per-step path can't
    observe the perturbation (those slots are outside its verify set) but
    the next sweep should pick it up and emit a ``sweep_*`` violation.

    Must be called AFTER the per-step head/tail freeze AND BEFORE
    :meth:`CanaryRunner._run_sweep`, so the freeze captures clean state and
    the sweep's next read sees the mutated bytes.
    """
    if runner is None or forward_batch is None or req_to_token_pool is None:
        return
    prob = runner.config.real_perturb_bytes_prob
    if prob <= 0.0:
        return

    rank = runner.tp_rank
    rng = _REAL_PERTURB_RNG_CACHE.get(rank)
    if rng is None:
        rng = random.Random(
            _rng_seed_for_rank(runner.config.real_perturb_bytes_seed, rank)
        )
        _REAL_PERTURB_RNG_CACHE[rank] = rng
    if rng.random() >= prob:
        return

    group = runner.buffer_group
    if group.real_kv_source is None or group.real_kv_slot_stride_bytes <= 0:
        return

    alive = compute_alive_owned_slots(
        req_to_token_pool=req_to_token_pool, forward_batch=forward_batch
    )
    if alive.numel() == 0:
        return

    last_plan = runner.last_plan
    verify_this_step = (
        set(last_plan.verify_slot_indices) if last_plan is not None else set()
    )
    alive_list = alive.detach().cpu().tolist()
    idle_alive: List[int] = [
        int(s) for s in alive_list if int(s) not in verify_this_step
    ]
    if not idle_alive:
        return

    slot_idx = idle_alive[rng.randrange(len(idle_alive))]
    stride = int(group.real_kv_slot_stride_bytes)
    byte_offset = rng.randrange(stride)
    flat = group.real_kv_source.view(torch.uint8).flatten()
    base = slot_idx * stride
    flat[base + byte_offset] ^= 0xFF
    logger.warning(
        "kv-canary self-test: perturbed real KV bytes at slot=%d offset=%d (rank=%d)",
        slot_idx,
        byte_offset,
        rank,
    )
