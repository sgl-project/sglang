from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.kv_cache_canary.runner import CanaryRunner
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_PERTURB_RNG_CACHE: Dict[int, random.Random] = {}


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
