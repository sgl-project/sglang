"""Self-test perturb helpers used by the canary install hooks.

These probabilistically corrupt either:

- the ``req_to_token`` index table (drives canary slot-pointer mismatch detection), or
- the real-KV bytes at an idle alive slot (drives the periodic real-data sweep).

Both are guarded by config knobs (``perturb_req_to_token_prob`` / ``real_perturb_bytes_prob``) and a
deterministic per-rank RNG so the perturbation sequence is reproducible across runs.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.kv_cache_canary.runner import CanaryRunner

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_PERTURB_RNG_CACHE: Dict[int, random.Random] = {}
_REAL_PERTURB_RNG_CACHE: Dict[int, random.Random] = {}


def maybe_perturb_hook(
    *,
    runner: Optional[CanaryRunner],
    model_runner: "ModelRunner",
    forward_batch: "ForwardBatch",
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
    req_to_token_pool: "ReqToTokenPool",
    rank: int,
    active_req_pool_indices: Optional[List[int]] = None,
    active_seq_lens: Optional[List[int]] = None,
) -> None:
    """Probabilistically swap slot pointers in ``req_to_token``.

    Active-row-aware: when ``active_req_pool_indices`` and ``active_seq_lens`` are provided, the
    perturbation only swaps within the in-use ``[0, seq_len)`` range of a randomly chosen active request —
    that's where the canary verify actually reads, so a swap there is observable.
    """
    if runner is None:
        return
    prob = runner.config.perturb_req_to_token_prob
    if prob <= 0.0:
        return

    rng = _get_or_init_rng(
        _PERTURB_RNG_CACHE, runner.config.perturb_req_to_token_seed, rank
    )
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


def maybe_perturb_real_kv_bytes(
    *,
    runner: Optional[CanaryRunner],
    req_to_token_pool: Optional["ReqToTokenPool"],
    forward_batch: Optional["ForwardBatch"],
) -> None:
    """Flip one byte of real KV at an alive slot, so the periodic sweep can catch it.

    Must run AFTER the per-step head/tail freeze and BEFORE the sweep launch.
    """
    if runner is None or forward_batch is None or req_to_token_pool is None:
        return
    prob = runner.config.real_perturb_bytes_prob
    if prob <= 0.0:
        return

    rank = runner.tp_rank
    rng = _get_or_init_rng(
        _REAL_PERTURB_RNG_CACHE, runner.config.real_perturb_bytes_seed, rank
    )
    if rng.random() >= prob:
        return

    group = runner.buffer_group
    sources = group.real_kv_sources_k
    if not sources:
        return
    source = sources[0]
    if source.read_bytes <= 0:
        return

    alive = _compute_alive_owned_slots(
        req_to_token_pool=req_to_token_pool, forward_batch=forward_batch
    )
    if alive.numel() == 0:
        return

    alive_list = alive.detach().cpu().tolist()
    if not alive_list:
        return
    slot_idx = int(alive_list[rng.randrange(len(alive_list))])
    byte_offset = rng.randrange(source.read_bytes)
    flat = source.tensor.view(torch.uint8)
    row = slot_idx // source.page_size
    col_base = (slot_idx % source.page_size) * source.num_bytes_per_token
    flat[row, col_base + byte_offset] ^= 0xFF
    logger.warning(
        "kv-canary self-test: perturbed real KV bytes at slot=%d offset=%d (rank=%d)",
        slot_idx,
        byte_offset,
        rank,
    )


def _extract_active_rows(
    forward_batch: Optional["ForwardBatch"],
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Pull ``(req_pool_indices, seq_lens)`` host lists from a forward batch."""
    if forward_batch is None:
        return None, None
    if forward_batch.req_pool_indices is None or forward_batch.seq_lens is None:
        return None, None
    indices = forward_batch.req_pool_indices.detach().cpu().tolist()
    seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
    return indices, seq_lens


def _compute_alive_owned_slots(
    *,
    req_to_token_pool: "ReqToTokenPool",
    forward_batch: "ForwardBatch",
) -> torch.Tensor:
    """Slot indices for every token owned by an alive req in ``forward_batch``."""
    device = req_to_token_pool.req_to_token.device
    indices_cpu, seq_lens_cpu = _extract_active_rows(forward_batch)
    if not indices_cpu or not seq_lens_cpu:
        return torch.empty(0, dtype=torch.int64, device=device)

    chunks: List[torch.Tensor] = []
    for r, n in zip(indices_cpu, seq_lens_cpu):
        r_int = int(r)
        n_int = int(n)
        if r_int <= 0 or n_int <= 0:
            continue
        chunks.append(req_to_token_pool.req_to_token[r_int, :n_int].to(torch.int64))

    if not chunks:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.cat(chunks, dim=0)


def _get_or_init_rng(
    cache: Dict[int, random.Random], base_seed: int, rank: int
) -> random.Random:
    rng = cache.get(rank)
    if rng is None:
        rng = random.Random(_rng_seed_for_rank(base_seed, rank))
        cache[rank] = rng
    return rng


def _rng_seed_for_rank(base_seed: int, rank: int) -> int:
    return (
        (base_seed & 0xFFFFFFFF) * 0x9E3779B1 + rank * 0xBF58476D1CE4E5B9
    ) & 0xFFFFFFFFFFFFFFFF
