"""Sampled EAGLE chain verify from one target sample per row
(SGLANG_OPT_SPEC_CHAIN_TOPP_SAMPLING).

For a topk-1 chain with both acceptance thresholds at 1, the target-only walk
(accept draft x with probability p'(x), else draw from p' without x) has the same
joint law as drawing y ~ p' on every verify row and accepting while y equals the
next draft token. The rows' samples come from one deterministic cluster kernel
(verify_topp_chain) and the chain acceptance from the fused chain kernel the
greedy path uses, so no rank-0 broadcast is needed.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Arbitrary; keeps these Philox draws apart from torch's streams under the same seed.
_KEY_SALT = 0x5EED_70B0_C4A1_7380
_INT63_MASK = (1 << 63) - 1


class _Stream:
    seed: Optional[int] = None
    logged: set = set()
    # Debug cross-rank check (the _CHECK switch): calls compared, calls differing.
    checked: int = 0
    differing: int = 0


def _seed() -> int:
    # TpModelWorker broadcasts the seed and calls manual_seed_all on every rank.
    if _Stream.seed is None:
        _Stream.seed = (torch.cuda.initial_seed() ^ _KEY_SALT) & _INT63_MASK
    return _Stream.seed


def _log_once(key: str, msg: str, *args) -> None:
    if key not in _Stream.logged:
        _Stream.logged.add(key)
        logger.info(msg, *args)


def chain_topp_applies(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    need_top_k: bool,
    need_min_p: bool,
    has_seed: bool,
    has_grammar: bool,
    chain: bool,
    threshold_single: float,
    threshold_acc: float,
    simulate_token_mode_ok: bool,
) -> bool:
    """Whether a sampled (not greedy) gathered-logits verify takes this path; the
    same on every rank."""
    if not envs.SGLANG_OPT_SPEC_CHAIN_TOPP_SAMPLING.get():
        return False
    from sglang.kernels.ops.speculative.verify_topp_chain import MAX_ROWS, MAX_VOCAB

    ok = (
        chain
        and simulate_token_mode_ok
        and not need_top_k
        and not need_min_p
        and not has_seed
        and not has_grammar
        and threshold_single == 1.0
        and threshold_acc == 1.0
        and logits.is_cuda
        and logits.dim() == 2
        and logits.dtype == torch.float32
        and logits.stride(-1) == 1
        and 0 < logits.shape[-1] <= MAX_VOCAB
        and 0 < logits.shape[0] <= MAX_ROWS
        and temperatures.dim() == 2
        and temperatures.dtype == torch.float32
    )
    if not ok:
        _log_once(
            "skip",
            "Chain top-p verify sampling: a batch took the reference path "
            "(chain=%s top_k=%s min_p=%s seed=%s grammar=%s thresholds=%s/%s "
            "logits=%s %s).",
            chain,
            need_top_k,
            need_min_p,
            has_seed,
            has_grammar,
            threshold_single,
            threshold_acc,
            tuple(logits.shape),
            logits.dtype,
        )
    return ok


def chain_topp_sample(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: Optional[torch.Tensor],
    num_draft: int,
    want_argmax: bool,
) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """(tokens [rows], argmax [rows] or None) int32, or None when the persistent
    state does not exist yet under graph capture (the caller then takes the
    reference path; every rank decides alike)."""
    from sglang.kernels.ops.speculative.verify_topp_chain import (
        state_for,
        verify_topp_chain_sample,
    )

    state = state_for(logits.device)
    if state is None:
        return None
    _log_once(
        "on",
        "Chain top-p verify sampling: on (%d verify rows, top_p %s); samples are "
        "drawn alike on every TP rank, not broadcast.",
        logits.shape[0],
        "per request" if top_ps is not None else "1",
    )
    tokens, argmax = verify_topp_chain_sample(
        logits,
        temperatures,
        top_ps,
        num_draft,
        _seed(),
        want_argmax=want_argmax,
        state=state,
    )
    if envs.SGLANG_OPT_SPEC_CHAIN_TOPP_SAMPLING_CHECK.get():
        _check_across_ranks(tokens)
    return tokens, argmax


def chain_topp_record(device: torch.device) -> Optional[dict]:
    """The device state's counters (host sync): calls, rows with more than one
    round, rows at the round cap; None before the first call."""
    from sglang.kernels.ops.speculative.verify_topp_chain import _STATES

    index = device.index if device.index is not None else torch.cuda.current_device()
    state = _STATES.get(index)
    if state is None:
        return None
    step, _, multi, capped = (int(x) for x in state.tolist())
    return {
        "kernel_calls": step,
        "multi_round_rows": multi,
        "capped_rows": capped,
        "checked_calls": _Stream.checked,
        "differing_calls": _Stream.differing,
    }


def _check_across_ranks(tokens: torch.Tensor) -> None:
    """Debug: log when the TP ranks drew different samples (blocks on the host)."""
    if torch.cuda.is_current_stream_capturing():
        return
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    if group.world_size == 1:
        return
    low, high = tokens.to(torch.int64), tokens.to(torch.int64).clone()
    torch.distributed.all_reduce(
        low, op=torch.distributed.ReduceOp.MIN, group=group.device_group
    )
    torch.distributed.all_reduce(
        high, op=torch.distributed.ReduceOp.MAX, group=group.device_group
    )
    diff = int((low != high).sum().item())
    _Stream.checked += 1
    if diff:
        _Stream.differing += 1
        logger.warning(
            "Chain top-p verify sampling: %d of %d samples differ across TP ranks.",
            diff,
            tokens.numel(),
        )
    # Arbitrary; one summary line per this many checked calls.
    if _Stream.checked % 100 == 0 and group.rank_in_group == 0:
        logger.info(
            "Chain top-p verify sampling check: %d calls compared across %d TP ranks, "
            "%d with differences.",
            _Stream.checked,
            group.world_size,
            _Stream.differing,
        )
