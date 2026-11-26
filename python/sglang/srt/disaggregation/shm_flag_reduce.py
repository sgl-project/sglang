"""PD decode queue flag reductions through the scheduler's shared-memory exchange.

On every pass with requests in its queues, a disaggregated decode scheduler reduces
one small flag per request across the attention-TP ranks: the MIN of the KV poll
states and of the publish and release flags, the MAX of the hold flags
(`utils.all_reduce_flags`). Over gloo each reduction is a CPU all_reduce. With
SGLANG_OPT_SCHEDULER_SHM_SYNC on, a decode instance on one or two hosts reduces
them through `shm_tag_exchange.get_shm_exchange` instead: every rank publishes its
flags, one byte each and five per 40-bit word behind a header word, and takes the
per-entry MIN or MAX over the gathered rows. The result equals the gloo all_reduce.

Every rank checks the gathered headers, so every rank takes the same path:
- a flag outside 0..255 on any rank sends the whole reduction through gloo;
- a list longer than one call holds is reduced in consecutive calls;
- lists of different lengths or ops raise in the first call. Where they take the
  same number of row words, every rank raises at once; otherwise the ranks with
  the shorter rows raise at once, and those with the longer rows wait for words
  that never come and raise when the exchange times out.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.shm_tag_exchange import (
    PD_DECODE_MAX_HOSTS,
    get_shm_exchange,
)
from sglang.srt.runtime_context import get_disagg, get_parallel

logger = logging.getLogger(__name__)

_KIND = "pdflags"
# Exchange words per call: one header and 63 data words of five one-byte flags.
_WIDTH = 64
PER_WORD = 5
_WORD_MASK = (1 << (8 * PER_WORD)) - 1

# Header word: bit 0 overflow, bits 2-3 op, bits 4-19 the entries of this call,
# bits 20-39 the length of the whole list. Ranks compare everything above bit 1.
_HDR_OVERFLOW = 1
_OP_SHIFT = 2
_LEN_SHIFT = 4
_TOTAL_SHIFT = 20
_MAX_TOTAL = (1 << 20) - 1
OP_MIN = 1
_OP_MAX = 2


def _op_code(op) -> int:
    if op == dist.ReduceOp.MIN:
        return OP_MIN
    if op == dist.ReduceOp.MAX:
        return _OP_MAX
    return 0


def pack_flags(flags: Sequence[int]) -> Tuple[List[int], int]:
    """Five flags per word, one byte each. Returns (words, overflow); overflow is 1
    (and the words zero) if any flag is outside 0..255."""
    n = len(flags)
    try:
        raw = bytes(flags)
    except (ValueError, TypeError):
        return [0] * -(-n // PER_WORD), _HDR_OVERFLOW
    # Word i holds bytes 5i..5i+4 of the little-endian whole.
    whole = int.from_bytes(raw, "little")
    return [(whole >> shift) & _WORD_MASK for shift in range(0, 8 * n, 8 * PER_WORD)], 0


def reduce_rows(rows: Sequence[Sequence[int]], *, n: int, op_code: int) -> List[int]:
    """Per-entry MIN or MAX of the first `n` flags over every rank's data words."""
    first = rows[0]
    if all(row == first for row in rows):
        raw = b"".join(w.to_bytes(PER_WORD, "little") for w in first)
        return list(raw[:n])
    # Each word as little-endian uint64: its five low bytes are the flags.
    lanes = np.array(rows, dtype="<u8").view(np.uint8).reshape(len(rows), -1, 8)
    lanes = lanes[:, :, :PER_WORD]
    out = lanes.min(axis=0) if op_code == OP_MIN else lanes.max(axis=0)
    return out.reshape(-1)[:n].tolist()


class ShmFlagReducer:
    """`reduce(flags, op)` = the gloo all_reduce MIN / MAX of `flags` over the
    exchange's group, or None where the caller must run that all_reduce."""

    def __init__(self, exchange, width: int = _WIDTH):
        self.exchange = exchange
        self.capacity = (width - 1) * PER_WORD

    def reduce(self, flags: Sequence[int], op) -> Optional[List[int]]:
        n = len(flags)
        code = _op_code(op)
        # Same on every rank: equal lengths and ops are the callers' contract.
        if code == 0 or n > _MAX_TOTAL:
            return None
        out: List[int] = []
        for start in range(0, n, self.capacity):
            part = self._reduce_call(
                flags[start : start + self.capacity], code=code, total=n
            )
            if part is None:
                return None
            out += part
        return out

    def _reduce_call(
        self, flags: Sequence[int], *, code: int, total: int
    ) -> Optional[List[int]]:
        n = len(flags)
        words, overflow = pack_flags(flags)
        header = (
            (total << _TOTAL_SHIFT) | (n << _LEN_SHIFT) | (code << _OP_SHIFT) | overflow
        )
        rows = self.exchange.allgather([header, *words])
        shape = rows[0][0] >> _OP_SHIFT
        if any(row[0] >> _OP_SHIFT != shape for row in rows):
            raise RuntimeError(
                "Decode flag reduction: ranks passed different lengths or ops "
                f"(headers {[row[0] for row in rows]}); every rank must pass "
                "equal-length lists in the same order"
            )
        if any(row[0] & _HDR_OVERFLOW for row in rows):
            return None
        return reduce_rows([row[1:] for row in rows], n=n, op_code=code)


# id(group) -> (group, reducer or None). Holding the group keeps its id unique.
_REDUCERS: Dict[int, Tuple[object, Optional[ShmFlagReducer]]] = {}


def get_flag_reducer(group) -> Optional[ShmFlagReducer]:
    entry = _REDUCERS.get(id(group))
    return entry[1] if entry is not None and entry[0] is group else None


def init_decode_flag_reducer(group, *, enable_overlap: bool) -> None:
    """Route the decode queues' flag reductions on `group` through the exchange
    where it is on. Collective on the first call per group: every rank of the group
    calls it at the same point."""
    if group is None:
        return
    entry = _REDUCERS.get(id(group))
    if entry is not None and entry[0] is group:
        return
    reducer = None
    if not envs.SGLANG_OPT_SCHEDULER_SHM_SYNC.get():
        _REDUCERS[id(group)] = (group, None)
        return
    reason = _decline_reason(group, enable_overlap=enable_overlap)
    if reason is None:
        exchange = get_shm_exchange(
            group,
            _KIND,
            multihost=True,
            width=_WIDTH,
            max_hosts=PD_DECODE_MAX_HOSTS,
        )
        if exchange is None:
            reason = (
                "no shared-memory exchange for this group (--sleep-on-idle, or "
                "hosts the exchange does not cover)"
            )
        else:
            reducer = ShmFlagReducer(exchange)
    if reducer is not None:
        logger.info(
            "Decode queue flags reduce through the shared-memory exchange (%s) for "
            "a %d-rank group",
            type(reducer.exchange).__name__,
            dist.get_world_size(group),
        )
    else:
        logger.info("Decode queue flags reduce over gloo: %s", reason)
    _REDUCERS[id(group)] = (group, reducer)


def release_decode_flag_reducers() -> None:
    """The decode queues are torn down (a PD role switch): their groups go back to
    the stock reductions."""
    _REDUCERS.clear()


def _decline_reason(group, *, enable_overlap: bool) -> Optional[str]:
    if get_disagg().disaggregation_mode != "decode":
        return "not a PD decode scheduler"
    if dist.get_world_size(group) <= 1:
        return "one rank, nothing to reduce"
    if get_parallel().pp_size != 1:
        return "pipeline parallelism"
    if not enable_overlap:
        return "no overlap scheduler"
    return None
