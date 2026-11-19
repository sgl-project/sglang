"""Cross-DP sync rows of a PD decode scheduler through the shared-memory exchange.

Under DP attention every scheduler pass, idle ones included, gathers one small
int64 row per rank of the TP group (`MLPSyncBatchInfo.all_gather` ->
`dp_attn.exchange_sync_rows`). On the CPU group that is a gloo gather plus a
broadcast. With SGLANG_OPT_SCHEDULER_SHM_SYNC_DP_ROWS, a PD decode scheduler
gathers the same rows through `shm_tag_exchange.get_shm_exchange` instead: one
shared-memory table on one host, the two-host exchange on two hosts. The result
is the stock result: every rank's row, in group-rank order.

Each row travels behind a header word with its column count and an overflow bit,
padded to a fixed width, and every rank reads every header, so all ranks take the
same path:
- a value outside [0, 2**40), or a row wider than WIDTH - 1 columns, on any rank
  sends that call through the stock gather on every rank;
- rows of different widths raise on every rank.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.shm_tag_exchange import (
    PD_DECODE_MAX_HOSTS,
    get_shm_exchange,
)
from sglang.srt.runtime_context import get_disagg, get_parallel

logger = logging.getLogger(__name__)

_KIND = "dprows"
# Exchange words per call: one header and up to 15 columns.
WIDTH = 16
_VAL_MAX = (1 << 40) - 1
_HDR_OVERFLOW = 1
_COLS_SHIFT = 1


class DPSyncRowExchange:
    """`exchange_rows` fills `global_flat` as the stock gather does, or returns
    False where every rank must run the stock gather."""

    def __init__(self, exchange, *, world: int):
        self.exchange = exchange
        self.world = world

    def exchange_rows(self, global_flat: torch.Tensor, local_row: torch.Tensor) -> bool:
        cols = local_row.numel()
        values = local_row.tolist()
        # A row the exchange cannot carry still sends its header, so every rank
        # sees it and takes the stock gather, or raises, with the others.
        overflow = (
            cols >= WIDTH
            or global_flat.numel() != self.world * cols
            or (bool(values) and not (min(values) >= 0 and max(values) <= _VAL_MAX))
        )
        header = (cols << _COLS_SHIFT) | (_HDR_OVERFLOW if overflow else 0)
        if overflow:
            payload = [0] * (WIDTH - 1)
        else:
            payload = values + [0] * (WIDTH - 1 - cols)
        rows = self.exchange.allgather([header, *payload])
        if any(row[0] >> _COLS_SHIFT != cols for row in rows):
            raise RuntimeError(
                "DP sync rows: ranks sent rows of different widths "
                f"(headers {[row[0] for row in rows]})"
            )
        if any(row[0] & _HDR_OVERFLOW for row in rows):
            return False
        gathered = [v for row in rows for v in row[1 : cols + 1]]
        global_flat.copy_(torch.tensor(gathered, dtype=global_flat.dtype))
        return True


# id(group) -> (group, exchange or None). Holding the group keeps its id unique.
_EXCHANGES: Dict[int, Tuple[object, Optional[DPSyncRowExchange]]] = {}


def exchange_dp_sync_rows(
    global_flat: torch.Tensor, local_row: torch.Tensor, *, group
) -> bool:
    """True if `global_flat` now holds every rank's `local_row` in group-rank
    order, gathered through the shared-memory exchange; False if the caller must
    run the stock gather. Collective over `group`, like the gather it replaces."""
    entry = _EXCHANGES.get(id(group))
    if entry is None or entry[0] is not group:
        entry = _EXCHANGES[id(group)] = (group, _build(group))
    sync = entry[1]
    return sync is not None and sync.exchange_rows(global_flat, local_row)


def _build(group) -> Optional[DPSyncRowExchange]:
    if not envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_DP_ROWS.get():
        return None
    reason = _decline_reason(group)
    exchange = None
    if reason is None:
        exchange = get_shm_exchange(
            group,
            _KIND,
            multihost=True,
            width=WIDTH,
            max_hosts=PD_DECODE_MAX_HOSTS,
        )
        if exchange is None:
            reason = (
                "no shared-memory exchange for this group (--sleep-on-idle, hosts "
                "without SGLANG_OPT_SCHEDULER_SHM_SYNC_MULTIHOST, or more than "
                f"{PD_DECODE_MAX_HOSTS} hosts)"
            )
    if reason is not None:
        logger.info("SGLANG_OPT_SCHEDULER_SHM_SYNC_DP_ROWS declined: %s", reason)
        return None
    world = dist.get_world_size(group)
    logger.info(
        "SGLANG_OPT_SCHEDULER_SHM_SYNC_DP_ROWS engaged: DP sync rows go through the "
        "shared-memory exchange (%s) for a %d-rank group",
        type(exchange).__name__,
        world,
    )
    return DPSyncRowExchange(exchange, world=world)


def _decline_reason(group) -> Optional[str]:
    if not envs.SGLANG_OPT_SCHEDULER_SHM_SYNC.get():
        return "needs SGLANG_OPT_SCHEDULER_SHM_SYNC"
    disagg = get_disagg()
    if disagg.disaggregation_mode != "decode":
        return "not a PD decode scheduler"
    if disagg.enable_pd_role_switch:
        return "PD role switch enabled"
    parallel = get_parallel()
    if parallel.pp_size != 1:
        return "pipeline parallelism"
    if not parallel.attn_dp_enabled:
        return "no attention DP"
    if dist.get_world_size(group) <= 1:
        return "one rank"
    return None
