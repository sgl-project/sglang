"""hier_2shot_push for CustomAllReduceV2 on ROCm gfx1250 DPX."""

import logging
from typing import List, Optional, Tuple

import msgspec
import torch
from torch.distributed import ProcessGroup

from sglang.kernels.ops.communication.all_reduce import (
    AllReduceAlgo,
    Communicator,
    PushPlane,
    custom_all_reduce,
)
from sglang.kernels.ops.communication.dpx_pair_push import custom_pair_stage

from .configs.custom_all_reduce_v2 import (
    AllReduceConfig,
    Range,
    two_shot_push_slot_bytes,
)

logger = logging.getLogger(__name__)

_ALIGN_BYTES = 1024


def _ceil_align(nbytes: int, align: int) -> int:
    return (nbytes + align - 1) // align * align


def _hier_max_bytes(config, max_size: int, *, sizes_own_push: bool) -> int:
    """Largest hier_2shot_push message the workspace should hold, or 0.

    A pair slot holds half a message and, like every push slot, stays within
    ``max_size``.
    """
    if sizes_own_push:
        return 0
    hier = max(
        (h.hier_2shot_push for h in (config.graph, config.eager)),
        key=lambda r: r.max_bytes,
    )
    hier_max = min(hier.max_bytes, 2 * max_size) // 32 * 32
    return hier_max if hier_max > hier.min_bytes else 0


def _fit_hier(config, hier_max: int):
    """Clip hier_2shot_push to ``hier_max`` (0 disables it).

    Where it then covers the whole 2shot_push range, 2shot_push keeps only the
    sizes below it, so the push slots need not grow for sizes hier takes; in
    that range, sizes it cannot halve into 16 B vectors fall back to NCCL.
    """

    def fit(heuristic):
        hier = heuristic.hier_2shot_push
        if not hier_max:
            return heuristic._replace(hier_2shot_push=Range(0, 0))
        hier = Range(hier.min_bytes, min(hier.max_bytes, hier_max))
        two_shot = heuristic.two_shot_push_threshold
        if hier.max_bytes >= two_shot:
            two_shot = min(two_shot, hier.min_bytes)
        return heuristic._replace(
            hier_2shot_push=hier, two_shot_push_threshold=two_shot
        )

    return config._replace(graph=fit(config.graph), eager=fit(config.eager))


class _Layout(msgspec.Struct, frozen=True, kw_only=True):
    """Per-rank layout: [2 * 2 pair slots | 2 * world_size / 2 cross slots]."""

    pair_slot: int
    cross_slot: int
    pair_bytes: int
    total_bytes: int


def _layout(world_size: int, max_bytes: int) -> _Layout:
    # A pair stage of N bytes moves one N/2 chunk per slot; the cross 2shot_push
    # all-reduces N/2.
    cross_world = world_size // 2
    pair_slot = _ceil_align(max_bytes // 2, _ALIGN_BYTES)
    cross_slot = _ceil_align(
        two_shot_push_slot_bytes(max_bytes // 2, cross_world), _ALIGN_BYTES
    )
    pair_bytes = 2 * 2 * pair_slot
    return _Layout(
        pair_slot=pair_slot,
        cross_slot=cross_slot,
        pair_bytes=pair_bytes,
        total_bytes=pair_bytes + 2 * cross_world * cross_slot,
    )


class DpxHierAllReduce:
    """hier_2shot_push on ROCm gfx1250 DPX: a 2shot_push split into stages.

    Each SoC's two partitions reduce-scatter between themselves (the pair plane),
    every partition all-reduces its half with its peers on the other SoCs (a
    2shot_push on the cross plane), and the pair all-gathers the result.
    """

    @staticmethod
    def plan(
        *,
        group: ProcessGroup,
        device: torch.device,
        config: AllReduceConfig,
        max_size: int,
        sizes_own_push: bool,
    ) -> Tuple[AllReduceConfig, Optional[List[List[int]]], int]:
        """``(config, DPX pairs, largest message)``; a largest message of 0 disables it.

        Runs before the push slots are sized: where hier takes over, 2shot_push keeps
        only the sizes below it. Callers that size their own push plane or run
        push-only get none; it needs both DPX partitions of every SoC in the group.
        """
        max_bytes = _hier_max_bytes(config, max_size, sizes_own_push=sizes_own_push)
        pairs = None
        if max_bytes:
            from .gfx1250_fabric import probe_dpx_pairs

            pairs = probe_dpx_pairs(group, device)
        max_bytes = max_bytes if pairs else 0
        return _fit_hier(config, max_bytes), pairs, max_bytes

    @staticmethod
    def workspace_bytes(*, world_size: int, max_bytes: int) -> int:
        return _layout(world_size, max_bytes).total_bytes

    def __init__(
        self,
        *,
        slabs: List[torch.Tensor],
        offset: int,
        device: torch.device,
        rank: int,
        world_size: int,
        pairs: List[List[int]],
        max_bytes: int,
        push_blocks: int,
    ) -> None:
        """Both planes live in ``[offset, offset + workspace_bytes)`` of ``slabs``."""
        layout = _layout(world_size, max_bytes)
        # partition p of a pair reduces half p, which the cross plane of every SoC's
        # partition p then all-reduces
        pair = next(ranks for ranks in pairs if rank in ranks)
        cross = [ranks[pair.index(rank)] for ranks in pairs]
        # one EpochCounter per plane: pair, cross
        self._counters = torch.zeros((2,), dtype=torch.int64, device=device)

        def plane(
            ranks: List[int], start: int, slot_bytes: int, counter
        ) -> Communicator:
            # 2 phases x len(ranks) peers, like the full push plane
            nbytes = 2 * len(ranks) * slot_bytes
            assert start + nbytes <= layout.total_bytes
            begin = offset + start
            workspaces = [
                slabs[r][begin : begin + nbytes].view(-1, slot_bytes) for r in ranks
            ]
            return Communicator(
                push=PushPlane(
                    ranks.index(rank),
                    len(ranks),
                    workspaces=workspaces,
                    counter=counter,
                    default_blocks=push_blocks,
                )
            )

        self._pair: Optional[Communicator] = plane(
            pair, 0, layout.pair_slot, self._counters[0:1]
        )
        self._cross: Optional[Communicator] = plane(
            cross, layout.pair_bytes, layout.cross_slot, self._counters[1:2]
        )

    def all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        half = custom_pair_stage(self._pair, input.view(-1), all_gather=False)
        half = custom_all_reduce(self._cross, half, algo=AllReduceAlgo.TWO_SHOT_PUSH)
        out = custom_pair_stage(self._pair, half, all_gather=True)
        return out.view(input.shape)

    def close(self) -> None:
        # drop the views; the slabs belong to the caller
        self._pair = self._cross = None
