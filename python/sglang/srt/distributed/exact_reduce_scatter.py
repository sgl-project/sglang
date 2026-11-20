"""A reduce-scatter that returns an all-reduce's bits.

``all_reduce(x)`` followed by a slice gives every rank the same sum for a row
whichever rank receives it, which deterministic inference needs. But every rank
then takes part in reducing the whole tensor to keep ``1 / world`` of it.

Here an all-to-all brings each rank the ``world`` versions of its own rows only,
and the rank adds them the way the all-reduce would have
(:mod:`sglang.srt.distributed.reduce_order`). The order is found once per tensor
shape by probing the group's own all-reduce and is then checked on random data
on every rank. A shape whose order is not found, or does not reproduce the
all-reduce bit for bit, keeps the all-reduce.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import torch

from sglang.srt.distributed.reduce_order import (
    ReduceOrder,
    discover_tree_order,
    fold,
    num_probes,
    wide_order,
)

logger = logging.getLogger(__name__)

# Random tensors a candidate order must reproduce before it is used.
_VERIFY_ROUNDS = 2


class ExactReduceScatter:
    """``output = all_reduce(input)[this rank's chunk]`` at a reduce-scatter's traffic.

    ``all_reduce(tensor) -> tensor`` and ``all_to_all_single(output, input)`` are
    the group's collectives. Every rank of the group must make the same calls in
    the same order, as the ranks of a tensor-parallel forward do.
    """

    def __init__(
        self,
        all_reduce: Callable[[torch.Tensor], torch.Tensor],
        all_to_all_single: Callable[[torch.Tensor, torch.Tensor], None],
        world_size: int,
        rank: int,
    ):
        self.all_reduce = all_reduce
        self.all_to_all_single = all_to_all_single
        self.world_size = world_size
        self.rank = rank
        # None: this shape keeps the all-reduce.
        self._orders: Dict[Tuple, Optional[ReduceOrder]] = {}
        self._seen = set()

    def __call__(self, output: torch.Tensor, input: torch.Tensor) -> bool:
        """Fill ``output`` and return True, or return False and leave it to the
        caller's all-reduce."""
        if input.dtype not in (torch.bfloat16, torch.float16):
            return False
        # The all-to-all sends whole rows.
        if self.world_size == 1 or input.shape[0] % self.world_size:
            return False
        key = (tuple(input.shape), input.dtype)
        if key not in self._orders:
            # Discovery runs collectives and reads their results, so it stays
            # out of a graph capture. It also costs many all-reduces, so a
            # shape pays for it only once it comes back.
            if key not in self._seen or _capturing(input):
                self._seen.add(key)
                return False
            self._orders[key] = self._discover(input)
        order = self._orders[key]
        if order is None:
            return False
        output.copy_(self._reduce(input, order).view_as(output))
        return True

    def _reduce(self, input: torch.Tensor, order: ReduceOrder) -> torch.Tensor:
        # Chunk j of `input` goes to rank j; chunk i of `landing` came from rank i.
        landing = torch.empty_like(input)
        self.all_to_all_single(landing, input.contiguous())
        chunk = input.numel() // self.world_size
        return fold(landing.view(self.world_size, chunk), order, self.rank * chunk)

    def _agreed(self, ok: bool, like: torch.Tensor) -> bool:
        votes = torch.full((1,), float(ok), dtype=torch.float32, device=like.device)
        return int(self.all_reduce(votes).item()) == self.world_size

    def _reproduces(self, input: torch.Tensor, order: Optional[ReduceOrder]) -> bool:
        """Whether every rank gets the all-reduce's bits with ``order``. Runs the
        same collectives on every rank, also on one that has no order."""
        chunk = input.numel() // self.world_size
        generator = torch.Generator(device=input.device)
        generator.manual_seed(0x5EED + self.rank)
        ok = order is not None
        for _ in range(_VERIFY_ROUNDS):
            data = torch.randn(
                input.shape, generator=generator, device=input.device
            ).to(input.dtype)
            expected = self.all_reduce(data.clone()).reshape(-1)
            expected = expected.narrow(0, self.rank * chunk, chunk)
            if order is None:
                landing = torch.empty_like(data)
                self.all_to_all_single(landing, data)
                continue
            got = self._reduce(data, order)
            ok = ok and bool(_same_bits(got, expected))
        return self._agreed(ok, input)

    def _discover(self, input: torch.Tensor) -> Optional[ReduceOrder]:
        world, shape = self.world_size, tuple(input.shape)
        num_elements = input.numel()
        if self._reproduces(input, wide_order(world, num_elements)):
            logger.info("Exact reduce-scatter of %s: one FP32 sum in rank order", shape)
            return wide_order(world, num_elements)
        if input.dtype != torch.bfloat16 or num_probes(world) == 0:
            return None
        order = discover_tree_order(
            self.all_reduce, world, self.rank, shape, input.device
        )
        if not self._reproduces(input, order):
            logger.info("Exact reduce-scatter of %s: order not found", shape)
            return None
        logger.info(
            "Exact reduce-scatter of %s: %d addition tree(s)",
            shape,
            len(order.programs),
        )
        return order


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.view(torch.int16) == b.view(torch.int16)).all()


def _capturing(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and torch.cuda.is_current_stream_capturing()
