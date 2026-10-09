"""Hand-tuned dispatch configs for the JIT custom all-reduce (v2).

Dispatch tables and block counts come from sweeps of
``test/registered/kernels/benchmark/communication/bench_custom_all_reduce.py``
on the listed GPUs; ``get_all_reduce_config`` picks the table for the current
arch and world size.
"""

from functools import cache
from typing import Callable, NamedTuple, Optional, Tuple

import torch

from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo

KB, MB = 1024, 1024 * 1024


class Row(NamedTuple):
    """One dispatch-table entry: run ``algo`` for ``min_bytes..max_bytes``."""

    algo: AllReduceAlgo
    max_bytes: int
    multicast: bool = False
    min_bytes: int = 0

    def contains(self, nbytes: int) -> bool:
        return self.min_bytes <= nbytes <= self.max_bytes

    def clip(self, *, max_push_bytes: int, max_pull_bytes: int) -> "Row":
        cap = max_push_bytes if self.algo.is_push() else max_pull_bytes
        return self._replace(
            max_bytes=min(self.max_bytes, cap), min_bytes=min(self.min_bytes, cap)
        )


# A table is tried in order and the first row containing ``nbytes`` wins; past
# the last row the caller falls back to NCCL. ``multicast`` rows are dropped
# when the group has no multicast.
Table = Tuple[Row, ...]


def pick(table: Table, nbytes: int) -> Optional[Row]:
    return next((row for row in table if row.contains(nbytes)), None)


def push1(max_bytes: float) -> Row:
    return Row(AllReduceAlgo.ONE_SHOT_PUSH, int(max_bytes))


def pull1(max_bytes: float) -> Row:
    return Row(AllReduceAlgo.ONE_SHOT_PULL, int(max_bytes))


def pull2(max_bytes: float) -> Row:
    return Row(AllReduceAlgo.TWO_SHOT_PULL, int(max_bytes))


def mc(max_bytes: float, min_bytes: float = 0) -> Row:
    return Row(AllReduceAlgo.TWO_SHOT_PULL, int(max_bytes), True, int(min_bytes))


class AllReduceConfig(NamedTuple):
    """All tuning knobs for a single (arch, world_size).

    ``graph`` / ``eager`` are the dispatch tables for CUDA-graph capture and
    eager mode. Block-count knobs apply to the kernel grid:
      - ``num_push_blocks``: 1shot_push grid (bound to the counter array)
      - ``num_pull_blocks``: 1shot_pull (any mode) and non-mc 2shot_pull
      - ``num_mc_blocks``  : mc 2shot_pull; ``None`` disables multicast
    """

    graph: Table
    eager: Table
    num_push_blocks: int
    num_pull_blocks: int
    num_mc_blocks: Optional[int]

    def _max_bytes(self, push: bool) -> int:
        rows = self.graph + self.eager
        return max((r.max_bytes for r in rows if r.algo.is_push() == push), default=0)

    @property
    def max_push_bytes(self) -> int:
        return self._max_bytes(push=True)

    @property
    def max_pull_bytes(self) -> int:
        return self._max_bytes(push=False)

    def map_tables(self, fn: Callable[[Table], Table]) -> "AllReduceConfig":
        return self._replace(graph=fn(self.graph), eager=fn(self.eager))

    def clip(self, **kwargs) -> "AllReduceConfig":
        return self.map_tables(lambda t: tuple(r.clip(**kwargs) for r in t))

    def without_multicast(self) -> "AllReduceConfig":
        return self.map_tables(lambda t: tuple(r for r in t if not r.multicast))

    def with_push_max(self, max_bytes: int) -> "AllReduceConfig":
        def replace(row: Row) -> Row:
            if row.algo is AllReduceAlgo.ONE_SHOT_PUSH:
                return row._replace(max_bytes=max_bytes)
            return row

        return self.map_tables(lambda t: tuple(map(replace, t)))

    def with_pull_fallback(self, max_bytes: int) -> "AllReduceConfig":
        return self.map_tables(lambda t: t + (pull2(max_bytes),))


def _config_builder(num_push_blocks, num_pull_blocks, num_mc_blocks):
    def config(world_size: int, *, graph: Table, eager: Optional[Table] = None):
        return AllReduceConfig(
            graph=graph,
            eager=graph if eager is None else eager,
            num_push_blocks=num_push_blocks,
            num_pull_blocks=num_pull_blocks(world_size),
            num_mc_blocks=num_mc_blocks(world_size),
        )

    return config


# SM100 (Blackwell, B200/B300/GB200). Tuned on B200 (148 SMs); world 16 on GB200.
@cache
def _sm100_configs(num_sm: int) -> dict[int, AllReduceConfig]:
    mc_blocks = {5: 64, 6: 48, 7: 48, 8: 32, 16: 32}
    config = _config_builder(
        num_sm, lambda ws: num_sm if ws == 2 else 96, mc_blocks.get
    )

    return {
        2: config(
            2,
            graph=(push1(8 * MB), pull1(32 * MB), pull2(128 * MB)),
            eager=(push1(16 * MB), pull1(128 * MB), pull2(128 * MB)),
        ),
        3: config(
            3,
            graph=(push1(4 * MB), pull2(128 * MB)),
            eager=(push1(8 * MB), pull2(32 * MB)),
        ),
        4: config(
            4,
            graph=(push1(2.25 * MB), pull2(128 * MB)),
            eager=(push1(3 * MB), pull2(32 * MB)),
        ),
        5: config(
            5,
            graph=(push1(1.5 * MB), pull2(128 * MB)),
            eager=(push1(2 * MB), mc(32 * MB), pull2(32 * MB)),
        ),
        6: config(
            6,
            graph=(push1(1 * MB), pull2(128 * MB)),
            eager=(push1(1.25 * MB), mc(64 * MB), pull2(64 * MB)),
        ),
        7: config(
            7,
            graph=(push1(640 * KB), pull2(128 * MB)),
            eager=(push1(1 * MB), mc(64 * MB), pull2(64 * MB)),
        ),
        8: config(
            8,
            graph=(push1(512 * KB), mc(128 * MB, 8 * MB), pull2(128 * MB)),
            eager=(push1(768 * KB), mc(128 * MB), pull2(128 * MB)),
        ),
        16: config(16, graph=(push1(256 * KB), mc(128 * MB), pull2(128 * MB))),
    }


# SM107 (Rubin, VR)
@cache
def _sm107_configs(num_sm: int) -> dict[int, AllReduceConfig]:
    mc_blocks = {4: 128, 5: 128, 6: 128, 7: 128, 8: 96, 16: 32}
    config = _config_builder(num_sm, lambda ws: min(192, num_sm), mc_blocks.get)

    return {
        2: config(2, graph=(push1(128 * MB), pull2(128 * MB))),
        3: config(
            3,
            graph=(push1(19.625 * MB), pull2(128 * MB)),
            eager=(push1(39.188 * MB), pull2(128 * MB)),
        ),
        4: config(
            4,
            graph=(push1(6.938 * MB), pull2(128 * MB)),
            eager=(push1(9.812 * MB), mc(128 * MB), pull2(128 * MB)),
        ),
        5: config(5, graph=(push1(6.938 * MB), mc(128 * MB), pull2(128 * MB))),
        6: config(6, graph=(push1(4.875 * MB), mc(128 * MB), pull2(128 * MB))),
        7: config(7, graph=(push1(3.438 * MB), mc(128 * MB), pull2(128 * MB))),
        8: config(8, graph=(push1(2.438 * MB), mc(128 * MB), pull2(128 * MB))),
        16: config(16, graph=(push1(832 * KB), mc(128 * MB), pull2(128 * MB))),
    }


# SM90 (Hopper, H100/H200). Tuned on H200.
@cache
def _sm90_configs(num_sm: int) -> dict[int, AllReduceConfig]:
    config = _config_builder(
        num_sm, lambda ws: 64, lambda ws: None if ws < 4 else 128 // ws
    )

    return {
        2: config(
            2,
            graph=(push1(16 * MB), pull1(128 * MB), pull2(128 * MB)),
            eager=(push1(32 * MB), pull1(128 * MB), pull2(128 * MB)),
        ),
        3: config(
            3,
            graph=(push1(1.25 * MB), pull2(128 * MB)),
            eager=(push1(3 * MB), pull2(16 * MB)),
        ),
        4: config(
            4,
            graph=(push1(384 * KB), pull2(128 * MB)),
            eager=(push1(896 * KB), mc(32 * MB), pull2(32 * MB)),
        ),
        5: config(
            5,
            graph=(push1(192 * KB), pull2(32 * MB)),
            eager=(push1(384 * KB), mc(32 * MB), pull2(32 * MB)),
        ),
        6: config(
            6,
            graph=(push1(128 * KB), mc(32 * MB, 8 * MB), pull2(32 * MB)),
            eager=(push1(192 * KB), mc(32 * MB), pull2(32 * MB)),
        ),
        7: config(
            7,
            graph=(push1(128 * KB), mc(32 * MB, 1 * MB), pull2(32 * MB)),
            eager=(push1(128 * KB), mc(32 * MB), pull2(32 * MB)),
        ),
        8: config(
            8,
            graph=(push1(128 * KB), mc(128 * MB, 512 * KB), pull2(32 * MB)),
            eager=(push1(128 * KB), mc(128 * MB), pull2(128 * MB)),
        ),
    }


@cache
def _get_all_reduce_configs() -> dict[int, AllReduceConfig]:
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    num_sm = torch.cuda.get_device_properties().multi_processor_count
    if cuda_major == 9:
        return _sm90_configs(num_sm)
    if cuda_major == 10:
        if cuda_minor >= 7:
            return _sm107_configs(num_sm)
        return _sm100_configs(num_sm)

    default = AllReduceConfig(
        graph=(push1(1 * MB), pull2(16 * MB)),
        eager=(push1(1 * MB), pull2(16 * MB)),
        num_push_blocks=num_sm,
        num_pull_blocks=num_sm,
        num_mc_blocks=None,
    )
    return {world_size: default for world_size in range(2, 17)}


@cache
def get_supported_world_sizes() -> tuple[int, ...]:
    return tuple(_get_all_reduce_configs())


@cache
def get_all_reduce_config(world_size: int) -> AllReduceConfig:
    """Tuned dispatch tables and block counts for the current arch / world size.

    Only SM90, SM100 and SM107 are benchmarked so far; other archs get a
    conservative default (1 MB one-shot crossovers, no multicast).
    """
    return _get_all_reduce_configs()[world_size]
