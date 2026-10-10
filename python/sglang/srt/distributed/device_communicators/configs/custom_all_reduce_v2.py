"""Hand-tuned dispatch configs for the JIT custom all-reduce (v2).

Dispatch tables and block counts come from sweeps of
``test/registered/kernels/benchmark/communication/bench_custom_all_reduce.py``
on the listed GPUs; ``get_all_reduce_config`` picks the table for the current
arch and world size.
"""

from functools import cache
from typing import NamedTuple, Optional

import torch

from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo

KB, MB = 1024, 1024 * 1024


# Algos as (AllReduceAlgo, use_multicast).
PUSH1 = (AllReduceAlgo.ONE_SHOT_PUSH, False)
PULL1 = (AllReduceAlgo.ONE_SHOT_PULL, False)
PULL2 = (AllReduceAlgo.TWO_SHOT_PULL, False)
MC = (AllReduceAlgo.TWO_SHOT_PULL, True)


class AllReduceConfig(NamedTuple):
    """All tuning knobs for a single (arch, world_size).

    ``graph`` / ``eager`` are the dispatch tables for CUDA-graph capture and
    eager mode: ``(max_bytes, algo)`` rows tried in order, the first with
    ``nbytes <= max_bytes`` wins, NCCL past the end. ``MC`` rows are skipped
    when multicast is unavailable. Block-count knobs apply to the kernel grid:
      - ``num_push_blocks``: 1shot_push grid (bound to the counter array)
      - ``num_pull_blocks``: 1shot_pull (any mode) and non-mc 2shot_pull
      - ``num_mc_blocks``  : mc 2shot_pull; ``None`` disables multicast
    """

    graph: tuple
    eager: tuple
    num_push_blocks: int
    num_pull_blocks: int
    num_mc_blocks: Optional[int]

    def max_bytes(self, push: bool) -> int:
        rows = self.graph + self.eager
        return int(max((b for b, (a, _) in rows if a.is_push() == push), default=0))


# SM100 (Blackwell, B200/B300/GB200). Tuned on B200 (148 SMs); world 16 on GB200.
@cache
def _sm100_configs(num_sm: int) -> dict[int, AllReduceConfig]:
    mc_blocks = {5: 64, 6: 48, 7: 48, 8: 32, 16: 32}

    def config(world_size: int, graph: tuple, eager: tuple = ()) -> AllReduceConfig:
        return AllReduceConfig(
            graph=graph,
            eager=eager or graph,
            num_push_blocks=num_sm,
            num_pull_blocks=num_sm if world_size == 2 else 96,
            num_mc_blocks=mc_blocks.get(world_size),
        )

    # fmt: off
    return {
        2: config(2, graph=((8 * MB, PUSH1), (32 * MB, PULL1), (128 * MB, PULL2)),
                     eager=((16 * MB, PUSH1), (128 * MB, PULL1))),
        3: config(3, graph=((4 * MB, PUSH1), (128 * MB, PULL2)),
                     eager=((8 * MB, PUSH1), (32 * MB, PULL2))),
        4: config(4, graph=((2.25 * MB, PUSH1), (128 * MB, PULL2)),
                     eager=((3 * MB, PUSH1), (32 * MB, PULL2))),
        5: config(5, graph=((1.5 * MB, PUSH1), (128 * MB, PULL2)),
                     eager=((2 * MB, PUSH1), (32 * MB, MC), (32 * MB, PULL2))),
        6: config(6, graph=((1 * MB, PUSH1), (128 * MB, PULL2)),
                     eager=((1.25 * MB, PUSH1), (64 * MB, MC), (64 * MB, PULL2))),
        7: config(7, graph=((640 * KB, PUSH1), (128 * MB, PULL2)),
                     eager=((1 * MB, PUSH1), (64 * MB, MC), (64 * MB, PULL2))),
        8: config(8, graph=((512 * KB, PUSH1), (8 * MB - 1, PULL2), (128 * MB, MC), (128 * MB, PULL2)),
                     eager=((768 * KB, PUSH1), (128 * MB, MC), (128 * MB, PULL2))),
        16: config(16, graph=((256 * KB, PUSH1), (128 * MB, MC), (128 * MB, PULL2))),
    }
    # fmt: on


# SM107 (Rubin, VR)
@cache
def _sm107_configs(num_sm: int) -> dict[int, AllReduceConfig]:
    mc_blocks = {4: 128, 5: 128, 6: 128, 7: 128, 8: 96, 16: 32}

    def config(world_size: int, graph: tuple, eager: tuple = ()) -> AllReduceConfig:
        return AllReduceConfig(
            graph=graph,
            eager=eager or graph,
            num_push_blocks=num_sm,
            num_pull_blocks=min(192, num_sm),
            num_mc_blocks=mc_blocks.get(world_size),
        )

    # fmt: off
    return {
        2: config(2, graph=((128 * MB, PUSH1), (128 * MB, PULL2))),
        3: config(3, graph=((19.625 * MB, PUSH1), (128 * MB, PULL2)),
                     eager=((39.188 * MB, PUSH1), (128 * MB, PULL2))),
        4: config(4, graph=((6.938 * MB, PUSH1), (128 * MB, PULL2)),
                     eager=((9.812 * MB, PUSH1), (128 * MB, MC), (128 * MB, PULL2))),
        5: config(5, graph=((6.938 * MB, PUSH1), (128 * MB, MC), (128 * MB, PULL2))),
        6: config(6, graph=((4.875 * MB, PUSH1), (128 * MB, MC), (128 * MB, PULL2))),
        7: config(7, graph=((3.438 * MB, PUSH1), (128 * MB, MC), (128 * MB, PULL2))),
        8: config(8, graph=((2.438 * MB, PUSH1), (128 * MB, MC), (128 * MB, PULL2))),
        16: config(16, graph=((832 * KB, PUSH1), (128 * MB, MC), (128 * MB, PULL2))),
    }
    # fmt: on


# SM90 (Hopper, H100/H200). Tuned on H200.
@cache
def _sm90_configs(num_sm: int) -> dict[int, AllReduceConfig]:
    def config(world_size: int, graph: tuple, eager: tuple = ()) -> AllReduceConfig:
        return AllReduceConfig(
            graph=graph,
            eager=eager or graph,
            num_push_blocks=num_sm,
            num_pull_blocks=64,
            num_mc_blocks=None if world_size < 4 else 128 // world_size,
        )

    # fmt: off
    return {
        2: config(2, graph=((16 * MB, PUSH1), (128 * MB, PULL1)),
                     eager=((32 * MB, PUSH1), (128 * MB, PULL1))),
        3: config(3, graph=((1.25 * MB, PUSH1), (128 * MB, PULL2)),
                     eager=((3 * MB, PUSH1), (16 * MB, PULL2))),
        4: config(4, graph=((384 * KB, PUSH1), (128 * MB, PULL2)),
                     eager=((896 * KB, PUSH1), (32 * MB, MC), (32 * MB, PULL2))),
        5: config(5, graph=((192 * KB, PUSH1), (32 * MB, PULL2)),
                     eager=((384 * KB, PUSH1), (32 * MB, MC), (32 * MB, PULL2))),
        6: config(6, graph=((128 * KB, PUSH1), (8 * MB - 1, PULL2), (32 * MB, MC), (32 * MB, PULL2)),
                     eager=((192 * KB, PUSH1), (32 * MB, MC), (32 * MB, PULL2))),
        7: config(7, graph=((128 * KB, PUSH1), (1 * MB - 1, PULL2), (32 * MB, MC), (32 * MB, PULL2)),
                     eager=((128 * KB, PUSH1), (32 * MB, MC), (32 * MB, PULL2))),
        8: config(8, graph=((128 * KB, PUSH1), (512 * KB - 1, PULL2), (128 * MB, MC), (32 * MB, PULL2)),
                     eager=((128 * KB, PUSH1), (128 * MB, MC), (128 * MB, PULL2))),
    }
    # fmt: on


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
        graph=((1 * MB, PUSH1), (16 * MB, PULL2)),
        eager=((1 * MB, PUSH1), (16 * MB, PULL2)),
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
