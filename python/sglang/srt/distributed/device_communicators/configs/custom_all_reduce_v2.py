"""Hand-tuned dispatch configs for the JIT custom all-reduce (v2).

Thresholds and block counts come from sweeps of
``test/registered/jit/benchmark/bench_custom_all_reduce.py`` on the listed
GPUs; ``get_all_reduce_config`` picks the table for the current arch and
world size.
"""

from functools import cache
from typing import NamedTuple, Optional

import torch

KB, MB = 1024, 1024 * 1024


class Range(NamedTuple):
    min_bytes: int
    max_bytes: int

    def contains(self, nbytes: int) -> bool:
        return self.min_bytes <= nbytes <= self.max_bytes

    def clip(self, max_bytes: int) -> "Range":
        return Range(
            min(self.min_bytes, max_bytes),
            min(self.max_bytes, max_bytes),
        )


class Heuristic(NamedTuple):
    """Self-contained algo ranges for one dispatch context (graph or eager).

    Four algos are tried in order of preference (fastest first):
      1. ``1shot_push``:    nbytes <= ``one_shot_push_threshold``
      2. ``1shot_pull``:    nbytes <= ``one_shot_pull_threshold``
      3. ``2shot_pull`` mc: nbytes in ``mc.min_bytes..mc.max_bytes``
                            (only when multicast is enabled at runtime)
      4. ``2shot_pull``:    nbytes <= ``two_shot_pull_threshold``
    Above all of these, the caller falls back to NCCL.

    Setting two adjacent thresholds equal effectively disables the middle
    algo; leaving ``mc`` at the default disables multicast.
    """

    one_shot_push_threshold: int
    one_shot_pull_threshold: int
    two_shot_pull_threshold: int
    mc: Range = Range(0, 0)  # default: multicast disabled in this context

    @property
    def max_push_bytes(self) -> int:
        return self.one_shot_push_threshold

    @property
    def max_pull_bytes(self) -> int:
        # The pull workspace hosts every pull-variant kernel, so it has to
        # fit whichever variant runs at the largest size.
        return max(
            self.one_shot_pull_threshold,
            self.two_shot_pull_threshold,
            self.mc.max_bytes,
        )

    def clip(self, *, max_push_bytes: int, max_pull_bytes: int) -> "Heuristic":
        return Heuristic(
            min(self.one_shot_push_threshold, max_push_bytes),
            min(self.one_shot_pull_threshold, max_pull_bytes),
            min(self.two_shot_pull_threshold, max_pull_bytes),
            self.mc.clip(max_pull_bytes),
        )


class AllReduceConfig(NamedTuple):
    """All tuning knobs for a single (arch, world_size).

    The two ``Heuristic`` entries describe the size crossover for each
    dispatch context (CUDA-graph capture vs eager). Block-count knobs apply
    to the kernel grid:
      - ``num_push_blocks``: 1shot_push grid (bound to the counter array)
      - ``num_pull_blocks``: 1shot_pull (any mode) and non-mc 2shot_pull
      - ``num_mc_blocks``  : mc 2shot_pull; ``None`` disables multicast
    """

    graph: Heuristic
    eager: Heuristic
    num_push_blocks: int
    num_pull_blocks: int
    num_mc_blocks: Optional[int]

    @property
    def max_push_bytes(self) -> int:
        return max(self.graph.max_push_bytes, self.eager.max_push_bytes)

    @property
    def max_pull_bytes(self) -> int:
        return max(self.graph.max_pull_bytes, self.eager.max_pull_bytes)

    def clip(self, *, max_push_bytes: int, max_pull_bytes: int) -> "AllReduceConfig":
        return self._replace(
            graph=self.graph.clip(
                max_push_bytes=max_push_bytes, max_pull_bytes=max_pull_bytes
            ),
            eager=self.eager.clip(
                max_push_bytes=max_push_bytes, max_pull_bytes=max_pull_bytes
            ),
        )


def _pack_heuristic(*args) -> Heuristic:
    arg_list: list = [int(p) if isinstance(p, float) else p for p in args]
    return Heuristic(*arg_list)


def _sm100_config(world_size: int, num_sm: int) -> AllReduceConfig:
    # SM100 (Blackwell, B200/B300). Tuned on B200 (148 SMs).
    graph_map = {
        2: (8.000 * MB, 32.00 * MB, 128.0 * MB),
        3: (4.000 * MB, 4.000 * MB, 128.0 * MB),
        4: (2.250 * MB, 2.250 * MB, 128.0 * MB),
        5: (1.500 * MB, 1.500 * MB, 128.0 * MB),
        6: (1.000 * MB, 1.000 * MB, 128.0 * MB),
        7: (0.625 * MB, 0.625 * MB, 128.0 * MB),
        8: (0.500 * MB, 0.500 * MB, 128.0 * MB, Range(8 * MB, 128 * MB)),
    }
    eager_map = {
        2: (16.00 * MB, 128.0 * MB, 128.0 * MB),
        3: (8.000 * MB, 8.000 * MB, 32.00 * MB),
        4: (3.000 * MB, 3.000 * MB, 32.00 * MB),
        5: (2.000 * MB, 2.000 * MB, 32.00 * MB, Range(0, 32 * MB)),
        6: (1.250 * MB, 1.250 * MB, 64.00 * MB, Range(0, 64 * MB)),
        7: (1.000 * MB, 1.000 * MB, 64.00 * MB, Range(0, 64 * MB)),
        8: (0.750 * MB, 0.750 * MB, 128.0 * MB, Range(0, 128 * MB)),
    }
    mc_blocks_map = {5: 64, 6: 48, 7: 48, 8: 32}
    return AllReduceConfig(
        graph=_pack_heuristic(*graph_map[world_size]),
        eager=_pack_heuristic(*eager_map[world_size]),
        num_push_blocks=num_sm,
        num_pull_blocks=num_sm if world_size == 2 else 96,
        num_mc_blocks=mc_blocks_map.get(world_size, None),
    )


def _sm90_config(world_size: int, num_sm: int) -> AllReduceConfig:
    # SM90 (Hopper, H100/H200). Tuned on H200.
    graph_map = {
        2: (16.00 * MB, 128.0 * MB, 128.0 * MB),
        3: (1.250 * MB, 1.250 * MB, 128.0 * MB),
        4: (384.0 * KB, 384.0 * KB, 128.0 * MB),
        5: (192.0 * KB, 192.0 * KB, 32.00 * MB),
        6: (128.0 * KB, 128.0 * KB, 32.00 * MB, Range(8 * MB, 32 * MB)),
        7: (128.0 * KB, 128.0 * KB, 32.00 * MB, Range(1 * MB, 32 * MB)),
        8: (128.0 * KB, 128.0 * KB, 32.00 * MB, Range(512 * KB, 128 * MB)),
    }
    eager_map = {
        2: (32.00 * MB, 128.0 * MB, 128.0 * MB),
        3: (3.000 * MB, 3.000 * MB, 16.00 * MB),
        4: (896.0 * KB, 896.0 * KB, 32.00 * MB, Range(0, 32 * MB)),
        5: (384.0 * KB, 384.0 * KB, 32.00 * MB, Range(0, 32 * MB)),
        6: (192.0 * KB, 192.0 * KB, 32.00 * MB, Range(0, 32 * MB)),
        7: (128.0 * KB, 128.0 * KB, 32.00 * MB, Range(0, 32 * MB)),
        8: (128.0 * KB, 128.0 * KB, 128.0 * MB, Range(0, 128 * MB)),
    }
    return AllReduceConfig(
        graph=_pack_heuristic(*graph_map[world_size]),
        eager=_pack_heuristic(*eager_map[world_size]),
        num_push_blocks=num_sm,
        num_pull_blocks=64,
        num_mc_blocks=None if world_size < 4 else 128 // world_size,
    )


@cache
def get_all_reduce_config(world_size: int) -> AllReduceConfig:
    """Tuned thresholds and block counts for the current arch / world size.

    Only SM90 and SM100 are benchmarked so far; other archs get a
    conservative default (1 MB one-shot crossovers, no multicast).
    """
    cuda_major, _ = torch.cuda.get_device_capability()
    num_sm = torch.cuda.get_device_properties().multi_processor_count
    if cuda_major == 9:
        return _sm90_config(world_size, num_sm)
    if cuda_major == 10:
        return _sm100_config(world_size, num_sm)
    default = Heuristic(1 * MB, 1 * MB, 16 * MB)
    return AllReduceConfig(
        graph=default,
        eager=default,
        num_push_blocks=num_sm,
        num_pull_blocks=num_sm,
        num_mc_blocks=None,
    )
