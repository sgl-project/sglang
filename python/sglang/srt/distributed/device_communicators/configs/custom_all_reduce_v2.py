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
        # `Range(0, 0)` is the "algo disabled here" convention, so it must not
        # match even a zero-byte message
        return self.max_bytes > 0 and self.min_bytes <= nbytes <= self.max_bytes

    def clip(self, max_bytes: int) -> "Range":
        return Range(
            min(self.min_bytes, max_bytes),
            min(self.max_bytes, max_bytes),
        )


class Heuristic(NamedTuple):
    """Self-contained algo ranges for one dispatch context (graph or eager).

    Five algos are tried in order of preference (fastest first):
      1. ``2shot_push``: nbytes in ``two_shot_push`` (a barrier-free band)
      2. ``1shot_push``:    nbytes <= ``one_shot_push_threshold``
      3. ``1shot_pull``:    nbytes <= ``one_shot_pull_threshold``
      4. ``2shot_pull`` mc: nbytes in ``mc.min_bytes..mc.max_bytes``
                            (only when multicast is enabled at runtime)
      5. ``2shot_pull``:    nbytes <= ``two_shot_pull_threshold``
    Above all of these, the caller falls back to NCCL.

    ``two_shot_push`` is tried first so its band may start below
    ``one_shot_push_threshold``.

    Setting two adjacent thresholds equal effectively disables the middle
    algo; leaving a ``Range`` at the default disables that algo.
    """

    one_shot_push_threshold: int
    one_shot_pull_threshold: int
    two_shot_pull_threshold: int
    mc: Range = Range(0, 0)  # default: multicast disabled in this context
    two_shot_push: Range = Range(0, 0)  # default: 2shot_push disabled here
    two_shot_push_mc: Range = Range(0, 0)  # multimem fan-out sub-band of 2shot_push

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

    @property
    def max_two_shot_push_bytes(self) -> int:
        return self.two_shot_push.max_bytes

    def clip(
        self,
        *,
        max_push_bytes: int,
        max_pull_bytes: int,
        max_two_shot_push_bytes: int,
    ) -> "Heuristic":
        return Heuristic(
            min(self.one_shot_push_threshold, max_push_bytes),
            min(self.one_shot_pull_threshold, max_pull_bytes),
            min(self.two_shot_pull_threshold, max_pull_bytes),
            self.mc.clip(max_pull_bytes),
            self.two_shot_push.clip(max_two_shot_push_bytes),
            self.two_shot_push_mc.clip(max_two_shot_push_bytes),
        )


class AllReduceConfig(NamedTuple):
    """All tuning knobs for a single (arch, world_size).

    The two ``Heuristic`` entries describe the size crossover for each
    dispatch context (CUDA-graph capture vs eager). Block-count knobs apply
    to the kernel grid:
      - ``num_push_blocks``: shared 1shot_push / 2shot_push grid (bound
        to the counter array)
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

    @property
    def max_two_shot_push_bytes(self) -> int:
        return max(
            self.graph.max_two_shot_push_bytes,
            self.eager.max_two_shot_push_bytes,
        )

    def clip(
        self,
        *,
        max_push_bytes: int,
        max_pull_bytes: int,
        max_two_shot_push_bytes: int,
    ) -> "AllReduceConfig":
        bounds = dict(
            max_push_bytes=max_push_bytes,
            max_pull_bytes=max_pull_bytes,
            max_two_shot_push_bytes=max_two_shot_push_bytes,
        )
        return self._replace(
            graph=self.graph.clip(**bounds),
            eager=self.eager.clip(**bounds),
        )


def _pack_heuristic(*args) -> Heuristic:
    arg_list: list = [int(p) if isinstance(p, float) else p for p in args]
    return Heuristic(*arg_list)


# SM100 (Blackwell, B200/B300/GB200). Tuned on B200 (148 SMs); world 16 on GB200.
@cache
def _sm100_configs(num_sm: int) -> dict[int, AllReduceConfig]:
    mc_blocks = {5: 64, 6: 48, 7: 48, 8: 32, 16: 32}

    def config(world_size: int, *, graph: tuple, eager: tuple) -> AllReduceConfig:
        return AllReduceConfig(
            graph=_pack_heuristic(*graph),
            eager=_pack_heuristic(*eager),
            num_push_blocks=num_sm,
            num_pull_blocks=num_sm if world_size == 2 else 96,
            num_mc_blocks=mc_blocks.get(world_size),
        )

    return {
        2: config(
            2,
            graph=(8.000 * MB, 32.00 * MB, 128.0 * MB),
            eager=(16.00 * MB, 128.0 * MB, 128.0 * MB),
        ),
        3: config(
            3,
            graph=(4.000 * MB, 4.000 * MB, 128.0 * MB),
            eager=(8.000 * MB, 8.000 * MB, 32.00 * MB),
        ),
        4: config(
            4,
            graph=(
                0.625 * MB,
                0.625 * MB,
                128.0 * MB,
                Range(0, 0),
                Range(int(0.625 * MB), 16 * MB),
            ),
            eager=(
                0.625 * MB,
                0.625 * MB,
                32.00 * MB,
                Range(0, 0),
                Range(int(0.625 * MB), 16 * MB),
            ),
        ),
        5: config(
            5,
            graph=(1.500 * MB, 1.500 * MB, 128.0 * MB),
            eager=(2.000 * MB, 2.000 * MB, 32.00 * MB, Range(0, 32 * MB)),
        ),
        6: config(
            6,
            graph=(1.000 * MB, 1.000 * MB, 128.0 * MB),
            eager=(1.250 * MB, 1.250 * MB, 64.00 * MB, Range(0, 64 * MB)),
        ),
        7: config(
            7,
            graph=(0.625 * MB, 0.625 * MB, 128.0 * MB),
            eager=(1.000 * MB, 1.000 * MB, 64.00 * MB, Range(0, 64 * MB)),
        ),
        8: config(
            8,
            graph=(0.500 * MB, 0.500 * MB, 128.0 * MB, Range(8 * MB, 128 * MB)),
            eager=(0.750 * MB, 0.750 * MB, 128.0 * MB, Range(0, 128 * MB)),
        ),
        16: config(
            16,
            graph=(0.250 * MB, 0.250 * MB, 128.0 * MB, Range(256 * KB, 128 * MB)),
            eager=(0.250 * MB, 0.250 * MB, 128.0 * MB, Range(256 * KB, 128 * MB)),
        ),
    }


# SM107 (Rubin, VR)
@cache
def _sm107_configs(num_sm: int) -> dict[int, AllReduceConfig]:
    mc_blocks = {4: 128, 5: 128, 6: 128, 7: 128, 8: 96, 16: 32}

    def config(world_size: int, *, graph: tuple, eager: tuple) -> AllReduceConfig:
        return AllReduceConfig(
            graph=_pack_heuristic(*graph),
            eager=_pack_heuristic(*eager),
            num_push_blocks=num_sm,
            num_pull_blocks=min(192, num_sm),
            num_mc_blocks=mc_blocks.get(world_size),
        )

    return {
        2: config(
            2,
            graph=(128.0 * MB, 128.0 * MB, 128.0 * MB),
            eager=(128.0 * MB, 128.0 * MB, 128.0 * MB),
        ),
        3: config(
            3,
            graph=(19.625 * MB, 19.625 * MB, 128.0 * MB),
            eager=(39.188 * MB, 39.188 * MB, 128.0 * MB),
        ),
        4: config(
            4,
            graph=(6.938 * MB, 6.938 * MB, 128.0 * MB),
            eager=(9.812 * MB, 9.812 * MB, 128.0 * MB, Range(9.812 * MB, 128 * MB)),
        ),
        5: config(
            5,
            graph=(6.938 * MB, 6.938 * MB, 128.0 * MB, Range(6.938 * MB, 128 * MB)),
            eager=(6.938 * MB, 6.938 * MB, 128.0 * MB, Range(6.938 * MB, 128 * MB)),
        ),
        6: config(
            6,
            graph=(4.875 * MB, 4.875 * MB, 128.0 * MB, Range(4.875 * MB, 128 * MB)),
            eager=(4.875 * MB, 4.875 * MB, 128.0 * MB, Range(4.875 * MB, 128 * MB)),
        ),
        7: config(
            7,
            graph=(3.438 * MB, 3.438 * MB, 128.0 * MB, Range(3.438 * MB, 128 * MB)),
            eager=(3.438 * MB, 3.438 * MB, 128.0 * MB, Range(3.438 * MB, 128 * MB)),
        ),
        8: config(
            8,
            graph=(2.438 * MB, 2.438 * MB, 128.0 * MB, Range(2.438 * MB, 128 * MB)),
            eager=(2.438 * MB, 2.438 * MB, 128.0 * MB, Range(2.438 * MB, 128 * MB)),
        ),
        16: config(
            16,
            graph=(832.0 * KB, 832.0 * KB, 128.0 * MB, Range(832 * KB, 128 * MB)),
            eager=(832.0 * KB, 832.0 * KB, 128.0 * MB, Range(832 * KB, 128 * MB)),
        ),
    }


# SM90 (Hopper, H100/H200). Tuned on H200.
@cache
def _sm90_configs(num_sm: int) -> dict[int, AllReduceConfig]:
    def config(world_size: int, *, graph: tuple, eager: tuple) -> AllReduceConfig:
        return AllReduceConfig(
            graph=_pack_heuristic(*graph),
            eager=_pack_heuristic(*eager),
            num_push_blocks=num_sm,
            num_pull_blocks=64,
            num_mc_blocks=None if world_size < 4 else 128 // world_size,
        )

    return {
        2: config(
            2,
            graph=(16.00 * MB, 128.0 * MB, 128.0 * MB),
            eager=(32.00 * MB, 128.0 * MB, 128.0 * MB),
        ),
        3: config(
            3,
            graph=(1.250 * MB, 1.250 * MB, 128.0 * MB),
            eager=(3.000 * MB, 3.000 * MB, 16.00 * MB),
        ),
        4: config(
            4,
            graph=(384.0 * KB, 384.0 * KB, 128.0 * MB),
            eager=(896.0 * KB, 896.0 * KB, 32.00 * MB, Range(0, 32 * MB)),
        ),
        5: config(
            5,
            graph=(192.0 * KB, 192.0 * KB, 32.00 * MB),
            eager=(384.0 * KB, 384.0 * KB, 32.00 * MB, Range(0, 32 * MB)),
        ),
        6: config(
            6,
            graph=(128.0 * KB, 128.0 * KB, 32.00 * MB, Range(8 * MB, 32 * MB)),
            eager=(192.0 * KB, 192.0 * KB, 32.00 * MB, Range(0, 32 * MB)),
        ),
        7: config(
            7,
            graph=(128.0 * KB, 128.0 * KB, 32.00 * MB, Range(1 * MB, 32 * MB)),
            eager=(128.0 * KB, 128.0 * KB, 32.00 * MB, Range(0, 32 * MB)),
        ),
        8: config(
            8,
            graph=(128.0 * KB, 128.0 * KB, 32.00 * MB, Range(512 * KB, 128 * MB)),
            eager=(128.0 * KB, 128.0 * KB, 128.0 * MB, Range(0, 128 * MB)),
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
        graph=Heuristic(1 * MB, 1 * MB, 16 * MB),
        eager=Heuristic(1 * MB, 1 * MB, 16 * MB),
        num_push_blocks=num_sm,
        num_pull_blocks=num_sm,
        num_mc_blocks=None,
    )
    return {world_size: default for world_size in range(2, 17)}


@cache
def get_supported_world_sizes() -> tuple[int, ...]:
    return tuple(_get_all_reduce_configs())


# A world size does not pin down the interconnect: 8 ranks on one board and 8
# ranks spanning two nodes have different crossovers, and the tables above are
# tuned on the former. Being barrier-free and remote-read-free, 2shot_push
# gains most exactly where fabric round-trips cost most, so its multi-node
# band runs both lower and wider than the single-node one.
class _MultinodePushBands(NamedTuple):
    """The eager 2shot_push retune a multi-node group swaps in."""

    two_shot_push: Range
    two_shot_push_mc: Range = Range(0, 0)


_MULTINODE_PUSH_BANDS: dict[tuple[int, int], _MultinodePushBands] = {
    # for 2 x GB200 (4 GPUs each).
    (10, 8): _MultinodePushBands(
        two_shot_push=Range(288 * KB, 16 * MB),
        two_shot_push_mc=Range(int(3.5 * MB), 16 * MB),
    ),
}


@cache
def get_all_reduce_config(world_size: int, multinode: bool = False) -> AllReduceConfig:
    """Tuned thresholds and block counts for the current arch / world size.

    Only SM90, SM100 and SM107 are benchmarked so far; other archs get a
    conservative default (1 MB one-shot crossovers, no multicast).

    ``multinode`` swaps in a separately tuned ``2shot_push`` band where
    one exists. Only the eager row can differ: a multi-node group forces eager anyway, because
    graph zero-copy input registration is node-local.
    """
    config = _get_all_reduce_configs()[world_size]
    if not multinode:
        return config
    cuda_major, _ = torch.cuda.get_device_capability()
    bands = _MULTINODE_PUSH_BANDS.get((cuda_major, world_size))
    if bands is None:
        return config
    return config._replace(eager=config.eager._replace(**bands._asdict()))
