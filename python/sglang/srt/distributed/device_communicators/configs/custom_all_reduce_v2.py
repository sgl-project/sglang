"""Hand-tuned dispatch configs for the JIT custom all-reduce (v2).

Thresholds and block counts come from sweeps of
``test/registered/kernels/benchmark/communication/bench_custom_all_reduce.py``
on the listed GPUs; ``get_all_reduce_config`` picks the table for the current
arch and world size.
"""

import os
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

    Seven algos are tried in order of preference (fastest first):
      1. ``1shot_push``:    nbytes <= ``one_shot_push_threshold``
      2. ``hier_2shot_push``: nbytes in ``hier_2shot_push.min_bytes..max_bytes``
                            (only when the group is made of DPX pairs)
      3. ``2shot_push``:    nbytes <= ``two_shot_push_threshold``
      4. ``1shot_pull``:    nbytes <= ``one_shot_pull_threshold``
      5. ``2shot_pull`` mc: nbytes in ``mc.min_bytes..mc.max_bytes``
                            (only when multicast is enabled at runtime)
      6. ``2shot_pull``:    nbytes <= ``two_shot_pull_threshold``
      7. ``ring_simple``:   nbytes in ``ring_simple.min_bytes..max_bytes``
    Above all of these, the caller falls back to NCCL.

    Setting two adjacent thresholds equal effectively disables the middle
    algo; leaving ``mc`` / ``two_shot_push_threshold`` / ``hier_2shot_push``
    / ``ring_simple`` at the default disables that algo.

    The field order below is the positional layout of the table rows
    (``1shot_push, 1shot_pull, 2shot_pull, mc``); ``two_shot_push_threshold``
    is set by keyword, see ``_sm100_configs``.
    """

    one_shot_push_threshold: int
    one_shot_pull_threshold: int
    two_shot_pull_threshold: int
    mc: Range = Range(0, 0)  # default: multicast disabled in this context
    two_shot_push_threshold: int = 0  # default: 2shot_push disabled
    hier_2shot_push: Range = Range(0, 0)  # default: hier_2shot_push disabled
    ring_simple: Range = Range(0, 0)  # default: ring_simple disabled

    def max_push_bytes(self, world_size: int) -> int:
        """Per-slot push plane bytes this heuristic needs."""
        return max(
            self.one_shot_push_threshold,
            two_shot_push_slot_bytes(self.two_shot_push_threshold, world_size),
        )

    @property
    def max_pull_bytes(self) -> int:
        # The pull workspace hosts every pull-variant kernel, so it has to
        # fit whichever variant runs at the largest size.
        return max(
            self.one_shot_pull_threshold,
            self.two_shot_pull_threshold,
            self.mc.max_bytes,
        )

    def clip(
        self, *, max_push_bytes: int, max_pull_bytes: int, world_size: int
    ) -> "Heuristic":
        return Heuristic(
            one_shot_push_threshold=min(self.one_shot_push_threshold, max_push_bytes),
            one_shot_pull_threshold=min(self.one_shot_pull_threshold, max_pull_bytes),
            two_shot_pull_threshold=min(self.two_shot_pull_threshold, max_pull_bytes),
            mc=self.mc.clip(max_pull_bytes),
            two_shot_push_threshold=min(
                self.two_shot_push_threshold,
                two_shot_push_max_bytes(max_push_bytes, world_size),
            ),
            # sized by its own range, outside the push slots
            hier_2shot_push=self.hier_2shot_push,
            # streams through its own FIFO, so no workspace bounds it
            ring_simple=self.ring_simple,
        )


def two_shot_push_slot_bytes(nbytes: int, world_size: int) -> int:
    """A 2shot_push slot holds a partial half and a gathered half of one
    ``ceil(nbytes / world_size)`` chunk, in 16 B vectors."""
    return 32 * -(-nbytes // (16 * world_size))


def two_shot_push_max_bytes(slot_bytes: int, world_size: int) -> int:
    """Inverse of ``two_shot_push_slot_bytes``: the largest tensor a slot fits."""
    return world_size * 16 * (slot_bytes // 32)


class AllReduceConfig(NamedTuple):
    """All tuning knobs for a single (arch, world_size).

    The two ``Heuristic`` entries describe the size crossover for each
    dispatch context (CUDA-graph capture vs eager). Block-count knobs apply
    to the kernel grid:
      - ``num_push_blocks``: 1shot / 2shot push grid
      - ``num_pull_blocks``: 1shot_pull (any mode) and non-mc 2shot_pull
      - ``num_mc_blocks``  : mc 2shot_pull; ``None`` disables multicast
      - ``num_ring_simple_channels``: ring_simple grid, one channel (ring) per
                             block
    """

    graph: Heuristic
    eager: Heuristic
    num_push_blocks: int
    num_pull_blocks: int
    num_mc_blocks: Optional[int]
    num_ring_simple_channels: int = 0

    def max_push_bytes(self, world_size: int) -> int:
        return max(
            self.graph.max_push_bytes(world_size),
            self.eager.max_push_bytes(world_size),
        )

    @property
    def max_pull_bytes(self) -> int:
        return max(self.graph.max_pull_bytes, self.eager.max_pull_bytes)

    def clip(self, **kwargs) -> "AllReduceConfig":
        return self._replace(
            graph=self.graph.clip(**kwargs), eager=self.eager.clip(**kwargs)
        )


def _pack_heuristic(*args) -> Heuristic:
    arg_list: list = [int(p) if isinstance(p, float) else p for p in args]
    return Heuristic(*arg_list)


# SM100 (Blackwell, B200/B300/GB200). Tuned on B200 (148 SMs); world 16 on GB200;
# the 2shot_push rows (world 4 / 8) on B300.
@cache
def _sm100_configs(num_sm: int) -> dict[int, AllReduceConfig]:
    mc_blocks = {5: 64, 6: 48, 7: 48, 8: 32, 16: 32}

    def config(
        world_size: int, *, graph: tuple, eager: tuple, two_shot_push: int = 0
    ) -> AllReduceConfig:
        return AllReduceConfig(
            graph=_pack_heuristic(*graph)._replace(
                two_shot_push_threshold=two_shot_push
            ),
            eager=_pack_heuristic(*eager)._replace(
                two_shot_push_threshold=two_shot_push
            ),
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
            graph=(0.625 * MB, 0.625 * MB, 128.0 * MB),
            eager=(0.625 * MB, 0.625 * MB, 32.00 * MB),
            two_shot_push=8 * MB,
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
            graph=(256 * KB, 256 * KB, 128.0 * MB, Range(8 * MB, 128 * MB)),
            eager=(256 * KB, 256 * KB, 128.0 * MB, Range(0, 128 * MB)),
            two_shot_push=8 * MB,
        ),
        16: config(
            16,
            graph=(0.250 * MB, 0.250 * MB, 128.0 * MB, Range(256 * KB, 128 * MB)),
            eager=(0.250 * MB, 0.250 * MB, 128.0 * MB, Range(256 * KB, 128 * MB)),
        ),
    }


# SM107 (Rubin, VR). World 2 / 4 / 8 / 16 tuned on VR;
@cache
def _sm107_configs(num_sm: int) -> dict[int, AllReduceConfig]:
    mc_blocks = {4: 128, 5: 128, 6: 128, 7: 128, 8: 96, 16: 32}

    def config(
        world_size: int, *, graph: tuple, eager: tuple, two_shot_push: int = 0
    ) -> AllReduceConfig:
        return AllReduceConfig(
            graph=_pack_heuristic(*graph)._replace(
                two_shot_push_threshold=two_shot_push
            ),
            eager=_pack_heuristic(*eager)._replace(
                two_shot_push_threshold=two_shot_push
            ),
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
            graph=(1.000 * MB, 1.000 * MB, 128.0 * MB),
            eager=(1.000 * MB, 1.000 * MB, 128.0 * MB, Range(9.812 * MB, 128 * MB)),
            two_shot_push=16 * MB,
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
            graph=(384 * KB, 384 * KB, 128.0 * MB, Range(2.438 * MB, 128 * MB)),
            eager=(384 * KB, 384 * KB, 128.0 * MB, Range(2.438 * MB, 128 * MB)),
            two_shot_push=16 * MB,
        ),
        16: config(
            16,
            graph=(256 * KB, 256 * KB, 128.0 * MB, Range(832 * KB, 128 * MB)),
            eager=(256 * KB, 256 * KB, 128.0 * MB, Range(832 * KB, 128 * MB)),
            two_shot_push=16 * MB,
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


# ROCm (MI350X / gfx950). Block counts tuned at TP2/4/6/8; the thresholds are
# still the conservative default.
@cache
def _rocm_configs(num_sm: int) -> dict[int, AllReduceConfig]:
    default = AllReduceConfig(
        graph=Heuristic(1 * MB, 1 * MB, 16 * MB),
        eager=Heuristic(1 * MB, 1 * MB, 16 * MB),
        num_push_blocks=num_sm,
        # each pull block runs its own cross-GPU barrier: 32 blocks are 40-65%
        # faster than one per CU at TP8 from 1 MB, and within a few % of the
        # best grid at TP2/4/6
        num_pull_blocks=32,
        num_mc_blocks=None,
    )
    return {world_size: default for world_size in range(2, 9)}


# ROCm (MI450 / gfx1250, 96 CUs). Every size up to 4 GB stays on carv2:
# 1shot_push, then 2shot_push, then ring_simple. The 2shot_push and ring_simple
# cutoffs are the last size each won on h13-13 (DPX, graph, bf16, median of 3,
# back to back, TP2-8). 1shot_push wins back to back up to 256 KB-8 MB, but
# stops at 64 KB: with a late rank and a GEMM before the all-reduce, as in
# serving, 2shot_push wins from 128 KB, and at TP8 a 256 KB cutoff cost
# DeepSeek-V4-Flash 18% of its C16 decode throughput. Pull stays off: it lost
# at every measured size.
@cache
def _rocm_gfx1250_configs(
    num_sm: int, dev_kernarg: bool = False
) -> dict[int, AllReduceConfig]:
    def config(one_shot: int, two_shot_push: int) -> AllReduceConfig:
        # 4 GB keeps every dtype's element count in 32 bits
        ring = Range(max(one_shot, two_shot_push) + 16, 4 * 1024 * MB)
        heuristic = Heuristic(
            one_shot, 0, 0, two_shot_push_threshold=two_shot_push, ring_simple=ring
        )
        return AllReduceConfig(
            graph=heuristic,
            eager=heuristic,
            num_push_blocks=min(num_sm, 64),
            num_pull_blocks=32,
            num_mc_blocks=None,
            # one ring channel per CU
            num_ring_simple_channels=num_sm,
        )

    # world size: (1shot_push up to, 2shot_push up to). TP8 stops 2shot_push at
    # 8 MB although it wins back to back to 32 MB: serving DeepSeek-V4-Flash,
    # its 16 MB prefill all-reduces halved prefill throughput, and ring_simple
    # there matches RCCL.
    cutoffs = {
        2: (64 * KB, 8 * MB),
        3: (64 * KB, 8 * MB),
        4: (64 * KB, 16 * MB),
        5: (64 * KB, 8 * MB),
        6: (64 * KB, 16 * MB),
        7: (64 * KB, 16 * MB),
        8: (64 * KB, 8 * MB),
    }
    # That stall needs kernel arguments in host memory (the cookbook's
    # HIP_FORCE_DEV_KERNARG=0). With HIP_FORCE_DEV_KERNARG=1 the kernels after
    # a 2shot_push run at full speed, so TP2/4/8 take 2shot_push to the last
    # size it wins back to back. 1shot_push also serves decode further, to the
    # last size it wins in CUDA graphs with a late rank or a GEMM before it
    # (768 KB at TP8, every decode size at TP2/4): DeepSeek-V4-Flash decode
    # gains about 2-4% at TP4/TP8.
    if dev_kernarg:
        cutoffs.update(
            {2: (2 * MB, 16 * MB), 4: (2 * MB, 32 * MB), 8: (768 * KB, 32 * MB)}
        )
    configs = {ws: config(*cutoff) for ws, cutoff in cutoffs.items()}
    # 16 ranks span two DPX hosts in one UALink fabric clique. Tuned there with
    # and without a late rank: across hosts 2shot_push pays two fabric hops, so
    # 1shot_push wins to 256 KB, and a block per CU keeps 2shot_push ahead of
    # RCCL to 64 MB. When every SoC's two DPX partitions are in the group,
    # hier_2shot_push reduces each pair first, so only half the tensor crosses
    # the fabric: above 256 KB it beats 2shot_push by 17-31% back to back and
    # 9-14% with a late rank, so it takes over to 64 MB. ring_simple takes
    # the sizes above that.
    tp16 = Heuristic(
        256 * KB,
        0,
        0,
        two_shot_push_threshold=64 * MB,
        hier_2shot_push=Range(256 * KB, 64 * MB),
        ring_simple=Range(64 * MB + 16, 4 * 1024 * MB),
    )
    configs[16] = configs[8]._replace(
        graph=tp16, eager=tp16, num_push_blocks=min(num_sm, 96)
    )
    return configs


@cache
def _get_all_reduce_configs() -> dict[int, AllReduceConfig]:
    props = torch.cuda.get_device_properties()
    num_sm = props.multi_processor_count
    if torch.version.hip:
        # ROCm reports gfx9xx as capability 9.x: never fall into the SM90 table
        if props.gcnArchName.startswith("gfx1250"):
            dev_kernarg = os.environ.get("HIP_FORCE_DEV_KERNARG") == "1"
            return _rocm_gfx1250_configs(num_sm, dev_kernarg)
        return _rocm_configs(num_sm)
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
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


@cache
def get_all_reduce_config(world_size: int) -> AllReduceConfig:
    """Tuned thresholds and block counts for the current arch / world size.

    Only SM90, SM100 and SM107 are benchmarked so far; other archs get a
    conservative default (1 MB one-shot crossovers, no multicast).
    """
    return _get_all_reduce_configs()[world_size]
