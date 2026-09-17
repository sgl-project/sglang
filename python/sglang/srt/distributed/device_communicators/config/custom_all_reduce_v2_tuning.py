"""Hand-tuned all-reduce crossovers, per arch and world size.

Each entry lists, per dispatch context, the size at which each algo stops being
the fastest -- in priority order, not in size order. Numbers come from sweeps of
``test/registered/kernels/benchmark/communication/bench_custom_all_reduce.py``
on the listed GPUs. ``_RawTuning.compile`` turns them into the band table the
hot path scans.

NOTE: never import this module unless you know what you are doing -- go through
the package, which hands out the compiled table.
"""

from functools import cache
from typing import Any, Mapping, NamedTuple, Optional

import torch

from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo

from .custom_all_reduce_v2_dispatch import Band, Dispatch, RawTuningSpec, Tuning

KB, MB = 1024, 1024 * 1024
UNTUNED_MAX = 128 * MB


def _make_band_default_policy(
    one_shot_push: float,
    one_shot_pull: float,
    two_shot_pull: float,
    # TODO: support more ways of expressing multicast
    # currently, we only support two shot pull + multicast
    two_shot_pull_multicast_range: Optional[tuple[float, float]] = None,
    *,
    can_use_graph: bool,
    can_use_multicast: bool,
) -> tuple[Band, ...]:
    # Default policy assume such a order
    assert one_shot_push <= one_shot_pull <= two_shot_pull
    bands = [
        Band(int(one_shot_push), Dispatch(AllReduceAlgo.ONE_SHOT_PUSH)),
        Band(int(one_shot_pull), Dispatch(AllReduceAlgo.ONE_SHOT_PULL, can_use_graph)),
        Band(int(two_shot_pull), Dispatch(AllReduceAlgo.TWO_SHOT_PULL, can_use_graph)),
    ]
    if can_use_multicast and two_shot_pull_multicast_range:
        # Multicast wins inside its range only; plain 2shot still covers the
        # sizes below it, and above it when the tuned ceiling reaches further.
        last = bands[-1]
        mc_min, mc_max = (int(v) for v in two_shot_pull_multicast_range)
        mc_dispatch = Dispatch(AllReduceAlgo.TWO_SHOT_PULL, False, True)
        bands.insert(-1, Band(max_bytes=mc_min - 1, dispatch=last.dispatch))
        bands.insert(-1, Band(max_bytes=mc_max, dispatch=mc_dispatch))
    return Band.normalize_list(bands)


# NOTE: this is only for the internal usage, the API may change easily
class _RawTuning(NamedTuple):
    graph: tuple[Any, ...]
    eager: tuple[Any, ...]
    num_push_blocks: int
    num_pull_blocks: int
    num_multicast_blocks: Optional[int]

    def compile(self, can_use_multicast: bool) -> Tuning:
        return Tuning(
            eager=_make_band_default_policy(
                *self.eager,
                can_use_graph=False,
                can_use_multicast=can_use_multicast,
            ),
            graph=_make_band_default_policy(
                *self.graph,
                can_use_graph=True,
                can_use_multicast=can_use_multicast,
            ),
            num_push_blocks=self.num_push_blocks,
            num_pull_blocks=self.num_pull_blocks,
            num_multicast_blocks=(
                self.num_multicast_blocks if can_use_multicast else None
            ),
        )


# SM90 (Hopper, H100/H200). Tuned on H200.
@cache
def _sm90_tunings(num_sm: int) -> dict[int, _RawTuning]:
    def tuning(world_size: int, *, graph: tuple, eager: tuple):
        return _RawTuning(
            graph=graph,
            eager=eager,
            num_push_blocks=num_sm,
            num_pull_blocks=64,
            num_multicast_blocks=None if world_size < 4 else 128 // world_size,
        )

    return {
        2: tuning(
            2,
            graph=(16.00 * MB, UNTUNED_MAX, UNTUNED_MAX),
            eager=(32.00 * MB, UNTUNED_MAX, UNTUNED_MAX),
        ),
        3: tuning(
            3,
            graph=(1.250 * MB, 1.250 * MB, UNTUNED_MAX),
            eager=(3.000 * MB, 3.000 * MB, 16.00 * MB),
        ),
        4: tuning(
            4,
            graph=(384.0 * KB, 384.0 * KB, UNTUNED_MAX),
            eager=(896.0 * KB, 896.0 * KB, 32.00 * MB, (0, 32 * MB)),
        ),
        5: tuning(
            5,
            graph=(192.0 * KB, 192.0 * KB, 32.00 * MB),
            eager=(384.0 * KB, 384.0 * KB, 32.00 * MB, (0, 32 * MB)),
        ),
        6: tuning(
            6,
            graph=(128.0 * KB, 128.0 * KB, 32.00 * MB, (8 * MB, 32 * MB)),
            eager=(192.0 * KB, 192.0 * KB, 32.00 * MB, (0, 32 * MB)),
        ),
        7: tuning(
            7,
            graph=(128.0 * KB, 128.0 * KB, 32.00 * MB, (1 * MB, 32 * MB)),
            eager=(128.0 * KB, 128.0 * KB, 32.00 * MB, (0, 32 * MB)),
        ),
        8: tuning(
            8,
            graph=(128.0 * KB, 128.0 * KB, 32.00 * MB, (512 * KB, UNTUNED_MAX)),
            eager=(128.0 * KB, 128.0 * KB, UNTUNED_MAX, (0, UNTUNED_MAX)),
        ),
    }


# SM100 (Blackwell, B200/B300/GB200). Tuned on B200 (148 SMs); world 16 on GB200.
@cache
def _sm100_tunings(num_sm: int) -> dict[int, _RawTuning]:
    mc_blocks = {5: 64, 6: 48, 7: 48, 8: 32, 16: 32}

    def tuning(world_size: int, *, graph: tuple, eager: tuple):
        return _RawTuning(
            graph=graph,
            eager=eager,
            num_push_blocks=num_sm,
            num_pull_blocks=num_sm if world_size == 2 else 96,
            num_multicast_blocks=mc_blocks.get(world_size),
        )

    return {
        2: tuning(
            2,
            graph=(8.000 * MB, 32.00 * MB, UNTUNED_MAX),
            eager=(16.00 * MB, UNTUNED_MAX, UNTUNED_MAX),
        ),
        3: tuning(
            3,
            graph=(4.000 * MB, 4.000 * MB, UNTUNED_MAX),
            eager=(8.000 * MB, 8.000 * MB, 32.00 * MB),
        ),
        4: tuning(
            4,
            graph=(2.250 * MB, 2.250 * MB, UNTUNED_MAX),
            eager=(3.000 * MB, 3.000 * MB, 48.00 * MB),
        ),
        5: tuning(
            5,
            graph=(1.500 * MB, 1.500 * MB, UNTUNED_MAX),
            eager=(2.000 * MB, 2.000 * MB, 48.00 * MB, (0, 56 * MB)),
        ),
        6: tuning(
            6,
            graph=(1.000 * MB, 1.000 * MB, UNTUNED_MAX),
            eager=(1.250 * MB, 1.250 * MB, 64.00 * MB, (0, 64 * MB)),
        ),
        7: tuning(
            7,
            graph=(0.625 * MB, 0.625 * MB, UNTUNED_MAX),
            eager=(1.000 * MB, 1.000 * MB, 64.00 * MB, (0, 64 * MB)),
        ),
        8: tuning(
            8,
            graph=(0.500 * MB, 0.500 * MB, UNTUNED_MAX, (8 * MB, UNTUNED_MAX)),
            eager=(0.750 * MB, 0.750 * MB, UNTUNED_MAX, (0, UNTUNED_MAX)),
        ),
        16: tuning(
            16,
            graph=(0.250 * MB, 0.250 * MB, UNTUNED_MAX, (256 * KB, UNTUNED_MAX)),
            eager=(0.250 * MB, 0.250 * MB, UNTUNED_MAX, (256 * KB, UNTUNED_MAX)),
        ),
    }


# SM107 (Rubin, VR)
@cache
def _sm107_tunings(num_sm: int) -> dict[int, _RawTuning]:
    mc_blocks = {4: 128, 5: 128, 6: 128, 7: 128, 8: 96, 16: 32}

    def tuning(world_size: int, *, graph: tuple, eager: tuple):
        return _RawTuning(
            graph=graph,
            eager=eager,
            num_push_blocks=num_sm,
            num_pull_blocks=min(192, num_sm),
            num_multicast_blocks=mc_blocks.get(world_size),
        )

    return {
        2: tuning(
            2,
            graph=(UNTUNED_MAX, UNTUNED_MAX, UNTUNED_MAX),
            eager=(UNTUNED_MAX, UNTUNED_MAX, UNTUNED_MAX),
        ),
        3: tuning(
            3,
            graph=(19.625 * MB, 19.625 * MB, UNTUNED_MAX),
            eager=(39.188 * MB, 39.188 * MB, UNTUNED_MAX),
        ),
        4: tuning(
            4,
            graph=(6.938 * MB, 6.938 * MB, UNTUNED_MAX),
            eager=(9.812 * MB, 9.812 * MB, UNTUNED_MAX, (9.812 * MB, UNTUNED_MAX)),
        ),
        5: tuning(
            5,
            graph=(6.938 * MB, 6.938 * MB, UNTUNED_MAX, (6.938 * MB, UNTUNED_MAX)),
            eager=(6.938 * MB, 6.938 * MB, UNTUNED_MAX, (6.938 * MB, UNTUNED_MAX)),
        ),
        6: tuning(
            6,
            graph=(4.875 * MB, 4.875 * MB, UNTUNED_MAX, (4.875 * MB, UNTUNED_MAX)),
            eager=(4.875 * MB, 4.875 * MB, UNTUNED_MAX, (4.875 * MB, UNTUNED_MAX)),
        ),
        7: tuning(
            7,
            graph=(3.438 * MB, 3.438 * MB, UNTUNED_MAX, (3.438 * MB, UNTUNED_MAX)),
            eager=(3.438 * MB, 3.438 * MB, UNTUNED_MAX, (3.438 * MB, UNTUNED_MAX)),
        ),
        8: tuning(
            8,
            graph=(2.438 * MB, 2.438 * MB, UNTUNED_MAX, (2.438 * MB, UNTUNED_MAX)),
            eager=(2.438 * MB, 2.438 * MB, UNTUNED_MAX, (2.438 * MB, UNTUNED_MAX)),
        ),
        16: tuning(
            16,
            graph=(832.0 * KB, 832.0 * KB, UNTUNED_MAX, (832 * KB, UNTUNED_MAX)),
            eager=(832.0 * KB, 832.0 * KB, UNTUNED_MAX, (832 * KB, UNTUNED_MAX)),
        ),
    }


@cache
def get_raw_tunings() -> Mapping[int, RawTuningSpec]:
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    num_sm = torch.cuda.get_device_properties().multi_processor_count
    if cuda_major == 9:
        return _sm90_tunings(num_sm)
    if cuda_major == 10:
        if cuda_minor >= 7:
            return _sm107_tunings(num_sm)
        return _sm100_tunings(num_sm)

    default = _RawTuning(
        graph=(128 * KB, 128 * KB, 16 * MB),
        eager=(128 * KB, 128 * KB, 16 * MB),
        num_push_blocks=num_sm,
        num_pull_blocks=num_sm,
        num_multicast_blocks=None,
    )
    return {world_size: default for world_size in range(2, 17)}


__all__ = ["get_raw_tunings"]
