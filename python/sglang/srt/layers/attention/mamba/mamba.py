import os
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np
import torch

from sglang.srt.distributed import divide

LoaderFunction = Callable[[torch.Tensor, torch.Tensor], None]


def mamba_v2_sharded_weight_loader(
    shard_spec: List[Tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
) -> LoaderFunction:
    """Create a weight loader for mamba v2. This ensures that the projections
    are correctly sharded so that they can be split into x, B, C. It also
    ensures the the all the groups corresponding to a head shard is placed
    together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:

        # - track boundary of (sharded) param, and loaded_weight, respectively
        boundary, loaded_boundary = 0, 0

        # - iterate over the shard specs
        for full_dim, extra, duplicate_groups in shard_spec:
            # - full dim is the model dim (before TP).
            # - extra > 0, means there is expected overall increase
            #   of dimensions. This is so because of replication.
            # - ratio is used map the tp_rank to the actual shard
            #   rank. This is useful when there is replication of
            #   groups to accompany head shards.

            # - size of the loaded shard
            shard_size = full_dim // tp_size

            # - compute the rank into the loaded shard.
            # - if there is replication, different TP shards will
            #   take from the same rank.
            # NOTE: currently we only support duplication
            # in the case where num_groups == 1
            rank = 0 if duplicate_groups else tp_rank

            # - leftmost boundary index into loaded weight.
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip

            # - take these many dims from the loaded weight.
            take = min(shard_size, full_dim - extra - loaded_skip)

            # - always shard on dim 0
            # - the ignore is for a mundane mypy error as it does not
            #   seem to handle slices well.
            # https://github.com/python/mypy/issues/2410
            param.data[
                boundary : (boundary + take), ...  # type: ignore[misc]
            ] = loaded_weight[
                loaded_start_idx : (loaded_start_idx + take)  # type: ignore[misc]
            ]  # type: ignore[misc]

            # move indexing boundaries
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


def extra_groups_for_head_shards(ngroups: int, tp_size: int):
    """Compute the increase in group numbers to account for
    replication in order to accompany the head shards."""

    # in the case ngoups % tp_size == 0, this will be zero
    if ngroups % tp_size == 0:
        return 0

    # for n_groups == 1, this is exactly tp_size - n_groups
    return tp_size - ngroups


@dataclass(kw_only=True, frozen=True)
class Mamba2StateShape:
    conv: tuple[int, int]
    temporal: tuple[int, int, int]


@dataclass(kw_only=True, frozen=True)
class Mamba2StateDType:
    conv: torch.dtype
    temporal: torch.dtype


CONV_DTYPE = torch.bfloat16


def mamba2_state_dtype() -> Mamba2StateDType:
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    ssm_dtype = dtype_map[os.environ["SGLANG_MAMBA_SSM_DTYPE"]]
    return Mamba2StateDType(conv=CONV_DTYPE, temporal=ssm_dtype)


@dataclass(kw_only=True, frozen=True)
class Mamba2CacheParams:
    shape: Mamba2StateShape
    dtype: Mamba2StateDType = field(default_factory=mamba2_state_dtype)
    layers: list[int]

    @staticmethod
    def shape(
        *,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
    ) -> Mamba2StateShape:
        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = n_groups + extra_groups_for_head_shards(n_groups, tp_world_size)
        # heads and n_groups are TP-ed
        conv_dim = intermediate_size + 2 * n_groups * state_size

        # contiguous along 'dim' axis
        conv_state_shape = divide(conv_dim, tp_world_size), conv_kernel - 1

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, head_dim, state_size) = (128, 64, 128)
        temporal_state_shape = (divide(num_heads, tp_world_size), head_dim, state_size)
        return Mamba2StateShape(conv=conv_state_shape, temporal=temporal_state_shape)

    @property
    def mamba_cache_per_req(self) -> int:
        return (
            int(np.prod(self.shape.conv)) * self.dtype.conv.itemsize
            + int(np.prod(self.shape.temporal)) * self.dtype.temporal.itemsize
        ) * len(self.layers)
