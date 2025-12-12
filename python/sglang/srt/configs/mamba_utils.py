# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common config utils for mamba2 - NemotronH, FalconH1, Qwen3Next, etc."""

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from sglang.srt.distributed.utils import divide


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

    intermediate_size: int
    conv_dim: int
    ssm_state_size: int
    num_heads: int
    head_dim: int
    state_size: int
    conv_kernel: int

    @staticmethod
    def create(
        *,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
    ) -> "Mamba2StateShape":
        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        if n_groups % tp_world_size != 0:
            extra_groups = extra_groups_for_head_shards(n_groups, tp_world_size)
            n_groups += extra_groups
        # heads and n_groups are TP-ed
        conv_dim = intermediate_size + 2 * n_groups * state_size

        # contiguous along 'dim' axis
        conv_state_shape = divide(conv_dim, tp_world_size), conv_kernel - 1

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., QWen3-Next: (32, 128, 128)
        temporal_state_shape = (divide(num_heads, tp_world_size), head_dim, state_size)
        return Mamba2StateShape(
            conv=conv_state_shape,
            temporal=temporal_state_shape,
            intermediate_size=intermediate_size,
            conv_dim=conv_dim,
            ssm_state_size=state_size,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
            conv_kernel=conv_kernel,
        )


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

    @property
    def mamba_cache_per_req(self) -> int:
        return (
            int(np.prod(self.shape.conv)) * self.dtype.conv.itemsize
            + int(np.prod(self.shape.temporal)) * self.dtype.temporal.itemsize
        ) * len(self.layers)
