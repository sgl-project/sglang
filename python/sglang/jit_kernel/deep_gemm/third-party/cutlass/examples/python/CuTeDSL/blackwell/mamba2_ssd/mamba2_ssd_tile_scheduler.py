# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Tuple

from cutlass.cutlass_dsl import (
    Boolean,
    Integer,
    Int32,
    min,
    extract_mlir_values,
    new_from_mlir_values,
    dsl_user_op,
)
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass.utils import WorkTileInfo


class Mamba2SSDTileSchedulerParams:
    def __init__(
        self,
        problem_shape_ntiles: int,
        eh: int,
        ngroup_ratio: int,
        *,
        loc=None,
        ip=None,
    ):
        self.problem_shape_ntiles = problem_shape_ntiles
        self.eh = eh
        self.ngroup_ratio = ngroup_ratio
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.problem_shape_ntiles, self.eh, self.ngroup_ratio]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.problem_shape_ntiles, self.eh, self.ngroup_ratio], self._values_pos
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return Mamba2SSDTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)

    @dsl_user_op
    def get_grid_shape(
        self, max_active_clusters: Int32, *, loc=None, ip=None
    ) -> Tuple[Integer, Integer, Integer]:
        return (min(self.problem_shape_ntiles, max_active_clusters), 1, 1)


class Mamba2SSDTileScheduler:
    def __init__(
        self,
        params: Mamba2SSDTileSchedulerParams,
        num_persistent_ctas: Int32,
        current_work_linear_idx: Int32,
        num_tiles_executed: Int32,
    ):
        self.params = params
        self.num_persistent_ctas = num_persistent_ctas
        self._current_work_linear_idx = current_work_linear_idx
        self._num_tiles_executed = num_tiles_executed

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.num_persistent_ctas)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        return values

    def __new_from_mlir_values__(
        self, values: list[ir.Value]
    ) -> "Mamba2SSDTileScheduler":
        assert len(values) == 3
        new_num_persistent_ctas = new_from_mlir_values(
            self.num_persistent_ctas, [values[0]]
        )
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[1]]
        )
        new_num_tiles_executed = new_from_mlir_values(
            self._num_tiles_executed, [values[2]]
        )
        return Mamba2SSDTileScheduler(
            self.params,
            new_num_persistent_ctas,
            new_current_work_linear_idx,
            new_num_tiles_executed,
        )

    # called by host
    @dsl_user_op
    @staticmethod
    def create(
        params: Mamba2SSDTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        *,
        loc=None,
        ip=None,
    ):
        params = params

        # Calculate the number of persistent clusters by dividing the total grid size
        # by the number of CTAs per cluster
        num_persistent_ctas = Int32(cute.size(grid_dim, loc=loc, ip=ip))

        bidx, bidy, bidz = block_idx

        # Initialize workload index equals to the cluster index in the grid
        current_work_linear_idx = Int32(bidx)

        # Initialize number of tiles executed to zero
        num_tiles_executed = Int32(0)
        return Mamba2SSDTileScheduler(
            params,
            num_persistent_ctas,
            current_work_linear_idx,
            num_tiles_executed,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Mamba2SSDTileSchedulerParams,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Integer, Integer, Integer]:
        return params.get_grid_shape(max_active_clusters, loc=loc, ip=ip)

    # private method
    def _get_current_work_for_linear_idx(
        self, current_work_linear_idx: Int32, *, loc=None, ip=None
    ) -> WorkTileInfo:
        is_valid = current_work_linear_idx < cute.size(
            self.params.problem_shape_ntiles, loc=loc, ip=ip
        )

        eh_idx = current_work_linear_idx % self.params.eh
        b_idx = current_work_linear_idx // self.params.eh
        g_idx = eh_idx // self.params.ngroup_ratio
        # cur_tile_coord is (b_idx, eh_idx, g_idx)
        cur_tile_coord = tuple(Int32(x) for x in (b_idx, eh_idx, g_idx))

        return WorkTileInfo(cur_tile_coord, is_valid)

    @dsl_user_op
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self._get_current_work_for_linear_idx(
            self._current_work_linear_idx, loc=loc, ip=ip
        )

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        self._current_work_linear_idx += Int32(advance_count) * Int32(
            self.num_persistent_ctas
        )
        self._num_tiles_executed += Int32(1)

    @property
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed
