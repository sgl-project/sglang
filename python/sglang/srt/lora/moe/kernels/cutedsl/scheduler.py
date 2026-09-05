# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Persistent scheduler for the packed work list from schedule_builder."""

from typing import List, Tuple

import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    Int64,
    Integer,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)

from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
    EXPERT_MASK,
    OUTPUT_CLUSTER_MASK,
    OUTPUT_CLUSTER_SHIFT,
    TOKEN_CLUSTER_MASK,
    TOKEN_CLUSTER_SHIFT,
)


class MoEWorkTileInfo:
    def __init__(
        self,
        expert_idx: Int32,  # -1 means invalid tile
        tile_m_idx: Int32,
        tile_n_idx: Int32,
        k_tile_cnt: Int32,
    ):
        self.expert_idx = expert_idx
        self.tile_m_idx = tile_m_idx
        self.tile_n_idx = tile_n_idx
        self.k_tile_cnt = k_tile_cnt

    @property
    def is_valid_tile(self) -> Boolean:
        return self.expert_idx >= Int32(0)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = extract_mlir_values(self.expert_idx)
        values.extend(extract_mlir_values(self.tile_m_idx))
        values.extend(extract_mlir_values(self.tile_n_idx))
        values.extend(extract_mlir_values(self.k_tile_cnt))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MoEWorkTileInfo":
        assert len(values) == 4
        return MoEWorkTileInfo(
            expert_idx=new_from_mlir_values(self.expert_idx, [values[0]]),
            tile_m_idx=new_from_mlir_values(self.tile_m_idx, [values[1]]),
            tile_n_idx=new_from_mlir_values(self.tile_n_idx, [values[2]]),
            k_tile_cnt=new_from_mlir_values(self.k_tile_cnt, [values[3]]),
        )


class MoESchedulerParams:
    def __init__(
        self,
        expert_shape: Tuple[
            int | Int32, int | Int32, int | Int32
        ],  # (expert_cnt, intermediate, hidden)
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ):
        e, i, h = expert_shape
        self.expert_cnt = e if isinstance(e, Int32) else Int32(e)
        self.intermediate = i if isinstance(i, Int32) else Int32(i)
        self.hidden = h if isinstance(h, Int32) else Int32(h)
        self.cta_tile_shape_mnk = cta_tile_shape_mnk
        self.cluster_shape_mn = cluster_shape_mn

    @property
    def cluster_tile_m(self) -> int:
        return self.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0]

    @property
    def cta_tile_k(self) -> int:
        return self.cta_tile_shape_mnk[2]

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.expert_cnt))
        values.extend(extract_mlir_values(self.intermediate))
        values.extend(extract_mlir_values(self.hidden))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MoESchedulerParams":
        assert len(values) == 3
        return MoESchedulerParams(
            expert_shape=(
                new_from_mlir_values(self.expert_cnt, [values[0]]),
                new_from_mlir_values(self.intermediate, [values[1]]),
                new_from_mlir_values(self.hidden, [values[2]]),
            ),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
        )

    @staticmethod
    def get_grid_shape(
        params: "MoESchedulerParams",
        max_active_clusters: int,
    ) -> Tuple[int, int, int]:
        # Device row counts determine which persistent CTAs have work.
        return (
            params.cluster_shape_mn[0],
            params.cluster_shape_mn[1],
            max_active_clusters,
        )


class MoEDirectPersistentTileScheduler:
    """Decode (expert, cluster_m, cluster_n) from one int64 per tile."""

    def __init__(
        self,
        params: MoESchedulerParams,
        schedule: cute.Tensor,
        schedule_tiles: cute.Tensor,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
    ):
        self.params = params
        self.schedule = schedule
        self.schedule_tiles = schedule_tiles
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self.cta_id_in_cluster = cta_id_in_cluster

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.params))
        values.extend(extract_mlir_values(self.schedule))
        values.extend(extract_mlir_values(self.schedule_tiles))
        values.extend(extract_mlir_values(self.num_persistent_clusters))
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "MoEDirectPersistentTileScheduler":
        idx = 0
        new_params = new_from_mlir_values(self.params, values[idx : idx + 3])
        idx += 3
        schedule_len = len(extract_mlir_values(self.schedule))
        new_schedule = new_from_mlir_values(
            self.schedule, values[idx : idx + schedule_len]
        )
        idx += schedule_len
        tiles_len = len(extract_mlir_values(self.schedule_tiles))
        new_schedule_tiles = new_from_mlir_values(
            self.schedule_tiles, values[idx : idx + tiles_len]
        )
        idx += tiles_len
        new_num_clusters = new_from_mlir_values(
            self.num_persistent_clusters, [values[idx]]
        )
        idx += 1
        new_work_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[idx]]
        )
        idx += 1
        new_cta_id = new_from_mlir_values(self.cta_id_in_cluster, values[idx : idx + 3])
        return MoEDirectPersistentTileScheduler(
            new_params,
            new_schedule,
            new_schedule_tiles,
            new_num_clusters,
            new_work_idx,
            new_cta_id,
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: MoESchedulerParams,
        schedule: cute.Tensor,
        schedule_tiles: cute.Tensor,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        *,
        loc=None,
        ip=None,
    ) -> "MoEDirectPersistentTileScheduler":
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(
            params.cluster_shape_mn, loc=loc, ip=ip
        )
        bidx, bidy, bidz = block_idx
        return MoEDirectPersistentTileScheduler(
            params,
            schedule,
            schedule_tiles,
            Int32(num_persistent_clusters),
            Int32(bidz),
            (
                Int32(bidx % params.cluster_shape_mn[0]),
                Int32(bidy % params.cluster_shape_mn[1]),
                Int32(0),
            ),
        )

    @dsl_user_op
    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> MoEWorkTileInfo:
        return self._get_current_work(loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None) -> MoEWorkTileInfo:
        self._current_work_linear_idx += self.num_persistent_clusters
        return self._get_current_work(loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def _get_current_work(self, *, loc=None, ip=None) -> MoEWorkTileInfo:
        work = MoEWorkTileInfo(Int32(-1), Int32(0), Int32(0), Int32(0))
        if self._current_work_linear_idx < self.schedule_tiles[0]:
            packed = Int64(self.schedule[self._current_work_linear_idx])
            expert = Int32(packed & Int64(EXPERT_MASK))
            cluster_m = Int32(
                (packed >> Int64(TOKEN_CLUSTER_SHIFT)) & Int64(TOKEN_CLUSTER_MASK)
            )
            cluster_n = Int32(
                (packed >> Int64(OUTPUT_CLUSTER_SHIFT)) & Int64(OUTPUT_CLUSTER_MASK)
            )
            tile_m = (
                cluster_m * self.params.cluster_shape_mn[0]
                + self.cta_id_in_cluster[0]  # type: ignore[index]
            )
            tile_n = (
                cluster_n * self.params.cluster_shape_mn[1]
                + self.cta_id_in_cluster[1]  # type: ignore[index]
            )
            k_tiles = (
                self.params.hidden + self.params.cta_tile_k - 1
            ) // self.params.cta_tile_k
            work = MoEWorkTileInfo(expert, tile_m, tile_n, Int32(k_tiles))
        return work


def resolve_scheduler_params_and_grid(
    *,
    a: cute.Tensor,
    c: cute.Tensor,
    cta_tile_shape_mnk,
    cluster_shape_mn,
    swap_ab: bool,
    contiguous_segments: bool,
    persistent_clusters,
    max_active_clusters,
):
    """Keep tokens on scheduler M, transposing tile/cluster axes under swap_ab."""
    m_max, n, expert_cnt = c.shape
    if contiguous_segments:
        # Flat output has one slot; the weight retains the real expert count.
        expert_cnt = cute.size(a.shape[2])
    # K can be hierarchical; the scheduler needs its scalar extent.
    k = cute.size(a.shape[1])
    cta_tile = cta_tile_shape_mnk
    cluster = cluster_shape_mn
    sched_n = n
    if swap_ab:
        cta_tile = (
            cta_tile_shape_mnk[1],
            cta_tile_shape_mnk[0],
            cta_tile_shape_mnk[2],
        )
        cluster = (cluster_shape_mn[1], cluster_shape_mn[0])
        sched_n = m_max
    params = MoESchedulerParams(
        expert_shape=(expert_cnt, sched_n, k),
        cta_tile_shape_mnk=cta_tile,
        cluster_shape_mn=cluster,
    )
    launch_clusters = (
        max_active_clusters if persistent_clusters is None else persistent_clusters
    )
    grid = MoESchedulerParams.get_grid_shape(params, launch_clusters)
    if swap_ab:
        grid = (cluster_shape_mn[0], cluster_shape_mn[1], launch_clusters)
    return params, grid


def create_moe_tile_scheduler(
    *,
    tile_sched_params,
    direct_schedule: cute.Tensor,
    schedule_tiles: cute.Tensor,
    swap_ab: bool,
):
    """Restore scheduler axis order under swap_ab at trace time."""
    scheduler_block_idx = cute.arch.block_idx()
    scheduler_grid_dim = cute.arch.grid_dim()
    if swap_ab:
        scheduler_block_idx = (
            scheduler_block_idx[1],
            scheduler_block_idx[0],
            scheduler_block_idx[2],
        )
        scheduler_grid_dim = (
            scheduler_grid_dim[1],
            scheduler_grid_dim[0],
            scheduler_grid_dim[2],
        )
    return MoEDirectPersistentTileScheduler.create(
        tile_sched_params,
        direct_schedule,
        schedule_tiles,
        scheduler_block_idx,
        scheduler_grid_dim,
    )
