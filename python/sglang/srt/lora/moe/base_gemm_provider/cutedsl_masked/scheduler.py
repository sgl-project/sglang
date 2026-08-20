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

import math
from typing import List, Literal, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    Int64,
    Integer,
    const_expr,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)

from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_abi import (
    EXPERT_MASK,
    OUTPUT_CLUSTER_MASK,
    OUTPUT_CLUSTER_SHIFT,
    TOKEN_CLUSTER_MASK,
    TOKEN_CLUSTER_SHIFT,
)


@cute.jit
def _warp_prefix_sum(value: Int32, lane: Int32) -> Int32:
    for index in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << index
        partial = cute.arch.shuffle_sync_up(value, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            value += partial
    return value


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


class MoEStaticSchedulerParams:
    def __init__(
        self,
        scenario: Literal["2Dx3D", "2Dx2D"],
        expert_shape: Tuple[
            int | Int32, int | Int32, int | Int32
        ],  # (expert_cnt, intermediate, hidden)
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        use_warp_scan: bool = False,
        uniform_m: int | None = None,
    ):
        self.scenario = scenario
        e, i, h = expert_shape
        self.expert_cnt = e if isinstance(e, Int32) else Int32(e)
        self.intermediate = i if isinstance(i, Int32) else Int32(i)
        self.hidden = h if isinstance(h, Int32) else Int32(h)
        self.cta_tile_shape_mnk = cta_tile_shape_mnk
        self.cluster_shape_mn = cluster_shape_mn
        self.use_warp_scan = use_warp_scan
        self.uniform_m = uniform_m

    @property
    def cluster_tile_m(self) -> int:
        return self.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0]

    @property
    def cluster_tile_n(self) -> int:
        return self.cta_tile_shape_mnk[1] * self.cluster_shape_mn[1]

    @property
    def cta_tile_k(self) -> int:
        return self.cta_tile_shape_mnk[2]

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.expert_cnt))
        values.extend(extract_mlir_values(self.intermediate))
        values.extend(extract_mlir_values(self.hidden))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "MoEStaticSchedulerParams":
        assert len(values) == 3
        return MoEStaticSchedulerParams(
            scenario=self.scenario,
            expert_shape=(
                new_from_mlir_values(self.expert_cnt, [values[0]]),
                new_from_mlir_values(self.intermediate, [values[1]]),
                new_from_mlir_values(self.hidden, [values[2]]),
            ),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            use_warp_scan=self.use_warp_scan,
            uniform_m=self.uniform_m,
        )

    @staticmethod
    def get_grid_shape(
        params: "MoEStaticSchedulerParams",
        max_active_clusters: int,
    ) -> Tuple[int, int, int]:
        """
        The host does not know the token distribution across experts, so launch
        max_active_clusters and let the device-side scheduler pick valid tiles.
        """
        return (
            params.cluster_shape_mn[0],
            params.cluster_shape_mn[1],
            max_active_clusters,
        )


class MoEStaticPersistentTileScheduler:
    """
    The shipped provider config always sets ``direct_schedule=True`` and compiles
    :class:`MoEDirectPersistentTileScheduler` instead; this variant is kept as the
    fallback schedule source for the planned SM90 port (plan section 54), where
    the host-built direct schedule is most likely to need rework first.

    The kernel constructs it outside any warp predicate and does no smem
    broadcast, so enumeration is replicated across all six warps (TMA, MMA, four
    epilogue). Every warp must derive an identical tile sequence or the mainloop
    pipelines deadlock; that holds only because the mapping is a pure function of
    ``offs`` (or of the packed schedule), which must not be mutated while the
    kernel runs. Per-tile resolution cost is also paid six times, the strongest
    argument for :class:`MoEDirectPersistentTileScheduler`.
    """

    def __init__(
        self,
        params: MoEStaticSchedulerParams,
        offs: cute.Tensor,  # (experts,) valid row counts
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        current_expert_idx: Int32,
        expert_tile_start: Int32,
        expert_tile_end: Int32,
    ):
        self.params = params
        self.offs = offs
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self.cta_id_in_cluster = cta_id_in_cluster
        self.current_expert_idx = current_expert_idx
        self.expert_tile_start = expert_tile_start
        self.expert_tile_end = expert_tile_end

    @property
    def scenario(self) -> Literal["2Dx3D", "2Dx2D"]:
        return self.params.scenario

    @property
    def expert_cnt(self) -> Int32:
        return self.params.expert_cnt

    @property
    def intermediate(self) -> Int32:
        return self.params.intermediate

    @property
    def hidden(self) -> Int32:
        return self.params.hidden

    @property
    def cta_tile_shape_mnk(self) -> Tuple[int, int, int]:
        return self.params.cta_tile_shape_mnk

    @property
    def cluster_shape_mn(self) -> Tuple[int, int]:
        return self.params.cluster_shape_mn

    @property
    def cluster_tile_m(self) -> int:
        return self.params.cluster_tile_m

    @property
    def cluster_tile_n(self) -> int:
        return self.params.cluster_tile_n

    @property
    def cta_tile_k(self) -> int:
        return self.params.cta_tile_k

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.params))
        values.extend(extract_mlir_values(self.offs))
        values.extend(extract_mlir_values(self.num_persistent_clusters))
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self.current_expert_idx))
        values.extend(extract_mlir_values(self.expert_tile_start))
        values.extend(extract_mlir_values(self.expert_tile_end))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "MoEStaticPersistentTileScheduler":
        idx = 0

        new_params = new_from_mlir_values(self.params, values[idx : idx + 3])
        idx += 3

        offs_len = len(extract_mlir_values(self.offs))
        new_offs = new_from_mlir_values(self.offs, values[idx : idx + offs_len])
        idx += offs_len

        new_num_persistent_clusters = new_from_mlir_values(
            self.num_persistent_clusters, [values[idx]]
        )
        idx += 1
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[idx]]
        )
        idx += 1

        new_cta_id_in_cluster = new_from_mlir_values(
            self.cta_id_in_cluster, values[idx : idx + 3]
        )
        idx += 3

        new_current_expert_idx = new_from_mlir_values(
            self.current_expert_idx, [values[idx]]
        )
        idx += 1
        new_expert_tile_start = new_from_mlir_values(
            self.expert_tile_start, [values[idx]]
        )
        idx += 1
        new_expert_tile_end = new_from_mlir_values(self.expert_tile_end, [values[idx]])
        idx += 1

        return MoEStaticPersistentTileScheduler(
            params=new_params,
            offs=new_offs,
            num_persistent_clusters=new_num_persistent_clusters,
            current_work_linear_idx=new_current_work_linear_idx,
            cta_id_in_cluster=new_cta_id_in_cluster,
            current_expert_idx=new_current_expert_idx,
            expert_tile_start=new_expert_tile_start,
            expert_tile_end=new_expert_tile_end,
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: MoEStaticSchedulerParams,
        offs: cute.Tensor,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        *,
        loc=None,
        ip=None,
    ) -> "MoEStaticPersistentTileScheduler":
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(
            params.cluster_shape_mn, loc=loc, ip=ip
        )

        bidx, bidy, bidz = block_idx
        current_work_linear_idx = Int32(bidz)

        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )

        current_expert_idx = Int32(0)
        expert_tile_start = Int32(0)
        expert_tile_end = Int32(0)

        return MoEStaticPersistentTileScheduler(
            params=params,
            offs=offs,
            num_persistent_clusters=num_persistent_clusters,
            current_work_linear_idx=current_work_linear_idx,
            cta_id_in_cluster=cta_id_in_cluster,
            current_expert_idx=current_expert_idx,
            expert_tile_start=expert_tile_start,
            expert_tile_end=expert_tile_end,
        )

    @dsl_user_op
    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> MoEWorkTileInfo:
        return self._get_work_tile_for_linear_idx(
            self._current_work_linear_idx, loc=loc, ip=ip
        )

    @dsl_user_op
    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None) -> MoEWorkTileInfo:
        self._current_work_linear_idx += self.num_persistent_clusters
        return self._get_work_tile_for_linear_idx(
            self._current_work_linear_idx, loc=loc, ip=ip
        )

    @dsl_user_op
    @cute.jit
    def _get_work_tile_for_linear_idx(
        self, cluster_linear_idx: Int32, *, loc=None, ip=None
    ) -> MoEWorkTileInfo:
        if const_expr(self.params.uniform_m is not None):
            self._resolve_uniform_expert(cluster_linear_idx, loc=loc, ip=ip)
        elif const_expr(self.params.use_warp_scan):
            self._warp_scan_expert(cluster_linear_idx, loc=loc, ip=ip)
        else:
            self._advance_expert_to_contain(cluster_linear_idx, loc=loc, ip=ip)

        is_valid = self.current_expert_idx < self.expert_cnt

        work_tile_info = MoEWorkTileInfo(
            expert_idx=Int32(-1),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            k_tile_cnt=Int32(0),
        )

        if is_valid:
            local_idx = cluster_linear_idx - self.expert_tile_start
            cluster_tile_m_idx, cluster_tile_n_idx = self._decompose_local_idx(
                local_idx, self.current_expert_idx, loc=loc, ip=ip
            )

            cta_tile_m_idx = (
                cluster_tile_m_idx * self.cluster_shape_mn[0]
                + self.cta_id_in_cluster[0]  # type: ignore[index]
            )
            cta_tile_n_idx = (
                cluster_tile_n_idx * self.cluster_shape_mn[1]
                + self.cta_id_in_cluster[1]  # type: ignore[index]
            )
            k_tile_cnt = self._compute_k_tile_cnt(
                self.current_expert_idx, loc=loc, ip=ip
            )

            work_tile_info = MoEWorkTileInfo(
                expert_idx=self.current_expert_idx,
                tile_m_idx=cta_tile_m_idx,
                tile_n_idx=cta_tile_n_idx,
                k_tile_cnt=k_tile_cnt,
            )
        return work_tile_info

    @dsl_user_op
    @cute.jit
    def _resolve_uniform_expert(
        self,
        cluster_linear_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> None:
        """Diagnostic fast path when every expert has the same valid M."""
        cluster_tile_m_cnt = (
            self.params.uniform_m + self.cluster_tile_m - 1
        ) // self.cluster_tile_m
        cluster_tile_n_cnt = (
            self.intermediate + self.cluster_tile_n - 1
        ) // self.cluster_tile_n
        tiles_per_expert = cluster_tile_m_cnt * cluster_tile_n_cnt
        expert_idx = cluster_linear_idx // tiles_per_expert
        self.current_expert_idx = expert_idx
        self.expert_tile_start = expert_idx * tiles_per_expert
        self.expert_tile_end = self.expert_tile_start + tiles_per_expert

    @dsl_user_op
    @cute.jit
    def _warp_scan_expert(
        self,
        cluster_linear_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> None:
        lane = cute.arch.lane_idx()
        expert_base = Int32(0)
        tiles_before_window = Int32(0)
        found = Boolean(False)
        resolved_expert = Int32(self.expert_cnt)
        resolved_start = Int32(0)
        resolved_end = Int32(0)

        while expert_base < self.expert_cnt and not found:
            expert_idx = expert_base + lane
            expert_tiles = Int32(0)
            if expert_idx < self.expert_cnt:
                expert_tiles = self._compute_tiles_for_expert(
                    expert_idx, loc=loc, ip=ip
                )
            inclusive_tiles = _warp_prefix_sum(expert_tiles, lane)
            window_tiles = cute.arch.shuffle_sync(
                inclusive_tiles, cute.arch.WARP_SIZE - 1
            )
            if cluster_linear_idx < tiles_before_window + window_tiles:
                prior_expert_mask = cute.arch.vote_ballot_sync(
                    tiles_before_window + inclusive_tiles <= cluster_linear_idx
                )
                expert_in_window = cute.arch.popc(prior_expert_mask)
                local_start = Int32(0)
                if expert_in_window > Int32(0):
                    local_start = cute.arch.shuffle_sync(
                        inclusive_tiles, expert_in_window - 1
                    )
                selected_tiles = cute.arch.shuffle_sync(expert_tiles, expert_in_window)
                resolved_expert = expert_base + expert_in_window
                resolved_start = tiles_before_window + local_start
                resolved_end = resolved_start + selected_tiles
                found = Boolean(True)
            else:
                tiles_before_window += window_tiles
                expert_base += cute.arch.WARP_SIZE

        self.current_expert_idx = resolved_expert
        self.expert_tile_start = resolved_start
        self.expert_tile_end = resolved_end

    @dsl_user_op
    @cute.jit
    def _advance_expert_to_contain(
        self,
        cluster_linear_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> None:
        # expert_tile_end == 0 means uninitialized, i.e. this is the first call.
        if self.expert_tile_end == Int32(0):
            tiles_for_expert_0 = self._compute_tiles_for_expert(
                Int32(0), loc=loc, ip=ip
            )
            self.expert_tile_end = tiles_for_expert_0

        while (
            cluster_linear_idx >= self.expert_tile_end
            and self.current_expert_idx < self.expert_cnt
        ):
            self.current_expert_idx = self.current_expert_idx + 1
            self.expert_tile_start = self.expert_tile_end

            if self.current_expert_idx < self.expert_cnt:
                tiles_for_expert = self._compute_tiles_for_expert(
                    self.current_expert_idx, loc=loc, ip=ip
                )
                self.expert_tile_end = self.expert_tile_end + tiles_for_expert

    @dsl_user_op
    @cute.jit
    def _compute_tiles_for_expert(
        self,
        expert_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Int32:
        if const_expr(self.scenario == "2Dx2D"):
            cluster_tile_m_cnt = (
                self.hidden + self.cluster_tile_m - 1
            ) // self.cluster_tile_m
            cluster_tile_n_cnt = (
                self.intermediate + self.cluster_tile_n - 1
            ) // self.cluster_tile_n
            return cluster_tile_m_cnt * cluster_tile_n_cnt
        else:
            # Unlike the upstream contiguous scheduler, ``offs`` is a count
            # vector here.
            tokens_i = self.offs[expert_idx]
            cluster_tile_m_cnt = (
                tokens_i + self.cluster_tile_m - 1  # type: ignore[operator]
            ) // self.cluster_tile_m
            cluster_tile_n_cnt = (
                self.intermediate + self.cluster_tile_n - 1
            ) // self.cluster_tile_n
            return cluster_tile_m_cnt * cluster_tile_n_cnt

    @dsl_user_op
    @cute.jit
    def _decompose_local_idx(
        self,
        local_idx: Int32,
        expert_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32]:
        """
        Short side first: the shorter dimension changes faster, which maximizes
        overlap between adjacent clusters in L2.
        """
        cluster_tile_m_cnt, cluster_tile_n_cnt = self._get_cluster_tile_counts(
            expert_idx, loc=loc, ip=ip
        )
        cluster_tile_m_idx = -1
        cluster_tile_n_idx = -1

        if cluster_tile_m_cnt <= cluster_tile_n_cnt:
            cluster_tile_m_idx = local_idx % cluster_tile_m_cnt
            cluster_tile_n_idx = local_idx // cluster_tile_m_cnt
        else:
            cluster_tile_n_idx = local_idx % cluster_tile_n_cnt
            cluster_tile_m_idx = local_idx // cluster_tile_n_cnt

        return (cluster_tile_m_idx, cluster_tile_n_idx)

    @dsl_user_op
    @cute.jit
    def _get_cluster_tile_counts(
        self,
        expert_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32]:
        if const_expr(self.scenario == "2Dx2D"):
            cluster_tile_m_cnt = (
                self.hidden + self.cluster_tile_m - 1
            ) // self.cluster_tile_m
            cluster_tile_n_cnt = (
                self.intermediate + self.cluster_tile_n - 1
            ) // self.cluster_tile_n
        else:
            # ``offs`` stores counts; the uniform specialization drops that load.
            tokens_i = (
                self.params.uniform_m
                if const_expr(self.params.uniform_m is not None)
                else self.offs[expert_idx]
            )
            cluster_tile_m_cnt = (
                tokens_i + self.cluster_tile_m - 1  # type: ignore[operator]
            ) // self.cluster_tile_m
            cluster_tile_n_cnt = (
                self.intermediate + self.cluster_tile_n - 1
            ) // self.cluster_tile_n
        return (cluster_tile_m_cnt, cluster_tile_n_cnt)

    @dsl_user_op
    @cute.jit
    def _compute_k_tile_cnt(
        self,
        expert_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Int32:
        if const_expr(self.scenario == "2Dx3D"):
            return (self.hidden + self.cta_tile_k - 1) // self.cta_tile_k
        else:
            tokens_i = self.offs[expert_idx]
            if expert_idx > cutlass.Int32(0):
                tokens_i = tokens_i - self.offs[expert_idx - 1]  # type: ignore[operator]
            return (tokens_i + self.cta_tile_k - 1) // self.cta_tile_k  # type: ignore[return-value, operator]


class MoEDirectPersistentTileScheduler:
    """Persistent scheduler over a routing-produced compact cluster map.

    ``schedule[i]`` packs ``(expert, cluster_m, cluster_n)`` into one int64 at the
    field widths declared in ``schedule_abi``. Building the map belongs to
    routing/permutation and is deliberately outside the GEMM boundary, like
    TRTLLM's CTA-to-expert map, so the GEMM does one coalesced load per work tile
    instead of rescanning every expert count in each specialized warp.
    """

    def __init__(
        self,
        params: MoEStaticSchedulerParams,
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
        params: MoEStaticSchedulerParams,
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
            # Extracted in 64-bit, narrowed once: every field is far inside
            # int32 range by construction (see schedule_abi).
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
