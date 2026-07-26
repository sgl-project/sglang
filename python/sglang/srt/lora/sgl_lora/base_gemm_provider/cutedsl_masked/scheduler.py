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

"""
Masked MoE Persistent Tile Scheduler

A specialized tile scheduler for MoE (Mixture of Experts) grouped GEMM operations.
This scheduler handles tile iteration across all experts, producing MoEWorkTileInfo
(expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt) for each tile.

This masked adaptation consumes per-expert valid row counts directly
instead of cumulative contiguous-row offsets.  That lets the GEMM keep its
fixed-stride ``[expert, m_max, ...]`` tensors and a single 3-D TMA descriptor
while the scheduler enumerates only tiles intersecting valid rows.

Scenarios:
- 2Dx3D (Forward): A(tokens_sum, hidden) x B(experts, intermediate, hidden) -> C(tokens_sum, intermediate)
- 2Dx2D (Backward): A(intermediate, tokens_sum) x B(hidden, tokens_sum) -> C(experts, intermediate, hidden)

Key design principle:
- Scheduler is ONLY responsible for tile iteration (tensor-agnostic, TMA-agnostic)
- Domain conversion (fake tensor -> real expert tensor) is handled by MoESchedExtension
- TMA descriptor management is handled by OnlineTensormapDescCreator
- The kernel orchestrates all three components
"""

import math
from typing import List, Literal, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    Integer,
    const_expr,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)


@cute.jit
def _warp_prefix_sum(value: Int32, lane: Int32) -> Int32:
    """Inclusive prefix sum across one warp."""
    for index in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << index
        partial = cute.arch.shuffle_sync_up(value, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            value += partial
    return value


# =============================================================================
# Work Tile Info
# =============================================================================


class MoEWorkTileInfo:
    """
    Work tile information for MoE scheduler.

    Contains CTA-level tile information for executor warps:
    - expert_idx: Which expert (-1 means invalid/done)
    - tile_m_idx: CTA tile index along GEMM M dimension
    - tile_n_idx: CTA tile index along GEMM N dimension
    - k_tile_cnt: Number of CTA tiles along K dimension

    Note: These are CTA-level indices, not cluster-level.
    tile_l_idx is always 0 for MoE, executor can hardcode it.

    For 2Dx3D (Forward):
        M = tokens_i (dynamic), N = intermediate (fixed), K = hidden (fixed)

    For 2Dx2D (Backward):
        M = intermediate (fixed), N = hidden (fixed), K = tokens_i (dynamic)
    """

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
        """Check if this is a valid work tile (expert_idx >= 0)."""
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


# =============================================================================
# Scheduler Parameters
# =============================================================================


class MoEStaticSchedulerParams:
    """
    Parameters for MoE tile scheduler.

    Uses unified semantics for both scenarios:
    - expert_shape: (expert_cnt, intermediate, hidden)

    For 2Dx3D: GEMM is (M=tokens_i, N=intermediate, K=hidden) per expert
    For 2Dx2D: GEMM is (M=hidden, N=intermediate, K=tokens_i) per expert

    Tile hierarchy:
    - cta_tile_shape_mnk: Single CTA tile shape (tile_m, tile_n, tile_k)
    - cluster_shape_mn: CTAs per cluster (cluster_m, cluster_n)
    - cluster_tile_shape_mn: Cluster tile shape = cta_tile_shape * cluster_shape

    This class is used both on host (for grid shape calculation) and on device
    (stored in scheduler). Codegen-time constants (scenario, cta_tile_shape_mnk,
    cluster_shape_mn) are NOT serialized to MLIR values.
    """

    def __init__(
        self,
        scenario: Literal["2Dx3D", "2Dx2D"],
        expert_shape: Tuple[
            int | Int32, int | Int32, int | Int32
        ],  # (expert_cnt, intermediate, hidden)
        cta_tile_shape_mnk: Tuple[int, int, int],  # (tile_m, tile_n, tile_k)
        cluster_shape_mn: Tuple[int, int],  # (cluster_m, cluster_n)
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
        """Cluster tile size along M = cta_tile_m * cluster_m."""
        return self.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0]

    @property
    def cluster_tile_n(self) -> int:
        """Cluster tile size along N = cta_tile_n * cluster_n."""
        return self.cta_tile_shape_mnk[1] * self.cluster_shape_mn[1]

    @property
    def cta_tile_k(self) -> int:
        """CTA tile size along K (same as cluster since cluster_k = 1)."""
        return self.cta_tile_shape_mnk[2]

    def __extract_mlir_values__(self) -> List[ir.Value]:
        """Only serialize runtime values, not codegen-time constants."""
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
        Compute grid shape for kernel launch.

        Since host doesn't know token distribution across experts,
        we launch max_active_clusters and let device-side scheduler
        determine which tiles are valid.
        """
        return (
            params.cluster_shape_mn[0],
            params.cluster_shape_mn[1],
            max_active_clusters,
        )


# =============================================================================
# Scheduler (Device-side)
# =============================================================================


class MoEStaticPersistentTileScheduler:
    """
    Persistent tile scheduler specialized for MoE grouped GEMM.

    NOTE: the shipped provider config always sets ``direct_schedule=True`` and
    compiles :class:`MoEDirectPersistentTileScheduler` instead; this on-device
    enumerating variant is kept as the fallback schedule source for the
    planned SM90 port (plan section 54), where the host-built direct schedule
    is the part most likely to need rework first.

    This scheduler is ONLY responsible for tile iteration. It does NOT know
    about tensor types, TMA descriptors, or domain conversion. Those concerns
    are handled by the consuming kernel.

    The scheduler handles:
    - 2Dx3D: Dynamic M per expert (from offs), fixed N (intermediate) and K (hidden)
    - 2Dx2D: Fixed M (intermediate) and N (hidden), dynamic K per expert (reduction axis)

    HOW THE SHIPPED KERNEL USES IT (corrected on review; an earlier version of
    this docstring described a scheduler-warp-broadcasts-through-smem design
    that ``kernel.py`` does NOT implement). The scheduler is constructed once
    OUTSIDE any warp predicate, and each specialized warp group then walks the
    same tile stream independently:

        scheduler = MoEStaticPersistentTileScheduler.create(params, offs, block_idx, grid_dim)
        work_tile = scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # ... this warp group's share of the work ...
            work_tile = scheduler.advance_to_next_work()

    There is no smem broadcast and no named barrier for work distribution, so
    the enumeration is REPLICATED across all six warps (TMA, MMA, and four
    epilogue warps). Two consequences: every warp must derive an identical tile
    sequence or the mainloop pipelines deadlock — which holds because the
    mapping is a pure function of ``offs`` (or of the packed schedule), so
    neither may be mutated while the kernel runs; and whatever per-tile
    resolution costs here is paid six times, which is the strongest argument
    for :class:`MoEDirectPersistentTileScheduler`.
    """

    def __init__(
        self,
        # Params (contains scenario, expert_cnt, intermediate, hidden, tile/cluster shapes)
        params: MoEStaticSchedulerParams,
        # Runtime tensor for scheduling
        offs: cute.Tensor,  # (experts,) valid row counts
        # Scheduling state
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        # Expert tracking state (for O(1) advance within same expert)
        current_expert_idx: Int32,
        expert_tile_start: Int32,  # cumsum of tiles before current expert
        expert_tile_end: Int32,  # cumsum of tiles including current expert
    ):
        self.params = params
        self.offs = offs
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self.cta_id_in_cluster = cta_id_in_cluster
        # Expert tracking
        self.current_expert_idx = current_expert_idx
        self.expert_tile_start = expert_tile_start
        self.expert_tile_end = expert_tile_end

    # =========================================================================
    # Convenience accessors for params
    # =========================================================================

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

    # =========================================================================
    # MLIR value serialization (for SSA value passing in device code)
    # =========================================================================

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        # Params (only runtime values are extracted)
        values.extend(extract_mlir_values(self.params))
        # Runtime tensor for scheduling
        values.extend(extract_mlir_values(self.offs))
        # Scheduling state
        values.extend(extract_mlir_values(self.num_persistent_clusters))
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        # Expert tracking state
        values.extend(extract_mlir_values(self.current_expert_idx))
        values.extend(extract_mlir_values(self.expert_tile_start))
        values.extend(extract_mlir_values(self.expert_tile_end))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "MoEStaticPersistentTileScheduler":
        idx = 0

        # Params (3 values: expert_cnt, intermediate, hidden)
        new_params = new_from_mlir_values(self.params, values[idx : idx + 3])
        idx += 3

        # Runtime tensor for scheduling (variable size)
        offs_len = len(extract_mlir_values(self.offs))
        new_offs = new_from_mlir_values(self.offs, values[idx : idx + offs_len])
        idx += offs_len

        # Scheduling state
        new_num_persistent_clusters = new_from_mlir_values(
            self.num_persistent_clusters, [values[idx]]
        )
        idx += 1
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[idx]]
        )
        idx += 1

        # cta_id_in_cluster (3 values for Coord)
        new_cta_id_in_cluster = new_from_mlir_values(
            self.cta_id_in_cluster, values[idx : idx + 3]
        )
        idx += 3

        # Expert tracking state
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

    # =========================================================================
    # Factory method
    # =========================================================================

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
        """
        Create a MoE persistent tile scheduler.

        :param params: Scheduler parameters (from host)
        :param offs: Valid row count per expert, shape (experts,)
        :param block_idx: CUDA block index
        :param grid_dim: CUDA grid dimensions
        """
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

        # Initialize expert tracking to "before expert 0"
        # The first call to _get_work_tile_for_linear_idx will advance to the correct expert
        current_expert_idx = Int32(0)
        expert_tile_start = Int32(0)
        expert_tile_end = Int32(0)  # Will be computed on first access

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

    # =========================================================================
    # Tile iteration methods
    # =========================================================================

    @dsl_user_op
    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> MoEWorkTileInfo:
        """Get the initial work tile info."""
        return self._get_work_tile_for_linear_idx(
            self._current_work_linear_idx, loc=loc, ip=ip
        )

    @dsl_user_op
    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None) -> MoEWorkTileInfo:
        """Advance to the next work tile and return its info."""
        self._current_work_linear_idx += self.num_persistent_clusters
        return self._get_work_tile_for_linear_idx(
            self._current_work_linear_idx, loc=loc, ip=ip
        )

    @dsl_user_op
    @cute.jit
    def _get_work_tile_for_linear_idx(
        self, cluster_linear_idx: Int32, *, loc=None, ip=None
    ) -> MoEWorkTileInfo:
        """
        Convert a linear cluster index to MoEWorkTileInfo.

        Uses cached expert tracking state for O(1) fast path when staying
        within the same expert. Advances expert state when needed.

        Returns an invalid tile (expert_idx = -1) if cluster_linear_idx is out of range.
        """
        # Resolve the expert either with the original cached serial walk or a
        # warp-cooperative scan over 32 expert counts at a time.
        if const_expr(self.params.uniform_m is not None):
            self._resolve_uniform_expert(cluster_linear_idx, loc=loc, ip=ip)
        elif const_expr(self.params.use_warp_scan):
            self._warp_scan_expert(cluster_linear_idx, loc=loc, ip=ip)
        else:
            self._advance_expert_to_contain(cluster_linear_idx, loc=loc, ip=ip)

        # Check if valid (still within expert range after advancing)
        is_valid = self.current_expert_idx < self.expert_cnt

        work_tile_info = MoEWorkTileInfo(
            expert_idx=Int32(-1),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            k_tile_cnt=Int32(0),
        )

        if is_valid:
            # Compute local cluster tile indices within current expert
            local_idx = cluster_linear_idx - self.expert_tile_start
            cluster_tile_m_idx, cluster_tile_n_idx = self._decompose_local_idx(
                local_idx, self.current_expert_idx, loc=loc, ip=ip
            )

            # Convert cluster tile indices to CTA tile indices
            # cta_tile_idx = cluster_tile_idx * cluster_shape + cta_id_in_cluster
            cta_tile_m_idx = (
                cluster_tile_m_idx * self.cluster_shape_mn[0]
                + self.cta_id_in_cluster[0]  # type: ignore[index]
            )
            cta_tile_n_idx = (
                cluster_tile_n_idx * self.cluster_shape_mn[1]
                + self.cta_id_in_cluster[1]  # type: ignore[index]
            )
            # Compute k_tile_cnt
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
        """Resolve a linear work index with a warp-cooperative expert scan."""
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
        """
        Advance expert tracking state until current expert contains cluster_linear_idx,
        or we run out of experts.

        Fast path: If already in correct expert, no work needed.
        """
        # Initialize expert_tile_end if this is the first call (expert_tile_end == 0)
        if self.expert_tile_end == Int32(0):
            tiles_for_expert_0 = self._compute_tiles_for_expert(
                Int32(0), loc=loc, ip=ip
            )
            self.expert_tile_end = tiles_for_expert_0

        # Advance until cluster_linear_idx < expert_tile_end or no more experts
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
        """Compute total cluster tiles for a given expert."""
        if const_expr(self.scenario == "2Dx2D"):
            # Fixed M=hidden, N=intermediate
            cluster_tile_m_cnt = (
                self.hidden + self.cluster_tile_m - 1
            ) // self.cluster_tile_m
            cluster_tile_n_cnt = (
                self.intermediate + self.cluster_tile_n - 1
            ) // self.cluster_tile_n
            return cluster_tile_m_cnt * cluster_tile_n_cnt
        else:  # 2Dx3D
            # Variable M (valid rows), fixed N.  Unlike the upstream
            # contiguous scheduler, ``offs`` is a count vector here.
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
        Decompose local cluster tile index within expert to (cluster_tile_m_idx, cluster_tile_n_idx).

        Uses "short side first" strategy: the shorter dimension changes faster.
        This maximizes overlap between adjacent clusters for better L2 cache utilization.

        For example, if m_cnt=2, n_cnt=8:
        - N is longer, so M changes faster: local_idx = n_idx * m_cnt + m_idx
        - Linearization order: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2), ...
        """
        # Get tile counts for M and N
        cluster_tile_m_cnt, cluster_tile_n_cnt = self._get_cluster_tile_counts(
            expert_idx, loc=loc, ip=ip
        )
        cluster_tile_m_idx = -1
        cluster_tile_n_idx = -1

        # Short side first: shorter dimension changes faster
        # If m_cnt <= n_cnt: m is shorter, m changes faster
        #   local_idx = n_idx * m_cnt + m_idx
        # If n_cnt < m_cnt: n is shorter, n changes faster
        #   local_idx = m_idx * n_cnt + n_idx
        if cluster_tile_m_cnt <= cluster_tile_n_cnt:
            # M is shorter or equal, M changes faster
            cluster_tile_m_idx = local_idx % cluster_tile_m_cnt
            cluster_tile_n_idx = local_idx // cluster_tile_m_cnt
        else:
            # N is shorter, N changes faster
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
        """Get (cluster_tile_m_cnt, cluster_tile_n_cnt) for a given expert."""
        if const_expr(self.scenario == "2Dx2D"):
            # Fixed M=hidden, N=intermediate
            cluster_tile_m_cnt = (
                self.hidden + self.cluster_tile_m - 1
            ) // self.cluster_tile_m
            cluster_tile_n_cnt = (
                self.intermediate + self.cluster_tile_n - 1
            ) // self.cluster_tile_n
        else:  # 2Dx3D
            # Variable M (valid rows), fixed N. ``offs`` stores counts. The
            # uniform diagnostic specialization removes the metadata load.
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
        """
        Compute the number of K tiles for this expert.

        2Dx3D: K = hidden (fixed) -> k_tile_cnt = ceil(hidden / cta_tile_k)
        2Dx2D: K = tokens_i (variable) -> k_tile_cnt = ceil(tokens_i / cta_tile_k)
        """
        if const_expr(self.scenario == "2Dx3D"):
            # K is hidden (fixed)
            return (self.hidden + self.cta_tile_k - 1) // self.cta_tile_k
        else:  # 2Dx2D
            # K is tokens_i (variable per expert)
            tokens_i = self.offs[expert_idx]
            if expert_idx > cutlass.Int32(0):
                tokens_i = tokens_i - self.offs[expert_idx - 1]  # type: ignore[operator]
            return (tokens_i + self.cta_tile_k - 1) // self.cta_tile_k  # type: ignore[return-value, operator]


class MoEDirectPersistentTileScheduler:
    """Persistent scheduler over a routing-produced compact cluster map.

    ``schedule[i]`` stores one packed int32 containing
    ``(expert, cluster_m, cluster_n)`` in ``(10, 10, 12)`` bits. Building this
    map belongs to routing/permutation and is deliberately outside the GEMM
    boundary, just like TRTLLM's CTA-to-expert map. The GEMM performs one
    coalesced int32 load per work tile instead of scanning every expert count
    independently in each specialized warp.
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
            packed = Int32(self.schedule[self._current_work_linear_idx])
            expert = packed & Int32(0x3FF)
            cluster_m = (packed >> Int32(10)) & Int32(0x3FF)
            cluster_n = (packed >> Int32(20)) & Int32(0xFFF)
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
