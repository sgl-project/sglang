# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

from typing import List, Tuple

import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, extract_mlir_values, new_from_mlir_values
from cutlass._mlir import ir

from cutlass.utils.static_persistent_tile_scheduler import PersistentTileSchedulerParams


class GroupSearchResult:
    """
    The result of the group search for grouped gemm.

    :param group_idx: The result group index
    :type group_idx: Int32
    :param cta_tile_idx_m: CTA tile index along M dimension after rasterization
    :type cta_tile_idx_m: Int32
    :param cta_tile_idx_n: CTA tile index along N dimension after rasterization
    :type cta_tile_idx_n: Int32
    :param problem_shape_m: The M dimension of the gemm problem
    :type problem_shape_m: Int32
    :param problem_shape_n: The N dimension of the gemm problem
    :type problem_shape_n: Int32
    :param problem_shape_k: The K dimension of the gemm problem
    :type problem_shape_k: Int32
    :param cta_tile_count_k: Number of tiles along K dimension
    :type cta_tile_count_k: Int32
    """

    def __init__(
        self,
        group_idx: Int32,
        cta_tile_idx_m: Int32,
        cta_tile_idx_n: Int32,
        problem_shape_m: Int32,
        problem_shape_n: Int32,
        problem_shape_k: Int32,
        cta_tile_count_k: Int32,
    ) -> None:
        self.group_idx = group_idx
        self.cta_tile_idx_m = cta_tile_idx_m
        self.cta_tile_idx_n = cta_tile_idx_n
        self.problem_shape_m = problem_shape_m
        self.problem_shape_n = problem_shape_n
        self.problem_shape_k = problem_shape_k
        self.cta_tile_count_k = cta_tile_count_k

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = extract_mlir_values(self.group_idx)
        values.extend(extract_mlir_values(self.cta_tile_idx_m))
        values.extend(extract_mlir_values(self.cta_tile_idx_n))
        values.extend(extract_mlir_values(self.problem_shape_m))
        values.extend(extract_mlir_values(self.problem_shape_n))
        values.extend(extract_mlir_values(self.problem_shape_k))
        values.extend(extract_mlir_values(self.cta_tile_count_k))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "GroupSearchResult":
        assert len(values) == 7
        return GroupSearchResult(*tuple(values))


class GroupedGemmGroupSearchState:
    """
    The state of group index search for grouped gemm.

    The state will be initialized once and updated in every round of group index search.

    :param start_group_idx: The group idx to start the search with
    :type start_group_idx: Int32
    :param tile_count_prev_group: Number of tiles before the matched group
    :type tile_count_prev_group: Int32
    :param tile_count_searched: Number of tiles we have searched. When the matched group is found,
                               it records the number of tiles including the matched group
    :type tile_count_searched: Int32
    """

    def __init__(
        self,
        start_group_idx: Int32,
        tile_count_prev_group: Int32,
        tile_count_searched: Int32,
    ) -> None:
        self.start_group_idx = start_group_idx
        self.tile_count_prev_group = tile_count_prev_group
        self.tile_count_searched = tile_count_searched

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = extract_mlir_values(self.start_group_idx)
        values.extend(extract_mlir_values(self.tile_count_prev_group))
        values.extend(extract_mlir_values(self.tile_count_searched))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "GroupedGemmGroupSearchState":
        start_group_idx = new_from_mlir_values(self.start_group_idx, [values[0]])
        tile_count_prev_group = new_from_mlir_values(
            self.tile_count_prev_group, [values[1]]
        )
        tile_count_searched = new_from_mlir_values(
            self.tile_count_searched, [values[2]]
        )
        return GroupedGemmGroupSearchState(
            start_group_idx, tile_count_prev_group, tile_count_searched
        )


def create_initial_search_state() -> GroupedGemmGroupSearchState:
    """
    Create an initial search state for grouped gemm.

    :return: A new search state with initial values
    :rtype: GroupedGemmGroupSearchState
    """
    return GroupedGemmGroupSearchState(
        start_group_idx=Int32(0),
        tile_count_prev_group=Int32(0),
        tile_count_searched=Int32(0),
    )


class GroupedGemmTileSchedulerHelper:
    """
    A helper to translate the raw block index (x, y, z) from tile scheduler to real CTA tile index for grouped gemm.

    :param group_count: Number of groups in current grouped gemm problem
    :type group_count: int
    :param tile_sched_params: Parameter used to create the tile scheduler this helper works with
    :type tile_sched_params: PersistentTileSchedulerParams
    :param cluster_tile_shape_mnk: The shape of cluster tile as (m, n, k)
    :type cluster_tile_shape_mnk: tuple[int, int, int]
    :param search_state: The initial search state
    :type search_state: GroupedGemmGroupSearchState
    """

    def __init__(
        self,
        group_count: int,
        tile_sched_params: PersistentTileSchedulerParams,
        cluster_tile_shape_mnk: tuple[int, int, int],
        search_state: GroupedGemmGroupSearchState,
    ) -> None:
        self.tile_sched_params = tile_sched_params
        self.group_count = group_count
        self.lane_idx = cute.arch.lane_idx()
        self.cluster_tile_shape_mnk = cluster_tile_shape_mnk
        self.search_state = search_state

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = extract_mlir_values(self.tile_sched_params)
        values.extend(extract_mlir_values(self.search_state))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "GroupedGemmTileSchedulerHelper":
        tile_sched_params = new_from_mlir_values(self.tile_sched_params, values)
        search_state = new_from_mlir_values(self.search_state, values[1:])
        return GroupedGemmTileSchedulerHelper(
            self.group_count,
            tile_sched_params,
            self.cluster_tile_shape_mnk,
            search_state,
        )

    def delinearize_z(
        self,
        cta_tile_coord: tuple,
        problem_shape_mnkl: cute.Tensor,
    ) -> GroupSearchResult:
        """
        Delinearize the linear z index and return GroupSearchResult.

        This function should be used by warps that need to know the CTA tile index on M and N dimensions.

        :param cta_tile_coord: The raw CTA coordinate from tile scheduler
        :type cta_tile_coord: tuple of Int32
        :param problem_shape_mnkl: Tensor containing gemm problem size (M, N, K, L) for each group
        :type problem_shape_mnkl: cute.Tensor
        :return: The search result containing group index and tile coordinates
        :rtype: GroupSearchResult
        """
        # delinear the z coord
        linear_idx = cta_tile_coord[2]
        group_idx, problem_mnkl = self._group_search_and_load_problem_shape(
            linear_idx,
            problem_shape_mnkl,
            self.search_state.start_group_idx,
            self.search_state.tile_count_prev_group,
        )
        # linear index local to current group
        cluster_tile_idx_in_current_group = (
            linear_idx - self.search_state.tile_count_prev_group
        )
        cluster_count_m, cluster_count_n, cluster_count_k = cute.ceil_div(
            (problem_mnkl[0], problem_mnkl[1], problem_mnkl[2]),
            (
                self.cluster_tile_shape_mnk[0],
                self.cluster_tile_shape_mnk[1],
                self.cluster_tile_shape_mnk[2],
            ),
        )
        # decompose to get indices on M and N
        cta_tile_idx_m, cta_tile_idx_n = self._compute_cta_tile_coord(
            cluster_tile_idx_in_current_group,
            cta_tile_coord,
            cluster_count_m,
            cluster_count_n,
        )
        return GroupSearchResult(
            group_idx,
            cta_tile_idx_m,
            cta_tile_idx_n,
            problem_mnkl[0],
            problem_mnkl[1],
            problem_mnkl[2],
            cluster_count_k,
        )

    def search_cluster_tile_count_k(
        self,
        cta_tile_coord: tuple,
        problem_shape_mnkl: cute.Tensor,
    ) -> Tuple[Int32, Int32]:
        """
        Search the matched group for given linear index and compute the number of tiles along K dimension for the matched group.

        This function should be used by warps that are only interested in the number of tiles along K dimension.

        :param cta_tile_coord: The raw CTA coordinate from tile scheduler
        :type cta_tile_coord: tuple of Int32
        :param problem_shape_mnkl: Tensor containing gemm problem size (M, N, K, L) for all groups
        :type problem_shape_mnkl: cute.Tensor
        :return: A tuple containing cluster count along K dimension and the group index
        :rtype: Tuple[Int32, Int32]
        """
        group_idx, problem_mnk = self._group_search_and_load_problem_shape(
            cta_tile_coord[2],
            problem_shape_mnkl,
            self.search_state.start_group_idx,
            self.search_state.tile_count_prev_group,
        )
        cluster_count_k = (
            problem_mnk[2] + self.cluster_tile_shape_mnk[2] - 1
        ) // self.cluster_tile_shape_mnk[2]
        return cluster_count_k, group_idx

    @cute.jit
    def _prefix_sum(self, value_per_thread: Int32) -> Int32:
        """
        Perform prefix sum within a full warp.

        :param value_per_thread: The value for this thread to contribute to the prefix sum
        :type value_per_thread: Int32
        :return: The prefix sum result for this thread
        :rtype: Int32
        """
        clamp_value = 0
        idx = 1
        sum_per_thread = value_per_thread
        while idx < cute.arch.WARP_SIZE:
            value = cute.arch.shuffle_sync_up(
                sum_per_thread, idx, mask_and_clamp=clamp_value
            )
            if self.lane_idx >= idx:
                sum_per_thread += value
            idx = idx << 1
        return sum_per_thread

    def _get_problem_for_group(
        self, problem_shape_mnkl: cute.Tensor, group_idx: Int32
    ) -> cute.Tensor:
        """
        Load gemm problem (m,n,k,l) for the specified group from global memory to register.

        :param problem_shape_mnkl: Tensor in global memory with layout (group_count, 4):(4, 1)
        :type problem_shape_mnkl: cute.Tensor
        :param group_idx: The index of the group to load
        :type group_idx: Int32
        :return: The problem shape tensor for the specified group
        :rtype: cute.Tensor
        """
        cur_problem_mnkl = cute.make_fragment(
            cute.make_layout(4), problem_shape_mnkl.element_type
        )
        cute.autovec_copy(problem_shape_mnkl[(group_idx, None)], cur_problem_mnkl)
        return cur_problem_mnkl

    def _get_cluster_tile_count_mn(self, problem_shape: cute.Tensor) -> Int32:
        """
        Compute total cluster count.

        :param problem_shape: Tensor containing problem shape (m, n, k, l)
        :type problem_shape: cute.Tensor
        :return: The total cluster tile count for M and N dimensions
        :rtype: Int32
        """
        cur_ntile_m = (
            problem_shape[0] + self.cluster_tile_shape_mnk[0] - 1
        ) // self.cluster_tile_shape_mnk[0]
        cur_ntile_n = (
            problem_shape[1] + self.cluster_tile_shape_mnk[1] - 1
        ) // self.cluster_tile_shape_mnk[1]
        cur_ntile_mn = cur_ntile_m * cur_ntile_n
        return cur_ntile_mn

    def _compute_cta_tile_coord(
        self,
        cluster_tile_idx: Int32,
        cta_tile_coord_in_cluster: tuple,
        cluster_tile_count_m: Int32,
        cluster_tile_count_n: Int32,
    ) -> tuple:
        """
        Compute CTA tile indices along M and N dimensions based on the linear index within a group.

        It uses the AlongM mode to decompose the linear index onto M and N dimensions.

        :param cluster_tile_idx: The linear index within a group
        :type cluster_tile_idx: Int32
        :param cta_tile_coord_in_cluster: CTA indices along M and N dimensions within a cluster
        :type cta_tile_coord_in_cluster: tuple of Int32
        :param cluster_tile_count_m: The number of clusters along M dimension of the matched group
        :type cluster_tile_count_m: Int32
        :param cluster_tile_count_n: The number of clusters along N dimension of the matched group
        :type cluster_tile_count_n: Int32
        :return: A tuple containing CTA tile indices along M and N dimensions
        :rtype: tuple of (Int32, Int32)
        """
        cluster_layout_mn = cute.make_layout(
            (cluster_tile_count_m, cluster_tile_count_n)
        )
        (mi, ni) = cluster_layout_mn.get_hier_coord(cluster_tile_idx)
        cta_tile_idx_m = (
            mi * self.tile_sched_params.cluster_shape_mn[0]
            + cta_tile_coord_in_cluster[0]
        )
        cta_tile_idx_n = (
            ni * self.tile_sched_params.cluster_shape_mn[1]
            + cta_tile_coord_in_cluster[1]
        )
        return (cta_tile_idx_m, cta_tile_idx_n)

    @cute.jit
    def _group_search(
        self,
        linear_idx: Int32,
        problem_shape_mnkl: cute.Tensor,
        init_group_idx: Int32,
        init_tile_count_searched: Int32,
    ) -> GroupedGemmGroupSearchState:
        """
        Search which group the linear index belongs to.

        :param linear_idx: The linear index to be decomposed
        :type linear_idx: Int32
        :param problem_shape_mnkl: Tensor containing gemm problem size (M, N, K, L) for all groups
        :type problem_shape_mnkl: cute.Tensor
        :param init_group_idx: The group idx to start the search with
        :type init_group_idx: Int32
        :param init_tile_count_searched: The number of tiles we have searched
        :type init_tile_count_searched: Int32
        :return: The updated search state
        :rtype: GroupedGemmGroupSearchState
        """
        c_0 = Int32(0).ir_value()
        last_lane_idx = cute.arch.WARP_SIZE - 1

        tile_count_searched = init_tile_count_searched
        start_group_idx = init_group_idx
        not_found = linear_idx >= tile_count_searched
        tile_count_prev_group = self.search_state.tile_count_prev_group
        while not_found:
            # get group to search for current lane
            cur_group_idx = start_group_idx + self.lane_idx
            # check if the group to be checked is out of range
            inside_group_bound = cur_group_idx < self.group_count
            cur_ntile_mn = c_0
            if inside_group_bound:
                # get problem size of current group
                cur_problem_mnkl = self._get_problem_for_group(
                    problem_shape_mnkl, cur_group_idx
                )
                cur_ntile_mn = self._get_cluster_tile_count_mn(cur_problem_mnkl)
            # compute tile count from beginning to current group(included)
            total_cluster_tile_count_ps_per_thread = self._prefix_sum(cur_ntile_mn)
            cluster_tile_count_end_per_thread = (
                total_cluster_tile_count_ps_per_thread + tile_count_searched
            )

            group_not_in_window = linear_idx >= cluster_tile_count_end_per_thread
            hitted_group_idx_in_search_window = cute.arch.popc(
                cute.arch.vote_ballot_sync(group_not_in_window)
            )
            not_found = hitted_group_idx_in_search_window == cute.arch.WARP_SIZE
            start_group_idx = hitted_group_idx_in_search_window + start_group_idx
            hit_the_1st_problem_in_search_window = (
                hitted_group_idx_in_search_window == c_0
            )
            tile_count_prev_group = tile_count_searched
            if hit_the_1st_problem_in_search_window == False:
                tile_count_prev_group = cute.arch.shuffle_sync(
                    cluster_tile_count_end_per_thread,
                    hitted_group_idx_in_search_window - 1,
                )

            # If no matched group, then get new_cluster_tile_count_end from last lane
            # Otherwise, get new_cluster_tile_count_end from the hitted group
            lane_idx_for_cluster_tile_count_end = hitted_group_idx_in_search_window
            if not_found:
                lane_idx_for_cluster_tile_count_end = last_lane_idx
            tile_count_searched = cute.arch.shuffle_sync(
                cluster_tile_count_end_per_thread,
                lane_idx_for_cluster_tile_count_end,
            )

        return GroupedGemmGroupSearchState(
            start_group_idx,
            tile_count_prev_group,
            tile_count_searched,
        )

    def _group_search_and_load_problem_shape(
        self,
        linear_idx: Int32,
        problem_shape_mnkl: cute.Tensor,
        start_group_idx: Int32,
        tile_count_searched: Int32,
    ) -> Tuple[Int32, cute.Tensor]:
        """
        Perform group search and load problem shape for the matched group.

        :param linear_idx: The linear index to be decomposed
        :type linear_idx: Int32
        :param problem_shape_mnkl: Tensor containing gemm problem size (M, N, K, L) for all groups
        :type problem_shape_mnkl: cute.Tensor
        :param start_group_idx: The group idx to start the search with
        :type start_group_idx: Int32
        :param tile_count_searched: The number of tiles we have searched
        :type tile_count_searched: Int32
        :return: A tuple containing the final group index and the problem shape tensor
        :rtype: Tuple[Int32, cute.Tensor]
        """
        self.search_state = self._group_search(
            linear_idx,
            problem_shape_mnkl,
            start_group_idx,
            tile_count_searched,
        )
        # get final group search state
        final_group_idx = self.search_state.start_group_idx
        # let's revisit if it's better to broadcast problem_shape_mnk in group_search
        problem_mnkl = self._get_problem_for_group(problem_shape_mnkl, final_group_idx)
        return final_group_idx, problem_mnkl
