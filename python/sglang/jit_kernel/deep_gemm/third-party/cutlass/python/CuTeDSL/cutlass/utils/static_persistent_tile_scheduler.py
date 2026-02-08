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

##############################################################################
# Static persistent tile scheduler
##############################################################################


class WorkTileInfo:
    """A class to represent information about a work tile.

    :ivar tile_idx: The index of the tile.
    :type tile_idx: cute.Coord
    :ivar is_valid_tile: Whether the tile is valid.
    :type is_valid_tile: Boolean
    """

    def __init__(self, tile_idx: cute.Coord, is_valid_tile: Boolean):
        self._tile_idx = tile_idx
        self._is_valid_tile = Boolean(is_valid_tile)

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.tile_idx)
        values.extend(extract_mlir_values(self.is_valid_tile))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 4
        new_tile_idx = new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)

    @property
    def is_valid_tile(self) -> Boolean:
        """Check latest tile returned by the scheduler is valid or not. Any scheduling
        requests after all tasks completed will return an invalid tile.

        :return: The validity of the tile.
        :rtype: Boolean
        """
        return self._is_valid_tile

    @property
    def tile_idx(self) -> cute.Coord:
        """
        Get the index of the tile.

        :return: The index of the tile.
        :rtype: cute.Coord
        """
        return self._tile_idx


class PersistentTileSchedulerParams:
    """A class to represent parameters for a persistent tile scheduler.

    This class is designed to manage and compute the layout of clusters and tiles
    in a batched gemm problem.

    :ivar cluster_shape_mn: Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).
    :type cluster_shape_mn: tuple
    :ivar problem_layout_ncluster_mnl: Layout of the problem in terms of
        number of clusters in (m, n, l) dimensions.
    :type problem_layout_ncluster_mnl: cute.Layout
    """

    def __init__(
        self,
        problem_shape_ntile_mnl: cute.Shape,
        cluster_shape_mnk: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        """
        Initializes the PersistentTileSchedulerParams with the given parameters.

        :param problem_shape_ntile_mnl: The shape of the problem in terms of
            number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
        :type problem_shape_ntile_mnl: cute.Shape
        :param cluster_shape_mnk: The shape of the cluster in (m, n) dimensions.
        :type cluster_shape_mnk: cute.Shape

        :raises ValueError: If cluster_shape_k is not 1.
        """

        if cluster_shape_mnk[2] != 1:
            raise ValueError(f"unsupported cluster_shape_k {cluster_shape_mnk[2]}")

        self.problem_shape_ntile_mnl = problem_shape_ntile_mnl
        # cluster_shape_mnk is kept for reconstruction
        self._cluster_shape_mnk = cluster_shape_mnk
        self.cluster_shape_mn = cluster_shape_mnk[:2]
        self._loc = loc

        # By default, we follow m major (col-major) raster order, so make a col-major layout
        self.problem_layout_ncluster_mnl = cute.make_layout(
            cute.ceil_div(
                self.problem_shape_ntile_mnl, cluster_shape_mnk[:2], loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.problem_shape_ntile_mnl, self._cluster_shape_mnk]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.problem_shape_ntile_mnl, self._cluster_shape_mnk], self._values_pos
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return PersistentTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)

    @dsl_user_op
    def get_grid_shape(
        self, max_active_clusters: Int32, *, loc=None, ip=None
    ) -> Tuple[Integer, Integer, Integer]:
        """
        Computes the grid shape based on the maximum active clusters allowed.

        :param max_active_clusters: The maximum number of active clusters that
            can run in one wave.
        :type max_active_clusters: Int32

        :return: A tuple containing the grid shape in (m, n, persistent_clusters).
            - m: self.cluster_shape_m.
            - n: self.cluster_shape_n.
            - persistent_clusters: Number of persistent clusters that can run.
        """

        # Total ctas in problem size
        num_ctas_mnl = tuple(
            x * y
            for x, y in zip(
                self.problem_layout_ncluster_mnl.shape, self.cluster_shape_mn
            )
        ) + (self.problem_layout_ncluster_mnl.shape[2],)

        num_ctas_in_problem = cute.size(num_ctas_mnl, loc=loc, ip=ip)

        num_ctas_per_cluster = cute.size(self.cluster_shape_mn, loc=loc, ip=ip)
        # Total ctas that can run in one wave
        num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster

        num_persistent_ctas = min(num_ctas_in_problem, num_ctas_per_wave)
        num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster

        return (*self.cluster_shape_mn, num_persistent_clusters)


class StaticPersistentTileScheduler:
    """A scheduler for static persistent tile execution in CUTLASS/CuTe kernels.

    :ivar params: Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl
    :type params: PersistentTileSchedulerParams
    :ivar num_persistent_clusters: Number of persistent clusters that can be launched
    :type num_persistent_clusters: Int32
    :ivar cta_id_in_cluster: ID of the CTA within its cluster
    :type cta_id_in_cluster: cute.Coord
    :ivar _num_tiles_executed: Counter for executed tiles
    :type _num_tiles_executed: Int32
    :ivar _current_work_linear_idx: Current cluster index
    :type _current_work_linear_idx: Int32
    """

    def __init__(
        self,
        params: PersistentTileSchedulerParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
    ):
        """
        Initializes the StaticPersistentTileScheduler with the given parameters.

        :param params: Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl.
        :type params: PersistentTileSchedulerParams
        :param num_persistent_clusters: Number of persistent clusters that can be launched.
        :type num_persistent_clusters: Int32
        :param current_work_linear_idx: Current cluster index.
        :type current_work_linear_idx: Int32
        :param cta_id_in_cluster: ID of the CTA within its cluster.
        :type cta_id_in_cluster: cute.Coord
        :param num_tiles_executed: Counter for executed tiles.
        :type num_tiles_executed: Int32
        """
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self.cta_id_in_cluster = cta_id_in_cluster
        self._num_tiles_executed = num_tiles_executed

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.num_persistent_clusters)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        return values

    def __new_from_mlir_values__(
        self, values: list[ir.Value]
    ) -> "StaticPersistentTileScheduler":
        assert len(values) == 6
        new_num_persistent_clusters = new_from_mlir_values(
            self.num_persistent_clusters, [values[0]]
        )
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[1]]
        )
        new_cta_id_in_cluster = new_from_mlir_values(
            self.cta_id_in_cluster, values[2:5]
        )
        new_num_tiles_executed = new_from_mlir_values(
            self._num_tiles_executed, [values[5]]
        )
        return StaticPersistentTileScheduler(
            self.params,
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
        )

    # called by host
    @dsl_user_op
    @staticmethod
    def create(
        params: PersistentTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        *,
        loc=None,
        ip=None,
    ):
        """Initialize the static persistent tile scheduler.

        :param params: Parameters for the persistent
            tile scheduler.
        :type params: PersistentTileSchedulerParams
        :param block_idx: The 3d block index in the format (bidx, bidy, bidz).
        :type block_idx: Tuple[Integer, Integer, Integer]
        :param grid_dim: The 3d grid dimensions for kernel launch.
        :type grid_dim: Tuple[Integer, Integer, Integer]

        :return: A StaticPersistentTileScheduler object.
        :rtype: StaticPersistentTileScheduler
        """
        params = params

        # Calculate the number of persistent clusters by dividing the total grid size
        # by the number of CTAs per cluster
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(
            params.cluster_shape_mn, loc=loc, ip=ip
        )

        bidx, bidy, bidz = block_idx

        # Initialize workload index equals to the cluster index in the grid
        current_work_linear_idx = Int32(bidz)

        # CTA id in the cluster
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        # Initialize number of tiles executed to zero
        num_tiles_executed = Int32(0)
        return StaticPersistentTileScheduler(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: PersistentTileSchedulerParams,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Integer, Integer, Integer]:
        """Calculates the grid shape to be launched on GPU using problem shape,
        threadblock shape, and active cluster size.

        :param params: Parameters for grid shape calculation.
        :type params: PersistentTileSchedulerParams
        :param max_active_clusters: Maximum active clusters allowed.
        :type max_active_clusters: Int32

        :return: The calculated 3d grid shape.
        :rtype: Tuple[Integer, Integer, Integer]
        """

        return params.get_grid_shape(max_active_clusters, loc=loc, ip=ip)

    # private method
    def _get_current_work_for_linear_idx(
        self, current_work_linear_idx: Int32, *, loc=None, ip=None
    ) -> WorkTileInfo:
        """Compute current tile coord given current_work_linear_idx and cta_id_in_cluster.

        :param current_work_linear_idx: The linear index of the current work.
        :type current_work_linear_idx: Int32

        :return: An object containing information about the current tile coordinates
            and validity status.
        :rtype: WorkTileInfo
        """

        is_valid = current_work_linear_idx < cute.size(
            self.params.problem_layout_ncluster_mnl, loc=loc, ip=ip
        )

        cur_cluster_coord = self.params.problem_layout_ncluster_mnl.get_hier_coord(
            current_work_linear_idx, loc=loc, ip=ip
        )

        # cur_tile_coord is a tuple of i32 values
        cur_tile_coord = tuple(
            Int32(x) * Int32(z) + Int32(y)
            for x, y, z in zip(
                cur_cluster_coord,
                self.cta_id_in_cluster,
                (*self.params.cluster_shape_mn, Int32(1)),
            )
        )

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
            self.num_persistent_clusters
        )
        self._num_tiles_executed += Int32(1)

    @property
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed


