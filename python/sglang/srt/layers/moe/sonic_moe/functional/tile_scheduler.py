# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from quack.pipeline import PipelineStateWAdvance
from quack.tile_scheduler import TileScheduler, VarlenMTileScheduler


class SonicMoETileScheduler(TileScheduler):
    @staticmethod
    @cute.jit
    def create(
        params: TileScheduler.Params,
        tile_count: cute.Tensor | None = None,
        scheduler_pipeline: cutlass.pipeline.PipelineAsync | None = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> SonicMoETileScheduler:
        """is_scheduler_warp should only be true for one warp in the whole cluster"""
        stages = 0
        if const_expr(not params.is_persistent):
            cidx, cidy, _ = cute.arch.cluster_idx()
            cdimx, _, _ = cute.arch.cluster_dim()
            cluster_id = cidx + cidy * cdimx
            current_work_linear_idx = Int32(cluster_id)
        else:
            _, _, bidz = cute.arch.block_idx()
            current_work_linear_idx = Int32(bidz)
            if const_expr(params.tile_count_semaphore is not None):
                assert tile_count is not None
                assert scheduler_pipeline is not None
                stages = const_expr(cute.size(tile_count))
        return SonicMoETileScheduler(
            current_work_linear_idx,
            Int32(0),  # num_tiles_executed
            tile_count,
            scheduler_pipeline,
            PipelineStateWAdvance(
                stages, Int32(0), Int32(0), Int32(1 if is_scheduler_warp else 0)
            ),
            params,
            loc=loc,
            ip=ip,
        )

    def prefetch_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        old_current_work_linear_idx = self._current_work_linear_idx
        if const_expr(self.params.is_persistent):
            num_persistent_clusters = cute.arch.grid_dim()[2]
            self._current_work_linear_idx += advance_count * Int32(
                num_persistent_clusters
            )
        future_tile_coord_mnkl = self.get_current_work()
        self._current_work_linear_idx = old_current_work_linear_idx
        return future_tile_coord_mnkl


class SonicMoEVarlenMTileScheduler(VarlenMTileScheduler, SonicMoETileScheduler):
    @staticmethod
    @cute.jit
    def create(
        params: VarlenMTileScheduler.Params,
        tile_count: cute.Tensor | None = None,
        scheduler_pipeline: cutlass.pipeline.PipelineAsync | None = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> SonicMoEVarlenMTileScheduler:
        stages = 0
        _, _, bidz = cute.arch.block_idx()
        current_work_linear_idx = Int32(bidz)
        if const_expr(params.tile_count_semaphore is not None):
            assert tile_count is not None
            assert scheduler_pipeline is not None
            stages = const_expr(cute.size(tile_count))
        return SonicMoEVarlenMTileScheduler(
            current_work_linear_idx,
            Int32(0),  # num_tiles_executed
            Int32(0),  # current_batch_idx
            Int32(0),  # num_work_idx_before_cur_batch
            tile_count,
            scheduler_pipeline,
            PipelineStateWAdvance(
                stages, Int32(0), Int32(0), Int32(1 if is_scheduler_warp else 0)
            ),
            params,
            loc=loc,
            ip=ip,
        )
