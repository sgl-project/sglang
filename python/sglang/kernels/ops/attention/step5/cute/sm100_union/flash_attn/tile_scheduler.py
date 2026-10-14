# Copyright (c) 2025, Tri Dao.

from typing import Optional, Tuple
from dataclasses import dataclass, fields

# try:
#     from typing import override
# except ImportError:  # Python < 3.12
#     from typing_extensions import override
from typing_extensions import override
import cutlass
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass import Int32, const_expr

import sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.utils as utils
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn import barrier
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.fast_math import FastDivmod, clz


class WorkTileInfo(cutlass.utils.WorkTileInfo):
    """Altered WorkTileInfo which includes four axes: (block, head, batch, split)"""

    def __init__(self, tile_idx: cute.Coord, is_valid_tile: cutlass.Boolean, actual_num_splits: Int32 | int = 1):
        super().__init__(tile_idx, is_valid_tile)
        self._actual_num_splits = Int32(actual_num_splits)

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = cutlass.extract_mlir_values(self.tile_idx)
        values.extend(cutlass.extract_mlir_values(self.is_valid_tile))
        values.extend(cutlass.extract_mlir_values(self.actual_num_splits))
        return values

    @override
    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 6
        new_tile_idx = cutlass.new_from_mlir_values(self._tile_idx, values[:4])
        new_is_valid_tile = cutlass.new_from_mlir_values(self._is_valid_tile, [values[4]])
        new_actual_num_splits = cutlass.new_from_mlir_values(self._actual_num_splits, [values[5]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile, new_actual_num_splits)

    @property
    def actual_num_splits(self) -> Int32:
        return self._actual_num_splits


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, cutlass.Constexpr)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, cutlass.Constexpr)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, cutlass.Constexpr)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    num_splits: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
    mCuSeqlensQ: Optional[cute.Tensor] = None
    mSeqUsedQ: Optional[cute.Tensor] = None
    mCuSeqlensK: Optional[cute.Tensor] = None
    mSeqUsedK: Optional[cute.Tensor] = None
    mNumSplitsDynamic: Optional[cute.Tensor] = None
    mSplitOffsets: Optional[cute.Tensor] = None
    mTotalSplitUnits: Optional[cute.Tensor] = None
    mNumSplitsActual: Optional[cute.Tensor] = None
    mPartialOrder: Optional[cute.Tensor] = None
    mPartialOrderState: Optional[cute.Tensor] = None
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    is_causal: cutlass.Constexpr[bool] = False
    is_local: cutlass.Constexpr[bool] = False
    split_kv_len: cutlass.Constexpr[Optional[int]] = None
    dynamic_splitkv_scheduler: cutlass.Constexpr[bool] = False
    gpu_partial_order: cutlass.Constexpr[bool] = False
    dynamic_splitkv_balance: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        num_splits: Int32
        num_splits_divmod: FastDivmod
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileScheduler.Params":
            return SingleTileScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.num_splits,
                FastDivmod.create(args.num_splits),
                args.is_split_kv,
                args.cluster_shape_mn,
            )

    def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileScheduler":
        blk_coord = cute.arch.block_idx()
        return SingleTileScheduler(params, blk_coord, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        # TODO: this hard-codes the fact that we only use cluster = (1, 1) or (2, 1)
        assert params.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
        return (
            cute.round_up(params.num_block, params.cluster_shape_mn[0]),
            params.num_head * params.num_splits,
            params.num_batch,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        block_idx, head_idx, batch_idx = self._blk_coord
        if const_expr(self.params.is_split_kv):
            head_idx, split_idx = self.params.num_splits_divmod.divmod(head_idx)
        else:
            split_idx = Int32(0)
        return WorkTileInfo(
            (block_idx, head_idx, batch_idx, split_idx),
            self._is_first_block,
            self.params.num_splits if const_expr(self.params.is_split_kv) else Int32(1),
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)


class StaticPersistentTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block_divmod: FastDivmod
        num_head_divmod: FastDivmod
        total_blocks: Int32

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "StaticPersistentTileScheduler.Params":
            total_blocks = args.num_block * args.num_head * args.num_batch
            return StaticPersistentTileScheduler.Params(
                FastDivmod.create(args.num_block), FastDivmod.create(args.num_head), total_blocks
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return StaticPersistentTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "StaticPersistentTileScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return StaticPersistentTileScheduler(params, tile_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        return (cutlass.min(sm_count, params.total_blocks), Int32(1), Int32(1))

    # @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        hn_idx, block_idx = self.params.num_block_divmod.divmod(self._tile_idx)
        batch_idx, head_idx = self.params.num_head_divmod.divmod(hn_idx)
        is_valid = self._tile_idx < self.params.total_blocks
        # if cute.arch.thread_idx()[0] == 0:
        #     cute.printf("TileScheduler: tile_idx=%d, hn_idx=%d, block_idx=%d, batch_idx=%d, head_idx=%d, is_valid=%d", self._tile_idx, hn_idx, block_idx, batch_idx, head_idx, is_valid)
        return WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid, Int32(1)
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._tile_idx += cute.arch.grid_dim()[0]

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.params, self._tile_idx],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)


class SingleTileLPTScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_splits: Int32
        num_block_divmod: FastDivmod
        num_head_divmod: FastDivmod
        l2_minor_divmod: FastDivmod
        l2_major_divmod: FastDivmod
        l2_minor_residual_divmod: FastDivmod
        num_hb_quotient: Int32
        is_split_kv: cutlass.Constexpr[bool] = False

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileLPTScheduler.Params":
            # cute.printf(args.num_block, args.num_head, args.num_batch, args.seqlen_k, args.headdim, args.headdim_v, args.total_q, args.tile_shape_mn, args.qhead_per_kvhead_packgqa, args.element_size)
            size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
            size_one_head = size_one_kv_head
            size_l2 = 50 * 1024 * 1024  # 40 MB for K & V
            # Swizzle is the size of each "section". Round swizzle to a power of 2
            # Need to be careful about the case where only one head will fit
            # swizzle is how many heads can fit in L2
            # swizzle = 1 if size_l2 < size_one_head else (size_l2 // size_one_head)
            # Seems faster if swizzle if a power of 2
            log2_floor = lambda n: 31 - clz(n)
            swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
            # swizzle = 1 if size_l2 < size_one_head else (size_l2 // size_one_head)
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            return SingleTileLPTScheduler.Params(
                total_blocks=args.num_block * args.num_head * args.num_batch,
                num_block_divmod=FastDivmod.create(args.num_block),
                num_head_divmod=FastDivmod.create(args.num_head),
                l2_minor_divmod=FastDivmod.create(swizzle),
                l2_major_divmod=FastDivmod.create(swizzle * args.num_block),
                l2_minor_residual_divmod=FastDivmod.create(
                    max(num_hb_remainder, 1)
                ),  # don't divide by 0
                num_hb_quotient=Int32(num_hb_quotient),
                num_splits=args.num_splits,
                is_split_kv=args.is_split_kv,
            )

    def __init__(self, params: Params, tile_idx: Int32, split_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileLPTScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTScheduler":
        tile_idx, split_idx, _ = cute.arch.block_idx()
        return SingleTileLPTScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return (params.total_blocks, params.num_splits, Int32(1))

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = params.l2_major_divmod.divmod(self._tile_idx)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = params.l2_minor_divmod.divmod(l2_mod)
        else:
            block, bidhb_residual = params.l2_minor_residual_divmod.divmod(l2_mod)
        bidhb_actual = bidhb * params.l2_minor_divmod.divisor + bidhb_residual
        batch_idx, head_idx = params.num_head_divmod.divmod(bidhb_actual)
        # Longest-processing-time-first
        block = params.num_block_divmod.divisor - 1 - block
        is_valid = self._tile_idx < params.total_blocks
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(self._split_idx)), is_valid, params.num_splits
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx, self._split_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx, self._split_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


class SingleTileLPTBwdScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_head_divmod: FastDivmod
        l2_minor_divmod: FastDivmod
        l2_major_divmod: FastDivmod
        l2_minor_residual_divmod: FastDivmod
        num_hb_quotient: Int32
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileLPTBwdScheduler.Params":
            swizzle = 8
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            num_block = cute.ceil_div(args.num_block, args.cluster_shape_mn[0])
            return SingleTileLPTBwdScheduler.Params(
                total_blocks=(num_block * args.cluster_shape_mn[0])
                * args.num_head
                * args.num_batch,
                num_head_divmod=FastDivmod.create(args.num_head),
                l2_minor_divmod=FastDivmod.create(swizzle),
                l2_major_divmod=FastDivmod.create(swizzle * num_block),
                l2_minor_residual_divmod=FastDivmod.create(
                    max(num_hb_remainder, 1)
                ),  # don't divide by 0
                num_hb_quotient=Int32(num_hb_quotient),
                cluster_shape_mn=args.cluster_shape_mn,
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileLPTBwdScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTBwdScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileLPTBwdScheduler(params, tile_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return (params.total_blocks, Int32(1), Int32(1))

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        cluster_idx = self._tile_idx // self.params.cluster_shape_mn[0]
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = params.l2_major_divmod.divmod(cluster_idx)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = params.l2_minor_divmod.divmod(l2_mod)
        else:
            block, bidhb_residual = params.l2_minor_residual_divmod.divmod(l2_mod)
        bidhb_actual = bidhb * params.l2_minor_divmod.divisor + bidhb_residual
        batch_idx, head_idx = params.num_head_divmod.divmod(bidhb_actual)
        is_valid = self._tile_idx < params.total_blocks
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        block = block * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid, Int32(1))

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


class SingleTileVarlenScheduler:
    @dataclass
    class Params(ParamsBase):
        num_head: Int32
        num_batch: Int32
        total_q: Int32
        num_splits: Int32
        num_splits_divmod: FastDivmod
        total_tiles_max: Int32
        max_kvblock_in_l2: Int32
        tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
        mCuSeqlensQ: Optional[cute.Tensor] = None
        mSeqUsedQ: Optional[cute.Tensor] = None
        mCuSeqlensK: Optional[cute.Tensor] = None
        mSeqUsedK: Optional[cute.Tensor] = None
        mNumSplitsDynamic: Optional[cute.Tensor] = None
        mSplitOffsets: Optional[cute.Tensor] = None
        mTotalSplitUnits: Optional[cute.Tensor] = None
        mNumSplitsActual: Optional[cute.Tensor] = None
        mPartialOrder: Optional[cute.Tensor] = None
        mPartialOrderState: Optional[cute.Tensor] = None
        window_size_left: Optional[Int32] = None
        window_size_right: Optional[Int32] = None
        qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
        lpt: cutlass.Constexpr[bool] = False
        is_split_kv: cutlass.Constexpr[bool] = False
        is_persistent: cutlass.Constexpr[bool] = False
        is_causal: cutlass.Constexpr[bool] = False
        is_local: cutlass.Constexpr[bool] = False
        seqlen_k_static: Int32 = 0
        split_kv_len: cutlass.Constexpr[Optional[int]] = None
        dynamic_splitkv_scheduler: cutlass.Constexpr[bool] = False
        gpu_partial_order: cutlass.Constexpr[bool] = False
        dynamic_splitkv_balance: cutlass.Constexpr[bool] = False

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileVarlenScheduler.Params":
            total_blocks_max = (
                args.total_q + args.num_batch * (args.tile_shape_mn[0] - 1)
            ) // args.tile_shape_mn[0]
            size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
            max_kvblock_in_l2 = size_l2 // (
                (args.headdim + args.headdim_v) * args.element_size * args.tile_shape_mn[1]
            )
            assert args.mCuSeqlensQ is not None or args.mSeqUsedQ is not None, (
                "At least one of mCuSeqlensQ or mSeqUsedQ must be provided"
            )
            return SingleTileVarlenScheduler.Params(
                num_head=args.num_head,
                num_batch=args.num_batch,
                total_q=args.total_q,
                num_splits=args.num_splits,
                num_splits_divmod=FastDivmod.create(args.num_splits),
                total_tiles_max=total_blocks_max * args.num_head * args.num_splits,
                max_kvblock_in_l2=max_kvblock_in_l2,
                tile_shape_mn=args.tile_shape_mn,
                mCuSeqlensQ=args.mCuSeqlensQ,
                mSeqUsedQ=args.mSeqUsedQ,
                mCuSeqlensK=args.mCuSeqlensK,
                mSeqUsedK=args.mSeqUsedK,
                mNumSplitsDynamic=args.mNumSplitsDynamic,
                mSplitOffsets=args.mSplitOffsets,
                mTotalSplitUnits=args.mTotalSplitUnits,
                mNumSplitsActual=args.mNumSplitsActual,
                mPartialOrder=args.mPartialOrder,
                mPartialOrderState=args.mPartialOrderState,
                window_size_left=args.window_size_left,
                window_size_right=args.window_size_right,
                qhead_per_kvhead_packgqa=args.qhead_per_kvhead_packgqa,
                lpt=args.lpt,
                is_split_kv=args.is_split_kv,
                is_persistent=args.is_persistent,
                is_causal=args.is_causal,
                is_local=args.is_local,
                seqlen_k_static=args.seqlen_k,
                split_kv_len=args.split_kv_len,
                dynamic_splitkv_scheduler=args.dynamic_splitkv_scheduler,
                gpu_partial_order=args.gpu_partial_order,
                dynamic_splitkv_balance=args.dynamic_splitkv_balance,
            )

    def __init__(self, params: Params, tile_idx: Int32, split_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileVarlenScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileVarlenScheduler":
        tile_idx, split_idx, _ = cute.arch.block_idx()
        return SingleTileVarlenScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        total_blocks_max = (
            params.total_q + params.num_batch * (params.tile_shape_mn[0] - 1)
        ) // params.tile_shape_mn[0]
        if params.is_persistent:
            hardware_info = cutlass.utils.HardwareInfo()
            sm_count = hardware_info.get_device_multiprocessor_count()
            return (cutlass.min(sm_count, params.total_tiles_max), Int32(1), Int32(1))
        return (total_blocks_max * params.num_head, params.num_splits, Int32(1))

    @cute.jit
    def _get_num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        params = self.params
        batch_idx = lane + bidb_start
        if cutlass.const_expr(params.mSeqUsedQ is not None):
            seqlen = Int32(0)
            if batch_idx < params.num_batch:
                seqlen = params.mSeqUsedQ[batch_idx]
        else:
            assert params.mCuSeqlensQ is not None
            cur_cu_seqlen = Int32(0)
            if batch_idx <= params.num_batch:
                cur_cu_seqlen = params.mCuSeqlensQ[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        if cutlass.const_expr(params.qhead_per_kvhead_packgqa > 1):
            seqlen *= params.qhead_per_kvhead_packgqa
        return (
            cute.ceil_div(seqlen, params.tile_shape_mn[0])
            if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def _get_seqlen_q(self, lane: Int32, bidb_start: Int32) -> Int32:
        params = self.params
        batch_idx = lane + bidb_start
        if cutlass.const_expr(params.mSeqUsedQ is not None):
            seqlen_q = Int32(0)
            if batch_idx < params.num_batch:
                seqlen_q = params.mSeqUsedQ[batch_idx]
        else:
            assert params.mCuSeqlensQ is not None
            cur_cu_seqlen = Int32(0)
            if batch_idx <= params.num_batch:
                cur_cu_seqlen = params.mCuSeqlensQ[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen_q = next_cu_seqlen - cur_cu_seqlen
        return seqlen_q if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1 else Int32(0)

    @cute.jit
    def _get_seqlen_k(self, lane: Int32, bidb_start: Int32) -> Int32:
        params = self.params
        batch_idx = lane + bidb_start
        if cutlass.const_expr(params.mSeqUsedK is not None):
            seqlen_k = Int32(0)
            if batch_idx < params.num_batch:
                seqlen_k = params.mSeqUsedK[batch_idx]
        elif cutlass.const_expr(params.mCuSeqlensK is not None):
            cur_cu_seqlen = Int32(0)
            if batch_idx <= params.num_batch:
                cur_cu_seqlen = params.mCuSeqlensK[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen_k = next_cu_seqlen - cur_cu_seqlen
        else:
            seqlen_k = params.seqlen_k_static
        return seqlen_k if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1 else Int32(0)

    @cute.jit
    def _get_current_work_dynamic_splitkv(self, *, loc=None, ip=None) -> WorkTileInfo:
        params = self.params
        tile_idx = self._tile_idx
        assert params.mNumSplitsDynamic is not None
        if const_expr(params.gpu_partial_order):
            assert params.mPartialOrder is not None
            assert params.mPartialOrderState is not None
            total_tasks = barrier.ld_acquire(
                utils.elem_pointer(params.mPartialOrderState, (Int32(0),))
            )
            encoded = Int32(0)
            is_valid = tile_idx < total_tasks
            if is_valid:
                encoded = params.mPartialOrder[tile_idx]
            split_idx = encoded % params.num_splits
            hs_block = encoded // params.num_splits
            head_idx = hs_block % params.num_head
            batch_idx = hs_block // params.num_head
            return WorkTileInfo(
                (Int32(0), Int32(head_idx), Int32(batch_idx), Int32(split_idx)),
                is_valid,
                params.mNumSplitsDynamic[batch_idx],
            )
        assert params.mSplitOffsets is not None
        assert params.mTotalSplitUnits is not None
        unit_idx = tile_idx // params.num_head
        head_idx = tile_idx - unit_idx * params.num_head
        total_split_units = barrier.ld_acquire(
            utils.elem_pointer(params.mTotalSplitUnits, (Int32(0),))
        )
        block = Int32(0)
        batch_idx = Int32(0)
        split_idx = Int32(0)
        actual_splits = Int32(0)
        is_valid = tile_idx < params.total_tiles_max and unit_idx < total_split_units
        if params.num_batch == Int32(1):
            actual_splits = params.mNumSplitsDynamic[Int32(0)]
            if is_valid and actual_splits > Int32(0):
                block = unit_idx // actual_splits
                split_idx = unit_idx - block * actual_splits
                is_valid = split_idx < actual_splits
        else:
            batch_lo = Int32(0)
            batch_hi = params.num_batch
            while batch_lo < batch_hi:
                batch_mid = (batch_lo + batch_hi) // Int32(2)
                if params.mSplitOffsets[batch_mid + Int32(1)] <= unit_idx:
                    batch_lo = batch_mid + Int32(1)
                else:
                    batch_hi = batch_mid
            batch_idx = batch_lo
            if is_valid and batch_idx < params.num_batch:
                actual_splits = params.mNumSplitsDynamic[batch_idx]
                local_unit = unit_idx - params.mSplitOffsets[batch_idx]
                if actual_splits > Int32(0):
                    block = local_unit // actual_splits
                    split_idx = local_unit - block * actual_splits
                is_valid = actual_splits > Int32(0) and split_idx < actual_splits
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(split_idx)),
            is_valid,
            actual_splits,
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        params = self.params
        if const_expr(params.dynamic_splitkv_scheduler):
            return self._get_current_work_dynamic_splitkv(loc=loc, ip=ip)
        tile_idx = self._tile_idx
        split_idx = self._split_idx if const_expr(params.is_split_kv) else Int32(0)
        is_valid_tile = self._is_first_block
        if const_expr(params.is_persistent):
            is_valid_tile = tile_idx < params.total_tiles_max
            if const_expr(params.is_split_kv):
                tile_idx, split_idx = params.num_splits_divmod.divmod(tile_idx)
        lane_idx = cute.arch.lane_idx()
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
        num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
        # Total number of blocks for the next 31 batches
        m_blocks_in_group = cute.arch.shuffle_sync(num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1)
        # Same for all lanes
        group_end_tile = m_blocks_in_group * params.num_head
        # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, group_end_tile = %d, num_m_blocks=%d, num_m_blocks_cumulative = %d, m_blocks_in_group = %d", self._tile_idx, group_end_tile, num_m_blocks, num_m_blocks_cumulative, m_blocks_in_group)
        block, head_idx, batch_idx = Int32(0), Int32(0), Int32(0)
        next_tile_idx = tile_idx
        while group_end_tile <= next_tile_idx:
            batch_idx += cute.arch.WARP_SIZE - 1
            if batch_idx >= params.num_batch:
                batch_idx = Int32(params.num_batch)
                group_end_tile = next_tile_idx + 1
            else:
                num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=batch_idx)
                num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
                m_blocks_in_group = cute.arch.shuffle_sync(
                    num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
                )
                group_end_tile += m_blocks_in_group * params.num_head
        is_valid = False
        if batch_idx >= params.num_batch:
            block, head_idx, batch_idx = Int32(0), Int32(0), Int32(params.num_batch)
        else:
            group_start_tile = group_end_tile - m_blocks_in_group * params.num_head
            # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, group_end_tile = %d, num_m_blocks=%d, batch_idx = %d", self._tile_idx, group_end_tile, num_m_blocks, batch_idx)
            # The next problem to process is the first one that does not have ending tile position
            # that is greater than or equal to tile index.
            batch_idx_in_group = cute.arch.popc(
                cute.arch.vote_ballot_sync(
                    group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx
                )
            )
            batch_idx += batch_idx_in_group
            num_m_blocks_prev_lane = (
                0
                if batch_idx_in_group == 0
                else cute.arch.shuffle_sync(num_m_blocks_cumulative, batch_idx_in_group - 1)
            )
            num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
            mh_block = next_tile_idx - group_start_tile - num_m_blocks_prev_lane * params.num_head
            if cutlass.const_expr(params.lpt):
                # This is a version of the SingleTileLPTScheduler, complicated by the fact that
                # the seqlen can vary per batch.
                # TODO: is there any case where num_m_blocks is 0?
                # TODO: by right we should read the seqlen_kv but we're assuming seqlen_q == seqlen_k here
                num_n_blocks = (
                    num_m_blocks
                    * params.tile_shape_mn[0]
                    // params.qhead_per_kvhead_packgqa
                    // params.tile_shape_mn[1]
                )
                # nheads_in_l2 = min(max(self.max_kvblock_in_l2 // num_n_blocks, 1), self.num_head)
                # Seems faster to have this be a power of 2
                nheads_in_l2 = (
                    16
                    if num_n_blocks * 16 <= params.max_kvblock_in_l2
                    else (
                        8
                        if num_n_blocks * 8 <= params.max_kvblock_in_l2
                        else (
                            4
                            if num_n_blocks * 4 <= params.max_kvblock_in_l2
                            else (2 if num_n_blocks * 2 <= params.max_kvblock_in_l2 else 1)
                        )
                    )
                )
                nheads_in_l2 = min(nheads_in_l2, params.num_head)
                mh_in_l2 = nheads_in_l2 * num_m_blocks
                section_idx = mh_block // mh_in_l2
                l2_mod = mh_block - section_idx * mh_in_l2
                # Deal with tail section
                nheads_in_this_section = (
                    nheads_in_l2
                    if nheads_in_l2 * (section_idx + 1) <= params.num_head
                    else params.num_head - section_idx * nheads_in_l2
                )
                block = l2_mod // nheads_in_this_section
                head_idx_residual = l2_mod - block * nheads_in_this_section
                head_idx = section_idx * nheads_in_l2 + head_idx_residual
                block = num_m_blocks - 1 - block
            else:
                head_idx = mh_block // num_m_blocks
                block = mh_block - head_idx * num_m_blocks
            is_valid = is_valid_tile and batch_idx < params.num_batch
        # if cute.arch.thread_idx()[0] == 128: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, batch_idx=%d, head_idx=%d, block=%d, is_valid = %d", self._tile_idx, batch_idx, head_idx, block, is_valid)
        return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), split_idx), is_valid, params.num_splits)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.is_persistent):
            self._tile_idx += cute.arch.grid_dim()[0]
        else:
            # Single tile scheduler - set to invalid tile_idx to indicate no more work
            self._is_first_block = False

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx, self._split_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.params, self._tile_idx, self._split_idx],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileVarlenScheduler(*(tuple(obj_list)), loc=self._loc)
