# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source Step5 bitset union specialized to explicit groups and exact masks.

Only unused single-request/range/causal-limit outputs are removed. See the source
ledger for provenance; query positions belong to the attention consumer.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .compile_cache import cached_compile_function

_UNION_FILL_THREADS = 256
_UNION_COMPACT_THREADS = 512


@dsl_user_op
def elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def _atomic_or_i32(ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.OR,
            ptr=ptr.llvm_ptr,
            a=Int32(val).ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def _popc_i32(x: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _make_grouped_union_logical_bitset_fill_kernel(
    *,
    q_group: int,
    topk_windows: int,
    topk_width: int,
    num_region_bins: int,
    bit_words: int,
    bit_words_width: int,
):

    @cute.kernel
    def _grouped_union_logical_bitset_fill_kernel(
        mRegionLogical: cute.Tensor,
        mRegionPhys: cute.Tensor,
        mRegionCounts: cute.Tensor,
        mBitset: cute.Tensor,
        mWorkQLocal: cute.Tensor,
        mWorkQLen: cute.Tensor,
        total_q: Int32,
    ):
        group_id, _, _ = cute.arch.block_idx()
        tx = cute.arch.thread_idx()[0]
        q_base = mWorkQLocal[group_id]
        q_len = mWorkQLen[group_id]
        for word_chunk in cutlass.range_constexpr(
            (bit_words_width + _UNION_FILL_THREADS - 1) // _UNION_FILL_THREADS
        ):
            word_idx = tx + Int32(word_chunk * _UNION_FILL_THREADS)
            if word_idx < Int32(bit_words):
                mBitset[group_id, word_idx] = Int32(0)
        cute.arch.sync_threads()
        num_values = Int32(q_group * topk_width)
        num_tiles = (num_values + Int32(_UNION_FILL_THREADS - 1)) // Int32(
            _UNION_FILL_THREADS
        )
        for tile in cutlass.range(num_tiles, unroll=1):
            value_offset = tile * Int32(_UNION_FILL_THREADS) + tx
            if value_offset < num_values:
                q_slot = value_offset // Int32(topk_width)
                win = value_offset - q_slot * Int32(topk_width)
                q_pos = q_base + q_slot
                if q_pos < total_q and q_pos < q_len:
                    count = mRegionCounts[q_pos]
                    if win < count and win < Int32(topk_windows):
                        logical_start = mRegionLogical[q_pos, win]
                        phys = mRegionPhys[q_pos, win]
                        logical_region = logical_start // Int32(8)
                        if (
                            logical_region >= Int32(0)
                            and logical_region < Int32(num_region_bins)
                            and (phys >= Int32(0))
                        ):
                            word_idx = logical_region // Int32(32)
                            bit_idx = logical_region - word_idx * Int32(32)
                            bit_mask = Int32(1) << bit_idx
                            old_word = mBitset[group_id, word_idx]
                            if (old_word & bit_mask) == Int32(0):
                                _atomic_or_i32(
                                    elem_pointer(mBitset, (group_id, word_idx)),
                                    bit_mask,
                                )

    return _grouped_union_logical_bitset_fill_kernel


def _make_grouped_union_logical_bitset_compact_kernel(
    *,
    q_group: int,
    max_union_windows: int,
    num_region_bins: int,
    bit_words: int,
    bit_words_width: int,
    topk_windows: int,
    topk_width: int,
):
    word_chunks = (
        bit_words_width + _UNION_COMPACT_THREADS - 1
    ) // _UNION_COMPACT_THREADS

    @cute.kernel
    def _grouped_union_logical_bitset_compact_kernel(
        mRegionLogical: cute.Tensor,
        mRegionPhys: cute.Tensor,
        mRegionCounts: cute.Tensor,
        mBitset: cute.Tensor,
        mWorkQLocal: cute.Tensor,
        mWorkQLen: cute.Tensor,
        mUnionPhys: cute.Tensor,
        mUnionLogical: cute.Tensor,
        mUnionCounts: cute.Tensor,
        mExactMaskBits: cute.Tensor,
        total_q: Int32,
    ):
        group_id, _, _ = cute.arch.block_idx()
        tx = cute.arch.thread_idx()[0]
        q_base = mWorkQLocal[group_id]
        q_len = mWorkQLen[group_id]

        @cute.struct
        class SharedStorage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, bit_words_width], 128
            ]
            prefix: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, bit_words_width], 128
            ]
            popc: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, bit_words_width], 128
            ]
            chunk_offsets: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, word_chunks], 128
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)
        sWords = storage.words.get_tensor(
            cute.make_layout((bit_words_width,), stride=(1,))
        )
        sPrefix = storage.prefix.get_tensor(
            cute.make_layout((bit_words_width,), stride=(1,))
        )
        sPopc = storage.popc.get_tensor(
            cute.make_layout((bit_words_width,), stride=(1,))
        )
        sChunkOffsets = storage.chunk_offsets.get_tensor(
            cute.make_layout((word_chunks,), stride=(1,))
        )
        for word_chunk in cutlass.range_constexpr(word_chunks):
            word_idx = tx + Int32(word_chunk * _UNION_COMPACT_THREADS)
            if word_idx < Int32(bit_words_width):
                word = Int32(0)
                if word_idx < Int32(bit_words):
                    word = mBitset[group_id, word_idx]
                pc = _popc_i32(word)
                sWords[word_idx] = word
                sPopc[word_idx] = pc
                sPrefix[word_idx] = pc
        cute.arch.sync_threads()
        for word_chunk in cutlass.range_constexpr(word_chunks):
            word_base = Int32(word_chunk * _UNION_COMPACT_THREADS)
            word_idx = word_base + tx
            for offset_i in cutlass.range_constexpr(
                (_UNION_COMPACT_THREADS - 1).bit_length()
            ):
                offset = Int32(1 << offset_i)
                addend = Int32(0)
                if word_idx < Int32(bit_words_width) and tx >= offset:
                    addend = sPrefix[word_idx - offset]
                cute.arch.sync_threads()
                if word_idx < Int32(bit_words_width) and tx >= offset:
                    sPrefix[word_idx] = sPrefix[word_idx] + addend
                cute.arch.sync_threads()
            if tx == Int32(0):
                chunk_last = Int32(
                    min((word_chunk + 1) * _UNION_COMPACT_THREADS, bit_words_width) - 1
                )
                sChunkOffsets[Int32(word_chunk)] = sPrefix[chunk_last]
            cute.arch.sync_threads()
        if tx == Int32(0):
            running = Int32(0)
            for word_chunk in cutlass.range_constexpr(word_chunks):
                chunk_total = sChunkOffsets[Int32(word_chunk)]
                sChunkOffsets[Int32(word_chunk)] = running
                running = running + chunk_total
        cute.arch.sync_threads()
        for word_chunk in cutlass.range_constexpr(word_chunks):
            word_idx = tx + Int32(word_chunk * _UNION_COMPACT_THREADS)
            if word_idx < Int32(bit_words_width):
                sPrefix[word_idx] = sPrefix[word_idx] + sChunkOffsets[Int32(word_chunk)]
        cute.arch.sync_threads()
        exact_values = Int32(max_union_windows)
        exact_tiles = (exact_values + Int32(_UNION_COMPACT_THREADS - 1)) // Int32(
            _UNION_COMPACT_THREADS
        )
        for tile in cutlass.range(exact_tiles, unroll=1):
            rank = tile * Int32(_UNION_COMPACT_THREADS) + tx
            if rank < exact_values:
                mExactMaskBits[group_id, rank] = Int32(0)
        cute.arch.sync_threads()
        num_values = Int32(q_group * topk_width)
        num_tiles = (num_values + Int32(_UNION_COMPACT_THREADS - 1)) // Int32(
            _UNION_COMPACT_THREADS
        )
        for tile in cutlass.range(num_tiles, unroll=1):
            value_offset = tile * Int32(_UNION_COMPACT_THREADS) + tx
            if value_offset < num_values:
                q_slot = value_offset // Int32(topk_width)
                win = value_offset - q_slot * Int32(topk_width)
                q_pos = q_base + q_slot
                if q_pos < total_q and q_pos < q_len:
                    count = mRegionCounts[q_pos]
                    if win < count and win < Int32(topk_windows):
                        logical_start = mRegionLogical[q_pos, win]
                        phys = mRegionPhys[q_pos, win]
                        logical_region = logical_start // Int32(8)
                        if (
                            logical_region >= Int32(0)
                            and logical_region < Int32(num_region_bins)
                            and (phys >= Int32(0))
                        ):
                            word_idx = logical_region // Int32(32)
                            bit_idx = logical_region - word_idx * Int32(32)
                            bit_mask = Int32(1) << bit_idx
                            word = sWords[word_idx]
                            if (word & bit_mask) != Int32(0):
                                before_words = Int32(0)
                                if word_idx > Int32(0):
                                    before_words = sPrefix[word_idx - Int32(1)]
                                rank = before_words + _popc_i32(
                                    word & (bit_mask - Int32(1))
                                )
                                if rank < Int32(max_union_windows):
                                    mUnionPhys[group_id, rank] = phys
                                    mUnionLogical[group_id, rank] = logical_region
                                    _atomic_or_i32(
                                        elem_pointer(mExactMaskBits, (group_id, rank)),
                                        Int32(1) << q_slot,
                                    )
        cute.arch.sync_threads()
        if tx == Int32(0):
            union_count = cutlass.min(
                sPrefix[Int32(bit_words_width - 1)], Int32(max_union_windows)
            )
            mUnionCounts[group_id] = union_count

    return _grouped_union_logical_bitset_compact_kernel


@cached_compile_function
def _compile_union(
    num_region_bins: int, capacity: int, device_key: tuple, selection_stride: int = 1
):
    bit_words = (num_region_bins + 31) // 32
    bit_words_width = 1 << (bit_words - 1).bit_length()
    fill = _make_grouped_union_logical_bitset_fill_kernel(
        q_group=16,
        topk_windows=513,
        topk_width=1024,
        num_region_bins=num_region_bins,
        bit_words=bit_words,
        bit_words_width=bit_words_width,
    )
    compact = _make_grouped_union_logical_bitset_compact_kernel(
        q_group=16,
        max_union_windows=capacity,
        num_region_bins=num_region_bins,
        bit_words=bit_words,
        bit_words_width=bit_words_width,
        topk_windows=513,
        topk_width=1024,
    )

    @cute.jit
    def run(
        logical,
        physical,
        counts,
        starts,
        ends,
        bits,
        union_physical,
        union_logical,
        union_counts,
        membership,
        groups: Int32,
        rows: Int32,
        stream,
    ):
        fill(logical, physical, counts, bits, starts, ends, rows).launch(
            grid=[groups, 1, 1],
            block=[_UNION_FILL_THREADS, 1, 1],
            stream=stream,
        )
        compact(
            logical,
            physical,
            counts,
            bits,
            starts,
            ends,
            union_physical,
            union_logical,
            union_counts,
            membership,
            rows,
        ).launch(
            grid=[groups, 1, 1],
            block=[_UNION_COMPACT_THREADS, 1, 1],
            stream=stream,
        )

    def fake(width=None, stride=1):
        rows = cute.sym_int()
        return cute.runtime.make_fake_tensor(
            Int32,
            (rows,) if width is None else (rows, width),
            (1,) if width is None else (width * stride, stride),
            assumed_align=4,
        )

    # Architecture is part of the cache key even though only integer ops occur.
    return cute.compile(
        run,
        fake(513, selection_stride),
        fake(513, selection_stride),
        fake(),
        fake(),
        fake(),
        fake(bit_words),
        fake(capacity),
        fake(capacity),
        fake(),
        fake(capacity),
        Int32(1),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 2",
    )
