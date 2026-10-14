# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 GQA union-exact sparse paged-KV prefill."""

from __future__ import annotations

import dataclasses
import functools
import math
import os
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync

from sglang.kernels.ops.attention.step5.cute.compile_cache import cached_compile_function
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn import copy_utils, utils as flash_utils
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.block_sparsity import BlockSparseTensors
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.flash_fwd_sm100 import FlashAttentionForwardSm100
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.mask import AttentionMask, mask_r2p
from sglang.kernels.ops.attention.step5.cute.sm100_union.sparse_gqa.kernel_utils import cute_utils as sparse_utils
from sglang.kernels.ops.attention.step5.cute.sm100_union.utils import elem_pointer


SUPPORTED_Q_HEADS = (8, 16)
DEFAULT_Q_HEADS = 16
HEAD_DIM = 192
REGION_TOKENS = 8
PAGE_TOKENS = 16
M_BLOCK = 128
N_BLOCK = 128
REGIONS_PER_BLOCK = N_BLOCK // REGION_TOKENS
MASK_WORD_BITS = 32
COPY_ELEMS = 8
GATHER_THREADS_PER_ROW = 8
SCHEDULE_THREADS = 512
REQUIRED_POINTER_ALIGN = 16
UNION_EXACT_KERNEL_REVISION = 10


def _debug_validate_schedule(
    union_counts: torch.Tensor,
    union_phys_indices: torch.Tensor,
    workspace: "Sm100UnionExactPrefillWorkspace",
    *,
    physical_region_count: int,
) -> None:
    if os.getenv("OPTIMUS_SM100_PREFILL_DEBUG_VALIDATE") != "1":
        return
    width = int(union_phys_indices.shape[1])
    columns = torch.arange(width, device=union_counts.device)[None, :]
    active = columns < union_counts[:, None]
    invalid_count = ((union_counts < 0) | (union_counts > width)).sum()
    invalid_phys = (
        active
        & (
            (union_phys_indices < 0)
            | (union_phys_indices >= int(physical_region_count))
        )
    ).sum()
    block_counts = workspace.block_counts
    invalid_blocks = (
        (block_counts < 0) | (block_counts > int(workspace.max_blocks))
    ).sum()
    block_columns = torch.arange(
        int(workspace.max_blocks), device=block_counts.device
    )[None, :]
    active_blocks = block_columns < block_counts[:, None]
    invalid_indices = (
        active_blocks
        & (
            (workspace.block_indices[:, : workspace.max_blocks] < 0)
            | (
                workspace.block_indices[:, : workspace.max_blocks]
                >= int(workspace.max_blocks)
            )
        )
    ).sum()
    failures = {
        "counts": int(invalid_count.item()),
        "phys": int(invalid_phys.item()),
        "blocks": int(invalid_blocks.item()),
        "indices": int(invalid_indices.item()),
    }
    if any(failures.values()):
        raise RuntimeError(
            "SM100 union-exact prefill schedule validation failed: "
            f"{failures}, groups={union_counts.numel()}, width={width}, "
            f"physical_region_count={physical_region_count}"
        )


@dataclasses.dataclass(frozen=True)
class Sm100UnionExactPrefillWorkspace:
    block_counts: torch.Tensor
    block_indices: torch.Tensor
    full_block_counts: torch.Tensor
    exact_region_words: torch.Tensor
    causal_token_limits: torch.Tensor
    max_union_windows: int
    q_heads: int

    @property
    def group_capacity(self) -> int:
        return int(self.block_counts.numel())

    @property
    def max_blocks(self) -> int:
        return (int(self.max_union_windows) + REGIONS_PER_BLOCK - 1) // REGIONS_PER_BLOCK

    @property
    def q_group(self) -> int:
        return M_BLOCK // int(self.q_heads)


def make_sm100_union_exact_prefill_workspace(
    *,
    group_capacity: int,
    max_union_windows: int,
    device: torch.device | str,
    q_heads: int = DEFAULT_Q_HEADS,
) -> Sm100UnionExactPrefillWorkspace:
    groups = int(group_capacity)
    windows = int(max_union_windows)
    heads = int(q_heads)
    if groups <= 0:
        raise ValueError(f"group_capacity must be positive, got {groups}")
    if windows <= 0:
        raise ValueError(f"max_union_windows must be positive, got {windows}")
    if heads not in SUPPORTED_Q_HEADS:
        raise ValueError(
            f"SM100 union-exact prefill supports q_heads in {SUPPORTED_Q_HEADS}, "
            f"got {heads}"
        )
    q_group = M_BLOCK // heads
    max_blocks = (windows + REGIONS_PER_BLOCK - 1) // REGIONS_PER_BLOCK
    return Sm100UnionExactPrefillWorkspace(
        block_counts=torch.empty((groups,), dtype=torch.int32, device=device),
        # The final column carries the exact region count for the gather producer.
        # Generic SM100 block-sparse consumers only visit the first max_blocks columns.
        block_indices=torch.arange(
            max_blocks + 2, dtype=torch.int32, device=device
        ).expand(groups, -1).clone(),
        # Keep all four BlockSparseTensors fields materialized. CuTeDSL drops
        # intermediate None values while rebuilding the aggregate ABI.
        full_block_counts=torch.zeros((groups,), dtype=torch.int32, device=device),
        exact_region_words=torch.empty(
            (groups, max_blocks, q_group),
            dtype=torch.int32,
            device=device,
        ),
        causal_token_limits=torch.empty(
            (groups, q_group), dtype=torch.int32, device=device
        ),
        max_union_windows=windows,
        q_heads=heads,
    )


def _make_schedule_kernel(max_blocks: int, q_group: int):
    @cute.kernel
    def _schedule_kernel(
        mUnionCounts: cute.Tensor,
        mUnionPhys: cute.Tensor,
        mUnionLogical: cute.Tensor,
        mExactMaskBits: cute.Tensor,
        mCausalLimits: cute.Tensor,
        mWorkQGlobal: cute.Tensor,
        mWorkQLocal: cute.Tensor,
        mWorkQLen: cute.Tensor,
        mBlockCounts: cute.Tensor,
        mBlockIndices: cute.Tensor,
        mExactRegionWords: cute.Tensor,
        mCausalTokenLimits: cute.Tensor,
    ):
        group, _, _ = cute.arch.block_idx()
        tid = cute.arch.thread_idx()[0]
        count = mUnionCounts[group]
        if tid == 0:
            mBlockCounts[group] = (
                count + Int32(REGIONS_PER_BLOCK - 1)
            ) // Int32(REGIONS_PER_BLOCK)
            mBlockIndices[group, Int32(max_blocks)] = count
            valid_rows = mWorkQLen[group] - mWorkQLocal[group]
            valid_rows = cutlass.select_(valid_rows < Int32(q_group), valid_rows, Int32(q_group))
            last_position = Int32(-1)
            if valid_rows > Int32(0):
                last_position = mWorkQGlobal[mWorkQLocal[group]] + valid_rows - Int32(1)
            last_window = cutlass.select_(count > Int32(0), count - Int32(1), Int32(0))
            last_logical = mUnionLogical[group, last_window]
            token_limit = count * Int32(REGION_TOKENS)
            if (count > Int32(0)) & (last_logical == last_position // Int32(REGION_TOKENS)):
                token_limit = (count - Int32(1)) * Int32(REGION_TOKENS) + last_position % Int32(REGION_TOKENS) + Int32(1)
            mBlockIndices[group, Int32(max_blocks + 1)] = cutlass.select_(valid_rows > Int32(0), token_limit, Int32(0))
        mask_word_count = int(max_blocks) * q_group
        for chunk in cutlass.range_constexpr(
            (mask_word_count + SCHEDULE_THREADS - 1) // SCHEDULE_THREADS
        ):
            flat_idx = tid + Int32(chunk * SCHEDULE_THREADS)
            if flat_idx < Int32(mask_word_count):
                block_idx = flat_idx // Int32(q_group)
                q_slot = flat_idx % Int32(q_group)
                q_valid = mWorkQLocal[group] + q_slot < mWorkQLen[group]
                word = Uint32(0)
                for region_in_block in cutlass.range_constexpr(REGIONS_PER_BLOCK):
                    window = block_idx * Int32(REGIONS_PER_BLOCK) + Int32(region_in_block)
                    window_valid = window < count
                    safe_window = cutlass.select_(window_valid, window, Int32(0))
                    exact_bits = mExactMaskBits[group, safe_window]
                    selected = (exact_bits & (Int32(1) << q_slot)) != Int32(0)
                    # The compact union contract guarantees valid logical and
                    # physical indices in [0, count). Re-reading both arrays for
                    # every q-slot would duplicate that validation eight times.
                    region_valid = q_valid & window_valid & selected
                    word = word | cutlass.select_(
                        region_valid,
                        Uint32(1) << Uint32(region_in_block),
                        Uint32(0),
                    )
                mExactRegionWords[group, block_idx, q_slot] = Int32(word)

        if tid < Int32(q_group):
            q_slot = tid
            q_global = Int32(-1)
            if mWorkQLocal[group] + q_slot < mWorkQLen[group]:
                q_global = mWorkQGlobal[mWorkQLocal[group] + q_slot]
            q_valid = mWorkQLocal[group] + q_slot < mWorkQLen[group]
            lo, hi = Int32(0), count
            while lo < hi:
                mid = (lo + hi) // Int32(2)
                if mUnionLogical[group, mid] <= q_global // Int32(REGION_TOKENS):
                    lo = mid + Int32(1)
                else:
                    hi = mid
            limit = lo * Int32(REGION_TOKENS)
            last_window = limit // Int32(REGION_TOKENS) - Int32(1)
            has_last = limit >= Int32(REGION_TOKENS)
            safe_last = cutlass.select_(has_last, last_window, Int32(0))
            current_region_in_union = (
                has_last
                & (mUnionLogical[group, safe_last] == q_global // Int32(REGION_TOKENS))
            )
            exact_limit = cutlass.select_(
                current_region_in_union,
                limit - Int32(REGION_TOKENS) + q_global % Int32(REGION_TOKENS) + Int32(1),
                limit,
            )
            mCausalTokenLimits[group, q_slot] = cutlass.select_(
                q_valid, exact_limit, Int32(0)
            )

    return _schedule_kernel


def _make_schedule_launch(max_blocks: int, q_group: int):
    kernel = _make_schedule_kernel(max_blocks, q_group)

    @cute.jit
    def _launch(
        mUnionCounts: cute.Tensor,
        mUnionPhys: cute.Tensor,
        mUnionLogical: cute.Tensor,
        mExactMaskBits: cute.Tensor,
        mCausalLimits: cute.Tensor,
        mWorkQGlobal: cute.Tensor,
        mWorkQLocal: cute.Tensor,
        mWorkQLen: cute.Tensor,
        mBlockCounts: cute.Tensor,
        mBlockIndices: cute.Tensor,
        mExactRegionWords: cute.Tensor,
        mCausalTokenLimits: cute.Tensor,
        stream,
    ):
        kernel(
            mUnionCounts,
            mUnionPhys,
            mUnionLogical,
            mExactMaskBits,
            mCausalLimits,
            mWorkQGlobal,
            mWorkQLocal,
            mWorkQLen,
            mBlockCounts,
            mBlockIndices,
            mExactRegionWords,
            mCausalTokenLimits,
        ).launch(
            grid=[cute.size(mUnionCounts.shape[0]), 1, 1],
            block=[SCHEDULE_THREADS, 1, 1],
            stream=stream,
        )

    return _launch


@cached_compile_function
def _get_compiled_schedule(
    max_union_windows: int,
    q_group: int,
    device_key: tuple[str, int | None],
):
    device = sparse_utils.device_from_cache_key(device_key)
    max_blocks = (
        int(max_union_windows) + REGIONS_PER_BLOCK - 1
    ) // REGIONS_PER_BLOCK
    counts = torch.empty((1,), dtype=torch.int32, device=device)
    metadata = torch.empty(
        (1, int(max_union_windows)), dtype=torch.int32, device=device
    )
    block_counts = torch.empty((1,), dtype=torch.int32, device=device)
    block_indices = torch.empty(
        (1, max_blocks + 2), dtype=torch.int32, device=device
    )
    exact_region_words = torch.empty(
        (1, max_blocks, int(q_group)),
        dtype=torch.int32,
        device=device,
    )
    causal_token_limits = torch.empty(
        (1, int(q_group)), dtype=torch.int32, device=device
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _make_schedule_launch(max_blocks, int(q_group)),
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            counts, alignment=4, dynamic_shape_dim=0
        ),
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            metadata,
            alignment=4,
            dynamic_shape_dim=0,
            dynamic_stride_dims=(0,),
        ),
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            metadata,
            alignment=4,
            dynamic_shape_dim=0,
            dynamic_stride_dims=(0,),
        ),
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            metadata,
            alignment=4,
            dynamic_shape_dim=0,
            dynamic_stride_dims=(0,),
        ),
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            causal_token_limits,
            alignment=4,
            dynamic_shape_dim=0,
            dynamic_stride_dims=(0,),
        ),
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            counts, alignment=4, dynamic_shape_dim=0
        ),
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            counts, alignment=4, dynamic_shape_dim=0
        ),
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            counts, alignment=4, dynamic_shape_dim=0
        ),
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            block_counts, alignment=4, dynamic_shape_dim=0
        ),
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            block_indices,
            alignment=4,
            dynamic_shape_dim=0,
            dynamic_stride_dims=(0,),
        ),
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            exact_region_words,
            alignment=4,
            dynamic_shape_dim=0,
            dynamic_stride_dims=(0,),
        ),
        sparse_utils.make_fake_tensor_like_with_dynamic_dim(
            causal_token_limits,
            alignment=4,
            dynamic_shape_dim=0,
            dynamic_stride_dims=(0,),
        ),
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


def schedule_sm100_union_exact_prefill(
    union_counts: torch.Tensor,
    union_phys_indices: torch.Tensor,
    union_logical_indices: torch.Tensor,
    exact_mask_bits: torch.Tensor,
    causal_limits: torch.Tensor,
    work_q_global: torch.Tensor,
    work_q_local: torch.Tensor,
    work_q_len: torch.Tensor,
    workspace: Sm100UnionExactPrefillWorkspace,
) -> None:
    if union_counts.dtype != torch.int32 or union_counts.ndim != 1:
        raise ValueError("union_counts must be a 1D int32 tensor")
    if not union_counts.is_cuda or not union_counts.is_contiguous():
        raise ValueError("union_counts must be contiguous CUDA storage")
    if union_counts.device != workspace.block_counts.device:
        raise ValueError("union_counts and workspace must be on the same device")
    if int(union_counts.numel()) != workspace.group_capacity:
        raise ValueError(
            "SM100 union-exact workspace capacity must equal the runtime metadata "
            f"capacity, got {workspace.group_capacity} and {int(union_counts.numel())}"
        )
    compiled = _get_compiled_schedule(
        int(workspace.max_union_windows),
        int(workspace.q_group),
        sparse_utils.device_cache_key(union_counts.device),
    )
    compiled(
        union_counts,
        union_phys_indices,
        union_logical_indices,
        exact_mask_bits,
        causal_limits,
        work_q_global,
        work_q_local,
        work_q_len,
        workspace.block_counts,
        workspace.block_indices,
        workspace.exact_region_words,
        workspace.causal_token_limits,
    )


class _UnionExactAttentionMask(AttentionMask):
    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
    ) -> None:
        del mask_seqlen, mask_causal, mask_local, mask_mod, batch_idx, head_idx, fastdiv_mods
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        ncol = const_expr(cute.size(tScS_t2r.shape))
        assert ncol == N_BLOCK
        global_row = tScS_t2r[0][0] + m_block * Int32(self.tile_m)
        mask_row = global_row // Int32(self.qhead_per_kvhead_packgqa)
        q_group = self.tile_m // self.qhead_per_kvhead_packgqa
        group = mask_row // Int32(q_group)
        q_slot = mask_row % Int32(q_group)
        mExactRegionWords = aux_tensors[7]
        mCausalTokenLimits = aux_tensors[8]
        region_word = Uint32(mExactRegionWords[group, n_block, q_slot])
        for region in cutlass.range_constexpr(REGIONS_PER_BLOCK):
            keep = cutlass.Boolean(region_word & (Uint32(1) << Uint32(region)))
            for token in cutlass.range_constexpr(REGION_TOKENS):
                score_idx = region * REGION_TOKENS + token
                acc_S[score_idx] = acc_S[score_idx] if keep else -Float32.inf
        causal_limit = mCausalTokenLimits[group, q_slot] - n_block * Int32(N_BLOCK)
        if causal_limit < Int32(N_BLOCK):
            mask_r2p(acc_S, causal_limit, arch=100, rank1=True)


class _UnionExactFlashAttentionForwardSm100(FlashAttentionForwardSm100):
    """SM100 FMHA pipeline with sparse union-region gather in the load role."""

    def __init__(self, *, dtype, max_union_windows: int, q_heads: int) -> None:
        self.input_dtype = dtype
        self.max_union_windows = int(max_union_windows)
        self.q_heads = int(q_heads)
        self.q_group = M_BLOCK // self.q_heads
        super().__init__(
            HEAD_DIM,
            HEAD_DIM,
            qhead_per_kvhead=self.q_heads,
            pack_gqa=True,
            m_block_size=M_BLOCK,
            n_block_size=N_BLOCK,
            is_persistent=True,
            has_aux_tensors=True,
            paged_kv_non_tma=True,
            force_non_tma_o=True,
            attention_mask_cls=_UnionExactAttentionMask,
        )
        self.prefetch_attention_mask = True
        self.num_regs_softmax = 256
        self.num_regs_other = 64
        if self.q_stage != 1:
            raise RuntimeError("SM100 union-exact prefill requires one Q stage")

    @cute.jit
    def prefetch_attention_mask_sm100(
        self,
        m_block: Int32,
        n_block: Int32,
        thr_mma_qk,
        thr_tmem_load,
        aux_tensors: list,
    ) -> tuple[Uint32, Int32]:
        cS = cute.make_identity_tensor((M_BLOCK, N_BLOCK))
        tScS = thr_mma_qk.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        global_row = tScS_t2r[0][0] + m_block * Int32(M_BLOCK)
        mask_row = global_row // Int32(self.q_heads)
        group = mask_row // Int32(self.q_group)
        q_slot = mask_row % Int32(self.q_group)
        region_word = Uint32(aux_tensors[7][group, n_block, q_slot])
        causal_limit = aux_tensors[8][group, q_slot] - n_block * Int32(N_BLOCK)
        return region_word, causal_limit

    @cute.jit
    def apply_prefetched_attention_mask_sm100(
        self,
        acc_S: cute.Tensor,
        prefetched_mask,
    ) -> None:
        region_word, causal_limit = prefetched_mask
        for region in cutlass.range_constexpr(REGIONS_PER_BLOCK):
            keep = cutlass.Boolean(region_word & (Uint32(1) << Uint32(region)))
            for token in cutlass.range_constexpr(REGION_TOKENS):
                score_idx = region * REGION_TOKENS + token
                acc_S[score_idx] = acc_S[score_idx] if keep else -Float32.inf
        if causal_limit < Int32(N_BLOCK):
            mask_r2p(acc_S, causal_limit, arch=100, rank1=True)

    @cute.jit
    def output_row_offset(self, m_block: Int32, aux_tensors: list) -> Int32:
        return aux_tensors[5][m_block] - m_block * Int32(self.q_group)

    @cute.jit
    def output_seqlen_q(
        self,
        seqlen_q: Int32,
        m_block: Int32,
        aux_tensors: list,
    ) -> Int32:
        return m_block * Int32(self.q_group) + aux_tensors[6][m_block] - aux_tensors[5][m_block]

    @cute.jit
    def _load_union_kv(
        self,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mX: cute.Tensor,
        source_physical_token: Int32,
        source_valid_i32: Int32,
        producer_state: pipeline.PipelineState,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        K_or_V: cutlass.Constexpr,
    ) -> None:
        stage, phase = producer_state.index, producer_state.phase
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        num_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_threads
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.input_dtype,
            num_bits_per_copy=COPY_ELEMS * self.input_dtype.width,
        )
        threads_per_row = GATHER_THREADS_PER_ROW
        tiled_copy = cute.make_tiled_copy_tv(
            atom,
            cute.make_ordered_layout(
                (num_threads // threads_per_row, threads_per_row), order=(1, 0)
            ),
            cute.make_layout((1, COPY_ELEMS)),
        )
        thr_copy = tiled_copy.get_slice(tidx)
        sX = sK if const_expr(K_or_V == "K") else sV
        sX_cur = sX[None, None, None, stage]
        sX_flat = cute.make_tensor(
            sX_cur.iterator,
            cute.make_layout(
                (sX_cur.shape[0][0], (sX_cur.shape[0][1], sX_cur.shape[2])),
                stride=(
                    sX_cur.stride[0][0],
                    (sX_cur.stride[0][1], sX_cur.stride[2]),
                ),
            ),
        )
        if const_expr(K_or_V == "V"):
            sX_flat = cute.make_tensor(
                sX_flat.iterator, cute.select(sX_flat.layout, mode=[1, 0])
            )
        cX = cute.make_identity_tensor((N_BLOCK, HEAD_DIM))
        tXsX = thr_copy.partition_D(sX_flat)
        tXcX = thr_copy.partition_S(cX)

        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            source_lane = m % threads_per_row
            source_token = flash_utils.shuffle_sync(
                source_physical_token, source_lane, width=threads_per_row
            )
            valid_i32 = flash_utils.shuffle_sync(
                source_valid_i32, source_lane, width=threads_per_row
            )
            src_ptr_i64 = (
                elem_pointer(mX, (source_token, 0, 0, 0)).toint()
                if const_expr(K_or_V == "K")
                else elem_pointer(mX, (0, source_token, 0, 0)).toint()
            )
            src_ptr = cute.make_ptr(
                self.input_dtype,
                src_ptr_i64,
                cute.AddressSpace.gmem,
                assumed_align=REQUIRED_POINTER_ALIGN,
            )
            src_row = cute.make_tensor(src_ptr, cute.make_layout((HEAD_DIM,)))
            src_vectors = cute.tiled_divide(src_row, (COPY_ELEMS,))
            pred = cute.make_fragment_like(tXsX[(0, None), m, 0], cute.Boolean)
            pred.fill(valid_i32 != Int32(0))
            for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                vector_idx = tXcX[0, 0, k][1] // COPY_ELEMS
                src = src_vectors[None, vector_idx]
                dst = tXsX[None, m, k]
                src = cute.make_tensor(src.iterator, dst.layout)
                cute.copy(tiled_copy, src, dst, pred=pred)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_mbarrier_arrive_noinc(mbar_full_ptr + stage)

    @cute.jit
    def load(
        self,
        thr_mma_qk,
        thr_mma_pv,
        mQ,
        mK,
        mV,
        mO,
        mReductionQueue,
        mFusedTimeline,
        sQ,
        sK,
        sV,
        mPageTable,
        tma_atom_Q,
        tma_atom_K,
        tma_atom_V,
        pipeline_kv,
        sPageRing,
        mbar_ptr,
        block_info,
        num_splits,
        mNumSplitsActual,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: BlockSparseTensors,
        aux_tensors: list,
    ) -> None:
        del (
            thr_mma_pv,
            mO,
            mReductionQueue,
            mFusedTimeline,
            mPageTable,
            tma_atom_K,
            tma_atom_V,
            sPageRing,
            block_info,
            num_splits,
            mNumSplitsActual,
        )
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % Int32(num_load_threads)
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        mBlockCounts, mBlockIndices, _, mUnionPhys4d = blocksparse_tensors

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[
                None, None, head_idx
            ]
            mQ_cur = cute.domain_offset(
                ((0, self.output_row_offset(m_block, aux_tensors)), 0), mQ_cur
            )
            gQ = cute.local_tile(
                mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0)
            )
            tSgQ = thr_mma_qk.partition_A(gQ)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )
            block_count = mBlockCounts[batch_idx, head_idx, m_block]
            if block_count > Int32(0):
                if tidx < Int32(cute.arch.WARP_SIZE):
                    self.load_Q(
                        load_Q_fn,
                        mbar_ptr + self.mbar_load_q_full_offset,
                        mbar_ptr + self.mbar_load_q_empty_offset,
                        block=m_block,
                        stage=0,
                        phase=q_producer_phase,
                    )
                union_count = mBlockIndices[
                    batch_idx,
                    head_idx,
                    m_block,
                    Int32(cute.size(mBlockIndices.shape[3]) - 2),
                ]
                union_token_limit = mBlockIndices[batch_idx, head_idx, m_block, Int32(cute.size(mBlockIndices.shape[3]) - 1)]
                source_row = (
                    (tidx % Int32(GATHER_THREADS_PER_ROW))
                    * Int32(num_load_threads // GATHER_THREADS_PER_ROW)
                    + tidx // Int32(GATHER_THREADS_PER_ROW)
                )
                for offset in cutlass.range(block_count, unroll=1):
                    sparse_block = mBlockIndices[
                        batch_idx,
                        head_idx,
                        m_block,
                        block_count - Int32(1) - offset,
                    ]
                    source_union_token = sparse_block * Int32(N_BLOCK) + source_row
                    source_window = source_union_token // Int32(REGION_TOKENS)
                    source_token_in_region = source_union_token % Int32(REGION_TOKENS)
                    source_window_valid = source_window < union_count
                    safe_source_window = cutlass.select_(
                        source_window_valid, source_window, Int32(0)
                    )
                    source_physical_region = mUnionPhys4d[
                        batch_idx, head_idx, m_block, safe_source_window
                    ]
                    source_valid = source_window_valid & (source_union_token < union_token_limit) & (
                        source_physical_region >= Int32(0)
                    )
                    safe_source_region = cutlass.select_(
                        source_valid, source_physical_region, Int32(0)
                    )
                    source_physical_token = (
                        safe_source_region * Int32(REGION_TOKENS)
                        + source_token_in_region
                    )
                    source_valid_i32 = Int32(
                        cutlass.select_(source_valid, Int32(1), Int32(0))
                    )
                    self._load_union_kv(
                        sK,
                        sV,
                        mK,
                        source_physical_token,
                        source_valid_i32,
                        kv_producer_state,
                        mbar_ptr + self.mbar_load_kv_full_offset,
                        mbar_ptr + self.mbar_load_kv_empty_offset,
                        "K",
                    )
                    kv_producer_state.advance()
                    self._load_union_kv(
                        sK,
                        sV,
                        mV,
                        source_physical_token,
                        source_valid_i32,
                        kv_producer_state,
                        mbar_ptr + self.mbar_load_kv_full_offset,
                        mbar_ptr + self.mbar_load_kv_empty_offset,
                        "V",
                    )
                    kv_producer_state.advance()
                q_producer_phase ^= Int32(1)

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()


@functools.cache
def _get_union_exact_op(dtype, max_union_windows: int, q_heads: int):
    return _UnionExactFlashAttentionForwardSm100(
        dtype=dtype,
        max_union_windows=int(max_union_windows),
        q_heads=int(q_heads),
    )


def _make_dynamic_fake(
    tensor: torch.Tensor,
    *,
    dynamic_shape_dims: tuple[int, ...],
    dynamic_stride_dims: tuple[int, ...] = (),
    alignment: int = REQUIRED_POINTER_ALIGN,
):
    return sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        tensor,
        alignment=alignment,
        dynamic_shape_dims=dynamic_shape_dims,
        dynamic_stride_dims=dynamic_stride_dims,
    )


@cached_compile_function
def _get_compiled_union_exact(
    dtype,
    max_union_windows: int,
    q_heads: int,
    has_lse: bool,
    softmax_scale: float,
    device_key: tuple[str, int | None],
    generate_line_info: bool,
    kernel_revision: int,
):
    if kernel_revision != UNION_EXACT_KERNEL_REVISION:
        raise RuntimeError(
            f"unexpected union-exact kernel revision {kernel_revision}, "
            f"expected {UNION_EXACT_KERNEL_REVISION}"
        )
    device = sparse_utils.device_from_cache_key(device_key)
    max_blocks = (
        int(max_union_windows) + REGIONS_PER_BLOCK - 1
    ) // REGIONS_PER_BLOCK
    q_group = M_BLOCK // int(q_heads)
    torch_dtype = torch.bfloat16 if dtype == cutlass.BFloat16 else torch.float16
    q = torch.empty(
        (1, q_group, int(q_heads), HEAD_DIM), dtype=torch_dtype, device=device
    )
    k = torch.empty((1, PAGE_TOKENS, 1, HEAD_DIM), dtype=torch_dtype, device=device)
    v = torch.empty_like(k)
    out = torch.empty_like(q)
    lse = (
        torch.empty((q_group, int(q_heads)), dtype=torch.float32, device=device)
        .transpose(0, 1)
        .unsqueeze(0)
        if has_lse
        else None
    )
    union_phys = torch.empty(
        (1, int(max_union_windows)), dtype=torch.int32, device=device
    )
    union_logical = torch.empty_like(union_phys)
    exact_mask = torch.empty_like(union_phys)
    union_counts = torch.empty((1,), dtype=torch.int32, device=device)
    work_q_global = torch.empty_like(union_counts)
    work_q_local = torch.empty_like(union_counts)
    work_q_len = torch.empty_like(union_counts)
    block_counts = torch.empty((1, 1, 1), dtype=torch.int32, device=device)
    block_indices = torch.empty(
        (1, 1, 1, max_blocks + 2), dtype=torch.int32, device=device
    )
    full_block_counts = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    exact_region_words = torch.empty(
        (1, max_blocks, q_group),
        dtype=torch.int32,
        device=device,
    )
    causal_token_limits = torch.empty(
        (1, q_group), dtype=torch.int32, device=device
    )
    union_phys4d = union_phys.view(1, 1, 1, int(max_union_windows))

    fq = _make_dynamic_fake(
        q, dynamic_shape_dims=(1,), dynamic_stride_dims=(0, 1, 2)
    )
    fk = _make_dynamic_fake(
        k, dynamic_shape_dims=(1,), dynamic_stride_dims=(0, 1, 2)
    )
    fv = _make_dynamic_fake(
        v, dynamic_shape_dims=(1,), dynamic_stride_dims=(0, 1, 2)
    )
    fout = _make_dynamic_fake(
        out, dynamic_shape_dims=(1,), dynamic_stride_dims=(0, 1, 2)
    )
    flse = (
        _make_dynamic_fake(
            lse,
            dynamic_shape_dims=(2,),
            dynamic_stride_dims=(0, 1),
            alignment=4,
        )
        if lse is not None
        else None
    )
    f_union_phys = _make_dynamic_fake(
        union_phys,
        dynamic_shape_dims=(0,),
        dynamic_stride_dims=(0,),
        alignment=4,
    )
    f_union_logical = _make_dynamic_fake(
        union_logical,
        dynamic_shape_dims=(0,),
        dynamic_stride_dims=(0,),
        alignment=4,
    )
    f_exact_mask = _make_dynamic_fake(
        exact_mask,
        dynamic_shape_dims=(0,),
        dynamic_stride_dims=(0,),
        alignment=4,
    )
    f_union_counts = _make_dynamic_fake(
        union_counts, dynamic_shape_dims=(0,), alignment=4
    )
    f_work_q_global = _make_dynamic_fake(
        work_q_global, dynamic_shape_dims=(0,), alignment=4
    )
    f_work_q_local = _make_dynamic_fake(
        work_q_local, dynamic_shape_dims=(0,), alignment=4
    )
    f_work_q_len = _make_dynamic_fake(
        work_q_len, dynamic_shape_dims=(0,), alignment=4
    )
    f_block_counts = _make_dynamic_fake(
        block_counts,
        dynamic_shape_dims=(2,),
        dynamic_stride_dims=(0, 1),
        alignment=4,
    )
    f_block_indices = _make_dynamic_fake(
        block_indices,
        dynamic_shape_dims=(2,),
        dynamic_stride_dims=(0, 1, 2),
        alignment=4,
    )
    f_full_block_counts = _make_dynamic_fake(
        full_block_counts,
        dynamic_shape_dims=(2,),
        dynamic_stride_dims=(0, 1),
        alignment=4,
    )
    f_exact_region_words = _make_dynamic_fake(
        exact_region_words,
        dynamic_shape_dims=(0,),
        dynamic_stride_dims=(0,),
        alignment=4,
    )
    f_causal_token_limits = _make_dynamic_fake(
        causal_token_limits,
        dynamic_shape_dims=(0,),
        dynamic_stride_dims=(0,),
        alignment=4,
    )
    f_union_phys4d = _make_dynamic_fake(
        union_phys4d,
        dynamic_shape_dims=(2,),
        dynamic_stride_dims=(0, 1, 2),
        alignment=4,
    )
    blocksparse = BlockSparseTensors(
        f_block_counts,
        f_block_indices,
        f_full_block_counts,
        f_union_phys4d,
    )
    aux = [
        f_union_phys,
        f_union_logical,
        f_exact_mask,
        f_union_counts,
        f_work_q_global,
        f_work_q_local,
        f_work_q_len,
        f_exact_region_words,
        f_causal_token_limits,
    ]
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    op = _get_union_exact_op(dtype, int(max_union_windows), int(q_heads))
    compile_options = "--enable-tvm-ffi --opt-level 2"
    if generate_line_info:
        compile_options += " --generate-line-info"
    return cute.compile(
        op,
        fq,
        fk,
        fv,
        fout,
        flse,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Int32(0),
        Float32(softmax_scale),
        stream_fake,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        blocksparse,
        aux,
        None,
        options=compile_options,
    )


def _validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    ndim: int,
    device: torch.device,
) -> None:
    if tensor.dtype != dtype or tensor.ndim != ndim:
        raise ValueError(
            f"{name} must be rank-{ndim} {dtype}, got shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype}"
        )
    if tensor.device != device or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA storage on {device}")


def _token_wise_flash_attn_prefill_union_sm100_gqa_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    union_counts: torch.Tensor,
    union_phys_indices: torch.Tensor,
    union_logical_indices: torch.Tensor,
    exact_mask_bits: torch.Tensor,
    causal_limits: torch.Tensor,
    work_q_global: torch.Tensor,
    work_q_local: torch.Tensor,
    work_q_len: torch.Tensor,
    workspace: Sm100UnionExactPrefillWorkspace,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    schedule_workspace: bool,
) -> torch.Tensor:
    device = q.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 10:
        raise RuntimeError("union-exact SM100 prefill requires compute capability 10.x")
    _validate_tensor(q, name="q", dtype=q.dtype, ndim=3, device=device)
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("union-exact SM100 prefill supports fp16/bf16")
    q_heads = int(q.shape[1])
    if q_heads not in SUPPORTED_Q_HEADS:
        raise ValueError(
            f"union-exact SM100 prefill supports q_heads in {SUPPORTED_Q_HEADS}, "
            f"got {q_heads}"
        )
    q_group = M_BLOCK // q_heads
    if tuple(q.shape[1:]) != (q_heads, HEAD_DIM):
        raise ValueError(
            f"union-exact SM100 prefill requires q shape (T, {q_heads}, {HEAD_DIM}), "
            f"got {tuple(q.shape)}"
        )
    for name, tensor in (("k_cache", k_cache), ("v_cache", v_cache)):
        _validate_tensor(tensor, name=name, dtype=q.dtype, ndim=4, device=device)
        if tuple(tensor.shape[1:]) != (PAGE_TOKENS, 1, HEAD_DIM):
            raise ValueError(
                f"{name} must have shape (pages, {PAGE_TOKENS}, 1, {HEAD_DIM}), "
                f"got {tuple(tensor.shape)}"
            )
    if int(k_cache.shape[0]) != int(v_cache.shape[0]):
        raise ValueError("k_cache and v_cache page counts must match")

    groups = int(union_counts.numel())
    # ``work_q_local/work_q_len`` are explicit request-local ranges.  A
    # multi-request MTP batch may therefore need one group per request even
    # when each request contributes fewer than ``q_group`` rows; requiring
    # ceil(total_q/q_group) here would merge request boundaries and drop rows.
    if groups <= 0 or groups > int(q.shape[0]):
        raise ValueError(
            "union metadata group count must be within the query-row range, "
            f"got groups={groups}, total_q={int(q.shape[0])}"
        )
    if workspace.group_capacity != groups:
        raise ValueError(
            f"workspace group capacity {workspace.group_capacity} != metadata groups {groups}"
        )
    if int(workspace.q_heads) != q_heads:
        raise ValueError(
            f"workspace q_heads {workspace.q_heads} != input q_heads {q_heads}"
        )
    if workspace.full_block_counts.device != device:
        raise ValueError("workspace tensors and inputs must be on the same device")
    if int(union_phys_indices.shape[1]) != int(workspace.max_union_windows):
        raise ValueError(
            "workspace max_union_windows does not match union metadata width, "
            f"got {workspace.max_union_windows} and {int(union_phys_indices.shape[1])}"
        )
    metadata_2d = (
        ("union_phys_indices", union_phys_indices),
        ("union_logical_indices", union_logical_indices),
        ("exact_mask_bits", exact_mask_bits),
    )
    for name, tensor in metadata_2d:
        _validate_tensor(tensor, name=name, dtype=torch.int32, ndim=2, device=device)
        if tuple(tensor.shape) != tuple(union_phys_indices.shape):
            raise ValueError(f"{name} shape must match union_phys_indices")
    for name, tensor in (
        ("union_counts", union_counts),
        ("work_q_local", work_q_local),
        ("work_q_len", work_q_len),
    ):
        _validate_tensor(tensor, name=name, dtype=torch.int32, ndim=1, device=device)
        if int(tensor.numel()) != groups:
            raise ValueError(f"{name} must have {groups} entries")
    _validate_tensor(
        causal_limits,
        name="causal_limits",
        dtype=torch.int32,
        ndim=2,
        device=device,
    )
    if tuple(causal_limits.shape) != (groups, q_group):
        raise ValueError(f"causal_limits must have shape {(groups, q_group)}")
    for name, tensor, align in (
        ("q", q, REQUIRED_POINTER_ALIGN),
        ("k_cache", k_cache, REQUIRED_POINTER_ALIGN),
        ("v_cache", v_cache, REQUIRED_POINTER_ALIGN),
        ("union_phys_indices", union_phys_indices, 4),
        ("union_logical_indices", union_logical_indices, 4),
        ("exact_mask_bits", exact_mask_bits, 4),
        ("union_counts", union_counts, 4),
    ):
        if int(tensor.data_ptr()) % align != 0:
            raise ValueError(f"{name} must be {align}-byte aligned")

    if out is None:
        out = torch.empty_like(q)
    _validate_tensor(out, name="out", dtype=q.dtype, ndim=3, device=device)
    if tuple(out.shape) != tuple(q.shape):
        raise ValueError("out shape must match q")
    if lse is not None:
        _validate_tensor(lse, name="lse", dtype=torch.float32, ndim=2, device=device)
        if tuple(lse.shape) != (int(q.shape[0]), q_heads):
            raise ValueError(f"lse must have shape {(int(q.shape[0]), q_heads)}")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(HEAD_DIM))

    if schedule_workspace:
        schedule_sm100_union_exact_prefill(
            union_counts,
            union_phys_indices,
            union_logical_indices,
            exact_mask_bits,
            causal_limits,
            work_q_global,
            work_q_local,
            work_q_len,
            workspace,
        )
    _debug_validate_schedule(
        union_counts,
        union_phys_indices,
        workspace,
        physical_region_count=(int(k_cache.numel()) // HEAD_DIM) // REGION_TOKENS,
    )
    q4 = q.unsqueeze(0)
    k4 = k_cache.view(1, -1, 1, HEAD_DIM)
    v4 = v_cache.view(1, -1, 1, HEAD_DIM)
    out4 = out.unsqueeze(0)
    lse4 = lse.transpose(0, 1).unsqueeze(0) if lse is not None else None
    blocksparse = BlockSparseTensors(
        workspace.block_counts.view(1, 1, groups),
        workspace.block_indices.view(
            1, 1, groups, workspace.max_blocks + 2
        ),
        workspace.full_block_counts.view(1, 1, groups),
        union_phys_indices.view(
            1, 1, groups, int(workspace.max_union_windows)
        ),
    )
    aux = [
        union_phys_indices,
        union_logical_indices,
        exact_mask_bits,
        union_counts,
        work_q_global,
        work_q_local,
        work_q_len,
        workspace.exact_region_words,
        workspace.causal_token_limits,
    ]
    dtype = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16
    compiled = _get_compiled_union_exact(
        dtype,
        int(workspace.max_union_windows),
        q_heads,
        bool(lse is not None),
        float(softmax_scale),
        sparse_utils.device_cache_key(device),
        os.getenv("OPTIMUS_SM100_PREFILL_GENERATE_LINE_INFO") == "1",
        UNION_EXACT_KERNEL_REVISION,
    )
    compiled(
        q4,
        k4,
        v4,
        out4,
        lse4,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Int32(0),
        Float32(softmax_scale),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        blocksparse,
        aux,
        None,
    )
    return out


def token_wise_flash_attn_prefill_union_sm100_gqa_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    union_counts: torch.Tensor,
    union_phys_indices: torch.Tensor,
    union_logical_indices: torch.Tensor,
    exact_mask_bits: torch.Tensor,
    causal_limits: torch.Tensor,
    work_q_global: torch.Tensor,
    work_q_local: torch.Tensor,
    work_q_len: torch.Tensor,
    workspace: Sm100UnionExactPrefillWorkspace,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    return _token_wise_flash_attn_prefill_union_sm100_gqa_impl(
        q,
        k_cache,
        v_cache,
        union_counts=union_counts,
        union_phys_indices=union_phys_indices,
        union_logical_indices=union_logical_indices,
        exact_mask_bits=exact_mask_bits,
        causal_limits=causal_limits,
        work_q_global=work_q_global,
        work_q_local=work_q_local,
        work_q_len=work_q_len,
        workspace=workspace,
        out=out,
        lse=lse,
        softmax_scale=softmax_scale,
        schedule_workspace=True,
    )


def token_wise_flash_attn_prefill_union_sm100_gqa_prepared_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    union_counts: torch.Tensor,
    union_phys_indices: torch.Tensor,
    union_logical_indices: torch.Tensor,
    exact_mask_bits: torch.Tensor,
    causal_limits: torch.Tensor,
    work_q_global: torch.Tensor,
    work_q_local: torch.Tensor,
    work_q_len: torch.Tensor,
    workspace: Sm100UnionExactPrefillWorkspace,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Launch attention after workspace preparation on the same CUDA stream."""
    return _token_wise_flash_attn_prefill_union_sm100_gqa_impl(
        q,
        k_cache,
        v_cache,
        union_counts=union_counts,
        union_phys_indices=union_phys_indices,
        union_logical_indices=union_logical_indices,
        exact_mask_bits=exact_mask_bits,
        causal_limits=causal_limits,
        work_q_global=work_q_global,
        work_q_local=work_q_local,
        work_q_len=work_q_len,
        workspace=workspace,
        out=out,
        lse=lse,
        softmax_scale=softmax_scale,
        schedule_workspace=False,
    )


__all__ = [
    "Sm100UnionExactPrefillWorkspace",
    "make_sm100_union_exact_prefill_workspace",
    "schedule_sm100_union_exact_prefill",
    "token_wise_flash_attn_prefill_union_sm100_gqa_func",
    "token_wise_flash_attn_prefill_union_sm100_gqa_prepared_func",
]
