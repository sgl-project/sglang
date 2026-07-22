from functools import partial
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, Int8, Int32, const_expr

from sglang.kernels.ops.attention.flash_attn.cute.block_sparse_utils import (
    get_curr_blocksparse_tensors,
)
from sglang.kernels.ops.attention.flash_attn.cute.block_sparsity import (
    BlockSparseTensors,
    BlockSparseTensorsTorch,
    to_cute_block_sparse_tensors,
)
from sglang.kernels.ops.attention.flash_attn.cute.cute_dsl_utils import (
    get_aux_tensor_metadata,
    to_cute_aux_tensor,
    to_cute_tensor,
)
from sglang.kernels.ops.attention.flash_attn.cute.mask import call_mask_mod
from sglang.kernels.ops.attention.flash_attn.cute.seqlen_info import SeqlenInfoQK
from sglang.kernels.ops.attention.flash_attn.cute.testing import is_fake_mode
from sglang.kernels.ops.attention.flash_attn.cute.utils import (
    AuxData,
    get_batch_from_cu_tensor,
    hash_callable,
    scalar_to_ssa,
    ssa_to_scalar,
)


class BlockSparsityKernel:
    """Block sparsity kernel for FlexAttention.

    This kernel computes `mask_mod` for every token of each block
    to determine if an n block is full, masked, or neither.

    Writes block counts and indices to a BlockSparseTensors object.

    When use_fast_sampling=True, uses 5-point sampling (4 corners + center)
    which is much faster but only suitable for masks where this is sufficient.

    TODO:
        - optimize mask_mod evaluation
        - transposed tensors for bwd pass
    """

    def __init__(
        self,
        mask_mod: Callable,
        tile_mn: Tuple[int, int],
        compute_full_blocks: bool = True,
        use_aux_tensors: bool = False,
        use_fast_sampling: bool = False,
    ):
        self.mask_mod = mask_mod
        self.tile_mn = tile_mn
        self.compute_full_blocks = compute_full_blocks
        self.use_aux_tensors = use_aux_tensors
        self.use_fast_sampling = use_fast_sampling

    @cute.jit
    def __call__(
        self,
        blocksparse_tensors: BlockSparseTensors,
        seqlen_q: Int32,
        seqlen_k: Int32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
    ):
        (
            mask_cnt,
            mask_idx,
            full_cnt,
            full_idx,
            mCuTotalMBlocks,
            mCuBlockIdxOffsets,
            *_,
        ) = blocksparse_tensors

        self.is_varlen_q = const_expr(mCuSeqlensQ is not None)

        if const_expr(self.compute_full_blocks):
            assert (
                full_cnt is not None and full_idx is not None
            ), "full block tensors must be provided when computing full blocks"
        if const_expr(not self.is_varlen_q):
            batch_size, num_heads, num_m_blocks, _ = mask_idx.shape
            total_m_blocks = batch_size * num_m_blocks
        else:
            assert const_expr(
                mCuTotalMBlocks is not None
            ), "mCuTotalMBlocks must be provided when varlen q"
            num_heads, total_m_blocks = mask_cnt.shape  # num_m_blocks is total_m_blocks
            batch_size = mCuSeqlensQ.shape[0] - 1

        if const_expr(self.use_fast_sampling):
            num_threads = 5
            self.num_warps = 1
        else:
            num_threads = self.tile_mn[0]
            self.num_warps = (num_threads + 32 - 1) // 32

        if const_expr(not self.is_varlen_q):
            grid = [num_m_blocks, num_heads, batch_size]
        else:
            grid = [total_m_blocks, num_heads, 1]

        self.kernel(
            blocksparse_tensors,
            seqlen_q,
            seqlen_k,
            batch_size,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mCuTotalMBlocks,
            mCuBlockIdxOffsets,
            aux_data,
        ).launch(grid=grid, block=[num_threads, 1, 1])

    @cute.kernel
    def kernel(
        self,
        blocksparse_tensors: BlockSparseTensors,
        seqlen_q: Int32,
        seqlen_k: Int32,
        batch_size: Int32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        mCuBlockIdxOffsets: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()

        ssa = partial(scalar_to_ssa, dtype=Int32)

        @cute.struct
        class SharedStorage:
            reduction_buffer_smem: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, 2 * self.num_warps], 1024
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)

        reduction_buffer = storage.reduction_buffer_smem.get_tensor(
            cute.make_layout((self.num_warps, 2))
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=seqlen_q,
            seqlen_k_static=seqlen_k,
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=mCuTotalMBlocks,
            mCuBlockIdxOffsets=mCuBlockIdxOffsets,
            tile_m=self.tile_mn[0],
            tile_n=self.tile_mn[1],
        )

        if const_expr(not self.is_varlen_q):
            m_block, head_idx, batch_idx = cute.arch.block_idx()
        else:
            global_m_block, head_idx, _ = cute.arch.block_idx()
            batch_idx = get_batch_from_cu_tensor(global_m_block, mCuTotalMBlocks)
            m_block = global_m_block - mCuTotalMBlocks[batch_idx]

        seqlen = SeqlenInfoCls(batch_idx)
        seqlen_q = seqlen.seqlen_q
        seqlen_k = seqlen.seqlen_k
        global_m_block = seqlen.m_block_offset + m_block

        num_n_blocks = (seqlen_k + self.tile_mn[1] - 1) // self.tile_mn[1]

        _, curr_mask_idx, _, curr_full_idx = get_curr_blocksparse_tensors(
            batch_idx, head_idx, m_block, blocksparse_tensors, seqlen
        )

        num_mask_blocks = Int32(0)
        num_full_blocks = Int32(0)

        m_base = m_block * self.tile_mn[0]
        if const_expr(self.use_fast_sampling):
            # Loop-invariant per-thread q_idx for the 5 sample points
            # (tidx 0, 1: top corners; 2, 3: bottom corners; 4: center).
            q_idx_sample = m_base
            if tidx == 2 or tidx == 3:
                q_idx_sample = cutlass.min(m_base + self.tile_mn[0] - 1, seqlen_q - 1)
            elif tidx == 4:
                q_idx_sample = (
                    m_base + cutlass.min(seqlen_q - m_base, self.tile_mn[0]) // 2
                )
        else:
            q_idx_thread = m_base + tidx
            thread_in_bounds = Boolean(
                tidx < self.tile_mn[0] and q_idx_thread < seqlen_q
            )

        for n_block in cutlass.range(num_n_blocks):
            n_base = n_block * self.tile_mn[1]

            if const_expr(self.use_fast_sampling):
                # 5-point sampling (4 corners + center). Interior n_blocks
                # (n_base + tile_n <= seqlen_k) skip the OOB clamp on the right /
                # center samples.
                is_interior = (n_base + self.tile_mn[1]) <= seqlen_k
                n_right = Int32(0)
                n_mid = Int32(0)
                if is_interior:
                    n_right = n_base + self.tile_mn[1] - 1
                    n_mid = n_base + self.tile_mn[1] // 2
                else:
                    n_right = cutlass.min(n_base + self.tile_mn[1] - 1, seqlen_k - 1)
                    n_mid = (
                        n_base + cutlass.min(seqlen_k - n_base, self.tile_mn[1]) // 2
                    )

                kv_idx = n_base
                if tidx == 1 or tidx == 3:
                    kv_idx = n_right
                elif tidx == 4:
                    kv_idx = n_mid

                thread_result = Boolean(False)
                thread_is_valid = Boolean(False)
                if tidx < 5:
                    thread_is_valid = Boolean(True)
                    thread_result = ssa_to_scalar(
                        call_mask_mod(
                            self.mask_mod,
                            ssa(batch_idx),
                            ssa(head_idx),
                            ssa(q_idx_sample),
                            ssa(kv_idx),
                            seqlen,
                            aux_data,
                        )
                    )

                has_unmasked = cute.arch.vote_any_sync(thread_result & thread_is_valid)
                has_masked = cute.arch.vote_any_sync(
                    Boolean(not thread_result) & thread_is_valid
                )

            else:
                # Full path. Interior blocks (n_base + tile_n <= seqlen_k) drop the
                # per-element bound check; the boundary block (at most one) keeps it.
                thread_has_unmasked = Boolean(False)
                thread_has_masked = Boolean(False)
                kv_idx = Int32(0)
                is_interior = (n_base + self.tile_mn[1]) <= seqlen_k

                if is_interior:
                    if thread_in_bounds:
                        for c in cutlass.range(self.tile_mn[1], unroll_full=True):
                            mask_val = ssa_to_scalar(
                                call_mask_mod(
                                    self.mask_mod,
                                    ssa(batch_idx),
                                    ssa(head_idx),
                                    ssa(q_idx_thread),
                                    ssa(n_base + c),
                                    seqlen,
                                    aux_data,
                                )
                            )
                            thread_has_unmasked |= Boolean(mask_val)
                            thread_has_masked |= Boolean(not mask_val)
                else:
                    if thread_in_bounds:
                        for c in cutlass.range(self.tile_mn[1], unroll_full=True):
                            kv_idx = n_base + c
                            if kv_idx < seqlen_k:
                                mask_val = ssa_to_scalar(
                                    call_mask_mod(
                                        self.mask_mod,
                                        ssa(batch_idx),
                                        ssa(head_idx),
                                        ssa(q_idx_thread),
                                        ssa(kv_idx),
                                        seqlen,
                                        aux_data,
                                    )
                                )
                                thread_has_unmasked |= Boolean(mask_val)
                                thread_has_masked |= Boolean(not mask_val)

                warp_unmasked = cute.arch.vote_any_sync(
                    thread_has_unmasked & thread_in_bounds
                )
                warp_masked = cute.arch.vote_any_sync(
                    thread_has_masked & thread_in_bounds
                )
                if lane_id == 0:
                    reduction_buffer[warp_idx, 0] = (
                        Int8(1) if warp_unmasked else Int8(0)
                    )
                    reduction_buffer[warp_idx, 1] = Int8(1) if warp_masked else Int8(0)
                cute.arch.sync_threads()

                # Cross-warp OR via warp 0; thread 0 (lane 0 of warp 0) holds the result.
                has_unmasked = Boolean(False)
                has_masked = Boolean(False)
                if warp_idx == 0:
                    lane_unmasked = Boolean(False)
                    lane_masked = Boolean(False)
                    if lane_id < self.num_warps:
                        lane_unmasked = reduction_buffer[lane_id, 0] != Int8(0)
                        lane_masked = reduction_buffer[lane_id, 1] != Int8(0)
                    has_unmasked = cute.arch.vote_any_sync(lane_unmasked)
                    has_masked = cute.arch.vote_any_sync(lane_masked)

            # Only thread 0 updates the output arrays (common to both paths)
            if tidx == 0:
                # Block classification based on what we found:
                # - If has_masked and has_unmasked: partial block (needs masking)
                # - If only has_unmasked: full block (no masking needed)
                # - If only has_masked: skip this block entirely
                is_partial = Boolean(has_masked and has_unmasked)
                is_full = Boolean(has_unmasked and (not has_masked))

                if is_partial:
                    curr_mask_idx[num_mask_blocks] = n_block
                    num_mask_blocks += 1
                elif is_full and const_expr(self.compute_full_blocks):
                    curr_full_idx[num_full_blocks] = n_block
                    num_full_blocks += 1

        # Only thread 0 writes back the counts
        if tidx == 0:
            mask_cnt, _, full_cnt, *_ = blocksparse_tensors
            if const_expr(self.is_varlen_q):
                mask_cnt[head_idx, global_m_block] = num_mask_blocks
                if const_expr(self.compute_full_blocks):
                    full_cnt[head_idx, global_m_block] = num_full_blocks
            else:
                mask_cnt[batch_idx, head_idx, m_block] = num_mask_blocks
                if const_expr(self.compute_full_blocks):
                    full_cnt[batch_idx, head_idx, m_block] = num_full_blocks


def compute_block_sparsity(
    tile_m,
    tile_n,
    batch_size,
    num_heads,
    seqlen_q,
    seqlen_k,
    mask_mod: Callable,
    aux_tensors: Optional[list],
    device,
    aux_scalars: Optional[tuple] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cu_total_m_blocks: Optional[torch.Tensor] = None,
    cu_block_idx_offsets: Optional[torch.Tensor] = None,
    compute_full_blocks: bool = True,
    use_fast_sampling: bool = False,
) -> BlockSparseTensorsTorch:
    """
    Computes block sparsity for a given `mask_mod`.

    Args:
        tile_m: The tile size for the m dimension.
        tile_n: The tile size for the n dimension.
        batch_size: The batch size.
        num_heads: The number of heads.
        seqlen_q: The sequence length for the query.
        seqlen_k: The sequence length for the key.
        mask_mod: The `mask_mod` callable to use.
        aux_tensors: A list of auxiliary tensors.
        device: The device to use.
        cu_seqlens_q: Cumulative q sequence lengths for varlen
        cu_seqlens_k: Cumulative k sequence lengths for varlen
        seqused_q: Per-batch effective q sequence lengths
        seqused_k: Per-batch effective k sequence lengths
        cu_total_m_blocks: Cumulative total m blocks tensor for varlen q
        cu_block_idx_offsets: Cumulative offsets into the packed mask_block_idx /
            full_block_idx tensors per batch (== cumsum of M_b * N_b).
        compute_full_blocks: Whether to compute full blocks. If False, only partially-masked blocks are computed.
        use_fast_sampling: Whether to use 5-point sampling (4 corners + center). This is much faster, but only suitable for masks where this check is sufficient.

    Returns:
        BlockSparseTensorsTorch
    """
    aux_scalars = tuple(aux_scalars) if aux_scalars else None

    # Check if mask_mod is marked as suitable for 5-point sampling
    use_fast_sampling = getattr(mask_mod, "use_fast_sampling", use_fast_sampling)

    num_m_blocks = (seqlen_q + tile_m - 1) // tile_m
    num_n_blocks = (seqlen_k + tile_n - 1) // tile_n

    if cu_seqlens_q is not None:
        assert (
            cu_total_m_blocks is not None
        ), "total m blocks must be provided when varlen q"
        total_m_blocks = cu_total_m_blocks[-1].item()
        if cu_block_idx_offsets is None and (
            cu_seqlens_k is not None or seqused_k is not None
        ):
            # Derive cu_block_idx_offsets from per-batch K seqlens.
            cu_block_idx_offsets_list = [0]
            for batch_idx in range(batch_size):
                batch_seqlen_q = (
                    cu_seqlens_q[batch_idx + 1].item() - cu_seqlens_q[batch_idx].item()
                )
                if cu_seqlens_k is not None:
                    batch_seqlen_k = (
                        cu_seqlens_k[batch_idx + 1].item()
                        - cu_seqlens_k[batch_idx].item()
                    )
                else:
                    batch_seqlen_k = seqused_k[batch_idx].item()
                num_m_blocks_batch = (batch_seqlen_q + tile_m - 1) // tile_m
                num_n_blocks_batch = (batch_seqlen_k + tile_n - 1) // tile_n
                cu_block_idx_offsets_list.append(
                    cu_block_idx_offsets_list[-1]
                    + num_m_blocks_batch * num_n_blocks_batch
                )
            cu_block_idx_offsets = torch.tensor(
                cu_block_idx_offsets_list, dtype=torch.int32, device=device
            )
        if cu_block_idx_offsets is not None:
            total_n_blocks = cu_block_idx_offsets[-1].item()
        else:
            # Uniform-K varlen-Q: every batch has the same K seqlen.
            total_n_blocks = total_m_blocks * num_n_blocks

        mask_block_cnt = torch.zeros(
            (num_heads, total_m_blocks), device=device, dtype=torch.int32
        )
        mask_block_idx = torch.zeros(
            (num_heads, total_n_blocks), device=device, dtype=torch.int32
        )
        full_block_cnt = (
            torch.zeros((num_heads, total_m_blocks), device=device, dtype=torch.int32)
            if compute_full_blocks
            else None
        )
        full_block_idx = (
            torch.zeros((num_heads, total_n_blocks), device=device, dtype=torch.int32)
            if compute_full_blocks
            else None
        )
    else:
        total_m_blocks = batch_size * num_m_blocks
        total_n_blocks = batch_size * num_m_blocks * num_n_blocks

        mask_block_cnt = torch.zeros(
            (batch_size, num_heads, num_m_blocks), device=device, dtype=torch.int32
        )
        mask_block_idx = torch.zeros(
            (batch_size, num_heads, num_m_blocks, num_n_blocks),
            device=device,
            dtype=torch.int32,
        )
        full_block_cnt = (
            torch.zeros(
                (batch_size, num_heads, num_m_blocks), device=device, dtype=torch.int32
            )
            if compute_full_blocks
            else None
        )
        full_block_idx = (
            torch.zeros(
                (batch_size, num_heads, num_m_blocks, num_n_blocks),
                device=device,
                dtype=torch.int32,
            )
            if compute_full_blocks
            else None
        )

    blocksparse_tensors_torch = BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        cu_total_m_blocks=cu_total_m_blocks,
        cu_block_idx_offsets=cu_block_idx_offsets,
        block_size=(tile_m, tile_n),
    )

    mask_mod_hash = hash_callable(mask_mod)
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None
    aux_scalar_metadata = (
        tuple(type(s) for s in aux_scalars) if aux_scalars is not None else None
    )

    compile_key = (
        tile_m,
        tile_n,
        mask_mod_hash,
        aux_tensor_metadata,
        aux_scalar_metadata,
        compute_full_blocks,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        aux_tensors is not None,
        use_fast_sampling,
    )
    if compile_key not in compute_block_sparsity.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
            for t in (
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
            )
        ]
        blocksparse_tensors = to_cute_block_sparse_tensors(
            blocksparse_tensors_torch, enable_tvm_ffi=True
        )
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]
        else:
            cute_aux_tensors = None
        kernel = BlockSparsityKernel(
            mask_mod,
            tile_mn=(tile_m, tile_n),
            compute_full_blocks=compute_full_blocks,
            use_aux_tensors=aux_tensors is not None,
            use_fast_sampling=use_fast_sampling,
        )

        compute_block_sparsity.compile_cache[compile_key] = cute.compile(
            kernel,
            blocksparse_tensors,
            seqlen_q,
            seqlen_k,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            AuxData(cute_aux_tensors, aux_scalars),
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        compute_block_sparsity.compile_cache[compile_key](
            (
                blocksparse_tensors_torch.mask_block_cnt,
                blocksparse_tensors_torch.mask_block_idx,
                blocksparse_tensors_torch.full_block_cnt,
                blocksparse_tensors_torch.full_block_idx,
                blocksparse_tensors_torch.cu_total_m_blocks,
                blocksparse_tensors_torch.cu_block_idx_offsets,
                blocksparse_tensors_torch.dq_write_order,
                blocksparse_tensors_torch.dq_write_order_full,
            ),
            seqlen_q,
            seqlen_k,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            AuxData(aux_tensors, aux_scalars),
        )

    return blocksparse_tensors_torch


compute_block_sparsity.compile_cache = {}
