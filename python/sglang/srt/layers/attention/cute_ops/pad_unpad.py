from __future__ import annotations

"""
CuTe implementations of the draft-extend padding/unpadding kernels.

These mirror the Triton kernels `pad_draft_extend_query_kernel` and
`unpad_draft_extend_output_kernel`, but are written in Python CuTe
for environments that prefer the CuTe DSL.

Inputs/outputs are torch tensors on CUDA; sequence/accept lengths are int32;
BLOCK_HEAD and BLOCK_DIM control per-CTA tiling over heads/dim.
"""

import math
from dataclasses import dataclass

import torch

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.runtime import make_fake_compact_tensor


@dataclass
class _PadKernelConfig:
    block_head: int = 32
    block_dim: int = 32


def _torch_dtype_to_cutlass_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    if dtype == torch.int32:
        return cutlass.Int32
    raise TypeError(f"Unsupported dtype for CuTe fake tensor: {dtype}")


class CutePadDraftExtendQueryKernel:
    """
    CuTe kernel that pads draft queries into a dense (batch, max_seq_len, num_heads, head_dim) tensor.
    """

    def __init__(self, block_head: int = 32, block_dim: int = 32):
        self.cfg = _PadKernelConfig(block_head=block_head, block_dim=block_dim)
        self._compiled = {}

    def __call__(
        self,
        q: torch.Tensor,  # [total_seq_len, num_heads, head_dim]
        padded_q: torch.Tensor,  # [batch_size, max_seq_len, num_heads, head_dim]
        seq_lens_q: torch.Tensor,  # [batch_size], int32
        cumsum: torch.Tensor,  # [batch_size + 1], int32
    ) -> None:
        assert q.is_cuda and padded_q.is_cuda and seq_lens_q.is_cuda and cumsum.is_cuda
        assert seq_lens_q.dtype == torch.int32 and cumsum.dtype == torch.int32
        # Kernel only reads cumsum[batch_id], so we can drop the trailing element.
        cumsum = cumsum[:-1].contiguous()

        # Compile once per (dtype, max_seq_len, heads, dim, tile)
        max_seq_len = padded_q.shape[1]
        num_heads = padded_q.shape[2]
        head_dim = padded_q.shape[3]
        compile_key = (
            q.dtype,
            max_seq_len,
            num_heads,
            head_dim,
            self.cfg.block_head,
            self.cfg.block_dim,
        )
        compiled = self._compiled.get(compile_key)
        if compiled is None:
            batch_sym = cute.sym_int32()
            total_seq_sym = cute.sym_int32()

            q_dt = _torch_dtype_to_cutlass_dtype(q.dtype)
            i32 = cutlass.Int32

            # Row-major compact tensors: last dim contiguous
            mQ_fake = make_fake_compact_tensor(
                q_dt, (total_seq_sym, num_heads, head_dim), stride_order=(2, 1, 0)
            )
            mPadded_fake = make_fake_compact_tensor(
                q_dt,
                (batch_sym, max_seq_len, num_heads, head_dim),
                stride_order=(3, 2, 1, 0),
            )
            mSeqLens_fake = make_fake_compact_tensor(
                i32, (batch_sym,), stride_order=(0,)
            )
            # Avoid SymInt + int; we only need cumsum[batch_id].
            mCumsum_fake = make_fake_compact_tensor(
                i32, (batch_sym,), stride_order=(0,)
            )

            compiled = cute.compile(
                self.op,
                mQ_fake,
                mPadded_fake,
                mSeqLens_fake,
                mCumsum_fake,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
            self._compiled[compile_key] = compiled

        # compiled callable accepts torch tensors directly
        compiled(q, padded_q, seq_lens_q, cumsum)

    @cute.jit
    def op(
        self,
        mQ: cute.Tensor,  # [total_seq, heads, dim]
        mPadded: cute.Tensor,  # [batch, max_seq, heads, dim]
        mSeqLens: cute.Tensor,  # [batch]
        mCumsum: cute.Tensor,  # [batch + 1]
        stream,
    ):
        batch_size = mPadded.shape[0]
        max_seq_len = mPadded.shape[1]
        num_heads = mPadded.shape[2]
        head_dim = mPadded.shape[3]

        # Vectorized copy along contiguous dim (up to 128-bit).
        # NOTE: These must be compile-time constants; don't pass into kernel args.
        elem_width = const_expr(mPadded.element_type.width)
        max_vec = const_expr(128 // elem_width)
        vec = const_expr(math.gcd(self.cfg.block_dim, max_vec))
        vec = const_expr(max(1, vec))
        gmem_threads_per_row = const_expr(self.cfg.block_dim // vec)
        num_threads = const_expr(self.cfg.block_head * gmem_threads_per_row)

        grid = (
            batch_size * max_seq_len,
            cute.ceil_div(num_heads, self.cfg.block_head),
            cute.ceil_div(head_dim, self.cfg.block_dim),
        )
        block = (num_threads, 1, 1)

        self.kernel(
            mQ,
            mPadded,
            mSeqLens,
            mCumsum,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # [total_seq, heads, dim]
        mPadded: cute.Tensor,  # [batch, max_seq, heads, dim]
        mSeqLens: cute.Tensor,  # [batch]
        mCumsum: cute.Tensor,  # [batch + 1]
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        block_head = const_expr(self.cfg.block_head)
        block_dim = const_expr(self.cfg.block_dim)

        max_seq_len = mPadded.shape[1]
        num_heads = mPadded.shape[2]
        head_dim = mPadded.shape[3]

        batch_id = bidx // max_seq_len
        seq_pos = bidx - batch_id * max_seq_len

        seq_len = mSeqLens[batch_id]
        token_valid = seq_pos < seq_len

        # Tile views over (head, dim)
        input_pos = mCumsum[batch_id] + seq_pos
        # Create rank-2 tiles (head, dim) by slicing away singleton modes.
        gQ = cute.local_tile(mQ, (1, block_head, block_dim), (input_pos, bidy, bidz))
        gQ = cute.slice_(gQ, (0, None, None))
        gP = cute.local_tile(
            mPadded, (1, 1, block_head, block_dim), (batch_id, seq_pos, bidy, bidz)
        )
        gP = cute.slice_(gP, (0, 0, None, None))

        # Identity coords for predication
        idHD = cute.make_identity_tensor((num_heads, head_dim))
        cHD = cute.local_tile(idHD, (block_head, block_dim), (bidy, bidz))

        elem_width = const_expr(mPadded.element_type.width)
        max_vec = const_expr(128 // elem_width)
        vec = const_expr(math.gcd(block_dim, max_vec))
        vec = const_expr(max(1, vec))
        gmem_threads_per_row = const_expr(block_dim // vec)
        num_copy_bits = const_expr(vec * elem_width)

        copy_op = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(
            copy_op, mPadded.element_type, num_bits_per_copy=num_copy_bits
        )

        # Thread/value layout: threads cover rows (heads) and contiguous dim vectors.
        thr_layout = cute.make_ordered_layout(
            (block_head, gmem_threads_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, num_copy_bits // mPadded.element_type.width))
        tiled_copy = cute.make_tiled_copy_tv(
            copy_atom, thr_layout, val_layout
        ).get_slice(tidx)

        tQgQ = tiled_copy.partition_S(gQ)
        tPgP = tiled_copy.partition_D(gP)
        tQc = tiled_copy.partition_S(cHD)
        # Per-element predicate: token valid + in-bounds head/dim
        pred = (tQc[0][0] < num_heads) & (tQc[0][1] < head_dim) & token_valid
        cute.copy(copy_atom, tQgQ, tPgP, pred=pred)


@dataclass
class _UnpadKernelConfig:
    block_head: int = 32
    block_dim: int = 32


class CuteUnpadDraftExtendOutputKernel:
    """
    CuTe kernel that unpads draft outputs back into a flat [total_tokens, heads, dim] tensor.
    """

    def __init__(self, block_head: int = 32, block_dim: int = 32):
        self.cfg = _UnpadKernelConfig(block_head=block_head, block_dim=block_dim)
        self._compiled = {}

    def __call__(
        self,
        raw_out: torch.Tensor,  # [batch_size, token_per_batch, tp_q_head_num, v_head_dim]
        output: torch.Tensor,  # [total_tokens, tp_q_head_num, v_head_dim]
        accept_lengths: torch.Tensor,  # [batch_size], int32
        cumsum: torch.Tensor,  # [batch_size + 1], int32
    ) -> None:
        assert (
            raw_out.is_cuda
            and output.is_cuda
            and accept_lengths.is_cuda
            and cumsum.is_cuda
        )
        assert accept_lengths.dtype == torch.int32 and cumsum.dtype == torch.int32
        # Kernel only reads cumsum[batch_id], so we can drop the trailing element.
        cumsum = cumsum[:-1].contiguous()

        token_per_batch = raw_out.shape[1]
        tp_q_head_num = raw_out.shape[2]
        v_head_dim = raw_out.shape[3]
        compile_key = (
            raw_out.dtype,
            token_per_batch,
            tp_q_head_num,
            v_head_dim,
            self.cfg.block_head,
            self.cfg.block_dim,
        )
        compiled = self._compiled.get(compile_key)
        if compiled is None:
            batch_sym = cute.sym_int32()
            total_tokens_sym = cute.sym_int32()

            out_dt = _torch_dtype_to_cutlass_dtype(raw_out.dtype)
            i32 = cutlass.Int32

            mRaw_fake = make_fake_compact_tensor(
                out_dt,
                (batch_sym, token_per_batch, tp_q_head_num, v_head_dim),
                stride_order=(3, 2, 1, 0),
            )
            mOut_fake = make_fake_compact_tensor(
                out_dt,
                (total_tokens_sym, tp_q_head_num, v_head_dim),
                stride_order=(2, 1, 0),
            )
            mAccept_fake = make_fake_compact_tensor(
                i32, (batch_sym,), stride_order=(0,)
            )
            # Avoid SymInt + int; we only need cumsum[batch_id].
            mCumsum_fake = make_fake_compact_tensor(
                i32, (batch_sym,), stride_order=(0,)
            )

            compiled = cute.compile(
                self.op,
                mRaw_fake,
                mOut_fake,
                mAccept_fake,
                mCumsum_fake,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
            self._compiled[compile_key] = compiled

        compiled(raw_out, output, accept_lengths, cumsum)

    @cute.jit
    def op(
        self,
        mRaw: cute.Tensor,  # [batch, token_per_batch, heads, dim]
        mOut: cute.Tensor,  # [total_tokens, heads, dim]
        mAccept: cute.Tensor,  # [batch]
        mCumsum: cute.Tensor,  # [batch + 1]
        stream,
    ):
        batch_size = mRaw.shape[0]
        token_per_batch = mRaw.shape[1]
        tp_q_head_num = mRaw.shape[2]
        v_head_dim = mRaw.shape[3]

        # Vectorized copy along contiguous dim (up to 128-bit).
        elem_width = const_expr(mRaw.element_type.width)
        max_vec = const_expr(128 // elem_width)
        vec = const_expr(math.gcd(self.cfg.block_dim, max_vec))
        vec = const_expr(max(1, vec))
        gmem_threads_per_row = const_expr(self.cfg.block_dim // vec)
        num_threads = const_expr(self.cfg.block_head * gmem_threads_per_row)

        grid = (
            batch_size * token_per_batch,
            cute.ceil_div(tp_q_head_num, self.cfg.block_head),
            cute.ceil_div(v_head_dim, self.cfg.block_dim),
        )
        block = (num_threads, 1, 1)

        self.kernel(
            mRaw,
            mOut,
            mAccept,
            mCumsum,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mRaw: cute.Tensor,  # [batch, token_per_batch, heads, dim]
        mOut: cute.Tensor,  # [total_tokens, heads, dim]
        mAccept: cute.Tensor,  # [batch]
        mCumsum: cute.Tensor,  # [batch + 1]
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        block_head = const_expr(self.cfg.block_head)
        block_dim = const_expr(self.cfg.block_dim)

        token_per_batch = mRaw.shape[1]
        tp_q_head_num = mRaw.shape[2]
        v_head_dim = mRaw.shape[3]

        batch_id = bidx // token_per_batch
        seq_pos = bidx - batch_id * token_per_batch

        accept_len = mAccept[batch_id]
        token_valid = seq_pos < accept_len

        output_pos = mCumsum[batch_id] + seq_pos
        # Create rank-2 tiles (head, dim) by slicing away singleton modes.
        gR = cute.local_tile(
            mRaw, (1, 1, block_head, block_dim), (batch_id, seq_pos, bidy, bidz)
        )
        gR = cute.slice_(gR, (0, 0, None, None))
        gO = cute.local_tile(mOut, (1, block_head, block_dim), (output_pos, bidy, bidz))
        gO = cute.slice_(gO, (0, None, None))

        idHD = cute.make_identity_tensor((tp_q_head_num, v_head_dim))
        cHD = cute.local_tile(idHD, (block_head, block_dim), (bidy, bidz))

        elem_width = const_expr(mRaw.element_type.width)
        max_vec = const_expr(128 // elem_width)
        vec = const_expr(math.gcd(block_dim, max_vec))
        vec = const_expr(max(1, vec))
        gmem_threads_per_row = const_expr(block_dim // vec)
        num_copy_bits = const_expr(vec * elem_width)

        copy_op = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(
            copy_op, mRaw.element_type, num_bits_per_copy=num_copy_bits
        )
        thr_layout = cute.make_ordered_layout(
            (block_head, gmem_threads_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, num_copy_bits // mRaw.element_type.width))
        tiled_copy = cute.make_tiled_copy_tv(
            copy_atom, thr_layout, val_layout
        ).get_slice(tidx)

        tRgR = tiled_copy.partition_S(gR)
        tOgO = tiled_copy.partition_D(gO)
        tRc = tiled_copy.partition_S(cHD)
        pred = (tRc[0][0] < tp_q_head_num) & (tRc[0][1] < v_head_dim) & token_valid
        cute.copy(copy_atom, tRgR, tOgO, pred=pred)
