from __future__ import annotations

"""
CuTe implementations of the draft-extend padding/unpadding kernels.

These mirror the Triton kernels `pad_draft_extend_query_kernel` and
`unpad_draft_extend_output_kernel`, but are written in Python CuTe
for environments that prefer the CuTe DSL.

Inputs/outputs are torch tensors on CUDA; sequence/accept lengths are int32;
BLOCK_HEAD and BLOCK_DIM control per-CTA tiling over heads/dim.
"""

from dataclasses import dataclass

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.runtime import from_dlpack


@dataclass
class _PadKernelConfig:
    block_head: int = 32
    block_dim: int = 32


class CutePadDraftExtendQueryKernel:
    """
    CuTe kernel that pads draft queries into a dense (batch, max_seq_len, num_heads, head_dim) tensor.
    """

    def __init__(self, block_head: int = 32, block_dim: int = 32):
        self.cfg = _PadKernelConfig(block_head=block_head, block_dim=block_dim)
        self._compiled_kernel = None
        self._compile_key = None

    def __call__(
        self,
        q: torch.Tensor,  # [total_seq_len, num_heads, head_dim]
        padded_q: torch.Tensor,  # [batch_size, max_seq_len, num_heads, head_dim]
        seq_lens_q: torch.Tensor,  # [batch_size], int32
        cumsum: torch.Tensor,  # [batch_size + 1], int32
    ) -> None:
        assert q.is_cuda and padded_q.is_cuda
        batch_size, max_seq_len, num_heads, head_dim = padded_q.shape

        # Compile kernel on first use or if shape changed
        compile_key = (num_heads, head_dim, self.cfg.block_head, self.cfg.block_dim)
        if self._compiled_kernel is None or self._compile_key != compile_key:
            # Create fake tensors for compilation
            batch_sym = cute.sym_int()
            total_seq_sym = cute.sym_int()
            mQ_fake = cute.runtime.make_fake_tensor(
                q.dtype, (total_seq_sym, num_heads, head_dim)
            )
            mPadded_fake = cute.runtime.make_fake_tensor(
                padded_q.dtype, (batch_sym, max_seq_len, num_heads, head_dim)
            )
            mSeqLens_fake = cute.runtime.make_fake_tensor(
                seq_lens_q.dtype, (batch_sym,)
            )
            mCumsum_fake = cute.runtime.make_fake_tensor(cumsum.dtype, (batch_sym + 1,))

            # Compile the kernel
            self._compiled_kernel = cute.compile(
                self.kernel,
                mQ_fake,
                mPadded_fake,
                mSeqLens_fake,
                mCumsum_fake,
                Int32(batch_size),
                Int32(max_seq_len),
                Int32(num_heads),
                Int32(head_dim),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
            self._compile_key = compile_key

        # Convert torch tensors to CuTe tensors
        mQ = from_dlpack(q)
        mPadded = from_dlpack(padded_q)
        mSeqLens = from_dlpack(seq_lens_q)
        mCumsum = from_dlpack(cumsum)

        # Call compiled kernel
        self._compiled_kernel(
            mQ,
            mPadded,
            mSeqLens,
            mCumsum,
            Int32(batch_size),
            Int32(max_seq_len),
            Int32(num_heads),
            Int32(head_dim),
        )

    @cute.jit
    def kernel(
        self,
        mQ: cute.Tensor,  # [total_seq, num_heads, head_dim]
        mPadded: cute.Tensor,  # [batch, max_seq, num_heads, head_dim]
        mSeqLens: cute.Tensor,  # [batch]
        mCumsum: cute.Tensor,  # [batch + 1]
        batch_size: Int32,
        max_seq_len: Int32,
        num_heads: Int32,
        head_dim: Int32,
    ):
        grid = (
            batch_size * max_seq_len,
            cute.ceil_div(num_heads, self.cfg.block_head),
            cute.ceil_div(head_dim, self.cfg.block_dim),
        )
        block = (self.cfg.block_head, self.cfg.block_dim, 1)

        self.kernel_impl(
            mQ,
            mPadded,
            mSeqLens,
            mCumsum,
            batch_size,
            max_seq_len,
            num_heads,
            head_dim,
        ).launch(grid=grid, block=block, stream=cutlass.get_default_stream())

    @cute.kernel
    def kernel_impl(
        self,
        mQ: cute.Tensor,  # [total_seq, num_heads, head_dim]
        mPadded: cute.Tensor,  # [batch, max_seq, num_heads, head_dim]
        mSeqLens: cute.Tensor,  # [batch]
        mCumsum: cute.Tensor,  # [batch + 1]
        batch_size: Int32,
        max_seq_len: Int32,
        num_heads: Int32,
        head_dim: Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        block_head = const_expr(self.cfg.block_head)
        block_dim = const_expr(self.cfg.block_dim)

        batch_id = bidx // max_seq_len
        seq_pos = bidx - batch_id * max_seq_len
        if batch_id >= batch_size:
            return

        seq_len = mSeqLens[batch_id]
        if seq_pos >= seq_len:
            return

        head_id = bidy * block_head + tidx
        dim_id = bidz * block_dim + tidy

        if head_id < num_heads and dim_id < head_dim:
            input_start = mCumsum[batch_id]
            input_pos = input_start + seq_pos
            mPadded[batch_id, seq_pos, head_id, dim_id] = mQ[input_pos, head_id, dim_id]


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
        self._compiled_kernel = None
        self._compile_key = None

    def __call__(
        self,
        raw_out: torch.Tensor,  # [batch_size, token_per_batch, tp_q_head_num, v_head_dim]
        output: torch.Tensor,  # [total_tokens, tp_q_head_num, v_head_dim]
        accept_lengths: torch.Tensor,  # [batch_size], int32
        cumsum: torch.Tensor,  # [batch_size + 1], int32
    ) -> None:
        assert raw_out.is_cuda and output.is_cuda
        batch_size, token_per_batch, tp_q_head_num, v_head_dim = raw_out.shape

        # Compile kernel on first use or if shape changed
        compile_key = (
            tp_q_head_num,
            v_head_dim,
            self.cfg.block_head,
            self.cfg.block_dim,
        )
        if self._compiled_kernel is None or self._compile_key != compile_key:
            # Create fake tensors for compilation
            batch_sym = cute.sym_int()
            total_tokens_sym = cute.sym_int()
            mRaw_fake = cute.runtime.make_fake_tensor(
                raw_out.dtype, (batch_sym, token_per_batch, tp_q_head_num, v_head_dim)
            )
            mOut_fake = cute.runtime.make_fake_tensor(
                output.dtype, (total_tokens_sym, tp_q_head_num, v_head_dim)
            )
            mAccept_fake = cute.runtime.make_fake_tensor(
                accept_lengths.dtype, (batch_sym,)
            )
            mCumsum_fake = cute.runtime.make_fake_tensor(cumsum.dtype, (batch_sym + 1,))

            # Compile the kernel
            self._compiled_kernel = cute.compile(
                self.kernel,
                mRaw_fake,
                mOut_fake,
                mAccept_fake,
                mCumsum_fake,
                Int32(batch_size),
                Int32(token_per_batch),
                Int32(tp_q_head_num),
                Int32(v_head_dim),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
            self._compile_key = compile_key

        # Convert torch tensors to CuTe tensors
        mRaw = from_dlpack(raw_out)
        mOut = from_dlpack(output)
        mAccept = from_dlpack(accept_lengths)
        mCumsum = from_dlpack(cumsum)

        # Call compiled kernel
        self._compiled_kernel(
            mRaw,
            mOut,
            mAccept,
            mCumsum,
            Int32(batch_size),
            Int32(token_per_batch),
            Int32(tp_q_head_num),
            Int32(v_head_dim),
        )

    @cute.jit
    def kernel(
        self,
        mRaw: cute.Tensor,  # [batch, token_per_batch, heads, dim]
        mOut: cute.Tensor,  # [total_tokens, heads, dim]
        mAccept: cute.Tensor,  # [batch]
        mCumsum: cute.Tensor,  # [batch + 1]
        batch_size: Int32,
        token_per_batch: Int32,
        tp_q_head_num: Int32,
        v_head_dim: Int32,
    ):
        grid = (
            batch_size * token_per_batch,
            cute.ceil_div(tp_q_head_num, self.cfg.block_head),
            cute.ceil_div(v_head_dim, self.cfg.block_dim),
        )
        block = (self.cfg.block_head, self.cfg.block_dim, 1)

        self.kernel_impl(
            mRaw,
            mOut,
            mAccept,
            mCumsum,
            batch_size,
            token_per_batch,
            tp_q_head_num,
            v_head_dim,
        ).launch(grid=grid, block=block, stream=cutlass.get_default_stream())

    @cute.kernel
    def kernel_impl(
        self,
        mRaw: cute.Tensor,  # [batch, token_per_batch, heads, dim]
        mOut: cute.Tensor,  # [total_tokens, heads, dim]
        mAccept: cute.Tensor,  # [batch]
        mCumsum: cute.Tensor,  # [batch + 1]
        batch_size: Int32,
        token_per_batch: Int32,
        tp_q_head_num: Int32,
        v_head_dim: Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        block_head = const_expr(self.cfg.block_head)
        block_dim = const_expr(self.cfg.block_dim)

        batch_id = bidx // token_per_batch
        seq_pos = bidx - batch_id * token_per_batch
        if batch_id >= batch_size:
            return

        accept_len = mAccept[batch_id]
        if seq_pos >= accept_len:
            return

        head_id = bidy * block_head + tidx
        dim_id = bidz * block_dim + tidy

        if head_id < tp_q_head_num and dim_id < v_head_dim:
            output_start = mCumsum[batch_id]
            output_pos = output_start + seq_pos
            mOut[output_pos, head_id, dim_id] = mRaw[batch_id, seq_pos, head_id, dim_id]
