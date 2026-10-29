# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This file has been modified from its original version (the channel tile and a token-major output).
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/mamba/causal_conv1d_triton.py (causal_conv1d_update).
"""The causal conv update of a GDN target-verify step on SGLang's Triton kernel, with
the two launch choices ``causal_conv1d_update`` fixes: the channel tile (``block_n``)
and a token-major output, which the token-major projection views need (their rows
have gaps, so ``empty_like`` would make the output channel-major).

Only the verify case: no circular cache_seqlens, no accept counts, no tree mask.
"""

from __future__ import annotations

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/mamba/causal_conv1d_triton.py": (
        "_causal_conv1d_update_kernel",
        "PAD_SLOT_ID",
    ),
    "kernels/jit/utils/__init__.py": ("is_arch_support_pdl",),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "kernels/ops/mamba/causal_conv1d_triton.py:causal_conv1d_update": "3394efa20c88d580",
}


def conv_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias,
    activation,
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    *,
    block_n: int = 256,
    token_major_out: bool = False,
    hoist: bool = False,
    num_warps: int = 4,
    window_mode: int = 0,
) -> torch.Tensor:
    """``causal_conv1d_update`` of x (batch, dim, seqlen) for a verify step, with the
    channel tile ``block_n`` and, with ``token_major_out``, an output whose
    transpose(1, 2) is contiguous."""
    import triton
    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.kernels.ops.mamba.causal_conv1d_triton import (
        PAD_SLOT_ID,
        _causal_conv1d_update_kernel,
    )

    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()
    out = (
        torch.empty((batch, seqlen, dim), dtype=x.dtype, device=x.device).transpose(
            1, 2
        )
        if token_major_out
        else torch.empty_like(x)
    )
    state_len = width - 1
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}

    def grid(meta):
        return (batch, triton.cdiv(dim, meta["BLOCK_N"]))

    if hoist:
        if width != 4 or seqlen > 8 or not seqlen > state_len:
            raise ValueError(f"the hoisted conv takes width 4 and {state_len} < rows <= 8, not {width} / {seqlen}")
        st = intermediate_conv_window.stride()
        if window_mode == 1 and not (st[1] == 1 and st[3] == 1 and st[2] >= seqlen + 2):
            raise ValueError(f"window rows need the deduplicated window layout, not strides {st}")
        _HOISTED[0] = _HOISTED[0] or _hoisted_kernel()
        _HOISTED[0][grid](
            x, weight, bias, conv_state, conv_state_indices, intermediate_conv_window,
            intermediate_state_indices, out,
            dim, seqlen, state_len, num_cache_lines,
            *x.stride(), *weight.stride(), *conv_state.stride(), conv_state_indices.stride(0),
            *intermediate_conv_window.stride(), intermediate_state_indices.stride(0), *out.stride(),
            PAD_SLOT_ID,
            HAS_BIAS=bias is not None,
            SILU_ACTIVATION=activation in ["silu", "swish"],
            NP2_STATELEN=triton.next_power_of_2(state_len),
            BLOCK_N=block_n,
            USE_GDC=bool(pdl_kwargs),
            WINDOW_MODE=window_mode,
            num_warps=num_warps,
            **({"launch_pdl": True} if pdl_kwargs else {}),
        )
        return out

    _causal_conv1d_update_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        None,
        conv_state_indices,
        None,
        intermediate_conv_window,
        intermediate_state_indices,
        None,
        None,
        None,
        out,
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        *x.stride(),
        *weight.stride(),
        *conv_state.stride(),
        conv_state_indices.stride(0),
        *intermediate_conv_window.stride(),
        intermediate_state_indices.stride(0),
        0,
        0,
        0,
        0,
        0,
        0,
        *out.stride(),
        PAD_SLOT_ID,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=True,
        IS_SPEC_DECODING=False,
        NP2_STATELEN=triton.next_power_of_2(state_len),
        NP2_SEQLEN=triton.next_power_of_2(seqlen),
        USE_PAD_SLOT=True,
        BLOCK_N=block_n,
        SAVE_INTERMEDIATE=True,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=False,
        num_warps=num_warps,
        **pdl_kwargs,
    )
    return out


_HOISTED = [None]


def _hoisted_kernel():
    """SGLang's ``_causal_conv1d_update_kernel`` for this verify case (continuous batching, pad slots, saved
    windows, no tree, width 4) with every token's x row loaded before the first store: the stores of a token
    no longer serialize the next token's load (the compiler cannot move a load above a possibly aliasing
    store). Same values, same arithmetic order: bit for bit SGLang's kernel."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        x_ptr, w_ptr, bias_ptr, conv_state_ptr, conv_state_indices_ptr, inter_ptr, inter_indices_ptr, o_ptr,
        dim: tl.constexpr, seqlen: tl.constexpr, state_len: tl.constexpr, num_cache_lines: tl.constexpr,
        stride_x_seq: tl.constexpr, stride_x_dim: tl.constexpr, stride_x_token: tl.constexpr,
        stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
        stride_conv_state_seq: tl.constexpr, stride_conv_state_dim: tl.constexpr, stride_conv_state_tok: tl.constexpr,
        stride_state_indices: tl.constexpr,
        stride_inter_seq: tl.constexpr, stride_inter_step: tl.constexpr, stride_inter_dim: tl.constexpr,
        stride_inter_win: tl.constexpr, stride_inter_indices: tl.constexpr,
        stride_o_seq: tl.constexpr, stride_o_dim: tl.constexpr, stride_o_token: tl.constexpr,
        pad_slot_id: tl.constexpr,
        HAS_BIAS: tl.constexpr, SILU_ACTIVATION: tl.constexpr, NP2_STATELEN: tl.constexpr, BLOCK_N: tl.constexpr,
        USE_GDC: tl.constexpr, WINDOW_MODE: tl.constexpr,
    ):
        if USE_GDC:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        idx_seq = tl.program_id(0)
        idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices).to(tl.int64)
        inter_coord = tl.load(inter_indices_ptr + idx_seq * stride_inter_indices).to(tl.int64)
        if conv_state_batch_coord == pad_slot_id:
            return
        # STEP 1: the prior window
        conv_states_base = (
            conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim)
        )
        mask_w = idx_feats < dim
        col0 = tl.load(conv_states_base, mask_w, 0.0)
        col1 = tl.load(conv_states_base + 1 * stride_conv_state_tok, mask_w, 0.0)
        col2 = tl.load(conv_states_base + 2 * stride_conv_state_tok, mask_w, 0.0)
        prior1 = col1
        prior2 = col2
        # every token's x row, before any store
        x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)
        x0 = tl.load(x_base + 0 * stride_x_token, mask=mask_w)
        if seqlen > 1:
            x1 = tl.load(x_base + 1 * stride_x_token, mask=mask_w)
        if seqlen > 2:
            x2 = tl.load(x_base + 2 * stride_x_token, mask=mask_w)
        if seqlen > 3:
            x3 = tl.load(x_base + 3 * stride_x_token, mask=mask_w)
        if seqlen > 4:
            x4 = tl.load(x_base + 4 * stride_x_token, mask=mask_w)
        if seqlen > 5:
            x5 = tl.load(x_base + 5 * stride_x_token, mask=mask_w)
        if seqlen > 6:
            x6 = tl.load(x_base + 6 * stride_x_token, mask=mask_w)
        if seqlen > 7:
            x7 = tl.load(x_base + 7 * stride_x_token, mask=mask_w)
        # STEP 2: the new conv state (state_len < seqlen: the last state_len x rows)
        idx_tokens = tl.arange(0, NP2_STATELEN)
        conv_state_ptrs_source = (
            conv_state_ptr
            + (conv_state_batch_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)[None, :]
            + ((idx_tokens + seqlen) * stride_conv_state_tok)[:, None]
        )
        mask = (
            (conv_state_batch_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :]
        )
        conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)
        VAL = state_len - seqlen
        x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
        mask_x = (
            (idx_tokens - VAL >= 0)[:, None] & (idx_tokens - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
        )
        loaded_x = tl.load(x_ptrs, mask_x, 0.0)
        tl.debug_barrier()
        new_conv_state = tl.where(mask, conv_state, loaded_x)
        conv_state_ptrs_target = (
            conv_state_ptr
            + (conv_state_batch_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )[None, :] + (idx_tokens * stride_conv_state_tok)[:, None]
        tl.store(conv_state_ptrs_target, new_conv_state, (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :])
        # STEP 3 / 4: bias and weights
        if HAS_BIAS:
            acc_preload = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
        else:
            acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
        w_base = w_ptr + (idx_feats * stride_w_dim)
        w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
        w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
        w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
        w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)
        # STEP 5: each token
        for idx_token in tl.static_range(seqlen):
            acc = acc_preload
            if idx_token == 0:
                xt = x0
            elif idx_token == 1:
                xt = x1
            elif idx_token == 2:
                xt = x2
            elif idx_token == 3:
                xt = x3
            elif idx_token == 4:
                xt = x4
            elif idx_token == 5:
                xt = x5
            elif idx_token == 6:
                xt = x6
            else:
                xt = x7
            acc += col0 * w_col0
            acc += col1 * w_col1
            acc += col2 * w_col2
            acc += xt * w_col3
            col0 = col1
            col1 = col2
            col2 = xt
            if WINDOW_MODE == 0:
                base_ptr = (
                    inter_ptr + inter_coord * stride_inter_seq + idx_token * stride_inter_step
                    + idx_feats * stride_inter_dim
                )
                tl.store(base_ptr + 0 * stride_inter_win, col0, mask=mask_w)
                tl.store(base_ptr + 1 * stride_inter_win, col1, mask=mask_w)
                tl.store(base_ptr + 2 * stride_inter_win, col2, mask=mask_w)
            if SILU_ACTIVATION:
                acc = acc / (1 + tl.exp(-acc))
            tl.store(
                o_ptr + idx_seq * stride_o_seq + idx_token * stride_o_token + (idx_feats * stride_o_dim),
                acc,
                mask=(idx_token < seqlen) & (idx_feats < dim),
            )
        if WINDOW_MODE == 1:
            # the deduplicated windows (window[t][w] = row[t + w]) as one row a channel: the prior two state
            # columns, then every x row; each element once, the values the per-token stores leave there
            j = tl.arange(0, 16)
            row = tl.where(j[None, :] == 0, prior1[:, None], tl.zeros((BLOCK_N, 16), dtype=prior1.dtype))
            row = tl.where(j[None, :] == 1, prior2[:, None], row)
            row = tl.where(j[None, :] == 2, x0[:, None], row)
            if seqlen > 1:
                row = tl.where(j[None, :] == 3, x1[:, None], row)
            if seqlen > 2:
                row = tl.where(j[None, :] == 4, x2[:, None], row)
            if seqlen > 3:
                row = tl.where(j[None, :] == 5, x3[:, None], row)
            if seqlen > 4:
                row = tl.where(j[None, :] == 6, x4[:, None], row)
            if seqlen > 5:
                row = tl.where(j[None, :] == 7, x5[:, None], row)
            if seqlen > 6:
                row = tl.where(j[None, :] == 8, x6[:, None], row)
            if seqlen > 7:
                row = tl.where(j[None, :] == 9, x7[:, None], row)
            tl.store(
                inter_ptr + inter_coord * stride_inter_seq + (idx_feats * stride_inter_dim)[:, None] + j[None, :],
                row,
                mask=(idx_feats < dim)[:, None] & (j < seqlen + 2)[None, :],
            )

    return kernel
