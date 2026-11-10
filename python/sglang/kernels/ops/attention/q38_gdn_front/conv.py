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

import logging
from typing import Optional

import msgspec
import torch

logger = logging.getLogger(__name__)

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

    if not hoist and token_major_out and _RETILE_CONFIG is not None:
        reason = retile_unsupported_reason(
            x=x, conv_state=conv_state, weight=weight,
            intermediate_conv_window=intermediate_conv_window,
        )
        _log_retile_once(reason=reason, batch=batch, dim=dim, seqlen=seqlen)
        if reason is None:
            launch_retiled(
                x=x, conv_state=conv_state, weight=weight, bias=bias,
                silu=activation in ["silu", "swish"],
                conv_state_indices=conv_state_indices,
                intermediate_conv_window=intermediate_conv_window,
                intermediate_state_indices=intermediate_state_indices,
                out=out, config=_RETILE_CONFIG, use_gdc=bool(pdl_kwargs),
            )
            return out

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


class RetileConfig(msgspec.Struct, frozen=True, kw_only=True):
    block_n: int
    num_warps: int
    # batch rows per program (a power of 2)
    rows: int = 1
    # 0: every load after the PDL wait; 1: the conv weights before it; 2: also the conv-state
    # indices and the prior conv state (written only by earlier forwards, never by a kernel
    # of this verify before the conv)
    prefetch: int = 0


# Measured on VR200 at verify batches 4-104; re-tune when the kernel changes.
RETILE_DEFAULT = RetileConfig(block_n=256, num_warps=4, rows=1, prefetch=2)


def parse_retile_config(*, enabled: bool, raw: Optional[str]) -> Optional[RetileConfig]:
    """SGLANG_GDN_VERIFY_CONV_RETILE_CONFIG: "block_n,num_warps[,rows[,prefetch]]"."""
    if not enabled:
        return None
    if raw is None or not raw.strip():
        return RETILE_DEFAULT
    vals = [int(v) for v in raw.split(",")]
    if not 2 <= len(vals) <= 4:
        raise ValueError(f"SGLANG_GDN_VERIFY_CONV_RETILE_CONFIG takes 2-4 integers, not {raw!r}")
    cfg = RetileConfig(**dict(zip(("block_n", "num_warps", "rows", "prefetch"), vals)))

    def pow2(v, lo, hi):
        return lo <= v <= hi and v & (v - 1) == 0

    if not (pow2(cfg.block_n, 16, 4096) and pow2(cfg.num_warps, 1, 16) and pow2(cfg.rows, 1, 16)
            and cfg.prefetch in (0, 1, 2)):
        raise ValueError(
            "SGLANG_GDN_VERIFY_CONV_RETILE_CONFIG: block_n a power of 2 in 16-4096, num_warps in "
            f"1-16, rows in 1-16 (powers of 2), prefetch 0-2; got {raw!r}"
        )
    return cfg


def _retile_config_from_env() -> Optional[RetileConfig]:
    from sglang.srt.environ import envs

    return parse_retile_config(
        enabled=envs.SGLANG_ENABLE_GDN_VERIFY_CONV_RETILE.get(),
        raw=envs.SGLANG_GDN_VERIFY_CONV_RETILE_CONFIG.get(),
    )


_RETILE_CONFIG = _retile_config_from_env()
_retile_logged: set = set()


def retile_unsupported_reason(*, x, conv_state, weight, intermediate_conv_window) -> Optional[str]:
    """None when the retiled kernel reproduces ``_causal_conv1d_update_kernel`` for this call."""
    seqlen, width = x.shape[2], weight.shape[1]
    if width != 4 or conv_state.shape[-1] != width - 1:
        return f"conv width {width}, state length {conv_state.shape[-1]} (needs 4, 3)"
    if not width - 1 < seqlen <= 8:
        return f"{seqlen} verify tokens (needs 4-8)"
    if not (x.dtype == conv_state.dtype == intermediate_conv_window.dtype):
        return f"dtypes x {x.dtype}, state {conv_state.dtype}, window {intermediate_conv_window.dtype}"
    if intermediate_conv_window.dim() != 4:
        return f"window of {intermediate_conv_window.dim()} dims (needs [slot, step, dim, win])"
    return None


def _log_retile_once(*, reason: Optional[str], batch: int, dim: int, seqlen: int) -> None:
    key = reason is None
    if key in _retile_logged:
        return
    _retile_logged.add(key)
    if reason is None:
        logger.info(
            "SGLANG_ENABLE_GDN_VERIFY_CONV_RETILE engaged: %s, dim %d, %d tokens, first batch %d",
            _RETILE_CONFIG, dim, seqlen, batch,
        )
    else:
        logger.warning("SGLANG_ENABLE_GDN_VERIFY_CONV_RETILE not taken (stock conv): %s", reason)


def launch_retiled(
    *, x, conv_state, weight, bias, silu: bool, conv_state_indices, intermediate_conv_window,
    intermediate_state_indices, out, config: RetileConfig, use_gdc: bool,
) -> None:
    """The verify conv of ``conv_update`` (continuous batching, pad slots, saved windows, no
    tree, width 4, more verify tokens than state columns) on a [rows, block_n] tile: the
    same values and the same arithmetic per element, so bit for bit SGLang's kernel. A
    deduplicated window view (step stride == window stride) is written as one row a channel,
    as sgl-project/sglang#46184 does in ``_causal_conv1d_update_kernel``."""
    import triton
    from sglang.kernels.ops.mamba.causal_conv1d_triton import PAD_SLOT_ID

    batch, dim, seqlen = x.shape
    win_strides = intermediate_conv_window.stride()
    _RETILED[0] = _RETILED[0] or _retiled_kernel()
    _RETILED[0][(triton.cdiv(batch, config.rows), triton.cdiv(dim, config.block_n))](
        x, weight, bias, conv_state, conv_state_indices, intermediate_conv_window,
        intermediate_state_indices, out,
        batch, dim, seqlen,
        *x.stride(), *weight.stride(), *conv_state.stride(), conv_state_indices.stride(0),
        *win_strides, intermediate_state_indices.stride(0), *out.stride(),
        PAD_SLOT_ID,
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=silu,
        BLOCK_N=config.block_n,
        ROWS=config.rows,
        PREFETCH=config.prefetch,
        WINDOW_TILE=win_strides[1] == win_strides[3],
        NP2_WIN=triton.next_power_of_2(seqlen + 2),
        USE_GDC=use_gdc,
        num_warps=config.num_warps,
        **({"launch_pdl": True} if use_gdc else {}),
    )


_RETILED = [None]


def _retiled_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _gdn_verify_conv_retiled_kernel(
        x_ptr, w_ptr, bias_ptr, conv_state_ptr, conv_state_indices_ptr, inter_ptr, inter_indices_ptr, o_ptr,
        batch, dim: tl.constexpr, seqlen: tl.constexpr,
        stride_x_seq: tl.constexpr, stride_x_dim: tl.constexpr, stride_x_token: tl.constexpr,
        stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
        stride_cs_seq: tl.constexpr, stride_cs_dim: tl.constexpr, stride_cs_tok: tl.constexpr,
        stride_csi: tl.constexpr,
        stride_inter_seq: tl.constexpr, stride_inter_step: tl.constexpr, stride_inter_dim: tl.constexpr,
        stride_inter_win: tl.constexpr, stride_ii: tl.constexpr,
        stride_o_seq: tl.constexpr, stride_o_dim: tl.constexpr, stride_o_token: tl.constexpr,
        pad_slot_id: tl.constexpr,
        HAS_BIAS: tl.constexpr, SILU_ACTIVATION: tl.constexpr, BLOCK_N: tl.constexpr, ROWS: tl.constexpr,
        PREFETCH: tl.constexpr, WINDOW_TILE: tl.constexpr, NP2_WIN: tl.constexpr, USE_GDC: tl.constexpr,
    ):
        feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        fmask = feats < dim
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        if USE_GDC and PREFETCH == 0:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        w_base = w_ptr + feats * stride_w_dim
        w0 = tl.load(w_base + 0 * stride_w_width, fmask, other=0.0)
        w1 = tl.load(w_base + 1 * stride_w_width, fmask, other=0.0)
        w2 = tl.load(w_base + 2 * stride_w_width, fmask, other=0.0)
        w3 = tl.load(w_base + 3 * stride_w_width, fmask, other=0.0)
        if USE_GDC and PREFETCH == 1:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        in_batch = rows < batch
        coord = tl.load(conv_state_indices_ptr + rows * stride_csi, mask=in_batch, other=pad_slot_id).to(tl.int64)
        # padded rows (pad slot) write nothing, as the stock kernel's early return
        m = (in_batch & (coord != pad_slot_id))[:, None] & fmask[None, :]
        cs = conv_state_ptr + coord[:, None] * stride_cs_seq + (feats * stride_cs_dim)[None, :]
        col0 = tl.load(cs, m, 0.0)
        col1 = tl.load(cs + 1 * stride_cs_tok, m, 0.0)
        col2 = tl.load(cs + 2 * stride_cs_tok, m, 0.0)
        if USE_GDC and PREFETCH == 2:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        # after the wait: under SGLANG_ENABLE_PP_SPEC the launch just before the conv writes it
        inter_coord = tl.load(inter_indices_ptr + rows * stride_ii, mask=in_batch, other=0).to(tl.int64)
        prior1 = col1
        prior2 = col2
        # every x row before the first store (a store may alias a later load for the compiler)
        xb = x_ptr + rows[:, None] * stride_x_seq + (feats * stride_x_dim)[None, :]
        x0 = tl.load(xb + 0 * stride_x_token, mask=m)
        x1 = tl.load(xb + 1 * stride_x_token, mask=m)
        x2 = tl.load(xb + 2 * stride_x_token, mask=m)
        x3 = tl.load(xb + 3 * stride_x_token, mask=m)
        if seqlen > 4:
            x4 = tl.load(xb + 4 * stride_x_token, mask=m)
        if seqlen > 5:
            x5 = tl.load(xb + 5 * stride_x_token, mask=m)
        if seqlen > 6:
            x6 = tl.load(xb + 6 * stride_x_token, mask=m)
        if seqlen > 7:
            x7 = tl.load(xb + 7 * stride_x_token, mask=m)
        # the new conv state: the last 3 x rows (stock STEP 2 with seqlen > state_len)
        n0 = tl.load(xb + (seqlen - 3) * stride_x_token, mask=m, other=0.0)
        n1 = tl.load(xb + (seqlen - 2) * stride_x_token, mask=m, other=0.0)
        n2 = tl.load(xb + (seqlen - 1) * stride_x_token, mask=m, other=0.0)
        if WINDOW_TILE:
            # step t's window is columns t..t+2 of [state[1], state[2], x_0, ..., x_{seqlen-1}]
            c = tl.arange(0, NP2_WIN)
            c_ok = c < seqlen + 2
            xw = tl.load(
                xb[:, :, None] + ((c - 2) * stride_x_token)[None, None, :],
                mask=m[:, :, None] & ((c >= 2) & c_ok)[None, None, :],
                other=0.0,
            )
            win = tl.where(
                (c == 0)[None, None, :],
                prior1[:, :, None],
                tl.where((c == 1)[None, None, :], prior2[:, :, None], xw),
            )
        tl.store(cs, n0, m)
        tl.store(cs + 1 * stride_cs_tok, n1, m)
        tl.store(cs + 2 * stride_cs_tok, n2, m)
        inter_base = inter_ptr + inter_coord[:, None] * stride_inter_seq + (feats * stride_inter_dim)[None, :]
        if WINDOW_TILE:
            tl.store(
                inter_base[:, :, None] + (c * stride_inter_win)[None, None, :],
                win,
                mask=m[:, :, None] & c_ok[None, None, :],
            )
        if HAS_BIAS:
            b = tl.load(bias_ptr + feats, mask=fmask, other=0.0).to(tl.float32)
            acc_preload = tl.broadcast_to(b[None, :], (ROWS, BLOCK_N))
        else:
            acc_preload = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
        ob = o_ptr + rows[:, None] * stride_o_seq + (feats * stride_o_dim)[None, :]
        for t in tl.static_range(seqlen):
            if t == 0:
                xt = x0
            elif t == 1:
                xt = x1
            elif t == 2:
                xt = x2
            elif t == 3:
                xt = x3
            elif t == 4:
                xt = x4
            elif t == 5:
                xt = x5
            elif t == 6:
                xt = x6
            else:
                xt = x7
            # the stock kernel's order: state columns oldest first, then the token
            acc = acc_preload
            acc += col0 * w0[None, :]
            acc += col1 * w1[None, :]
            acc += col2 * w2[None, :]
            acc += xt * w3[None, :]
            col0 = col1
            col1 = col2
            col2 = xt
            if not WINDOW_TILE:
                step = inter_base + t * stride_inter_step
                tl.store(step + 0 * stride_inter_win, col0, mask=m)
                tl.store(step + 1 * stride_inter_win, col1, mask=m)
                tl.store(step + 2 * stride_inter_win, col2, mask=m)
            if SILU_ACTIVATION:
                acc = acc / (1 + tl.exp(-acc))
            tl.store(ob + t * stride_o_token, acc, mask=m)

    return _gdn_verify_conv_retiled_kernel


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
