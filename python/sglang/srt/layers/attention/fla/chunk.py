# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
from einops import rearrange

from sglang.srt.layers.attention.fla.chunk_bwd import (
    chunk_bwd_dqkwg,
    chunk_bwd_dv_local,
)
from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_delta_h_bwd import (
    chunk_gated_delta_rule_bwd_dhu,
)
from sglang.srt.layers.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
)
from sglang.srt.layers.attention.fla.l2norm import l2norm_bwd, l2norm_fwd, l2norm_fwd_with_rstd
from sglang.srt.layers.attention.fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
)
from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd
from sglang.srt.layers.attention.fla.wy_fast_bwd import prepare_wy_repr_bwd

CHUNK_SIZE = 64


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: torch.LongTensor | None = None,
):
    g = chunk_local_cumsum(
        g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )

    # fused kkt + solve_tril + recompute_w_u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    return g, o, A, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        q_orig = q
        k_orig = k
        q_rstd = None
        k_rstd = None

        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd_with_rstd(q)
            k, k_rstd = l2norm_fwd_with_rstd(k)

        B, T, H, K = q.shape
        V = v.shape[-1]
        if initial_state_indices is None:
            initial_state_indices = torch.arange(B, device=q.device, dtype=torch.long)
        has_initial_state = initial_state is not None
        if not has_initial_state:
            N = B if cu_seqlens is None else len(cu_seqlens) - 1
            initial_state = q.new_zeros(N, H, V, K, dtype=torch.float32)

        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
            if cu_seqlens is not None
            else None
        )
        initial_state_saved = initial_state.clone() if has_initial_state else initial_state
        g, o, A, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        ctx.save_for_backward(q, k, v, g, beta, A, initial_state_saved,
                              initial_state_indices, cu_seqlens, chunk_indices)
        if use_qk_l2norm_in_kernel:
            ctx.q_rstd = q_rstd
            ctx.k_rstd = k_rstd
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.has_initial_state = has_initial_state

        return o.to(q.dtype), h

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dh_all):
        q, k, v, g, beta, A, initial_state, initial_state_indices, cu_seqlens, chunk_indices = ctx.saved_tensors
        scale = ctx.scale

        # dh_all is grad w.r.t. h [B, NT, H, V, K] -- typically None in training
        # We don't propagate gradients through intermediate hidden states here;
        # only final state gradient (dht) matters for the recurrence backward.
        dht = None

        # Recompute w, u from forward
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            g_cumsum=g,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        # Recompute h and v_new (do NOT update initial_state in-place)
        h, v_new = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            save_new_value=True,
            inplace_update=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        # Step 1: dv from local attention (intra-chunk)
        dv = chunk_bwd_dv_local(
            q=q,
            k=k,
            do=do,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        # Step 2: backward hidden state recurrence
        dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            h0=initial_state,
            dht=dht,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        # Step 3: dq, dk, dw, dg from output and hidden gradients
        dq, dk, dw, dg = chunk_bwd_dqkwg(
            q=q,
            k=k,
            v=v_new,
            do=do,
            h=h,
            dh=dh,
            w=w,
            g=g,
            dv=dv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        # Step 4: WY representation backward
        dk2, dv, db, dg2 = prepare_wy_repr_bwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            dw=dw,
            du=dv,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        dk.add_(dk2)
        if dg2 is not None:
            dg.add_(dg2)

        # Reverse cumsum for dg
        dg = chunk_local_cumsum(
            dg, chunk_size=CHUNK_SIZE, reverse=True,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        )

        # l2norm backward if needed
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, ctx.q_rstd, dq)
            dk = l2norm_bwd(k, ctx.k_rstd, dk)

        return dq.to(q), dk.to(k), dv.to(v), dg.to(beta), db.to(beta), None, dh0 if ctx.has_initial_state else None, None, None, None


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert (
        q.dtype != torch.float32
    ), "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert (
        len(beta.shape) == 3
    ), "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta, g = map(
            lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g)
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if (
            initial_state_indices is not None
            and initial_state_indices.shape[0] != len(cu_seqlens) - 1
        ):
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state_indices.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, h = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, None, h
