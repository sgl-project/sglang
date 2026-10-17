# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
from typing import Optional

import torch
from einops import rearrange

from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.kernels.ops.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.fla.utils import (
    SUPPRESS_LEVEL,
    autocast_custom_fwd,
    input_guard,
    is_intel,
)
from sglang.srt.utils import is_hip, rank0_log

if is_intel:
    from sglang.srt.hardware_backend.xpu.kernels.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )
    from sglang.srt.hardware_backend.xpu.kernels.fla.chunk_fwd import (
        chunk_gated_delta_rule_fwd_intra,
    )

CHUNK_SIZE = 64

# Optional FlyDSL (aiter) prefill state-matrix kernel. Drop-in for the Triton
# `chunk_gated_delta_rule_fwd_h` (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`),
# ~1.5x faster at Qwen3.5 shapes on gfx950. Enable with SGLANG_GDN_PREFILL_FLYDSL=1
# (HIP only), with a silent fallback to Triton whenever the kernel is unavailable
# or the call does not match `_flydsl_fwd_h_eligible`.
#
# The kernel consumes SGLang's native GDN layouts directly -- token-major w/u
# ([B, T, H, K/V], `wu_head_major=False`), token-major cumulative g ([B, T, H]),
# VK-ordered state and the indexed state pool -- so `h` and `v_new` come back in
# the Triton layouts with no transposes and no state gather/scatter.
#
# Caveat: unlike the Triton kernel, the FlyDSL indexed path has no `-1`
# padded-slot guard; every row of `initial_state_indices` is dereferenced. Only
# enable this for batches whose state slots are all live.
_GDN_PREFILL_FLYDSL = os.getenv("SGLANG_GDN_PREFILL_FLYDSL", "0") == "1"
_flydsl_fwd_h = None
_flydsl_probed = False
_flydsl_logged = set()


def _log_flydsl_once(key: str, msg: str):
    """Log a FlyDSL dispatch decision once per process, on rank 0.

    The decision is identical for every call and every layer, so logging it
    per call would flood the serving log.
    """
    if key not in _flydsl_logged:
        _flydsl_logged.add(key)
        rank0_log(msg)


def _get_flydsl_fwd_h():
    global _flydsl_fwd_h, _flydsl_probed
    if _flydsl_probed:
        return _flydsl_fwd_h
    _flydsl_probed = True
    if _GDN_PREFILL_FLYDSL and is_hip():
        try:
            # Importing aiter.ops.flydsl already raises when flydsl is missing
            # or older than the minimum aiter supports.
            from aiter.ops.flydsl.linear_attention_prefill_kernels import (
                chunk_gated_delta_rule_fwd_h_flydsl_opt,
            )

            _flydsl_fwd_h = chunk_gated_delta_rule_fwd_h_flydsl_opt
        except Exception as e:
            _flydsl_fwd_h = None
            rank0_log(
                "GDN prefill: SGLANG_GDN_PREFILL_FLYDSL=1 but the aiter FlyDSL "
                f"state-matrix kernel is unavailable ({e!r}); using Triton."
            )
    return _flydsl_fwd_h


def _flydsl_fwd_h_reject_reason(
    k, w, u, g, initial_state, initial_state_indices, cu_seqlens, inplace_update
):
    """Return why the FlyDSL kernel cannot serve this call, or None if it can.

    Checks the kernel's preconditions without touching device memory. aiter's
    own audit of these is behind `AITER_K5_OPT_CHECK` and off by default, so an
    unchecked mismatch is a wrong-result or out-of-bounds bug rather than an
    exception.
    """
    if k.dim() != 4 or w.dim() != 4 or u.dim() != 4:
        return f"k/w/u must be 4-D, got {k.dim()}/{w.dim()}/{u.dim()}"
    if not (k.dtype == w.dtype == u.dtype == torch.bfloat16):
        return f"k/w/u must be bf16, got {k.dtype}/{w.dtype}/{u.dtype}"
    if not (k.is_contiguous() and w.is_contiguous() and u.is_contiguous()):
        return "k/w/u must be contiguous"

    B, T, Hg, K = k.shape
    H, V = u.shape[-2], u.shape[-1]
    # Only the 16x16x16 bf16 MFMA tile is compiled.
    if K != 128 or V != 128 or CHUNK_SIZE != 64:
        return f"only K=V=128 and chunk_size=64 are compiled, got K={K} V={V}"
    if H % Hg != 0:
        return f"H ({H}) must be a multiple of Hg ({Hg})"
    if w.shape != (B, T, H, K) or u.shape != (B, T, H, V):
        return (
            f"w/u must be token-major [B,T,H,K/V], got "
            f"{tuple(w.shape)}/{tuple(u.shape)}"
        )
    # Varlen packs the whole batch into one row.
    if cu_seqlens is not None and B != 1:
        return f"varlen requires B=1, got B={B}"
    # Scalar token-major g only; the per-channel gk path is not wired up here.
    if g is None:
        return "scalar g is required"
    if g.dtype != torch.float32 or g.shape != (B, T, H):
        return f"g must be fp32 [B,T,H]={(B, T, H)}, got {g.dtype} {tuple(g.shape)}"
    # The indexed pool is the only state contract the kernel offers, and it
    # requires the in-place write-back that `inplace_update` also asks for.
    if initial_state is None or initial_state_indices is None:
        return "an indexed state pool (initial_state + indices) is required"
    if not inplace_update:
        return "inplace_update=False is not supported by the indexed FlyDSL path"
    if not initial_state.is_contiguous() or initial_state.dim() != 4:
        return "the state pool must be contiguous and 4-D"
    if initial_state.shape[1:] != (H, V, K):
        return (
            f"the state pool must be [pool,H,V,K] with (H,V,K)={(H, V, K)}, "
            f"got {tuple(initial_state.shape)}"
        )
    return None


def _chunk_gated_delta_rule_fwd_h_flydsl(
    k,
    w,
    u,
    g,
    initial_state,
    initial_state_indices,
    cu_seqlens,
    fwd_h_flydsl,
):
    """FlyDSL drop-in for `chunk_gated_delta_rule_fwd_h`.

    Returns `(h, v_new)` in the Triton layouts -- `h` [B, NT, H, V, K] and
    `v_new` [B, T, H, V] -- with the final state written straight back into the
    `initial_state` pool slots, mirroring the Triton kernel's INPLACE_UPDATE.
    """
    h, v_new, _final_state = fwd_h_flydsl(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        output_final_state=True,
        inplace_final_state=True,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        # g arrives as a natural-log cumsum, so the gate is exp, not exp2.
        use_exp2=False,
        wu_head_major=False,
        g_head_major=False,
        # RNE matches Triton's fp32 -> bf16 conversion. The kernel defaults to
        # truncation instead, to stay bit-identical to aiter's HIP kernel.
        bf16_convert_trunc=False,
    )
    # `_final_state` aliases `initial_state`, which the kernel already updated.
    return h, v_new


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
    inplace_update: bool = True,
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

    fwd_h_flydsl = _get_flydsl_fwd_h()
    reject_reason = (
        _flydsl_fwd_h_reject_reason(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            inplace_update=inplace_update,
        )
        if fwd_h_flydsl is not None
        else None
    )
    if fwd_h_flydsl is not None and reject_reason is None:
        _log_flydsl_once(
            "hit",
            "GDN prefill: using the aiter FlyDSL state-matrix kernel "
            f"(token-major, H={u.shape[-2]} Hg={k.shape[-2]} K={k.shape[-1]} "
            f"V={u.shape[-1]}).",
        )
        h, v_new = _chunk_gated_delta_rule_fwd_h_flydsl(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            fwd_h_flydsl=fwd_h_flydsl,
        )
    else:
        if reject_reason is not None:
            _log_flydsl_once(
                "reject",
                "GDN prefill: the aiter FlyDSL state-matrix kernel does not fit "
                f"this call ({reject_reason}); using Triton.",
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
            inplace_update=inplace_update,
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
    if SUPPRESS_LEVEL < 3:
        return g, o, A, None, h, None
    elif SUPPRESS_LEVEL >= 3:
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
        inplace_update: bool = True,
    ):
        q_orig = q
        k_orig = k

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
            if cu_seqlens is not None
            else None
        )
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
            inplace_update=inplace_update,
        )
        return o.to(q.dtype), h


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
    inplace_update: bool = True,
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
        inplace_update (Optional[bool]):
            Whether to write final states back to `initial_state`. Default: `True`.

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
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, (
        "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
    )

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta, g = map(
            lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g)
        )
    # if not head_first and q.shape[1] < q.shape[2]:
    #     warnings.warn(
    #         f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
    #         "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
    #         "when head_first=False was specified. "
    #         "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
    #     )
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
        inplace_update,
    )
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, None, h
