# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Modified by SGLang team.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'd need to install nvidia-cutlass-dsl==4.1.0.

import math
from typing import Optional

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from sglang.srt.layers.attention.cute_ops.flash_fwd_sm100 import FlashAttentionForwardSm100


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    # m_block_size: int = 128,
    # n_block_size: int = 64,
    # num_threads: int = 128,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    _compute_capability: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    seqlen_k, num_head_kv, _ = k.shape[-3:]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (
            batch_size + 1,
        ), "cu_seqlens_k must have shape (batch_size + 1,)"
    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (
            batch_size + 1,
        ), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert seqused_q is None or seqused_q.shape == (
        batch_size,
    ), "seqused_q must have shape (batch_size,)"
    assert seqused_k is None or seqused_k.shape == (
        batch_size,
    ), "seqused_k must have shape (batch_size,)"
    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert (
                t.dtype == torch.int32
            ), "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            assert (
                t.stride(0) == 1
            ), "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"
    assert all(
        t is None or t.is_cuda
        for t in (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            learnable_sink,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv

    out_torch_dtype = q.dtype
    device = q.device
    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    )
    out = torch.empty(
        *q_batch_seqlen_shape,
        num_head,
        head_dim_v,
        dtype=out_torch_dtype,
        device=device,
    )
    lse_shape = (
        (batch_size, num_head, seqlen_q)
        if cu_seqlens_q is None
        else (num_head, total_q)
    )
    lse = torch.empty(lse_shape, dtype=torch.float32, device=device)

    dtype = torch2cute_dtype_map[q.dtype]
    q_tensor, k_tensor, v_tensor, o_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=t.ndim - 1
        )
        for t in (q, k, v, out)
    ]
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=lse.ndim - 1
        )
        if lse is not None
        else None
    )
    (
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        additive_sink_tensor,
    ) = [
        (
            from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
            if t is not None
            else None
        )
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
    ]
    if causal:
        window_size_right = 0
    local = window_size_left is not None or window_size_right is not None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
        else:
            causal, local = False, True
    compute_capability = (
        torch.cuda.get_device_capability()[0]
        if _compute_capability is None
        else _compute_capability
    )
    assert compute_capability == 10, "Unsupported compute capability. Supported: 10.x"
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        softcap is not None,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        m_block_size,
        n_block_size,
        num_threads,
        compute_capability,
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        fa_fwd = FlashAttentionForwardSm100(
            head_dim,
            head_dim_v,
            is_causal=causal,
            is_local=local,
            qhead_per_kvhead=qhead_per_kvhead,
            is_persistent=not causal
            and not local
            and cu_seqlens_q is None
            and seqused_q is None,
        )
        # TODO: check @can_implement
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            softcap,
            window_size_left,
            window_size_right,
            additive_sink_tensor,
        )
    _flash_attn_fwd.compile_cache[compile_key](
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        softmax_scale,
        current_stream,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        softcap,
        window_size_left,
        window_size_right,
        additive_sink_tensor,
    )
    return out, lse


_flash_attn_fwd.compile_cache = {}


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert causal, "Only support causal."
    assert (
        window_size[0] is None and window_size[1] is None
    ), "window_size is not supported."

    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        learnable_sink=learnable_sink,
        softcap=softcap,
    )
    return out, lse
