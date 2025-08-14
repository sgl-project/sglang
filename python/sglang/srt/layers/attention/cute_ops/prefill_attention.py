# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Modified by SGLang team.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'd need to install nvidia-cutlass-dsl==4.1.0.

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flash_fwd_sm100 import FlashAttentionForwardSm100

def _green(x: str) -> str:
    return f"\033[1;32m{x}\033[0m"


def _red(x: str) -> str:
    return f"\033[1;31m{x}\033[0m"


def _yellow(x: str) -> str:
    return f"\033[1;33m{x}\033[0m"


torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)

np.set_printoptions(
    suppress=True, precision=3, linewidth=120, formatter={"float": "{:>8.3f}".format}
)


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


def _ref_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> torch.Tensor:
    head_dim = q.shape[-1]
    softmax_scale = head_dim**-0.5 if softmax_scale is None else softmax_scale

    qo_len = q.shape[0]
    kv_len = k.shape[0]
    logits = torch.einsum("qhd,khd->qhk", q, k).to(torch.float32) * softmax_scale
    # print(_yellow(f"logits: {logits.shape=}"), "\n", logits[:, 0, :], flush=True)

    if causal:
        mask = (
            torch.arange(qo_len, dtype=torch.int32, device=logits.device)[:, None]
            >= torch.arange(kv_len, dtype=torch.int32, device=logits.device)[None, :]
        )

        logits = torch.where(
            mask[:, None, :],
            logits,
            torch.tensor(float("-inf"), dtype=torch.float32, device=logits.device),
        )

        # print(_yellow(f"mask: {mask.shape=}"), "\n", mask.to(torch.float32), flush=True)
        # print(_yellow(f"logits: {logits.shape=}"), "\n", logits[:, 0, :], flush=True)

    scores = F.softmax(logits, dim=-1).to(v.dtype)
    # print(_yellow(f"scores: {scores.shape=}"), "\n", scores[:, 0, :], flush=True)

    out = torch.einsum("qhv,vhd->qhd", scores, v)
    return out


def _flash_attn_varlen_func(
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


def test_ragged(
    qo_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    init_range: float = 0.5,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 31415,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    qo_len = sum(qo_lens)
    kv_len = sum(kv_lens)
    seqlens_q = torch.tensor(list(qo_lens), dtype=torch.int32, device="cuda")
    seqlens_k = torch.tensor(list(qo_lens), dtype=torch.int32, device="cuda")
    cu_seqlens_q = F.pad(
        torch.cumsum(seqlens_q, dim=0, dtype=torch.int32), pad=(1, 0), mode="constant", value=0)
    cu_seqlens_k = F.pad(
        torch.cumsum(seqlens_k, dim=0, dtype=torch.int32), pad=(1, 0), mode="constant", value=0)

    q = torch.empty(
        size=(qo_len, num_qo_heads, head_dim), dtype=dtype, device="cuda"
    ).uniform_(-init_range, init_range)
    k = torch.empty(
        size=(kv_len, num_kv_heads, head_dim), dtype=dtype, device="cuda"
    ).uniform_(-init_range, init_range)
    v = torch.empty(
        size=(kv_len, num_kv_heads, head_dim), dtype=dtype, device="cuda"
    ).uniform_(-init_range, init_range)

    out, lse, *rest = _flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
    )

    # out = flash_attn_varlen_func(
    #     q=q,
    #     k=k,
    #     v=v,
    #     cu_seqlens_q=cu_seqlens_q,
    #     cu_seqlens_k=cu_seqlens_k,
    #     softmax_scale=softmax_scale,
    # )
    ref = _ref_impl(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
    )
    diff = (out - ref).abs_().max().item()

    print(_green(f"--> {q.shape=} {k.shape=} {v.shape=}"), f"{ref.shape=}", f"{out.shape=}")
    print(_green("max_diff: "), f"{diff:<.5f}", flush=True)


if __name__ == "__main__":
    test_ragged(
        qo_lens=(8,),
        kv_lens=(8,),
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=128,
        softmax_scale=None,
    )

    # test_ragged(
    #     qo_lens=(128,),
    #     kv_lens=(1024,),
    #     num_qo_heads=4,
    #     num_kv_heads=4,
    #     head_dim=128,
    #     softmax_scale=None,
    # )

    # test_ragged(
    #     qo_lens=(1024,),
    #     kv_lens=(1024,),
    #     num_qo_heads=8,
    #     num_kv_heads=1,
    #     head_dim=128,
    #     softmax_scale=None,
    # )
