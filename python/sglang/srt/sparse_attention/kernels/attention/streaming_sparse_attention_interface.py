import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .flash_streaming_fwd_sm90 import FlashStreamingForwardSm90

# Mapping from PyTorch dtypes to Cutlass dtypes
torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _streaming_sparse_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    sink_size: Optional[int] = None,
    enable_streaming: bool = True,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    pack_gqa: Optional[bool] = None,
    _compute_capability: Optional[int] = None,
    groupwise: Optional[bool] = False,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for streaming sparse attention.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        cu_seqlens_q: Cumulative sequence lengths for queries (optional)
        cu_seqlens_k: Cumulative sequence lengths for keys (optional)
        seqused_q: Used sequence lengths for queries (optional)
        seqused_k: Used sequence lengths for keys (optional)
        page_table: Page table for paged KV cache (optional)
        softmax_scale: Scaling factor for softmax (optional, defaults to 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        softcap: Softcap value for attention scores (optional)
        window_size_left: Left window size for local attention (optional)
        window_size_right: Right window size for local attention (optional)
        learnable_sink: Learnable sink tokens (optional)
        sink_size: Number of sink tokens (optional)
        enable_streaming: Whether to enable streaming mode
        m_block_size: Block size for M dimension
        n_block_size: Block size for N dimension
        num_threads: Number of threads per block
        pack_gqa: Whether to pack GQA (optional, auto-detected if None)
        _compute_capability: Override compute capability (for testing)
        groupwise: Whether to use groupwise paged KV cache

    Returns:
        Tuple of (output, lse) where:
        - output: Attention output tensor
        - lse: Log-sum-exp values for backward pass (if requires_grad)
    """

    def maybe_contiguous(x):
        return x.contiguous() if x is not None and x.stride(-1) != 1 else x

    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    qhead_per_kvhead = num_head // num_head_kv

    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]

    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert (
            page_table.stride(-1) == 1
        ), "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        if groupwise:
            assert page_table.shape == (batch_size * num_head_kv, max_num_pages_per_seq)
        else:
            assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]

    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
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
            page_table,
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
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

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
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
    lse = (
        torch.empty(lse_shape, dtype=torch.float32, device=device)
        if requires_grad
        else None
    )

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
        learnable_sink_tensor,
    ) = [
        (
            from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
            if t is not None
            else None
        )
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
    ]

    page_table_tensor = (
        from_dlpack(page_table.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=1
        )
        if page_table is not None
        else None
    )

    position_ids_tensor = (
        from_dlpack(position_ids.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=1
        )
        if position_ids is not None
        else None
    )

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
    assert (
        compute_capability == 9
    ), "Streaming sparse attention only supports compute capability 9.x (Hopper)"
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if (
        pack_gqa
        and (128 % qhead_per_kvhead != 0)
        or (cu_seqlens_q is not None or seqused_q is not None)
    ):
        pack_gqa = False

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
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        sink_size,
        enable_streaming,
        m_block_size,
        n_block_size,
        num_threads,
        pack_gqa,
        compute_capability,
        groupwise,
    )

    if compile_key not in _streaming_sparse_attn_forward.compile_cache:
        fa_fwd = FlashStreamingForwardSm90(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=causal,
            is_local=local,
            pack_gqa=pack_gqa,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_stages=2,
            num_threads=num_threads,
            Q_in_regs=False,
            sink_size=sink_size,
            enable_streaming=enable_streaming,
            groupwise=groupwise,
            intra_wg_overlap=False,
        )

        _streaming_sparse_attn_forward.compile_cache[compile_key] = cute.compile(
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
            page_table_tensor,
            softcap,
            window_size_left,
            window_size_right,
            learnable_sink_tensor,
            position_ids_tensor,
        )

    _streaming_sparse_attn_forward.compile_cache[compile_key](
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
        page_table_tensor,
        softcap,
        window_size_left,
        window_size_right,
        learnable_sink_tensor,
        position_ids_tensor,
    )

    return out, lse


_streaming_sparse_attn_forward.compile_cache = {}


class StreamingSparseAttnFunc(torch.autograd.Function):
    """Autograd function for streaming sparse attention.

    This class provides automatic differentiation support for streaming sparse attention,
    wrapping the forward pass and providing hooks for backward pass.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        sink_size: Optional[int] = None,
        enable_streaming: bool = True,
        softcap: float = 0.0,
        pack_gqa: Optional[bool] = None,
        groupwise: bool = False,
    ):
        """Forward pass for streaming sparse attention.

        Args:
            ctx: Context object for saving tensors
            q: Query tensor (batch, seqlen_q, num_heads, head_dim) or (total_q, num_heads, head_dim)
            k: Key tensor (batch, seqlen_k, num_heads_kv, head_dim) or (total_k, num_heads_kv, head_dim)
            v: Value tensor (batch, seqlen_k, num_heads_kv, head_dim_v) or (total_k, num_heads_kv, head_dim_v)
            cu_seqlens_q: Cumulative sequence lengths for queries (optional)
            cu_seqlens_k: Cumulative sequence lengths for keys (optional)
            seqused_q: Used sequence lengths for queries (optional)
            seqused_k: Used sequence lengths for keys (optional)
            page_table: Page table for paged KV cache (optional)
            softmax_scale: Scaling factor for softmax (optional)
            causal: Whether to apply causal masking
            window_size: Tuple of (left, right) window sizes for local attention
            learnable_sink: Learnable sink tokens (optional)
            sink_size: Number of sink tokens (optional)
            enable_streaming: Whether to enable streaming mode
            softcap: Softcap value for attention scores
            pack_gqa: Whether to pack GQA (optional)
            groupwise: Whether to use groupwise paged KV cache

        Returns:
            Tuple of (output, lse) where:
            - output: Attention output tensor
            - lse: Log-sum-exp values for backward pass
        """
        out, lse = _streaming_sparse_attn_forward(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            sink_size=sink_size,
            enable_streaming=enable_streaming,
            softcap=softcap,
            pack_gqa=pack_gqa,
            groupwise=groupwise,
        )

        # Save tensors for backward pass
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.sink_size = sink_size
        ctx.enable_streaming = enable_streaming
        ctx.groupwise = groupwise

        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        """Backward pass for streaming sparse attention.

        Note: Backward pass for streaming sparse attention is not yet implemented.
        """
        raise NotImplementedError(
            "Backward pass for streaming sparse attention is not implemented yet. "
            "If you need gradients, please consider using standard flash attention or "
            "implementing a custom backward pass."
        )


def streaming_sparse_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    sink_size: Optional[int] = None,
    enable_streaming: bool = True,
    softcap: float = 0.0,
    pack_gqa: Optional[bool] = None,
    groupwise: bool = False,
    position_ids: Optional[torch.Tensor] = None,
    m_block_size: int = 128,
    n_block_size: int = 128,
):
    """User-facing function for streaming sparse attention.

    This function provides a convenient interface for streaming sparse attention,
    which is optimized for long sequences with sparse attention patterns.

    Args:
        q: Query tensor (batch, seqlen_q, num_heads, head_dim) or (total_q, num_heads, head_dim)
        k: Key tensor (batch, seqlen_k, num_heads_kv, head_dim) or (total_k, num_heads_kv, head_dim)
        v: Value tensor (batch, seqlen_k, num_heads_kv, head_dim_v) or (total_k, num_heads_kv, head_dim_v)
        cu_seqlens_q: Cumulative sequence lengths for queries (optional)
        cu_seqlens_k: Cumulative sequence lengths for keys (optional)
        seqused_q: Used sequence lengths for queries (optional)
        seqused_k: Used sequence lengths for keys (optional)
        page_table: Page table for paged KV cache (optional)
        softmax_scale: Scaling factor for softmax (optional, defaults to 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        window_size: Tuple of (left, right) window sizes for local attention
        learnable_sink: Learnable sink tokens (optional)
        sink_size: Number of sink tokens (optional)
        enable_streaming: Whether to enable streaming mode
        softcap: Softcap value for attention scores
        pack_gqa: Whether to pack GQA (optional, auto-detected if None)
        groupwise: Whether to use groupwise paged KV cache
        position_ids: Position IDs for Chunked Attention

    Returns:
        Tuple of (output, lse) where:
        - output: Attention output tensor with shape matching input q
        - lse: Log-sum-exp values for backward pass (or None if not requires_grad)

    Example:
        >>> q = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.bfloat16)
        >>> k = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.bfloat16)
        >>> v = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.bfloat16)
        >>> out, lse = streaming_sparse_attn_func(
        ...     q, k, v,
        ...     causal=True,
        ...     sink_size=4,
        ...     enable_streaming=True
        ... )
    """

    return _streaming_sparse_attn_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table,
        softmax_scale,
        causal,
        softcap,
        window_size[0],
        window_size[1],
        learnable_sink,
        sink_size,
        enable_streaming,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        position_ids=position_ids,
    )
