import torch
import triton
import triton.language as tl


def ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


@triton.jit
def _dynamic_conv_fwd_kernel(
    X_ptr,
    K_ptr,
    Out_ptr,
    Cache_ptr,
    Cu_seqlens_ptr,
    Seq_idx_ptr,
    Cache_idx_ptr,
    Has_init_ptr,
    D,
    X_stride_t,
    X_stride_d,
    K_stride_t,
    K_stride_d,
    K_stride_w,
    Out_stride_t,
    Out_stride_d,
    Cache_stride_b,
    Cache_stride_d,
    Cache_stride_t,
    W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_total_time_idx = tl.program_id(0)
    pid_d_block = tl.program_id(1)

    batch_idx = tl.load(Seq_idx_ptr + pid_total_time_idx)
    seq_start_idx = tl.load(Cu_seqlens_ptr + batch_idx)
    has_init = tl.load(Has_init_ptr + batch_idx)

    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D

    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    offs_w = tl.arange(0, W)

    k_ptrs = K_ptr + (
        pid_total_time_idx * K_stride_t
        + offs_d[:, None] * K_stride_d
        + offs_w[None, :] * K_stride_w
    )
    k_vals = tl.load(
        k_ptrs, mask=d_mask[:, None], other=0.0
    )  # Shape: [BLOCK_SIZE_D, W]

    x_abs_indices = pid_total_time_idx + offs_w - W + 1
    x_ptrs = X_ptr + (
        x_abs_indices[None, :] * X_stride_t + offs_d[:, None] * X_stride_d
    )
    x_final_load_mask = d_mask[:, None] & (x_abs_indices >= seq_start_idx)[None, :]
    x_input_vals = tl.load(x_ptrs, mask=x_final_load_mask, other=0.0)

    cache_batch_idx = tl.load(Cache_idx_ptr + batch_idx)
    cache_ptrs = Cache_ptr + (
        cache_batch_idx * Cache_stride_b
        + (x_abs_indices + W - seq_start_idx)[None, :] * Cache_stride_t
        + offs_d[:, None] * Cache_stride_d
    )
    cache_final_load_mask = (
        d_mask[:, None] & (x_abs_indices < seq_start_idx)[None, :] & (has_init != 0)
    )
    vals_from_cache = tl.load(cache_ptrs, mask=cache_final_load_mask, other=0.0)

    product = k_vals * (x_input_vals + vals_from_cache)
    accumulator += tl.sum(product, axis=1)

    out_ptrs = Out_ptr + (pid_total_time_idx * Out_stride_t + offs_d * Out_stride_d)
    tl.store(out_ptrs, accumulator, mask=d_mask)


def dynamic_conv_triton_cache_varlen(
    x: torch.Tensor,
    kernels: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    cache: torch.Tensor = None,
    cache_indices: torch.Tensor = None,
    has_initial_state: torch.Tensor = None,
    seq_idx: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fused dynamic convolution.
    Assumes W <= 4.

    Args:
        x: Input tensor of shape [T, D].
        kernels: Dynamic kernels of shape [T, D, W].
        cu_seqlens: Cumulative sequence lengths for each batch. Shape: [B+1].
        cache: Optional past context tensor of shape [N, D, W].
               If provided, treated as concatenated before x for convolution input.
        cache_indices: Indices of the cache for each sequence. Shape: [B].
        has_initial_state: Whether the initial state is provided. Shape: [B].
        seq_idx: Indices of the sequence for each token. Shape: [T].
    Returns:
        Output tensor of shape [T, D].
    """

    x = ensure_contiguous(x)
    kernels = ensure_contiguous(kernels)
    cache = ensure_contiguous(cache)

    T, D = x.shape
    W = kernels.shape[2]
    assert W <= 4, "Kernel W > 4 not expected for this version"

    out = torch.empty_like(x)

    grid = lambda meta: (T, triton.cdiv(D, meta["BLOCK_SIZE_D"]))
    BLOCK_SIZE_D = 128

    _dynamic_conv_fwd_kernel[grid](
        x,
        kernels,
        out,
        cache,
        cu_seqlens,
        seq_idx,
        cache_indices,
        has_initial_state,
        D,
        x.stride(0),
        x.stride(1),
        kernels.stride(0),
        kernels.stride(1),
        kernels.stride(2),
        out.stride(0),
        out.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        W=tl.constexpr(W),
        BLOCK_SIZE_D=tl.constexpr(BLOCK_SIZE_D),
    )

    return out
