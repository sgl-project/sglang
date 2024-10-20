import triton
import triton.language as tl


@triton.jit
def _segment_gemm_expand_kernel(
    output,  # (s, h) = (m, k)
    x,  # (s, r) = (m, n)
    weights,  # (num_lora, h, r) = (..., k, n)
    x_stride_0,
    w_stride_0,
    w_stride_1,
    output_stride_0,
    seg_lens,
    seg_start,
    weight_indices,
    K,
    N,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    divisible,
    output_add,
    input_slice_offset,
    output_slice_offset,
    scaling,
):
    batch_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    width = tl.cdiv(K, BLOCK_K)
    pid_s = pid // width
    pid_h = pid % width

    seg_len = tl.load(seg_lens + batch_id)
    w_index = tl.load(weight_indices + batch_id)
    seg_start = tl.load(seg_start + batch_id)

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    h_offset = tl.arange(0, BLOCK_K) + pid_h * BLOCK_K
    r_offset = tl.arange(0, BLOCK_N)

    # (BLOCK_S, BLOCK_N)
    x_ptrs = (
        x
        + seg_start * x_stride_0
        + s_offset[:, None] * x_stride_0
        + r_offset[None, :]
        + input_slice_offset
    )
    # (BLOCK_N, BLOCK_K)
    w_ptrs = (
        weights
        + w_index * w_stride_0
        + r_offset[:, None]
        + h_offset[None, :] * w_stride_1
    )

    partial_sum = tl.zeros((BLOCK_S, BLOCK_K), dtype=tl.float32)
    for rid in range(tl.cdiv(N, BLOCK_N)):
        if divisible:
            tiled_x = tl.load(x_ptrs)
            tiled_w = tl.load(w_ptrs)
        else:
            tiled_x = tl.load(
                x_ptrs, mask=r_offset[None, :] < N - rid * BLOCK_N, other=0
            )
            tiled_w = tl.load(
                w_ptrs, mask=r_offset[:, None] < N - rid * BLOCK_N, other=0
            )
        partial_sum += tl.dot(tiled_x, tiled_w) * scaling
        x_ptrs += BLOCK_N
        w_ptrs += BLOCK_N

    partial_sum = partial_sum.to(x.dtype.element_ty)
    seg_start_offset = seg_start * output_stride_0
    out_ptr = (
        output
        + seg_start_offset
        + s_offset[:, None] * output_stride_0
        + h_offset[None, :]
        + output_slice_offset
    )
    s_mask = s_offset[:, None] < seg_len
    if output_add:
        partial_sum += tl.load(out_ptr, mask=s_mask)
    tl.store(out_ptr, partial_sum, mask=s_mask)


def segment_gemm_expand(
    output,  # (s, h)
    x,  # (s, r)
    weights,  # (num_lora, h, r)
    batch_size,
    # weight_column_major,
    seg_lens,
    seg_start,
    weight_indices,
    max_len,
    input_slice_offset=0,
    output_slice_offset=0,
    output_add=False,
    scaling=1,
):
    assert weights.ndim == 3
    assert batch_size == seg_lens.shape[0] == weight_indices.shape[0]
    # assert weight_column_major
    # assert x.shape[-1] == weights.shape[-1]
    assert x.is_contiguous()
    assert weights.is_contiguous()

    H = weights.shape[-2]
    R = weights.shape[-1]

    BLOCK_S = 16
    BLOCK_H = 32
    BLOCK_R = 16
    divisible = R % BLOCK_R == 0
    assert H % BLOCK_H == 0

    grid = (
        triton.cdiv(max_len, BLOCK_S) * triton.cdiv(H, BLOCK_H),
        batch_size,
    )

    _segment_gemm_expand_kernel[grid](
        output,  # (s, h)
        x,  # (s, r)
        weights,  # (num_lora, h, r)
        x.stride(0),
        weights.stride(0),
        weights.stride(1),
        output.stride(0),
        seg_lens,
        seg_start,
        weight_indices,
        H,
        R,
        BLOCK_S,
        BLOCK_H,
        BLOCK_R,
        divisible,
        output_add=output_add,
        input_slice_offset=input_slice_offset,
        output_slice_offset=output_slice_offset,
        scaling=scaling,
    )
    return output
