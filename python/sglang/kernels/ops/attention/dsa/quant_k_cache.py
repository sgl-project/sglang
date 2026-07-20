import torch
import triton
import triton.language as tl


def quantize_k_cache(cache_k):
    return _quantize_k_cache_fast_wrapped(cache_k)


def quantize_k_cache_separate(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    tile_size: int = 128,
):
    """
    Quantize k_nope and k_rope separately without concat, returns two tensors.

    This avoids the concat operation and enables direct reuse of set_mla_kv_buffer_triton
    by returning two separate byte tensors for the nope and rope parts.

    Args:
        k_nope: (num_tokens, dim_nope) or (num_tokens, 1, dim_nope)
                Must have dim_nope=512 for FP8 MLA quantization
        k_rope: (num_tokens, dim_rope) or (num_tokens, 1, dim_rope)
                Must have dim_rope=64 for FP8 MLA quantization
        tile_size: quantization tile size (default 128)

    Returns:
        Tuple of (nope_part, rope_part) where:
        - nope_part: (num_tokens, 1, 528) as uint8 view, contains [nope_fp8(512) | scales(16)]
        - rope_part: (num_tokens, 1, 128) as uint8 view, contains [rope_bf16_bytes(128)]

        These two tensors can be directly passed to set_mla_kv_buffer_triton(kv_buffer, loc, nope_part, rope_part)
    """
    # Squeeze middle dimension if present
    k_nope_2d = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope_2d = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope

    num_tokens = k_nope_2d.shape[0]
    dim_nope = k_nope_2d.shape[1]
    dim_rope = k_rope_2d.shape[1]

    # Validate dimensions for FP8 MLA
    if dim_nope != 512:
        raise ValueError(f"Expected dim_nope=512 for FP8 MLA, got {dim_nope}")
    if dim_rope != 64:
        raise ValueError(f"Expected dim_rope=64 for FP8 MLA, got {dim_rope}")
    if k_rope_2d.shape[0] != num_tokens:
        raise ValueError(
            f"k_nope and k_rope must have same num_tokens, got {num_tokens} vs {k_rope_2d.shape[0]}"
        )

    return _quantize_k_cache_fast_separate(
        k_nope=k_nope_2d, k_rope=k_rope_2d, group_size=tile_size
    )


# Copied from original
def _quantize_k_cache_ref(
    input_k_cache: torch.Tensor,  # (num_blocks, block_size, h_k, d)
    dv: int = 512,
    tile_size: int = 128,
) -> torch.Tensor:
    """
    Quantize the k-cache
    Return a tensor with shape (num_blocks, block_size, h_k, dv + 4(dv/tile_size) + t(d-dv)) of dtype uint8_t, where t = input_k_cache.element_size()
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py or README.md
    """
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, d = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)  # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    result = torch.empty(
        (num_blocks, block_size, dv + num_tiles * 4 + input_elem_size * (d - dv)),
        dtype=torch.float8_e4m3fn,
        device=input_k_cache.device,
    )
    result_k_nope_part = result[..., :dv]
    result_k_scale_factor = result[..., dv : dv + num_tiles * 4].view(torch.float32)
    result_k_rope_part = result[..., dv + num_tiles * 4 :].view(input_k_cache.dtype)
    result_k_rope_part[:] = input_k_cache[..., dv:]

    for tile_idx in range(0, num_tiles):
        cur_scale_factors_inv = (
            torch.abs(
                input_k_cache[..., tile_idx * tile_size : (tile_idx + 1) * tile_size]
            )
            .max(dim=-1)
            .values
            / 448.0
        )  # [num_blocks, block_size]
        result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

        cur_scale_factors_inv.unsqueeze_(-1)  # [num_blocks, block_size, 1]
        cur_quantized_nope = (
            input_k_cache[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].float()
            / cur_scale_factors_inv.float()
        ).to(torch.float8_e4m3fn)
        result_k_nope_part[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
            cur_quantized_nope
        )

    result = result.view(num_blocks, block_size, 1, -1)
    return result


def _quantize_k_cache_fast_wrapped(
    input_k_cache: torch.Tensor,
    dv: int = 512,
    tile_size: int = 128,
) -> torch.Tensor:
    # TODO the final API may be 2D instead of 4D, thus we convert them here
    num_blocks, block_size, _, dim_nope_and_rope = input_k_cache.shape
    assert dv == 512
    assert dim_nope_and_rope == 512 + 64
    assert tile_size == 128
    input_k_cache = input_k_cache.view((-1, dim_nope_and_rope))

    # TODO deliberately split into two tensors, then upstream can provide the two tensors instead of concat into one
    k_nope = input_k_cache[:, :dv]
    k_rope = input_k_cache[:, dv:]

    output = _quantize_k_cache_fast(k_nope=k_nope, k_rope=k_rope)

    return output.view(num_blocks, block_size, 1, -1)


def _quantize_k_cache_fast(k_nope, k_rope, group_size: int = 128):
    """
    :param k_nope: (num_tokens, dim_nope 512)
    :param k_rope: (num_tokens, dim_rope 64)
    """

    assert k_nope.dtype == torch.bfloat16
    assert k_rope.dtype == torch.bfloat16

    num_tokens, dim_nope = k_nope.shape
    num_tokens_, dim_rope = k_rope.shape
    assert num_tokens == num_tokens_
    assert dim_nope == 512
    assert dim_rope == 64
    assert k_nope.dtype == k_rope.dtype
    num_tiles = dim_nope // group_size

    assert k_nope.stride(1) == 1
    assert k_rope.stride(1) == 1

    output = torch.empty(
        (num_tokens, dim_nope + num_tiles * 4 + k_rope.element_size() * dim_rope),
        dtype=torch.float8_e4m3fn,
        device=k_nope.device,
    )
    output_nope_q = output[..., :dim_nope]
    output_nope_s = output[..., dim_nope : dim_nope + num_tiles * 4].view(torch.float32)
    output_rope = output[..., dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    assert num_blocks_per_token == 5

    assert dim_nope % group_size == 0
    NUM_NOPE_BLOCKS = dim_nope // group_size

    _quantize_k_cache_fast_kernel[(num_tokens, num_blocks_per_token)](
        output_nope_q,
        output_nope_s,
        output_rope,
        k_nope,
        k_rope,
        output_nope_q.stride(0),
        output_nope_s.stride(0),
        output_rope.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        NUM_NOPE_BLOCKS=NUM_NOPE_BLOCKS,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
    )

    return output


def _quantize_k_cache_fast_separate(k_nope, k_rope, group_size: int = 128):
    """
    Quantize k_nope and k_rope in a single Triton kernel, directly outputting two separate tensors.

    This avoids packing/unpacking and enables direct use with set_mla_kv_buffer_triton.

    :param k_nope: (num_tokens, dim_nope 512) bfloat16
    :param k_rope: (num_tokens, dim_rope 64) bfloat16
    :param group_size: quantization tile size (default 128, kernel is tuned for this value)
    :return: Tuple of (nope_part_u8, rope_part_u8)
        - nope_part_u8: (num_tokens, 1, nope_part_bytes) uint8, layout [nope_fp8(dim_nope) | scales(num_tiles*4)]
        - rope_part_u8: (num_tokens, 1, rope_part_bytes) uint8, layout [rope_bf16_bytes(dim_rope*2)]
    """
    num_tokens, dim_nope = k_nope.shape
    num_tokens_, dim_rope = k_rope.shape

    assert num_tokens == num_tokens_, f"k_nope and k_rope must have same num_tokens"

    # Ensure contiguous tensors for kernel
    k_nope = k_nope.contiguous()
    k_rope = k_rope.contiguous()

    num_tiles = dim_nope // group_size

    # Calculate byte sizes based on validated dimensions
    # nope_part: [FP8 quantized data (dim_nope bytes)] + [FP32 scales (num_tiles * 4 bytes)]
    # rope_part: [BF16 raw data (dim_rope * 2 bytes)]
    nope_part_bytes = (
        dim_nope + num_tiles * 4
    )  # e.g., 512 + 4*4 = 528 for dim_nope=512, group_size=128
    rope_part_bytes = (
        dim_rope * k_rope.element_size()
    )  # e.g., 64 * 2 = 128 for dim_rope=64, BF16

    # Allocate two separate output buffers (as uint8 for direct byte-level access)
    nope_part_u8 = torch.empty(
        (num_tokens, nope_part_bytes), dtype=torch.uint8, device=k_nope.device
    )
    rope_part_u8 = torch.empty(
        (num_tokens, rope_part_bytes), dtype=torch.uint8, device=k_rope.device
    )

    # Create typed views for the kernel to write into
    # Fixed byte layout for nope_part: [nope_fp8 (dim_nope bytes) | scales_fp32 (num_tiles*4 bytes)]
    # Fixed byte layout for rope_part: [rope_bf16 (dim_rope*2 bytes)]
    nope_q_view = nope_part_u8[:, :dim_nope].view(torch.float8_e4m3fn)
    nope_s_view = nope_part_u8[:, dim_nope:].view(torch.float32)
    rope_view = rope_part_u8.view(torch.bfloat16)

    # Kernel launch parameters
    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    NUM_NOPE_BLOCKS = dim_nope // group_size

    # Use the same kernel as _quantize_k_cache_fast (reuse existing implementation)
    _quantize_k_cache_fast_kernel[(num_tokens, num_blocks_per_token)](
        nope_q_view,
        nope_s_view,
        rope_view,
        k_nope,
        k_rope,
        nope_q_view.stride(0),
        nope_s_view.stride(0),
        rope_view.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        NUM_NOPE_BLOCKS=NUM_NOPE_BLOCKS,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
    )

    # Add middle dimension for compatibility with set_mla_kv_buffer_triton
    return nope_part_u8.unsqueeze(1), rope_part_u8.unsqueeze(1)


@triton.jit
def _quantize_k_cache_fast_kernel(
    output_nope_q_ptr,
    output_nope_s_ptr,
    output_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    output_nope_q_stride_0: int,
    output_nope_s_stride_0: int,
    output_rope_stride_0: int,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        # a. quant nope
        effective_block_id = raw_block_id

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_NOPE
        ptr = k_nope_ptr + token_id * k_nope_stride_0 + offs

        y = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

        # the ref impl do not have a `tl.maximum(... eps)`, so we remove it here
        y_s = tl.max(tl.abs(y)) / FP8_MAX
        y_s_inv = 1.0 / y_s
        y_q = tl.clamp(y * y_s_inv, FP8_MIN, FP8_MAX).to(
            output_nope_q_ptr.dtype.element_ty
        )

        dst_q_ptr = output_nope_q_ptr + token_id * output_nope_q_stride_0 + offs
        dst_s_ptr = (
            output_nope_s_ptr + token_id * output_nope_s_stride_0 + effective_block_id
        )

        tl.store(dst_q_ptr, y_q, mask=mask)
        tl.store(dst_s_ptr, y_s)
    else:
        # b. copy rope
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = k_rope_ptr + token_id * k_rope_stride_0 + offs
        dst_ptr = output_rope_ptr + token_id * output_rope_stride_0 + offs

        data = tl.load(src_ptr, mask=mask)
        tl.store(dst_ptr, data, mask=mask)


if __name__ == "__main__":
    import dequant_k_cache

    for num_blocks, block_size in [
        (1, 1),
        (10, 64),
    ]:
        dim_nope_and_rope = 512 + 64

        input_k_cache = torch.randn(
            (num_blocks, block_size, 1, dim_nope_and_rope),
            dtype=torch.bfloat16,
            device="cuda",
        )

        ref_quant = _quantize_k_cache_ref(input_k_cache)
        actual_quant = _quantize_k_cache_fast_wrapped(input_k_cache)

        ref_ref_dequant = dequant_k_cache._dequantize_k_cache_slow(ref_quant)
        ref_actual_dequant = dequant_k_cache._dequantize_k_cache_fast_wrapped(ref_quant)
        actual_actual_dequant = dequant_k_cache._dequantize_k_cache_fast_wrapped(
            actual_quant
        )

        print(f"{ref_ref_dequant=}")
        print(f"{actual_actual_dequant=}")
        print(f"{actual_actual_dequant - ref_ref_dequant=}")
        print(f"{torch.mean(ref_ref_dequant - actual_actual_dequant)=}")

        # TODO too different?
        torch.testing.assert_close(
            ref_ref_dequant, ref_actual_dequant, atol=0.2, rtol=0.2
        )
        torch.testing.assert_close(
            ref_ref_dequant, actual_actual_dequant, atol=0.2, rtol=0.2
        )

        # test dequant_k_cache_paged
        page_table_1 = torch.arange(
            num_blocks * block_size, dtype=torch.int32, device="cuda"
        )
        actual_dequant_paged = dequant_k_cache.dequantize_k_cache_paged(
            actual_quant, page_table_1
        ).reshape(actual_actual_dequant.shape)
        print(f"{torch.mean(actual_actual_dequant - actual_dequant_paged)=}")
        torch.testing.assert_close(
            ref_ref_dequant, actual_dequant_paged, atol=0.2, rtol=0.2
        )

    print("Passed")

    # Test quantize_k_cache_separate: verify output matches concat path
    print("\nTesting quantize_k_cache_separate...")
    for num_tokens in [64, 100]:
        dim_nope = 512
        dim_rope = 64

        k_nope = torch.randn(
            num_tokens, 1, dim_nope, dtype=torch.bfloat16, device="cuda"
        )
        k_rope = torch.randn(
            num_tokens, 1, dim_rope, dtype=torch.bfloat16, device="cuda"
        )

        # Old path: concat then quantize
        k_concat = torch.cat([k_nope, k_rope], dim=-1).squeeze(1)  # (num_tokens, 576)
        old_output = quantize_k_cache(k_concat.unsqueeze(1).unsqueeze(1))  # 4D input
        old_output = old_output.squeeze(1).squeeze(1)  # Back to (num_tokens, 656)

        # New path: quantize separately
        nope_part, rope_part = quantize_k_cache_separate(k_nope, k_rope)
        new_bytes = torch.cat([nope_part.squeeze(1), rope_part.squeeze(1)], dim=-1)

        # Compare byte-level equality
        old_bytes = old_output.view(torch.uint8)

        if old_bytes.shape != new_bytes.shape:
            raise RuntimeError(
                f"Shape mismatch: {old_bytes.shape} vs {new_bytes.shape}"
            )

        diff_bytes = (old_bytes != new_bytes).sum().item()
        if diff_bytes > 0:
            max_diff = (old_bytes.float() - new_bytes.float()).abs().max().item()
            raise RuntimeError(
                f"quantize_k_cache_separate output doesn't match concat path: "
                f"{diff_bytes} differing bytes, max_diff={max_diff}"
            )

        print(f"  num_tokens={num_tokens}: PASSED (outputs match byte-wise)")

    print("quantize_k_cache_separate tests passed!")

    print("\nDo benchmark...")

    for num_blocks, block_size in [
        (1, 64),
        (64, 64),
        (128, 64),
        (256, 64),
        (512, 64),
        (1024, 64),
        (2048, 64),
    ]:
        dim_nope_and_rope = 512 + 64

        input_k_cache = torch.randn(
            (num_blocks, block_size, 1, dim_nope_and_rope),
            dtype=torch.bfloat16,
            device="cuda",
        )

        actual_quant = _quantize_k_cache_fast_wrapped(input_k_cache)

        page_table_1 = torch.arange(
            num_blocks * block_size, dtype=torch.int32, device="cuda"
        )

        def run_ans():
            return dequant_k_cache.dequantize_k_cache_paged(actual_quant, page_table_1)

        ans_time: float = triton.testing.do_bench(run_ans, warmup=10, rep=20) / 1000  # type: ignore
        print(f"seq_kv: {num_blocks * block_size}, time: {ans_time * 1e6: 4.0f} us")
