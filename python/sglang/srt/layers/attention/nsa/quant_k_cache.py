import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.nsa.utils import NSA_QUANT_K_CACHE_FAST


def quantize_k_cache(cache_k):
    # TODO upstream can skip concat([k_nope, k_pe]) since we split them here
    if NSA_QUANT_K_CACHE_FAST:
        return _quantize_k_cache_fast_wrapped(cache_k)
    else:
        return _quantize_k_cache_slow(cache_k)


# Copied from original
def _quantize_k_cache_slow(
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
        # temp debug
        # input_k_cache = (576 - torch.arange(num_blocks * block_size * 1 * dim_nope_and_rope, device="cuda")).to(torch.bfloat16).reshape(num_blocks, block_size, 1, dim_nope_and_rope)

        ref_quant = _quantize_k_cache_slow(input_k_cache)
        actual_quant = _quantize_k_cache_fast_wrapped(input_k_cache)
        # print(f"{input_k_cache=}")
        # print(f"{ref_quant=}")
        # print(f"{actual_quant=}")
        # print(f"{ref_quant == actual_quant=}")
        # print(f"{actual_quant.to(torch.float32) - ref_quant.to(torch.float32)=}")
        # print(f"{ref_quant.view(torch.bfloat16)=}")
        # print(f"{actual_quant.view(torch.bfloat16)=}")
        # assert torch.all(ref_quant == actual_quant)

        import dequant_k_cache

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

    print("Passed")
