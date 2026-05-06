import enum
from typing import Tuple

import torch

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

# from sglang.srt.utils import is_hip
# FP8_DTYPE = torch.float8_e4m3fnuz if is_hip() else torch.float8_e4m3fn
FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


class FP8KVCacheLayout(enum.Enum):
    V32_FP8Sparse = 1
    MODEL1_FP8Sparse = 2

    def get_meta(self) -> Tuple[int, int, int, int, int]:
        # Return: (d, d_nope, d_rope, tile_size, num_tiles)
        return {
            FP8KVCacheLayout.V32_FP8Sparse: (576, 512, 64, 128, 4),
            FP8KVCacheLayout.MODEL1_FP8Sparse: (512, 448, 64, 64, 7),
        }[self]


def _cast_scale_inv_to_ue8m0(
    scales_inv: torch.Tensor, out_dtype=torch.float32
) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(out_dtype)


def quantize_k_cache(
    input_k_cache: torch.Tensor,  # (num_blocks, block_size, h_k, d)
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    Quantize the k-cache
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    assert input_k_cache.shape[-1] == d
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)  # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        bytes_per_token = d_nope + num_tiles * 4 + input_elem_size * d_rope
        result = torch.empty(
            (num_blocks, block_size + 1, bytes_per_token),
            dtype=FP8_DTYPE,
            device=input_k_cache.device,
        )[:, :block_size, :]
        result_k_nope_part = result[..., :d_nope]
        result_k_scale_factor = result[..., d_nope : d_nope + num_tiles * 4].view(
            torch.float32
        )
        result_k_rope_part = result[..., d_nope + num_tiles * 4 :].view(
            input_k_cache.dtype
        )
        result_k_rope_part[:] = input_k_cache[..., d_nope:]

        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = (
                torch.abs(
                    input_k_cache[
                        ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                    ]
                )
                .max(dim=-1)
                .values.float()
                / 448.0
            )  # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

            cur_scale_factors_inv.unsqueeze_(-1)  # [num_blocks, block_size, 1]
            cur_quantized_nope = (
                input_k_cache[
                    ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].float()
                / cur_scale_factors_inv.float()
            ).to(FP8_DTYPE)
            result_k_nope_part[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ] = cur_quantized_nope

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        bytes_per_token = d_nope + 2 * d_rope + num_tiles + 1
        size_per_block_padded = (block_size * bytes_per_token + 576 - 1) // 576 * 576
        result = torch.empty(
            (num_blocks, size_per_block_padded),
            dtype=FP8_DTYPE,
            device=input_k_cache.device,
        )[:, : block_size * bytes_per_token]
        result_k_nope_rope_part = result[:, : block_size * (d_nope + 2 * d_rope)].view(
            num_blocks, block_size, d_nope + 2 * d_rope
        )
        result_k_nope = result_k_nope_rope_part[
            :, :, :d_nope
        ]  # [num_blocks, block_size, d_nope]
        result_k_rope = result_k_nope_rope_part[:, :, d_nope:].view(
            input_k_cache.dtype
        )  # [num_blocks, block_size, d_rope]
        result_k_scale_factor = (
            result[:, block_size * (d_nope + 2 * d_rope) :]
            .view(num_blocks, block_size, 8)[:, :, :7]
            .view(torch.float8_e8m0fnu)
        )  # [num_blocks, block_size, num_tiles]

        result_k_rope[:] = input_k_cache[..., d_nope:]
        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = (
                torch.abs(
                    input_k_cache[
                        ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                    ]
                )
                .max(dim=-1)
                .values.float()
                / 448.0
            )  # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv.to(
                torch.float8_e8m0fnu
            )

            cur_scale_factors_inv = cur_scale_factors_inv.view(
                num_blocks, block_size, 1
            )
            cur_quantized_nope = (
                input_k_cache[
                    ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].float()
                / cur_scale_factors_inv.float()
            ).to(FP8_DTYPE)
            result_k_nope[:, :, tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_quantized_nope
            )

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")


def dequantize_k_cache(
    quant_k_cache: torch.Tensor,  # (num_blocks, block_size, 1, bytes_per_token)
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    # NOTE ADD
    assert quant_k_cache.dtype == FP8_DTYPE

    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty(
        (num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device
    )

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

        input_nope = quant_k_cache[..., :d_nope]
        input_scale = quant_k_cache[..., d_nope : d_nope + num_tiles * 4].view(
            torch.float32
        )
        input_rope = quant_k_cache[..., d_nope + num_tiles * 4 :].view(torch.bfloat16)
        result[..., d_nope:] = input_rope

        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].to(torch.float32)
            cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
            result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_nope * cur_scales
            )

    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, -1)  # [num_blocks, ...]
        input_nope_rope = quant_k_cache[:, : block_size * (d_nope + 2 * d_rope)].view(
            num_blocks, block_size, d_nope + 2 * d_rope
        )
        input_nope = input_nope_rope[:, :, :d_nope]
        input_rope = input_nope_rope[:, :, d_nope:].view(torch.bfloat16)
        input_scale = (
            quant_k_cache[:, block_size * (d_nope + 2 * d_rope) :]
            .view(num_blocks, block_size, 8)[:, :, :7]
            .view(torch.float8_e8m0fnu)
        )  # [num_blocks, block_size, num_tiles]

        result[..., d_nope:] = input_rope
        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].to(torch.bfloat16)
            cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
            result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_nope * cur_scales
            )

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")

    result = result.view(num_blocks, block_size, 1, d)
    return result


def abs_indices2indices_in_kvcache(
    abs_indices: torch.Tensor,  # [b, s_q, topk]
    block_table: torch.Tensor,  # [b, /]
    block_size: int,
) -> torch.Tensor:
    """
    Convert abs_indices (logical index, ranging from 0 to s_k-1) to index expected by the sparse attn kernel
    Equivalent to:

    b, s_q, topk = abs_indices.shape
    indices_in_kvcache = torch.empty_like(abs_indices)
    for i in range(b):
        cur_abs_indices = abs_indices[i, :, :].clone()  # [s_q, topk]
        invalid_mask = cur_abs_indices == -1
        cur_abs_indices[invalid_mask] = 0
        cur_indices_in_kvcache = block_table[i].index_select(0, cur_abs_indices.flatten()//block_size).view(s_q, topk)*block_size + cur_abs_indices%block_size
        cur_indices_in_kvcache[invalid_mask] = -1
        indices_in_kvcache[i] = cur_indices_in_kvcache
    return indices_in_kvcache

    """
    b, s_q, topk = abs_indices.shape
    _, max_blocks_per_seq = block_table.shape

    abs_indices = abs_indices.clone()
    invalid_mask = abs_indices == -1
    abs_indices[invalid_mask] = 0

    real_block_idxs = block_table.view(-1).index_select(
        0,
        (
            abs_indices // block_size
            + torch.arange(0, b).view(b, 1, 1) * max_blocks_per_seq
        ).view(-1),
    )
    indices_in_kvcache = (
        real_block_idxs.view(b, s_q, topk) * block_size + abs_indices % block_size
    )
    indices_in_kvcache[invalid_mask] = -1

    return indices_in_kvcache
