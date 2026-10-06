# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import triton
import triton.language as tl

from sglang.srt.runtime_context import get_platform

E2M1_MAX = 6.0
MAX_BLOCK_SCALE_FP8 = 448.0  # Maximum FP8 E4M3 value
# E2M1 format: 1 sign bit + 2 exponent bits + 1 mantissa bit = 4 bits
# 16 possible values: 0x0-0xF
# Negative values: 0x8-0xF (sign bit = 1)
# Positive values: 0x0-0x7 (sign bit = 0)
# Keep constants as Python literals. Compiled helpers materialize them with
# input.new_tensor(), so they follow the caller device without a global GPU tensor
# or a CPU tensor .to(device) in the hot path.
E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


class FP4MXBlock16KVQuantizeUtil:
    """Block-wise FP4 (E2M1) quantization for KV cache.

    Similar to MXFP4 but uses block_size=16 (MXFP4 spec defines block_size=32).
    Each block of 16 elements shares one uint8 exponent-only scale factor.
    """

    @staticmethod
    @torch.compile
    def batched_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Quantize tensor to KVFP4 format
        Args:
            tensor: Input tensor of shape [B, M, N]

        Returns:
            quant_tensor: Quantized tensor of shape [B, M, N/2]
            scale_factors: Scale factors of shape [B, M*N/16]
        """
        b, m, n = tensor.shape

        # Reshape to [B, M*N/16, 16] for block-wise quantization
        reshaped = tensor.view(b, m * n // 16, 16)

        # Compute scale factors per block
        block_max = reshaped.abs().max(dim=-1, keepdim=True).values
        scale_exp = torch.ceil(torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10)))
        scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)

        # Apply scaling
        scaled = reshaped / torch.exp2(scale_exp)

        # Quantize to FP4
        sign_bits = (scaled < 0).to(torch.uint8) << 3
        abs_vals = scaled.abs()

        # Pure tensor version (CUDA Graph safe)
        bounds = tensor.new_tensor(E2M1_BOUNDS, dtype=torch.float32)
        magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= bounds, dim=-1)

        # Combine sign and magnitude
        fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)

        # Pack two FP4 values into one uint8
        fp4_reshaped = fp4_vals.view(b, m, n)
        packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]

        return packed, scale_factors

    @staticmethod
    @torch.compile
    def batched_dequantize(
        quant_tensor: torch.Tensor,
        scale_factors: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize KVFP4 tensor
        Args:
            quant_tensor: Quantized tensor of shape [B, M, N/2]
            scale_factors: Scale factors of shape [B, M*N/16]
            dtype: Target dtype for output

        Returns:
            Dequantized tensor of shape [B, M, N]
        """
        b, m, n_half = quant_tensor.shape
        n = n_half * 2

        # More efficient unpacking using bit operations
        fp4_vals = torch.empty(b, m, n, dtype=torch.uint8, device=quant_tensor.device)
        fp4_vals[..., 0::2] = quant_tensor & 0x0F
        fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F

        # Extract sign and magnitude
        sign_mask = (fp4_vals & 0x08) != 0
        magnitude_idx = fp4_vals & 0x07

        # Convert to float values
        values = quant_tensor.new_tensor(E2M1_VALUES[:8], dtype=torch.float32)
        float_vals = values[magnitude_idx.long()]
        float_vals = torch.where(sign_mask, -float_vals, float_vals)

        # Reshape for block-wise scaling
        reshaped = float_vals.view(b, m * n // 16, 16)

        # Apply scale factors
        scale_exp = scale_factors.float() - 127
        scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))

        return scaled.view(b, m, n).to(dtype)


class MXFP4KVQuantizeUtil:
    """OCP MXFP4 quantization with 32-value blocks and E8M0 scales.

    The packed data uses two E2M1 values per byte (the even element in the
    low nibble). Scale factors are stored as raw E8M0 exponent bytes with a
    bias of 127.
    """

    BLOCK_SIZE = 32

    @staticmethod
    def _validate_input(tensor: torch.Tensor) -> tuple[int, int, int]:
        if tensor.ndim != 3:
            raise ValueError(
                f"MXFP4 KV cache expects a 3-D [B, M, N] tensor, got {tensor.shape}"
            )
        b, m, n = tensor.shape
        if n % MXFP4KVQuantizeUtil.BLOCK_SIZE != 0:
            raise ValueError(f"MXFP4 last dimension must be divisible by 32, got {n}")
        return b, m, n

    @staticmethod
    def _quantize_torch(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Portable reference path used on CPU and as a CUDA fallback."""
        b, m, n = MXFP4KVQuantizeUtil._validate_input(tensor)
        blocks = tensor.reshape(-1, MXFP4KVQuantizeUtil.BLOCK_SIZE).float()
        block_max = blocks.abs().amax(dim=-1, keepdim=True)

        # E8M0 has exponents [-127, 127]. A zero block uses the minimum scale;
        # all encoded values remain zero, so its exact scale is immaterial.
        scale_exp = torch.ceil(torch.log2(block_max / E2M1_MAX)).clamp(-127, 127)
        scale_exp = torch.where(
            block_max == 0, torch.full_like(scale_exp, -127), scale_exp
        )
        scaled = blocks / torch.exp2(scale_exp)

        bounds = tensor.new_tensor(E2M1_BOUNDS, dtype=torch.float32)
        magnitude_bits = torch.sum(scaled.abs().unsqueeze(-1) > bounds, dim=-1).to(
            torch.uint8
        )
        fp4_vals = magnitude_bits | ((scaled < 0).to(torch.uint8) << 3)
        fp4_vals = fp4_vals.reshape(b, m, n)
        packed = fp4_vals[..., 0::2] | (fp4_vals[..., 1::2] << 4)
        scales = (
            (scale_exp.squeeze(-1) + 127)
            .to(torch.uint8)
            .reshape(b, m, n // MXFP4KVQuantizeUtil.BLOCK_SIZE)
        )
        return packed, scales

    @staticmethod
    def batched_quantize(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize ``[B, M, N]`` BF16/FP16 data to linear-layout MXFP4."""
        b, m, n = MXFP4KVQuantizeUtil._validate_input(tensor)
        if not tensor.is_cuda:
            return MXFP4KVQuantizeUtil._quantize_torch(tensor)
        tensor = tensor.contiguous()
        packed = torch.empty((b, m, n // 2), dtype=torch.uint8, device=tensor.device)
        scales = torch.empty(
            (b, m, n // MXFP4KVQuantizeUtil.BLOCK_SIZE),
            dtype=torch.uint8,
            device=tensor.device,
        )
        _launch_mxfp4_quantize_kernel(tensor, packed, scales)
        return packed, scales

    @staticmethod
    def quantize_and_store_paged(
        tensor: torch.Tensor,
        bf16_tail: torch.Tensor,
        locations: torch.Tensor,
        quant_output: torch.Tensor,
        scale_output: torch.Tensor,
        tail_output: torch.Tensor,
    ) -> None:
        """Quantize and write DSA KV rows directly into paged cache buffers."""
        _, head_num, head_dim = MXFP4KVQuantizeUtil._validate_input(tensor)
        if bf16_tail.shape[:2] != tensor.shape[:2]:
            raise ValueError(
                "MXFP4 BF16 tail and latent tensor shapes are inconsistent"
            )
        if locations.numel() != tensor.shape[0]:
            raise ValueError("MXFP4 cache locations must contain one entry per token")
        expected_data_shape = (quant_output.shape[0], head_num, head_dim // 2)
        expected_scale_shape = (
            scale_output.shape[0],
            head_num,
            head_dim // MXFP4KVQuantizeUtil.BLOCK_SIZE,
        )
        if tuple(quant_output.shape) != expected_data_shape:
            raise ValueError(
                f"MXFP4 output must have shape {expected_data_shape}, got "
                f"{tuple(quant_output.shape)}"
            )
        if tuple(scale_output.shape) != expected_scale_shape:
            raise ValueError(
                f"MXFP4 scale output must have shape {expected_scale_shape}, got "
                f"{tuple(scale_output.shape)}"
            )
        if tail_output.shape[1:] != bf16_tail.shape[1:]:
            raise ValueError("MXFP4 BF16 tail output shape is inconsistent")

        if not tensor.is_cuda:
            packed, scales = MXFP4KVQuantizeUtil._quantize_torch(tensor)
            output_locations = locations.reshape(-1).long()
            quant_output[output_locations] = packed
            scale_output[output_locations] = scales
            tail_output[output_locations] = bf16_tail.to(tail_output.dtype)
            return

        _launch_mxfp4_quantize_kernel(
            tensor.contiguous(),
            quant_output,
            scale_output,
            locations=locations,
            bf16_tail=bf16_tail.contiguous(),
            tail_output=tail_output,
        )

    @staticmethod
    def batched_dequantize(
        quant_tensor: torch.Tensor,
        scale_factors: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Dequantize a dense batch of linear-layout MXFP4 values."""
        if quant_tensor.ndim != 3:
            raise ValueError(
                f"MXFP4 packed tensor must be 3-D, got {quant_tensor.shape}"
            )
        b, m, n_half = quant_tensor.shape
        n = n_half * 2
        expected_scales = (b, m, n // MXFP4KVQuantizeUtil.BLOCK_SIZE)
        if tuple(scale_factors.shape) != expected_scales:
            raise ValueError(
                f"MXFP4 scales must have shape {expected_scales}, got "
                f"{tuple(scale_factors.shape)}"
            )

        fp4_vals = torch.empty(b, m, n, dtype=torch.uint8, device=quant_tensor.device)
        fp4_vals[..., 0::2] = quant_tensor & 0x0F
        fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F
        float_vals = quant_tensor.new_tensor(E2M1_VALUES, dtype=torch.float32)[fp4_vals.long()]
        blocks = float_vals.reshape(-1, MXFP4KVQuantizeUtil.BLOCK_SIZE)
        scale = torch.exp2(scale_factors.reshape(-1, 1).float() - 127)
        return (blocks * scale).reshape(b, m, n).to(dtype)

    @staticmethod
    def dequantize_paged(
        quant_tensor: torch.Tensor,
        scale_factors: torch.Tensor,
        locations: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        bf16_tail: torch.Tensor | None = None,
        return_compact_page_table: bool = False,
        separate_tail: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Gather token locations and dequantize them into a compact tensor.

        ``bf16_tail`` supports MLA/DSA layouts that keep the RoPE dimensions in
        BF16. On CUDA, decoding the MXFP4 prefix and copying that tail are fused.
        """
        if locations.ndim != 1:
            locations = locations.reshape(-1)
        if quant_tensor.is_cuda:
            return _mxfp4_dequantize_paged_cuda(
                quant_tensor,
                scale_factors,
                locations,
                dtype,
                bf16_tail,
                return_compact_page_table,
                separate_tail,
            )
        valid = locations >= 0
        gather_locations = (
            locations.clamp_min(0) if return_compact_page_table else locations
        )
        output = MXFP4KVQuantizeUtil.batched_dequantize(
            quant_tensor[gather_locations.long()],
            scale_factors[gather_locations.long()],
            dtype,
        )
        if bf16_tail is not None:
            output = torch.cat(
                [output, bf16_tail[gather_locations.long()].to(dtype)], dim=-1
            )
        if return_compact_page_table:
            compact_page_table = torch.arange(
                locations.numel(), dtype=torch.int32, device=locations.device
            )
            compact_page_table.masked_fill_(~valid, -1)
            return output, compact_page_table
        return output


@triton.jit
def _mxfp4_e2m1_code(x):
    ax = tl.minimum(tl.abs(x), 6.0)
    code = (ax > 0.25).to(tl.uint8)
    code += (ax > 0.75).to(tl.uint8)
    code += (ax > 1.25).to(tl.uint8)
    code += (ax > 1.75).to(tl.uint8)
    code += (ax > 2.5).to(tl.uint8)
    code += (ax > 3.5).to(tl.uint8)
    code += (ax > 5.0).to(tl.uint8)
    sign = (x < 0).to(tl.uint8)
    return code | (sign << 3)


@triton.jit
def _mxfp4_quantize_kernel(
    input_ptr,
    tail_input_ptr,
    location_ptr,
    data_output_ptr,
    scale_output_ptr,
    tail_output_ptr,
    input_stride_token,
    input_stride_head,
    tail_input_stride_token,
    tail_input_stride_head,
    data_output_stride_token,
    data_output_stride_head,
    scale_output_stride_token,
    scale_output_stride_head,
    tail_output_stride_token,
    tail_output_stride_head,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TAIL_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_LOCATIONS: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    source_token = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)
    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    pair_input_base = (
        input_ptr
        + source_token * input_stride_token
        + head * input_stride_head
        + block_id * BLOCK_SIZE
    )
    value0 = tl.load(pair_input_base + pair_offsets * 2).to(tl.float32)
    value1 = tl.load(pair_input_base + pair_offsets * 2 + 1).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(value0), axis=0), tl.max(tl.abs(value1), axis=0))
    min_normal = 1.1754943508222875e-38
    scale_log2 = tl.ceil(tl.log2(tl.maximum(amax / 6.0, min_normal)))
    scale_byte = tl.minimum(tl.maximum(scale_log2 + 127.0, 1.0), 254.0).to(tl.int32)
    scale_byte = tl.where(amax == 0.0, 0, scale_byte)
    scale = (scale_byte << 23).to(tl.float32, bitcast=True)
    inv_scale = tl.where(amax == 0.0, 0.0, 1.0 / scale)

    code0 = _mxfp4_e2m1_code(value0 * inv_scale)
    code1 = _mxfp4_e2m1_code(value1 * inv_scale)
    packed = (code0 & 0x0F) | ((code1 & 0x0F) << 4)

    output_token = source_token
    if HAS_LOCATIONS:
        output_token = tl.load(location_ptr + source_token).to(tl.int64)
    tl.store(
        data_output_ptr
        + output_token * data_output_stride_token
        + head * data_output_stride_head
        + block_id * (BLOCK_SIZE // 2)
        + pair_offsets,
        packed,
    )
    tl.store(
        scale_output_ptr
        + output_token * scale_output_stride_token
        + head * scale_output_stride_head
        + block_id,
        scale_byte,
    )

    if HAS_TAIL and block_id == 0:
        tail_offsets = tl.arange(0, TAIL_DIM)
        tail = tl.load(
            tail_input_ptr
            + source_token * tail_input_stride_token
            + head * tail_input_stride_head
            + tail_offsets,
        )
        tl.store(
            tail_output_ptr
            + output_token * tail_output_stride_token
            + head * tail_output_stride_head
            + tail_offsets,
            tail,
        )


def _launch_mxfp4_quantize_kernel(
    tensor: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    locations: torch.Tensor | None = None,
    bf16_tail: torch.Tensor | None = None,
    tail_output: torch.Tensor | None = None,
) -> None:
    b, head_num, head_dim = tensor.shape
    has_locations = locations is not None
    has_tail = bf16_tail is not None
    if has_tail != (tail_output is not None):
        raise ValueError("MXFP4 tail input and output must be provided together")

    use_cuda_dsa_kernel = (
        has_locations
        and has_tail
        and tensor.dtype == torch.bfloat16
        and head_num == 1
        and head_dim == 512
        and bf16_tail.shape[-1] == 64
        and locations.dtype in (torch.int32, torch.int64)
        and locations.is_contiguous()
        and packed.is_contiguous()
        and scales.is_contiguous()
        and tail_output.is_contiguous()
    )
    if use_cuda_dsa_kernel:
        from sglang.kernels.jit.mxfp4 import mxfp4_quantize_and_store_paged

        mxfp4_quantize_and_store_paged(
            tensor,
            bf16_tail,
            locations,
            packed,
            scales,
            tail_output,
            warps_per_block=16 if tensor.shape[0] <= 512 else 4,
        )
        return

    location_arg = locations if locations is not None else tensor
    tail_input_arg = bf16_tail if bf16_tail is not None else tensor
    tail_output_arg = tail_output if tail_output is not None else packed
    _mxfp4_quantize_kernel[(b, head_num, head_dim // MXFP4KVQuantizeUtil.BLOCK_SIZE)](
        tensor,
        tail_input_arg,
        location_arg,
        packed,
        scales,
        tail_output_arg,
        tensor.stride(0),
        tensor.stride(1),
        bf16_tail.stride(0) if bf16_tail is not None else 0,
        bf16_tail.stride(1) if bf16_tail is not None else 0,
        packed.stride(0),
        packed.stride(1),
        scales.stride(0),
        scales.stride(1),
        tail_output.stride(0) if tail_output is not None else 0,
        tail_output.stride(1) if tail_output is not None else 0,
        HEAD_NUM=head_num,
        HEAD_DIM=head_dim,
        TAIL_DIM=bf16_tail.shape[-1] if bf16_tail is not None else 0,
        BLOCK_SIZE=MXFP4KVQuantizeUtil.BLOCK_SIZE,
        HAS_LOCATIONS=has_locations,
        HAS_TAIL=has_tail,
        num_warps=1,
    )


def _mxfp4_dequantize_paged_cuda(
    quant_tensor: torch.Tensor,
    scale_factors: torch.Tensor,
    locations: torch.Tensor,
    dtype: torch.dtype,
    bf16_tail: torch.Tensor | None,
    return_compact_page_table: bool,
    separate_tail: bool,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"MXFP4 dequantization only supports BF16/FP16, got {dtype}")
    if quant_tensor.ndim != 3 or scale_factors.ndim != 3:
        raise ValueError("MXFP4 paged buffers must have shape [tokens, heads, dim]")

    num_tokens = locations.numel()
    head_num = quant_tensor.shape[1]
    packed_dim = quant_tensor.shape[2]
    head_dim = packed_dim * 2
    if head_dim % MXFP4KVQuantizeUtil.BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP4 head dimension must be divisible by 32, got {head_dim}"
        )
    if scale_factors.shape != (
        quant_tensor.shape[0],
        head_num,
        head_dim // MXFP4KVQuantizeUtil.BLOCK_SIZE,
    ):
        raise ValueError("MXFP4 data and scale buffer shapes are inconsistent")

    tail_dim = 0
    if bf16_tail is not None:
        if bf16_tail.ndim != 3 or bf16_tail.shape[:2] != quant_tensor.shape[:2]:
            raise ValueError("MXFP4 BF16 tail shape is inconsistent with data buffer")
        if bf16_tail.dtype != torch.bfloat16:
            raise ValueError(f"MXFP4 tail must be BF16, got {bf16_tail.dtype}")
        tail_dim = bf16_tail.shape[-1]

    use_cuda_dsa_kernel = (
        dtype == torch.bfloat16
        and head_num == 1
        and head_dim == 512
        and tail_dim == 64
        and locations.dtype == torch.int32
    )

    # Target verification can contain several highly overlapping top-k tables
    # per request. Once the number of occurrences exceeds the physical cache
    # capacity, claim each source row once and retain its original page number.
    # This avoids repeatedly expanding the same row for adjacent draft tokens.
    if (
        use_cuda_dsa_kernel
        and return_compact_page_table
        and num_tokens > quant_tensor.shape[0]
    ):
        from sglang.kernels.jit.mxfp4 import (
            mxfp4_dequantize_paged_dedup,
            mxfp4_dequantize_paged_dedup_latent,
        )

        assert bf16_tail is not None
        output = torch.empty(
            (
                quant_tensor.shape[0],
                head_num,
                head_dim if separate_tail else head_dim + tail_dim,
            ),
            dtype=dtype,
            device=quant_tensor.device,
        )
        page_table = torch.empty(
            num_tokens, dtype=torch.int32, device=quant_tensor.device
        )
        claimed = torch.zeros(
            quant_tensor.shape[0], dtype=torch.int32, device=quant_tensor.device
        )
        dequantize = (
            mxfp4_dequantize_paged_dedup_latent
            if separate_tail
            else mxfp4_dequantize_paged_dedup
        )
        dequantize(
            quant_tensor,
            scale_factors,
            bf16_tail,
            locations,
            output,
            page_table,
            claimed,
        )
        if separate_tail:
            return output, bf16_tail, page_table
        return output, page_table

    output = torch.empty(
        (num_tokens, head_num, head_dim + tail_dim),
        dtype=dtype,
        device=quant_tensor.device,
    )
    compact_page_table = torch.empty(
        num_tokens, dtype=torch.int32, device=quant_tensor.device
    )

    # The GLM-5.2 DSA layout is common enough to justify a warp-specialized
    # CUDA path. Packing several independent rows into one CTA substantially
    # reduces scheduling overhead and uses vectorized BF16 stores. Keep the
    # generic Triton implementation below for other layouts and dtypes.
    if use_cuda_dsa_kernel:
        from sglang.kernels.jit.mxfp4 import mxfp4_dequantize_paged

        assert bf16_tail is not None
        mxfp4_dequantize_paged(
            quant_tensor,
            scale_factors,
            bf16_tail,
            locations,
            output,
            compact_page_table,
            warps_per_block=4,
        )
        if return_compact_page_table:
            return output, compact_page_table
        return output

    tail_arg = bf16_tail if bf16_tail is not None else quant_tensor
    _mxfp4_dequantize_paged_kernel[(num_tokens, head_num)](
        quant_tensor,
        scale_factors,
        tail_arg,
        locations,
        output,
        compact_page_table,
        quant_tensor.stride(0),
        quant_tensor.stride(1),
        scale_factors.stride(0),
        scale_factors.stride(1),
        bf16_tail.stride(0) if bf16_tail is not None else 0,
        bf16_tail.stride(1) if bf16_tail is not None else 0,
        output.stride(0),
        output.stride(1),
        HEAD_DIM=head_dim,
        TAIL_DIM=tail_dim,
        SCALE_BLOCK_SIZE=MXFP4KVQuantizeUtil.BLOCK_SIZE,
        HAS_TAIL=bf16_tail is not None,
        WRITE_COMPACT_PAGE_TABLE=return_compact_page_table,
        num_warps=1,
    )
    if return_compact_page_table:
        return output, compact_page_table
    return output


@triton.jit
def _mxfp4_dequantize_paged_kernel(
    data_ptr,
    scale_ptr,
    tail_ptr,
    location_ptr,
    output_ptr,
    compact_page_table_ptr,
    data_stride_token,
    data_stride_head,
    scale_stride_token,
    scale_stride_head,
    tail_stride_token,
    tail_stride_head,
    output_stride_token,
    output_stride_head,
    HEAD_DIM: tl.constexpr,
    TAIL_DIM: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    WRITE_COMPACT_PAGE_TABLE: tl.constexpr,
):
    output_token = tl.program_id(0)
    head = tl.program_id(1)
    source_token_raw = tl.load(location_ptr + output_token).to(tl.int64)
    source_token = tl.maximum(source_token_raw, 0)
    if WRITE_COMPACT_PAGE_TABLE:
        tl.store(
            compact_page_table_ptr + output_token,
            tl.where(source_token_raw >= 0, output_token, -1),
            mask=head == 0,
        )
    offsets = tl.arange(0, SCALE_BLOCK_SIZE)
    for block_id in tl.static_range(HEAD_DIM // SCALE_BLOCK_SIZE):
        packed_offsets = block_id * (SCALE_BLOCK_SIZE // 2) + offsets // 2
        packed = tl.load(
            data_ptr
            + source_token * data_stride_token
            + head * data_stride_head
            + packed_offsets
        ).to(tl.int32)
        nibble = tl.where((offsets & 1) == 0, packed & 0xF, (packed >> 4) & 0xF)
        magnitude = nibble & 0x7
        value = tl.where(
            magnitude < 4,
            magnitude.to(tl.float32) * 0.5,
            tl.exp2(((magnitude - 2) // 2).to(tl.float32))
            * (1.0 + (magnitude & 1).to(tl.float32) * 0.5),
        )
        value = tl.where((nibble & 0x8) != 0, -value, value)

        scale_byte = tl.load(
            scale_ptr
            + source_token * scale_stride_token
            + head * scale_stride_head
            + block_id
        ).to(tl.int32)
        value *= tl.exp2((scale_byte - 127).to(tl.float32))
        tl.store(
            output_ptr
            + output_token * output_stride_token
            + head * output_stride_head
            + block_id * SCALE_BLOCK_SIZE
            + offsets,
            value,
        )

    if HAS_TAIL:
        tail_offsets = tl.arange(0, TAIL_DIM)
        tail = tl.load(
            tail_ptr
            + source_token * tail_stride_token
            + head * tail_stride_head
            + tail_offsets,
        )
        tl.store(
            output_ptr
            + output_token * output_stride_token
            + head * output_stride_head
            + HEAD_DIM
            + tail_offsets,
            tail,
        )


class NVFP4KVQuantizeUtil:
    """Utility class for NVFP4 quantization and dequantization with two-level scaling
    (global FP32 + block FP8 E4M3).

    Quantize formula:  x_fp4 * block_scale * global_scale = x_bf16
    - Quantize: ``nvfp4_kv_quantize`` (SM100+), fallback ``fp4_quantize`` (SM90)
    - Dequantize: ``nvfp4_kv_dequantize`` (SM100+)
    """

    @staticmethod
    def quantize(
        tensor: torch.Tensor, global_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize BF16/FP16 tensor to NVFP4 format.

        Requires SM90+.  Uses ``nvfp4_kv_quantize`` on SM100+ (native PTX),
        falls back to ``fp4_quantize`` on SM90.

        Args:
            tensor: Input tensor of shape [B, M, N]
            global_scale: Global scale factor (float32 scalar or 1-element tensor)

        Returns:
            (fp4_data, block_scales, global_scale):
                fp4_data: shape [B, M, N/2], dtype uint8
                block_scales: shape [B, M, N/16], dtype float8_e4m3fn
                global_scale: passthrough
        """

        assert (
            get_platform().is_sm100 or get_platform().is_sm120 or get_platform().is_sm90
        ), "NVFP4 KV cache quantize requires SM100/SM120 or SM90 fallback GPU"

        b, m, n = tensor.shape
        tensor_2d = tensor.reshape(b * m, n)

        # The KV cache path passes preloaded per-layer scales already on device.
        # Keep scalar/0-d support for tests and future fallback paths, but do not
        # silently move tensor scales here.
        if isinstance(global_scale, (int, float)):
            global_scale = torch.tensor(
                [global_scale], dtype=torch.float32, device=tensor.device
            )
        elif global_scale.dim() == 0:
            global_scale = global_scale.unsqueeze(0)
        elif global_scale.device != tensor.device:
            raise ValueError(
                "NVFP4 global scale tensor must already be on the KV tensor device."
            )

        if get_platform().is_sm100 or get_platform().is_sm120:
            from flashinfer import nvfp4_kv_quantize

            # nvfp4_kv_quantize takes global_scale directly (not inverted)
            fp4_2d, scales_2d = nvfp4_kv_quantize(tensor_2d, global_scale)
        else:
            # SM90: fp4_quantize takes inverted global_scale
            from flashinfer import fp4_quantize

            global_scale_inv = 1.0 / global_scale
            fp4_2d, scales_2d = fp4_quantize(
                tensor_2d,
                global_scale_inv,
                sf_vec_size=16,
                sf_use_ue8m0=False,
                is_sf_swizzled_layout=False,
                is_sf_8x4_layout=False,
                enable_pdl=None,
            )

        fp4_data = fp4_2d.view(b, m, fp4_2d.shape[-1])
        block_scales = scales_2d.view(b, m, scales_2d.shape[-1]).view(
            torch.float8_e4m3fn
        )
        return fp4_data, block_scales, global_scale

    @staticmethod
    def dequantize(
        quant_tensor: torch.Tensor,
        block_scales: torch.Tensor,
        global_scale: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Dequantize NVFP4 tensor to BF16/FP16.

        Uses ``nvfp4_kv_dequantize`` on SM100+, falls back to pure PyTorch
        E2M1 LUT on SM90.

        Args:
            quant_tensor: Packed FP4 data of shape [B, M, N/2] (uint8)
            block_scales: Per-block FP8 E4M3 scales of shape [B, M, N/16]
            global_scale: Global scale factor (float32)
            dtype: Output dtype (bfloat16 or float16)

        Returns:
            Dequantized tensor of shape [B, M, N]
        """

        b, m, n_half = quant_tensor.shape

        # The KV cache path passes preloaded per-layer scales already on device.
        # Keep scalar/0-d support for tests and future fallback paths, but do not
        # silently move tensor scales here.
        if isinstance(global_scale, (int, float)):
            global_scale = torch.tensor(
                [global_scale], dtype=torch.float32, device=quant_tensor.device
            )
        elif global_scale.dim() == 0:
            global_scale = global_scale.unsqueeze(0)
        elif global_scale.device != quant_tensor.device:
            raise ValueError(
                "NVFP4 global scale tensor must already be on the KV tensor device."
            )

        if get_platform().is_sm100 or get_platform().is_sm120:
            from flashinfer import nvfp4_kv_dequantize

            quant_2d = quant_tensor.view(torch.uint8).reshape(b * m, n_half)
            scales_2d = block_scales.view(torch.uint8).reshape(b * m, -1)
            output_2d = nvfp4_kv_dequantize(
                quant_2d, scales_2d, global_scale, output_dtype=dtype
            )
            return output_2d.reshape(b, m, -1)
        else:
            assert get_platform().is_sm90, (
                "NVFP4 KV cache dequantize requires SM100/SM120 or SM90 fallback GPU"
            )
            # Pure PyTorch fallback for SM90
            n = n_half * 2
            fp4_vals = torch.empty(
                b, m, n, dtype=torch.uint8, device=quant_tensor.device
            )
            fp4_vals[..., 0::2] = quant_tensor & 0x0F
            fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F
            values = quant_tensor.new_tensor(E2M1_VALUES, dtype=torch.float32)
            float_vals = values[fp4_vals.long()]
            reshaped = float_vals.view(b, m * n // 16, 16)
            block_scales_float = block_scales.float().unsqueeze(-1)
            scaled = reshaped * block_scales_float
            return (scaled.view(b, m, n) * global_scale).to(dtype)
