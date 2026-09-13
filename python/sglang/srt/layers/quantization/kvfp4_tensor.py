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
    """OCP MXFP4: block-32 E2M1 values with one E8M0 scale per head block.

    The scale follows OCP MX v1.0 section 6.3: the largest power of two not
    greater than the block amax, divided by the largest power of two
    representable by E2M1 (4). E2M1 conversion is saturating round-to-nearest,
    ties-to-even. Scale tensors contain raw E8M0 bytes.
    """

    BLOCK_SIZE = 32
    E8M0_MIN_EXP = -127
    E8M0_MAX_EXP = 127
    E8M0_NAN_BYTE = 0xFF

    @staticmethod
    def batched_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: deliberately eager, unlike FP4MXBlock16KVQuantizeUtil. Under
        # torch 2.11 inductor, the compiled slice[...,:n] + pad-to-even +
        # nibble-pack graph skipped the final output byte, so callers observed
        # stale buffer contents that varied per process. The golden contract is
        # bit-exactness, so this util stays on eager pure-tensor ops (still
        # CUDA-graph capturable).
        if tensor.ndim != 3:
            raise ValueError(
                f"MXFP4 expects a 3-D [tokens, heads, dim] tensor, got {tensor.shape}"
            )
        b, h, n = tensor.shape
        if n <= 0:
            raise ValueError("MXFP4 head_dim must be positive")

        num_blocks = (
            n + MXFP4KVQuantizeUtil.BLOCK_SIZE - 1
        ) // MXFP4KVQuantizeUtil.BLOCK_SIZE
        padded_n = num_blocks * MXFP4KVQuantizeUtil.BLOCK_SIZE
        values = tensor.to(torch.float32)
        if padded_n != n:
            values = torch.nn.functional.pad(values, (0, padded_n - n))
        blocks = values.reshape(b, h, num_blocks, MXFP4KVQuantizeUtil.BLOCK_SIZE)

        nan_blocks = torch.isnan(blocks).any(dim=-1)
        inf_blocks = torch.isinf(blocks).any(dim=-1) & ~nan_blocks
        finite_abs = torch.nan_to_num(blocks.abs(), nan=0.0, posinf=0.0, neginf=0.0)
        amax = finite_abs.amax(dim=-1)
        scale_exp = torch.floor(torch.log2(amax)) - 2.0
        scale_exp = torch.where(
            amax == 0,
            scale_exp.new_full((), MXFP4KVQuantizeUtil.E8M0_MIN_EXP),
            scale_exp,
        )
        scale_exp = torch.where(
            inf_blocks,
            scale_exp.new_full((), MXFP4KVQuantizeUtil.E8M0_MAX_EXP),
            scale_exp,
        ).clamp(
            MXFP4KVQuantizeUtil.E8M0_MIN_EXP,
            MXFP4KVQuantizeUtil.E8M0_MAX_EXP,
        )
        scale_bytes = (scale_exp.to(torch.int32) + 127).to(torch.uint8)
        scale_bytes = torch.where(
            nan_blocks,
            scale_bytes.new_full((), MXFP4KVQuantizeUtil.E8M0_NAN_BYTE),
            scale_bytes,
        )

        scaled = blocks / torch.exp2(scale_exp).unsqueeze(-1)
        scaled = torch.nan_to_num(scaled, nan=0.0, posinf=6.0, neginf=-6.0)
        abs_vals = scaled.abs().clamp(max=E2M1_MAX)
        e2m1_values = tensor.new_tensor(E2M1_VALUES[:8], dtype=torch.float32)
        distances = (abs_vals.unsqueeze(-1) - e2m1_values).abs()
        min_distances = distances.amin(dim=-1, keepdim=True)
        tied = distances == min_distances
        code_ids = torch.arange(8, dtype=torch.int64, device=tensor.device)
        even_tied = tied & ((code_ids & 1) == 0)
        magnitude_bits = torch.where(
            even_tied.any(dim=-1),
            even_tied.to(torch.uint8).argmax(dim=-1).to(torch.uint8),
            tied.to(torch.uint8).argmax(dim=-1).to(torch.uint8),
        )
        fp4_vals = magnitude_bits | (torch.signbit(scaled).to(torch.uint8) << 3)
        fp4_vals = torch.where(
            nan_blocks.unsqueeze(-1), torch.zeros_like(fp4_vals), fp4_vals
        )
        fp4_vals = fp4_vals.reshape(b, h, padded_n)[..., :n]
        if n % 2:
            fp4_vals = torch.nn.functional.pad(fp4_vals, (0, 1))
        packed = fp4_vals[..., 0::2] | (fp4_vals[..., 1::2] << 4)
        return packed.contiguous(), scale_bytes.contiguous()

    @staticmethod
    def batched_dequantize(
        quant_tensor: torch.Tensor,
        scale_factors: torch.Tensor,
        *,
        logical_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        # See the note on batched_quantize: kept eager for bit-exactness.
        if quant_tensor.ndim != 3 or scale_factors.ndim != 3:
            raise ValueError("MXFP4 packed data and scales must both be 3-D")
        if logical_dim <= 0 or quant_tensor.shape[-1] != (logical_dim + 1) // 2:
            raise ValueError(
                f"packed last dim {quant_tensor.shape[-1]} does not match "
                f"logical_dim {logical_dim}"
            )
        num_blocks = (
            logical_dim + MXFP4KVQuantizeUtil.BLOCK_SIZE - 1
        ) // MXFP4KVQuantizeUtil.BLOCK_SIZE
        if (
            scale_factors.shape[:-1] != quant_tensor.shape[:-1]
            or scale_factors.shape[-1] != num_blocks
        ):
            raise ValueError(
                f"scale shape {scale_factors.shape} does not match packed shape "
                f"{quant_tensor.shape} and logical_dim {logical_dim}"
            )

        quant_bytes = quant_tensor.view(torch.uint8)
        fp4_vals = torch.empty(
            *quant_bytes.shape[:-1],
            quant_bytes.shape[-1] * 2,
            dtype=torch.uint8,
            device=quant_tensor.device,
        )
        fp4_vals[..., 0::2] = quant_bytes & 0x0F
        fp4_vals[..., 1::2] = (quant_bytes >> 4) & 0x0F
        fp4_vals = fp4_vals[..., :logical_dim]

        magnitude_idx = fp4_vals & 0x07
        values = quant_tensor.new_tensor(E2M1_VALUES[:8], dtype=torch.float32)
        float_vals = values[magnitude_idx.long()]
        float_vals = torch.where((fp4_vals & 0x08) != 0, -float_vals, float_vals)

        padded_n = num_blocks * MXFP4KVQuantizeUtil.BLOCK_SIZE
        if padded_n != logical_dim:
            float_vals = torch.nn.functional.pad(
                float_vals, (0, padded_n - logical_dim)
            )
        blocks = float_vals.reshape(
            *float_vals.shape[:-1], num_blocks, MXFP4KVQuantizeUtil.BLOCK_SIZE
        )

        scale_bytes = scale_factors.view(torch.uint8)
        nan_blocks = scale_bytes == MXFP4KVQuantizeUtil.E8M0_NAN_BYTE
        scale_exp = scale_bytes.to(torch.int16) - 127
        scales = torch.exp2(scale_exp.to(torch.float32))
        scales = torch.where(nan_blocks, scales.new_full((), float("nan")), scales)
        output = (blocks * scales.unsqueeze(-1)).flatten(-2)[..., :logical_dim]
        return output.to(dtype)


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
