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

# Define a enum class for FP4 formats, including MXFP4, NVFP4 and future formats
from enum import Enum

import torch


class FP4KVCacheRecipe(Enum):
    MXFP4 = 1  # KVFP4: block-wise scaling
    NVFP4 = 2  # two-level scaling: global FP32 + block FP8 E4M3


E2M1_MAX = 6.0
MAX_BLOCK_SCALE_FP8 = 448.0  # Maximum FP8 E4M3 value
# Put constants directly on CUDA if available
_device = "cuda" if torch.cuda.is_available() else "cpu"
# E2M1 format: 1 sign bit + 2 exponent bits + 1 mantissa bit = 4 bits
# 16 possible values: 0x0-0xF
# Negative values: 0x8-0xF (sign bit = 1)
# Positive values: 0x0-0x7 (sign bit = 0)
E2M1_VALUES = torch.tensor(
    [
        0,
        0.5,
        1,
        1.5,
        2,
        3,
        4,
        6,  # 0x0-0x7: positive values
        -0,
        -0.5,
        -1,
        -1.5,
        -2,
        -3,
        -4,
        -6,
    ],  # 0x8-0xF: negative values
    dtype=torch.float32,
    device=_device,
)
E2M1_BOUNDS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device=_device
)


class KVFP4QuantizeUtil:
    """Utility class for MXFP4 quantization and dequantization operations."""

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
        magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1)

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
        float_vals = E2M1_VALUES[magnitude_idx.long()]
        float_vals = torch.where(sign_mask, -float_vals, float_vals)

        # Reshape for block-wise scaling
        reshaped = float_vals.view(b, m * n // 16, 16)

        # Apply scale factors
        scale_exp = scale_factors.float() - 127
        scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))

        return scaled.view(b, m, n).to(dtype)


class NVFP4QuantizeUtil:
    """Utility class for NVFP4 quantization and dequantization with two-level scaling (global FP32 + block FP8)."""

    @staticmethod
    # @torch.compile
    def batched_quantize(
        tensor: torch.Tensor, global_scale: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to NVFP4 format with two-level scaling (global FP32 + block FP8 E4M3)

        Formula: x_fp4 * block_scale * global_scale = x_bf16

        Args:
            tensor: Input tensor of shape [B, M, N]
            global_scale: Optional global scale factor (float32 scalar).
                         If None, will auto-compute per-tensor global scale.
                         If provided, will use the given global scale.

        Returns:
            quant_tensor: Quantized E2M1 tensor of shape [B, M, N/2] (packed uint8)
            block_scales: Block scale factors of shape [B, M*N/16] (FP8 E4M3)
            global_scale: Global scale factor (float32 scalar)
        """
        b, m, n = tensor.shape
        device = tensor.device

        # Step 1: Calculate global_scale
        if global_scale is None:
            global_max = tensor.abs().amax()
            global_scale = torch.tensor(
                global_max.item() / (E2M1_MAX * MAX_BLOCK_SCALE_FP8),
                dtype=torch.float32,
                device=device,
            )
        else:
            # Use provided global scale
            if not isinstance(global_scale, torch.Tensor):
                global_scale = torch.tensor(
                    global_scale, dtype=torch.float32, device=device
                )
            else:
                global_scale = global_scale.to(device=device, dtype=torch.float32)

        if global_scale < 1e-6:
            global_scale = torch.tensor(1e-6, dtype=torch.float32, device=device)

        # Step 2: Scale x_bf16 to FP4 range [-6, 6]
        # First, reshape to blocks [B, M*N/16, 16]
        reshaped = tensor.float().view(b, m * n // 16, 16)
        block_max = reshaped.abs().amax(dim=-1, keepdim=True)
        block_scales = block_max.squeeze(-1) / (E2M1_MAX * global_scale)
        block_scales = torch.clamp(block_scales, 0.0, MAX_BLOCK_SCALE_FP8)
        block_scales_fp8 = block_scales.to(torch.float8_e4m3fn)

        # Scale each block to FP4 range: x_scaled = x / block_max * E2M1_MAX
        # This ensures values are in [-6, 6] range
        block_scales_fixed = block_scales.unsqueeze(-1)
        x_scaled = reshaped / (block_scales_fixed * global_scale)

        # Step 3: Convert scaled values (x_scaled) to packed FP4
        # x_scaled is already in FP4 range [-6, 6] in bf16 representation
        # Now quantize to E2M1 format

        # E2M1 format: bit 3 = sign, bits 2-0 = magnitude (exponent + mantissa)
        sign_bits = (x_scaled < 0).to(torch.uint8) << 3  # bit 3: sign bit
        abs_vals = x_scaled.abs()
        # Find nearest E2M1 magnitude (0-7) using boundaries
        magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1).to(
            torch.uint8
        )
        # Combine sign and magnitude: 4-bit value = sign_bit | magnitude
        fp4_vals = sign_bits | magnitude_bits
        # Pack two FP4 values into one uint8
        fp4_reshaped = fp4_vals.view(b, m, n)
        packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]

        return packed, block_scales_fp8, global_scale

    @staticmethod
    # @torch.compile
    def batched_dequantize(
        quant_tensor: torch.Tensor,
        block_scales: torch.Tensor,
        global_scale: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize NVFP4 tensor with two-level scaling (global FP32 + block FP8 E4M3)

        Args:
            quant_tensor: Quantized E2M1 tensor of shape [B, M, N/2] (packed uint8)
            block_scales: Block scale factors of shape [B, M*N/16] (FP8 E4M3)
            global_scale: Global scale factor (float32 scalar)
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

        # Directly map 4-bit E2M1 values (0x0-0xF) to float
        # E2M1_VALUES[0-7] = positive, E2M1_VALUES[8-15] = negative
        float_vals = E2M1_VALUES[fp4_vals.long()]

        # Reshape for block-wise scaling
        reshaped = float_vals.view(b, m * n // 16, 16)

        # Apply block scale factors (inverse scaling: divide by FP8 block scales)
        # Convert FP8 back to float32 for computation
        block_scales_float = block_scales.float().unsqueeze(-1)  # [B, M*N/16, 1]
        scaled = reshaped * block_scales_float

        # Apply inverse global scaling
        dequantized = scaled.view(b, m, n) * global_scale

        return dequantized.to(dtype)
