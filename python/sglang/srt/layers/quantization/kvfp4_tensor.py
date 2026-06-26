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


class BlockFP4KVQuantizeUtil:
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
        from sglang.srt.utils import is_sm90_supported, is_sm100_supported

        assert is_sm90_supported(), "NVFP4 KV cache quantize requires SM90+ GPU"

        b, m, n = tensor.shape
        tensor_2d = tensor.reshape(b * m, n)

        if isinstance(global_scale, (int, float)):
            global_scale = torch.tensor(
                [global_scale], dtype=torch.float32, device=tensor.device
            )
        elif global_scale.dim() == 0:
            global_scale = global_scale.unsqueeze(0)

        if is_sm100_supported():
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
        from sglang.srt.utils import is_sm100_supported

        b, m, n_half = quant_tensor.shape

        if isinstance(global_scale, (int, float)):
            global_scale = torch.tensor(
                [global_scale], dtype=torch.float32, device=quant_tensor.device
            )
        elif global_scale.dim() == 0:
            global_scale = global_scale.unsqueeze(0)

        if is_sm100_supported():
            from flashinfer import nvfp4_kv_dequantize

            quant_2d = quant_tensor.view(torch.uint8).reshape(b * m, n_half)
            scales_2d = block_scales.view(torch.uint8).reshape(b * m, -1)
            output_2d = nvfp4_kv_dequantize(
                quant_2d, scales_2d, global_scale, output_dtype=dtype
            )
            return output_2d.reshape(b, m, -1)
        else:
            # Pure PyTorch fallback for SM90
            n = n_half * 2
            fp4_vals = torch.empty(
                b, m, n, dtype=torch.uint8, device=quant_tensor.device
            )
            fp4_vals[..., 0::2] = quant_tensor & 0x0F
            fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F
            float_vals = E2M1_VALUES[fp4_vals.long()]
            reshaped = float_vals.view(b, m * n // 16, 16)
            block_scales_float = block_scales.float().unsqueeze(-1)
            scaled = reshaped * block_scales_float
            return (scaled.view(b, m, n) * global_scale).to(dtype)
