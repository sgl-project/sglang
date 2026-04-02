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
    """Utility class for NVFP4 quantization and dequantization with two-level scaling (global FP32 + block FP8).

    Quantize formula:  x_fp4 * block_scale * global_scale = x_bf16
    - Quantize: via FlashInfer ``fp4_quantize`` (fi_nvfp4_quantize)
    - Dequantize: pure PyTorch E2M1 LUT lookup (dequantize)
    """

    @staticmethod
    def fi_nvfp4_quantize(tensor: torch.Tensor, global_scale: torch.Tensor):
        # input and output shape [B, M, N]
        # return uint8 cache and fp8 block scales
        try:
            from flashinfer import fp4_quantize
        except ImportError:
            raise ImportError(
                "flashinfer is installed correctly. Please install flashinfer to use NVFP4 KV cache."
            )
        global_scale_inv = 1.0 / global_scale
        if isinstance(global_scale_inv, float):
            global_scale_inv = torch.tensor(
                global_scale_inv, dtype=torch.float32, device=tensor.device
            )
        assert (
            global_scale_inv.device == tensor.device
        ), "global_scale and tensor must be on the same device"
        b, m, n = tensor.shape
        tensor_reshaped = tensor.reshape(b * m, n)
        tensor_fp4, tensor_fp4_sf = fp4_quantize(
            tensor_reshaped,
            global_scale_inv,
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
            is_sf_8x4_layout=False,
            enable_pdl=None,
        )
        tensor_fp4 = tensor_fp4.view(b, m, tensor_fp4.shape[-1])
        tensor_fp4_sf = tensor_fp4_sf.view(b, m, tensor_fp4_sf.shape[-1]).view(
            torch.float8_e4m3fn
        )
        return tensor_fp4, tensor_fp4_sf, global_scale

    @staticmethod
    def dequantize(
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
