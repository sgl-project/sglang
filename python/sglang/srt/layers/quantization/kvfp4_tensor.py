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

"""Low-level tensor helpers for FP4 KV cache quantization.

Recipe selection and buffer ownership live in ``fp4_kv_cache_quant_method.py``. This
module only implements the packing, unpacking, and FlashInfer calls used by the
concrete quantization methods.
"""

import torch

E2M1_MAX = 6.0
# Keep constants as Python literals. The compiled helpers materialize them with
# input.new_tensor(), so they follow the caller device without a global GPU tensor
# or a CPU tensor .to(device) in the hot path.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


class FP4MXBlock16KVQuantizeUtil:
    """Utility class for block-16 FP4 quantization and dequantization operations."""

    @staticmethod
    @torch.compile
    def batched_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Quantize tensor to block-16 FP4 format
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
        Dequantize block-16 FP4 tensor
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
        values = quant_tensor.new_tensor(E2M1_VALUES, dtype=torch.float32)
        float_vals = values[magnitude_idx.long()]
        float_vals = torch.where(sign_mask, -float_vals, float_vals)

        # Reshape for block-wise scaling
        reshaped = float_vals.view(b, m * n // 16, 16)

        # Apply scale factors
        scale_exp = scale_factors.float() - 127
        scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))

        return scaled.view(b, m, n).to(dtype)


class NVFP4KVQuantizeUtil:
    """Utility wrapper for flashinfer NVFP4 KV quantization APIs."""

    @staticmethod
    def _as_global_scale_tensor(global_scale, device) -> torch.Tensor:
        if global_scale is None:
            raise ValueError("NVFP4 KV cache requires a per-layer global scale.")
        if isinstance(global_scale, (int, float)):
            return torch.tensor([global_scale], dtype=torch.float32, device=device)
        if global_scale.dim() == 0:
            global_scale = global_scale.unsqueeze(0)
        return global_scale.contiguous()

    @staticmethod
    def quantize(tensor: torch.Tensor, global_scale: torch.Tensor):
        try:
            from flashinfer import nvfp4_kv_quantize
        except ImportError as exc:
            raise ImportError(
                "flashinfer is required to use NVFP4 KV cache quantization."
            ) from exc

        b, m, n = tensor.shape
        global_scale = NVFP4KVQuantizeUtil._as_global_scale_tensor(
            global_scale, tensor.device
        )
        tensor_2d = tensor.reshape(b * m, n).contiguous()
        tensor_fp4, tensor_fp4_sf = nvfp4_kv_quantize(tensor_2d, global_scale)
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
        try:
            from flashinfer import nvfp4_kv_dequantize
        except ImportError as exc:
            raise ImportError(
                "flashinfer is required to use NVFP4 KV cache dequantization."
            ) from exc

        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"Unsupported dtype: {dtype}. Only torch.bfloat16 and torch.float16 are supported."
            )

        b, m, n_half = quant_tensor.shape
        global_scale = NVFP4KVQuantizeUtil._as_global_scale_tensor(
            global_scale, quant_tensor.device
        )
        quant_2d = quant_tensor.view(torch.uint8).reshape(b * m, n_half).contiguous()
        scales_2d = block_scales.view(torch.uint8).reshape(b * m, -1).contiguous()
        output_2d = nvfp4_kv_dequantize(quant_2d, scales_2d, global_scale, dtype)
        return output_2d.reshape(b, m, n_half * 2)
