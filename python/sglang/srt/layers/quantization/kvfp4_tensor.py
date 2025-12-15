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

from sglang.srt.utils.common import is_flashinfer_available, is_sm100_supported

if is_flashinfer_available() and is_sm100_supported():
    from flashinfer import e2m1_and_ufp8sf_scale_to_float, fp4_quantize
else:
    e2m1_and_ufp8sf_scale_to_float = None
    fp4_quantize = None

E2M1_MAX = 6.0
# Put constants directly on CUDA if available
_device = "cuda" if torch.cuda.is_available() else "cpu"
E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=_device
)
E2M1_BOUNDS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device=_device
)


class KVMXFP4QuantizeUtil:
    """Utility class for MXFP4 quantization and dequantization operations.

    MXFP4 uses:
    - E8M0 scale factors (exponent-only, stored as uint8)
    - Block size: 16 elements
    """

    @staticmethod
    @torch.compile
    def batched_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to MXFP4 format
        Args:
            tensor: Input tensor of shape [B, M, N]

        Returns:
            quant_tensor: Quantized tensor of shape [B, M, N/2]
            scale_factors: E8M0 scale factors of shape [B, M*N/16] (uint8)
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
        Dequantize MXFP4 tensor
        Args:
            quant_tensor: Quantized tensor of shape [B, M, N/2]
            scale_factors: E8M0 scale factors of shape [B, M*N/16] (uint8)
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

        scale_exp = scale_factors.float() - 127
        scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))

        return scaled.view(b, m, n).to(dtype)


class KVNVFP4QuantizeUtil:
    """Utility class for NVFP4 quantization and dequantization operations.

    NVFP4 uses:
    - FP8 E4M3FN scale factors (full FP8 format, stored as uint8)
    - Block size: 16 elements
    - Global FP32 scale (optional, loaded from checkpoint, per layer)
    """

    @staticmethod
    def batched_quantize(
        tensor: torch.Tensor,
        global_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to NVFP4 format.

        Block scales are stored compactly as
        (num_heads * head_dim / 16) bytes per token (per K or V), i.e. no padding-to-128
        and no swizzled layout.
        Args:
            tensor: Input tensor of shape [B, M, N]
            global_scale: Optional global FP32 scale factor (scalar tensor).
                         If None, defaults to 1.0 (no scaling).

        Returns:
            quant_tensor: Quantized tensor of shape [B, M, N/2]
            scale_factors: FP8 E4M3FN scale factors (uint8)
        """
        if fp4_quantize is None:
            raise RuntimeError("fp4_quantize requires flashinfer and SM100 support")

        b, m, n = tensor.shape

        # Ensure tensor is contiguous (required by flashinfer)
        tensor = tensor.contiguous()

        # Always pass a global scale (default to 1.0 if not provided)
        if global_scale is None:
            global_scale = torch.tensor(
                [1.0], dtype=torch.float32, device=tensor.device
            )
        elif not isinstance(global_scale, torch.Tensor):
            global_scale = torch.tensor(
                [global_scale], dtype=torch.float32, device=tensor.device
            )
        elif global_scale.dim() == 0:
            global_scale = global_scale.unsqueeze(0)

        # Use flashinfer fp4_quantize on a flattened (B*M, N) matrix with
        # is_sf_swizzled_layout=False to get compact per-16 block scales.
        flat = tensor.reshape(b * m, n)
        q_flat, sf_flat = fp4_quantize(
            flat,
            global_scale,
            sf_vec_size=16,
            sf_use_ue8m0=False,  # UE4M3 scales (NVFP4)
            is_sf_swizzled_layout=False,
            is_sf_8x4_layout=False,
        )

        quant_tensor = q_flat.view(b, m, n // 2)
        scale_factors = sf_flat.view(b, (m * n) // 16)
        return quant_tensor, scale_factors

    @staticmethod
    def batched_dequantize(
        quant_tensor: torch.Tensor,
        scale_factors: torch.Tensor,
        global_scale: torch.Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize NVFP4 tensor using flashinfer's e2m1_and_ufp8sf_scale_to_float.
        Args:
            quant_tensor: Quantized tensor of shape [B, M, N/2]
            scale_factors: FP8 E4M3FN scale factors (uint8) in swizzled layout
            global_scale: Optional global FP32 scale factor (scalar tensor).
                         If None, defaults to 1.0 (no scaling).
            dtype: Target dtype for output

        Returns:
            Dequantized tensor of shape [B, M, N]
        """
        if e2m1_and_ufp8sf_scale_to_float is None:
            raise RuntimeError(
                "e2m1_and_ufp8sf_scale_to_float requires flashinfer and SM100 support"
            )

        b, m, n_half = quant_tensor.shape

        # Ensure tensors are contiguous (required by flashinfer)
        quant_tensor = quant_tensor.contiguous()
        scale_factors = scale_factors.contiguous()

        # Always pass a global scale (default to 1.0 if not provided)
        if global_scale is None:
            global_scale = torch.tensor(
                [1.0], dtype=torch.float32, device=quant_tensor.device
            )
        elif not isinstance(global_scale, torch.Tensor):
            global_scale = torch.tensor(
                [global_scale], dtype=torch.float32, device=quant_tensor.device
            )
        elif global_scale.dim() == 0:
            global_scale = global_scale.unsqueeze(0)

        # Process each batch element (e2m1_and_ufp8sf_scale_to_float works on [M, K/2])
        dequantized_batches = []
        for i in range(b):
            dequantized = e2m1_and_ufp8sf_scale_to_float(
                quant_tensor[i],  # [M, N/2]
                scale_factors[i].flatten(),  # compact (M*N/16)
                global_scale,
                sf_vec_size=16,
                ufp8_type=1,  # 1 for E4M3 (NVFP4)
                is_sf_swizzled_layout=False,
            )
            dequantized_batches.append(dequantized)

        # Stack batches and convert to target dtype
        result = torch.stack(dequantized_batches, dim=0)  # [B, M, N]
        return result.to(dtype)


# Backward compatibility alias. TODO: remove this in the future.
KVFP4QuantizeUtil = KVMXFP4QuantizeUtil
