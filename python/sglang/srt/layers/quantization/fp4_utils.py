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

    # Cached kernel modules
    _nvfp4_dequant_module = None
    _nvfp4_quant_sm100_module = None

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
    def cuda_nvfp4_dequantize(
        quant_tensor: torch.Tensor,
        block_scales: torch.Tensor,
        global_scale: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize NVFP4 tensor using optimized CUDA kernel with vectorization and shared memory.

        This is a fast kernel-based implementation that provides significant performance improvements
        over the pure PyTorch implementation.

        Args:
            quant_tensor: Quantized E2M1 tensor of shape [B, M, N/2] (packed uint8)
            block_scales: Block scale factors of shape [B, M, N/16] (FP8 E4M3)
            global_scale: Global scale factor (float32 scalar or tensor)
            dtype: Target dtype for output (torch.bfloat16 or torch.float16)

        Returns:
            Dequantized tensor of shape [B, M, N]
        """
        b, m, n_half = quant_tensor.shape
        n = n_half * 2

        # Ensure inputs are on CUDA
        assert quant_tensor.is_cuda, "quant_tensor must be on CUDA"
        assert block_scales.is_cuda, "block_scales must be on CUDA"
        assert quant_tensor.dtype == torch.uint8, "quant_tensor must be uint8"

        # Handle global_scale conversion - ensure it's a CUDA tensor
        if isinstance(global_scale, (int, float)):
            global_scale = torch.tensor(
                [global_scale], dtype=torch.float32, device=quant_tensor.device
            )
        else:
            # Ensure global_scale is on CUDA and is a 1D tensor with 1 element
            if not global_scale.is_cuda:
                global_scale = global_scale.to(quant_tensor.device, non_blocking=True)
            if global_scale.dim() == 0:
                global_scale = global_scale.unsqueeze(0)
            global_scale = global_scale.contiguous()

        # Get the kernel module
        module = NVFP4QuantizeUtil._get_dequant_module()

        # Reshape to 2D for kernel: [B*M, N/2] and [B*M, N/16]
        quant_2d = quant_tensor.reshape(b * m, n_half)

        # Convert FP8 E4M3 block scales to uint8 view for kernel
        if block_scales.dtype == torch.float8_e4m3fn:
            scales_2d = block_scales.view(torch.uint8).reshape(b * m, -1)
        else:
            scales_2d = block_scales.reshape(b * m, -1)

        # Call appropriate kernel based on dtype
        if dtype == torch.bfloat16:
            output_2d = module.nvfp4_dequant_to_bf16(quant_2d, scales_2d, global_scale)
        elif dtype == torch.float16:
            output_2d = module.nvfp4_dequant_to_fp16(quant_2d, scales_2d, global_scale)
        else:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Only torch.bfloat16 and torch.float16 are supported."
            )

        # Reshape back to 3D: [B, M, N]
        output = output_2d.reshape(b, m, n)

        return output

    @staticmethod
    def cuda_nvfp4_quantize_sm100(
        tensor: torch.Tensor,
        global_scale: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to NVFP4 format using optimized SM100 CUDA kernel.

        This is a fast kernel-based implementation optimized for SM100 architecture
        that uses native E2M1 conversion instructions.

        Args:
            tensor: Input tensor of shape [B, M, N] (bfloat16 or float16)
            global_scale: Global scale factor (float32 scalar or tensor)
            dtype: Input dtype (must match tensor dtype)

        Returns:
            quant_tensor: Quantized E2M1 tensor of shape [B, M, N/2] (packed uint8)
            block_scales: Block scale factors of shape [B, M, N/16] (FP8 E4M3)
            global_scale: Global scale factor (float32 scalar tensor)
        """
        b, m, n = tensor.shape

        # Ensure inputs are on CUDA
        assert tensor.is_cuda, "tensor must be on CUDA"
        assert dtype in [
            torch.bfloat16,
            torch.float16,
        ], "Only bfloat16 and float16 are supported"
        assert (
            tensor.dtype == dtype
        ), f"tensor dtype {tensor.dtype} must match specified dtype {dtype}"
        assert n % 16 == 0, "N dimension must be divisible by 16"

        # Handle global_scale conversion - ensure it's a CUDA tensor
        if isinstance(global_scale, (int, float)):
            global_scale = torch.tensor(
                [global_scale], dtype=torch.float32, device=tensor.device
            )
        else:
            # Ensure global_scale is on CUDA and is a 1D tensor with 1 element
            if not global_scale.is_cuda:
                global_scale = global_scale.to(tensor.device, non_blocking=True)
            if global_scale.dim() == 0:
                global_scale = global_scale.unsqueeze(0)
            global_scale = global_scale.contiguous()

        # Get the kernel module
        module = NVFP4QuantizeUtil._get_quant_sm100_module()

        # Reshape to 2D for kernel: [B*M, N]
        tensor_2d = tensor.reshape(b * m, n).contiguous()

        # Call appropriate kernel based on dtype
        if dtype == torch.bfloat16:
            quant_2d, scales_2d = module.nvfp4_quant_from_bf16(tensor_2d, global_scale)
        elif dtype == torch.float16:
            quant_2d, scales_2d = module.nvfp4_quant_from_fp16(tensor_2d, global_scale)
        else:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Only torch.bfloat16 and torch.float16 are supported."
            )

        # Reshape back to 3D: [B, M, N/2] and [B, M, N/16]
        quant_tensor = quant_2d.reshape(b, m, n // 2)

        # Convert uint8 block scales back to FP8 E4M3
        block_scales = scales_2d.view(torch.float8_e4m3fn).reshape(b, m, n // 16)

        return quant_tensor, block_scales, global_scale

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

    @staticmethod
    def _get_dequant_module():
        """Load and cache the NVFP4 dequantization kernel module."""
        if NVFP4QuantizeUtil._nvfp4_dequant_module is None:
            from torch.utils.cpp_extension import load_inline

            CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_fp8.h>
typedef __nv_fp8_e4m3 fp8_e4m3;
typedef __nv_fp8x2_e4m3 fp8x2_e4m3;
#define HAS_FP8_SUPPORT 1
#else
typedef uint8_t fp8_e4m3;
typedef uint16_t fp8x2_e4m3;
#define HAS_FP8_SUPPORT 0
#endif

// E2M1 lookup table
__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Dequantize 4 FP4 values (packed in uint16_t) to float2x2
__device__ __forceinline__ void dequant_fp4x4_to_float2x2(
    uint16_t packed_fp4,
    float scale_0,
    float scale_1,
    float global_scale,
    float2& out0,
    float2& out1
) {
    uint8_t fp4_0 = (packed_fp4 >> 0) & 0xF;
    uint8_t fp4_1 = (packed_fp4 >> 4) & 0xF;
    uint8_t fp4_2 = (packed_fp4 >> 8) & 0xF;
    uint8_t fp4_3 = (packed_fp4 >> 12) & 0xF;

    out0.x = E2M1_LUT[fp4_0] * scale_0 * global_scale;
    out0.y = E2M1_LUT[fp4_1] * scale_0 * global_scale;
    out1.x = E2M1_LUT[fp4_2] * scale_1 * global_scale;
    out1.y = E2M1_LUT[fp4_3] * scale_1 * global_scale;
}

template<typename OutType, int BLOCK_SIZE = 128, int ELTS_PER_THREAD = 16>
__global__ void nvfp4_dequant_vectorized_kernel(
    const uint8_t* __restrict__ fp4_data,
    const uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scale_ptr,
    OutType* __restrict__ output,
    const int M,
    const int K
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= M) return;

    // Load global_scale from device memory once per block
    __shared__ float global_scale;
    __shared__ uint8_t smem_scales[512];

    if (tid == 0) {
        global_scale = *global_scale_ptr;
    }

    const int K_scales = K / 16;
    const int K_packed = K / 2;

    for (int i = tid; i < K_scales; i += BLOCK_SIZE) {
        smem_scales[i] = block_scales[row * K_scales + i];
    }
    __syncthreads();

    constexpr int PACKED_PER_THREAD = ELTS_PER_THREAD / 2;
    const int elts_per_block = BLOCK_SIZE * ELTS_PER_THREAD;

    const uint8_t* row_fp4 = fp4_data + row * K_packed;
    OutType* row_output = output + row * K;

    for (int base_col = 0; base_col < K; base_col += elts_per_block) {
        const int col_start = base_col + tid * ELTS_PER_THREAD;

        if (col_start >= K) break;

        #pragma unroll
        for (int i = 0; i < PACKED_PER_THREAD / 2; ++i) {
            const int col = col_start + i * 4;
            if (col + 3 >= K) break;

            const int packed_idx = col / 2;
            uint16_t packed_fp4 = *reinterpret_cast<const uint16_t*>(&row_fp4[packed_idx]);

            const int scale_idx_0 = col / 16;
            const int scale_idx_1 = (col + 2) / 16;

            const uint8_t scale_fp8_0 = smem_scales[scale_idx_0];
            const uint8_t scale_fp8_1 = smem_scales[scale_idx_1];

#if HAS_FP8_SUPPORT
            const float scale_0 = static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3*>(&scale_fp8_0));
            const float scale_1 = static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3*>(&scale_fp8_1));
#else
            const float scale_0 = 1.0f;
            const float scale_1 = 1.0f;
#endif

            float2 out0, out1;
            dequant_fp4x4_to_float2x2(packed_fp4, scale_0, scale_1, global_scale, out0, out1);

            if constexpr (std::is_same_v<OutType, __nv_bfloat16>) {
                __nv_bfloat162 bf16_0 = __float22bfloat162_rn(out0);
                __nv_bfloat162 bf16_1 = __float22bfloat162_rn(out1);

                *reinterpret_cast<__nv_bfloat162*>(&row_output[col]) = bf16_0;
                *reinterpret_cast<__nv_bfloat162*>(&row_output[col + 2]) = bf16_1;
            } else if constexpr (std::is_same_v<OutType, half>) {
                half2 h2_0 = __float22half2_rn(out0);
                half2 h2_1 = __float22half2_rn(out1);

                *reinterpret_cast<half2*>(&row_output[col]) = h2_0;
                *reinterpret_cast<half2*>(&row_output[col + 2]) = h2_1;
            }
        }
    }
}

torch::Tensor nvfp4_dequant_to_bf16_cuda_v2(
    torch::Tensor fp4_data,
    torch::Tensor block_scales,
    torch::Tensor global_scale
) {
    TORCH_CHECK(fp4_data.is_cuda(), "fp4_data must be CUDA tensor");
    TORCH_CHECK(fp4_data.dtype() == torch::kUInt8, "fp4_data must be uint8");
    TORCH_CHECK(global_scale.is_cuda(), "global_scale must be CUDA tensor");
    TORCH_CHECK(global_scale.dtype() == torch::kFloat32, "global_scale must be float32");

    const int M = fp4_data.size(0);
    const int K = fp4_data.size(1) * 2;

    auto output = torch::empty({M, K}, torch::TensorOptions()
        .dtype(torch::kBFloat16).device(fp4_data.device()));

    constexpr int BLOCK_SIZE = 128;
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_dequant_vectorized_kernel<__nv_bfloat16, BLOCK_SIZE, 16><<<grid, block, 0, stream>>>(
        fp4_data.data_ptr<uint8_t>(),
        block_scales.data_ptr<uint8_t>(),
        global_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        M, K
    );

    return output;
}

torch::Tensor nvfp4_dequant_to_fp16_cuda_v2(
    torch::Tensor fp4_data,
    torch::Tensor block_scales,
    torch::Tensor global_scale
) {
    TORCH_CHECK(fp4_data.is_cuda(), "fp4_data must be CUDA tensor");
    TORCH_CHECK(fp4_data.dtype() == torch::kUInt8, "fp4_data must be uint8");
    TORCH_CHECK(global_scale.is_cuda(), "global_scale must be CUDA tensor");
    TORCH_CHECK(global_scale.dtype() == torch::kFloat32, "global_scale must be float32");

    const int M = fp4_data.size(0);
    const int K = fp4_data.size(1) * 2;

    auto output = torch::empty({M, K}, torch::TensorOptions()
        .dtype(torch::kFloat16).device(fp4_data.device()));

    constexpr int BLOCK_SIZE = 128;
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_dequant_vectorized_kernel<half, BLOCK_SIZE, 16><<<grid, block, 0, stream>>>(
        fp4_data.data_ptr<uint8_t>(),
        block_scales.data_ptr<uint8_t>(),
        global_scale.data_ptr<float>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, K
    );

    return output;
}
"""

            CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor nvfp4_dequant_to_bf16_cuda_v2(
    torch::Tensor fp4_data,
    torch::Tensor block_scales,
    torch::Tensor global_scale
);

torch::Tensor nvfp4_dequant_to_fp16_cuda_v2(
    torch::Tensor fp4_data,
    torch::Tensor block_scales,
    torch::Tensor global_scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_dequant_to_bf16", &nvfp4_dequant_to_bf16_cuda_v2, "NVFP4 dequantize to BF16 V2");
    m.def("nvfp4_dequant_to_fp16", &nvfp4_dequant_to_fp16_cuda_v2, "NVFP4 dequantize to FP16 V2");
}
"""

            NVFP4QuantizeUtil._nvfp4_dequant_module = load_inline(
                name="nvfp4_dequant_v2",
                cpp_sources=[CPP_SOURCE],
                cuda_sources=[CUDA_SOURCE],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-DENABLE_BF16",
                ],
                verbose=False,
                with_cuda=True,
            )

        return NVFP4QuantizeUtil._nvfp4_dequant_module

    @staticmethod
    def _get_quant_sm100_module():
        """Load and cache the NVFP4 quantization kernel module for SM100 architecture."""
        if NVFP4QuantizeUtil._nvfp4_quant_sm100_module is None:
            from torch.utils.cpp_extension import load_inline

            CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_fp8.h>
typedef __nv_fp8_e4m3 fp8_e4m3;
#define HAS_FP8_SUPPORT 1
#else
typedef uint8_t fp8_e4m3;
#define HAS_FP8_SUPPORT 0
#endif

// Helper functions
__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ __nv_bfloat162 cuda_abs(__nv_bfloat162 a) {
    __nv_bfloat162 result;
    float fx = fabsf(__bfloat162float(a.x));
    float fy = fabsf(__bfloat162float(a.y));
    result.x = __float2bfloat16(fx);
    result.y = __float2bfloat16(fy);
    return result;
}

__device__ __forceinline__ half2 cuda_abs(half2 a) {
    return __habs2(a);
}

__device__ __forceinline__ __nv_bfloat162 cuda_max(__nv_bfloat162 a, __nv_bfloat162 b) {
    __nv_bfloat162 result;
    result.x = __bfloat162float(a.x) > __bfloat162float(b.x) ? a.x : b.x;
    result.y = __bfloat162float(a.y) > __bfloat162float(b.y) ? a.y : b.y;
    return result;
}

__device__ __forceinline__ half2 cuda_max(half2 a, half2 b) {
    return __hmax2(a, b);
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y), "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y));
    return val;
#else
    // Fallback for non-SM100
    return 0;
#endif
}

// Quantize 8 FP16/BF16 values to E2M1 with FP8 E4M3 block scaling
template<typename InType>
__device__ uint32_t quantize_fp16_to_e2m1_with_scaling(
    InType (&vec)[4],  // 4 x Vec2 = 8 values
    float global_scale,
    uint8_t* block_scale_out
) {
    constexpr int SF_VEC_SIZE = 16;
    constexpr int CVT_ELTS_PER_THREAD = 8;
    constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_ELTS_PER_THREAD;

    // Get absolute maximum values among the local 8 values
    auto localMax = cuda_abs(vec[0]);

    #pragma unroll
    for (int i = 1; i < CVT_ELTS_PER_THREAD / 2; i++) {
        localMax = cuda_max(localMax, cuda_abs(vec[i]));
    }

    // Get the absolute maximum among all 16 values (two threads for 16)
    localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
    if constexpr (CVT_NUM_THREADS_PER_SF == 4) {
        localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
    }

    // Get the final absolute maximum value
    float vecMax;
    if constexpr (std::is_same_v<InType, __nv_bfloat162>) {
        auto max_single = __bfloat162float(localMax.x) > __bfloat162float(localMax.y) ? localMax.x : localMax.y;
        vecMax = __bfloat162float(max_single);
    } else {
        vecMax = fmaxf(__half2float(localMax.x), __half2float(localMax.y));
    }

    // Calculate block scale factor (FP8 E4M3)
    uint8_t fp8_scale_val = 0;
    float output_scale = 0.0f;

    // Get the SF (max value of the vector / max value of e2m1)
    // maximum value of e2m1 = 6.0
    auto sf_value = reciprocal_approximate_ftz(global_scale) * (vecMax * reciprocal_approximate_ftz(6.0f));

#if HAS_FP8_SUPPORT
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(sf_value);
    fp8_scale_val = tmp.__x;
    sf_value = static_cast<float>(tmp);
#else
    // Fallback: clamp to uint8 range
    fp8_scale_val = static_cast<uint8_t>(fminf(fmaxf(sf_value, 0.0f), 255.0f));
    sf_value = static_cast<float>(fp8_scale_val);
#endif

    // Get the output scale
    output_scale = vecMax != 0 ? reciprocal_approximate_ftz(sf_value * global_scale) : 0.0f;

    // Write block scale
    if (block_scale_out) {
        *block_scale_out = fp8_scale_val;
    }

    // Convert the input to float and apply scaling
    float2 fp2_vals[CVT_ELTS_PER_THREAD / 2];

    #pragma unroll
    for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
        if constexpr (std::is_same_v<InType, __nv_bfloat162>) {
            fp2_vals[i] = __bfloat1622float2(vec[i]);
        } else {
            fp2_vals[i] = __half22float2(vec[i]);
        }
        fp2_vals[i].x *= output_scale;
        fp2_vals[i].y *= output_scale;
    }

    // Convert to e2m1 values (FP4)
    uint32_t e2m1_vec = fp32_vec_to_e2m1(fp2_vals);

    return e2m1_vec;
}

// Quantization kernel for BF16 to NVFP4
template<int BLOCK_SIZE = 128, int ELTS_PER_THREAD = 16>
__global__ void nvfp4_quant_from_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ global_scale_ptr,
    uint8_t* __restrict__ fp4_output,
    uint8_t* __restrict__ block_scales,
    const int M,
    const int K
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= M) return;

    // Load global_scale from device memory once per block
    __shared__ float global_scale;

    if (tid == 0) {
        global_scale = *global_scale_ptr;
    }
    __syncthreads();

    constexpr int CVT_ELTS_PER_THREAD = 8;
    constexpr int PACKED_PER_THREAD = CVT_ELTS_PER_THREAD / 2;
    const int elts_per_block = BLOCK_SIZE * CVT_ELTS_PER_THREAD;

    const __nv_bfloat16* row_input = input + row * K;
    uint8_t* row_fp4 = fp4_output + row * (K / 2);
    uint8_t* row_scales = block_scales + row * (K / 16);

    for (int base_col = 0; base_col < K; base_col += elts_per_block) {
        const int col_start = base_col + tid * CVT_ELTS_PER_THREAD;

        if (col_start >= K) break;

        // Load 8 BF16 values as 4 x BF16x2
        __nv_bfloat162 vec[4];

        #pragma unroll
        for (int i = 0; i < PACKED_PER_THREAD; ++i) {
            const int col = col_start + i * 2;
            if (col + 1 < K) {
                vec[i] = *reinterpret_cast<const __nv_bfloat162*>(&row_input[col]);
            } else if (col < K) {
                vec[i].x = row_input[col];
                vec[i].y = __float2bfloat16(0.0f);
            } else {
                vec[i] = __float2bfloat162_rn(0.0f);
            }
        }

        // Quantize to E2M1 with block scaling
        const int block_idx = col_start / 16;
        uint8_t* scale_out = (tid % 2 == 0) ? &row_scales[block_idx] : nullptr;

        uint32_t e2m1_vals = quantize_fp16_to_e2m1_with_scaling(vec, global_scale, scale_out);

        // Pack into output (4 bytes = 8 FP4 values)
        const int packed_idx = col_start / 2;
        if (packed_idx + 3 < K / 2) {
            *reinterpret_cast<uint32_t*>(&row_fp4[packed_idx]) = e2m1_vals;
        } else {
            // Handle boundary case
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&e2m1_vals);
            for (int i = 0; i < 4 && packed_idx + i < K / 2; ++i) {
                row_fp4[packed_idx + i] = bytes[i];
            }
        }
    }
}

// Quantization kernel for FP16 to NVFP4
template<int BLOCK_SIZE = 128, int ELTS_PER_THREAD = 16>
__global__ void nvfp4_quant_from_fp16_kernel(
    const half* __restrict__ input,
    const float* __restrict__ global_scale_ptr,
    uint8_t* __restrict__ fp4_output,
    uint8_t* __restrict__ block_scales,
    const int M,
    const int K
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= M) return;

    // Load global_scale from device memory once per block
    __shared__ float global_scale;

    if (tid == 0) {
        global_scale = *global_scale_ptr;
    }
    __syncthreads();

    constexpr int CVT_ELTS_PER_THREAD = 8;
    constexpr int PACKED_PER_THREAD = CVT_ELTS_PER_THREAD / 2;
    const int elts_per_block = BLOCK_SIZE * CVT_ELTS_PER_THREAD;

    const half* row_input = input + row * K;
    uint8_t* row_fp4 = fp4_output + row * (K / 2);
    uint8_t* row_scales = block_scales + row * (K / 16);

    for (int base_col = 0; base_col < K; base_col += elts_per_block) {
        const int col_start = base_col + tid * CVT_ELTS_PER_THREAD;

        if (col_start >= K) break;

        // Load 8 FP16 values as 4 x FP16x2
        half2 vec[4];

        #pragma unroll
        for (int i = 0; i < PACKED_PER_THREAD; ++i) {
            const int col = col_start + i * 2;
            if (col + 1 < K) {
                vec[i] = *reinterpret_cast<const half2*>(&row_input[col]);
            } else if (col < K) {
                vec[i].x = row_input[col];
                vec[i].y = __float2half(0.0f);
            } else {
                vec[i] = __float2half2_rn(0.0f);
            }
        }

        // Quantize to E2M1 with block scaling
        const int block_idx = col_start / 16;
        uint8_t* scale_out = (tid % 2 == 0) ? &row_scales[block_idx] : nullptr;

        uint32_t e2m1_vals = quantize_fp16_to_e2m1_with_scaling(vec, global_scale, scale_out);

        // Pack into output (4 bytes = 8 FP4 values)
        const int packed_idx = col_start / 2;
        if (packed_idx + 3 < K / 2) {
            *reinterpret_cast<uint32_t*>(&row_fp4[packed_idx]) = e2m1_vals;
        } else {
            // Handle boundary case
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&e2m1_vals);
            for (int i = 0; i < 4 && packed_idx + i < K / 2; ++i) {
                row_fp4[packed_idx + i] = bytes[i];
            }
        }
    }
}

std::vector<torch::Tensor> nvfp4_quant_from_bf16_cuda(
    torch::Tensor input,
    torch::Tensor global_scale
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(global_scale.is_cuda(), "global_scale must be CUDA tensor");
    TORCH_CHECK(global_scale.dtype() == torch::kFloat32, "global_scale must be float32");

    const int M = input.size(0);
    const int K = input.size(1);

    TORCH_CHECK(K % 16 == 0, "K dimension must be divisible by 16");

    auto fp4_output = torch::empty({M, K / 2}, torch::TensorOptions()
        .dtype(torch::kUInt8).device(input.device()));

    auto block_scales = torch::empty({M, K / 16}, torch::TensorOptions()
        .dtype(torch::kUInt8).device(input.device()));

    constexpr int BLOCK_SIZE = 128;
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_quant_from_bf16_kernel<BLOCK_SIZE, 16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        global_scale.data_ptr<float>(),
        fp4_output.data_ptr<uint8_t>(),
        block_scales.data_ptr<uint8_t>(),
        M, K
    );

    return {fp4_output, block_scales};
}

std::vector<torch::Tensor> nvfp4_quant_from_fp16_cuda(
    torch::Tensor input,
    torch::Tensor global_scale
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input must be float16");
    TORCH_CHECK(global_scale.is_cuda(), "global_scale must be CUDA tensor");
    TORCH_CHECK(global_scale.dtype() == torch::kFloat32, "global_scale must be float32");

    const int M = input.size(0);
    const int K = input.size(1);

    TORCH_CHECK(K % 16 == 0, "K dimension must be divisible by 16");

    auto fp4_output = torch::empty({M, K / 2}, torch::TensorOptions()
        .dtype(torch::kUInt8).device(input.device()));

    auto block_scales = torch::empty({M, K / 16}, torch::TensorOptions()
        .dtype(torch::kUInt8).device(input.device()));

    constexpr int BLOCK_SIZE = 128;
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_quant_from_fp16_kernel<BLOCK_SIZE, 16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        global_scale.data_ptr<float>(),
        fp4_output.data_ptr<uint8_t>(),
        block_scales.data_ptr<uint8_t>(),
        M, K
    );

    return {fp4_output, block_scales};
}
"""

            CPP_SOURCE = r"""
#include <torch/extension.h>

std::vector<torch::Tensor> nvfp4_quant_from_bf16_cuda(
    torch::Tensor input,
    torch::Tensor global_scale
);

std::vector<torch::Tensor> nvfp4_quant_from_fp16_cuda(
    torch::Tensor input,
    torch::Tensor global_scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_quant_from_bf16", &nvfp4_quant_from_bf16_cuda, "NVFP4 quantize from BF16 (SM100)");
    m.def("nvfp4_quant_from_fp16", &nvfp4_quant_from_fp16_cuda, "NVFP4 quantize from FP16 (SM100)");
}
"""

            NVFP4QuantizeUtil._nvfp4_quant_sm100_module = load_inline(
                name="nvfp4_quant_sm100",
                cpp_sources=[CPP_SOURCE],
                cuda_sources=[CUDA_SOURCE],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-DENABLE_BF16",
                    "--expt-relaxed-constexpr",
                    "-gencode=arch=compute_100a,code=sm_100a",
                ],
                verbose=False,
                with_cuda=True,
            )

        return NVFP4QuantizeUtil._nvfp4_quant_sm100_module
