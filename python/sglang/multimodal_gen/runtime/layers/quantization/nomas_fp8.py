# SPDX-License-Identifier: Apache-2.0
# Adapted from Tencent-Hunyuan/Hy-Nomas PixelUMM cutlass_fp8.py.
"""Serialized PixelUMM FP8 weights and official 128x128 activation scaling."""

import torch
import torch.nn.functional as F
from torch import nn


def quantize_nomas_fp8_activation(x):
    """Match the official SM89 backend, including a partial final token block."""
    m, k = x.shape
    blocks = F.pad(x, (0, 0, 0, (-m) % 128)).float().reshape(-1, 128, k // 128, 128)
    scales = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-4) / 448.0
    quantized = (blocks / scales[:, None, :, None]).to(torch.float8_e4m3fn)
    return (
        quantized.reshape(-1, k)[:m].contiguous(),
        scales.repeat_interleave(128, dim=0)[:m].contiguous(),
    )


class NomasFP8Linear(nn.Module):
    """Inference linear using existing Triton FP8 GEMM, without a BF16 cache.

    Activation blocks match the upstream SM89 path. GEMM scaling/accumulation
    order differs from CUTLASS, so decoded outputs are not bitwise identical.
    """

    def __init__(self, in_features, out_features, *, device=None):
        super().__init__()
        if in_features % 128 or out_features % 128:
            raise ValueError("PixelUMM FP8 dimensions must be divisible by 128")
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer(
            "fp8_weight",
            torch.empty(
                out_features, in_features, dtype=torch.float8_e4m3fn, device=device
            ),
        )
        self.register_buffer(
            "fp8_scale",
            torch.empty(
                out_features // 128,
                in_features // 128,
                dtype=torch.float32,
                device=device,
            ),
        )

    def _apply(self, fn, recurse=True):
        # Move raw bytes through fn to preserve non_blocking and original dtypes.
        for name, tensor in self._buffers.items():
            self._buffers[name] = fn(tensor.view(torch.uint8)).view(tensor.dtype)
        return self

    def forward(self, x):
        from sglang.kernels import KernelBackend, get_kernel

        if self.training or torch.is_grad_enabled():
            raise RuntimeError("PixelUMM FP8 requires eval and no_grad inference")
        if x.shape[-1] != self.in_features:
            raise ValueError("PixelUMM FP8 input dimension does not match the weight")
        if not x.is_cuda or torch.cuda.get_device_capability(x.device) < (8, 9):
            raise RuntimeError("PixelUMM FP8 requires a CUDA GPU with SM89 or newer")
        if x.dtype != torch.bfloat16:
            if (
                not torch.is_autocast_enabled("cuda")
                or torch.get_autocast_dtype("cuda") != torch.bfloat16
            ):
                raise TypeError(
                    "PixelUMM FP8 requires BF16 input or BF16 CUDA autocast"
                )
            x = x.to(torch.bfloat16)
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], self.out_features))
        quantized, scales = quantize_nomas_fp8_activation(
            x.reshape(-1, self.in_features)
        )
        output = get_kernel("gemm.w8a8_block_fp8_matmul_triton", KernelBackend.TRITON)(
            quantized,
            self.fp8_weight,
            scales,
            self.fp8_scale,
            [128, 128],
            output_dtype=torch.bfloat16,
        )
        return output.reshape(*x.shape[:-1], self.out_features)
