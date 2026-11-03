# SPDX-License-Identifier: Apache-2.0
"""Online W8A8 FP8 for plain ``nn.Linear`` modules of diffusers-native components.

``OnlineFP8Linear.from_linear`` replaces a prepared fp16/bf16 ``nn.Linear`` at
load with per-output-channel e4m3 weights (the recipe of ``Fp8LinearMethod``'s
online path), dynamic per-token activation scales and the shared
``apply_fp8_linear`` dispatch. ``forward_quantized`` takes rows a fused producer
already quantized. Inference only.
"""

from __future__ import annotations

from math import prod

import torch
from torch import nn

from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
)

FP8_DTYPE = torch.float8_e4m3fn


class OnlineFP8Linear(nn.Module):
    """Static per-channel FP8 weights, dynamic per-token FP8 activations."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if weight.dtype != FP8_DTYPE or weight.dim() != 2:
            raise ValueError("OnlineFP8Linear takes a 2-D e4m3 weight")
        self.out_features, self.in_features = weight.shape
        self.compute_dtype = compute_dtype
        self.weight = nn.Parameter(weight, requires_grad=False)
        # [1, out]: the per-channel layout apply_fp8_linear's GEMMs take.
        self.register_buffer(
            "weight_scale", weight_scale.reshape(1, -1).to(torch.float32).contiguous()
        )
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self._cutlass = cutlass_fp8_supported()

    @classmethod
    def from_linear(cls, source: nn.Linear) -> OnlineFP8Linear:
        compute_dtype = source.weight.dtype
        if compute_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("online FP8 quantizes an fp16 or bf16 linear")
        if not source.weight.is_cuda:
            raise ValueError("online FP8 quantizes a device-resident linear")
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            weight = source.weight.detach().contiguous()
            qweight, scale = per_token_group_quant_fp8(weight, weight.shape[-1])
            bias = None if source.bias is None else source.bias.detach()
        module = cls(qweight, scale, bias, compute_dtype=compute_dtype)
        module.train(source.training)
        return module

    def _apply(self, fn, recurse=True):
        # Whole-component dtype scopes must not expand the FP8 payload or cast
        # its scale, so moves are handled here and dtypes are pinned.
        device = fn(
            torch.empty(0, dtype=self.weight.dtype, device=self.weight.device)
        ).device
        if device.type not in ("cuda", "cpu"):
            raise ValueError("OnlineFP8Linear supports CPU/CUDA moves only")
        self.weight = nn.Parameter(
            self.weight.detach().to(device=device, dtype=FP8_DTYPE),
            requires_grad=False,
        )
        self.weight_scale = self.weight_scale.to(device=device, dtype=torch.float32)
        if self.bias is not None:
            self.bias = nn.Parameter(
                self.bias.detach().to(device=device, dtype=self.compute_dtype),
                requires_grad=False,
            )
        return self

    def _check_inference(self) -> None:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("OnlineFP8Linear is inference-only")

    def forward_quantized(self, values, scales, shape):
        """GEMM on rows a producer already quantized (per-token fp32 scales)."""
        self._check_inference()
        if (
            len(shape) < 2
            or shape[-1] != self.in_features
            or values.shape != (prod(shape[:-1]), self.in_features)
            or values.dtype != FP8_DTYPE
            or not values.is_contiguous()
            or scales.shape != (values.shape[0], 1)
            or scales.dtype != torch.float32
            or not scales.is_contiguous()
            or not values.is_cuda
            or values.device != self.weight.device
            or scales.device != values.device
        ):
            raise ValueError("unsupported pre-quantized FP8 activation")
        with torch.autocast("cuda", enabled=False):
            out = apply_fp8_linear(
                values,
                self.weight.t(),
                self.weight_scale,
                input_scale=scales,
                bias=self.bias,
                cutlass_fp8_supported=self._cutlass,
                use_per_token_if_dynamic=True,
                pre_quant_output_dtype=self.compute_dtype,
            )
        return out.reshape(*shape[:-1], self.out_features)

    def forward(self, value):
        self._check_inference()
        if (
            not value.is_cuda
            or value.device != self.weight.device
            or value.shape[-1] != self.in_features
        ):
            raise ValueError("unsupported OnlineFP8Linear activation")
        with torch.autocast("cuda", enabled=False):
            # Round the way the original Linear's autocast did before quantizing.
            rows = value.to(self.compute_dtype).reshape(-1, self.in_features)
            out = apply_fp8_linear(
                rows.contiguous(),
                self.weight.t(),
                self.weight_scale,
                bias=self.bias,
                cutlass_fp8_supported=self._cutlass,
                use_per_token_if_dynamic=True,
            )
        return out.reshape(*value.shape[:-1], self.out_features)
