# SPDX-License-Identifier: Apache-2.0
"""Online A6W4 for diffusion linears: ``--quantization a6w4`` on a bf16 checkpoint.

Weights quantize to MXFP4 and activations to MXFP6, and AITER's A6W4 ASM GEMM
multiplies the two. That keeps MXFP4's weight footprint -- which is what the
memory and weight-bandwidth savings come from -- while spending the extra bits
on the activations. See :mod:`aiter_packed_mx` for the shared contract.
"""

import torch

from sglang.multimodal_gen.runtime.layers.quantization.aiter_packed_mx import (
    PackedMXConfig,
    PackedMXLinearMethod,
    aiter_import_error,
)
from sglang.srt.utils.custom_op import direct_register_custom_op

_IMPORT_ERROR = aiter_import_error("aiter.ops.gemm_op_a6w4") or aiter_import_error(
    "aiter.ops.gemm_op_a6w6"
)

if _IMPORT_ERROR is None:
    from aiter.ops.gemm_op_a6w4 import gemm_a6w4, quant_mxfp4_gemm
    from aiter.ops.gemm_op_a6w6 import quant_mxfp6_gemm

    def _a6w4_quant_gemm(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        x_fp6, x_scale = quant_mxfp6_gemm(x)
        y = gemm_a6w4(
            x_fp6,
            weight,
            x_scale,
            weight_scale,
            x.shape[0],
            out_features,
            in_features,
            dtype=torch.bfloat16,
        )
        # The gemm returns a slice of its 256-padded output, which is strided
        # when out_features is not a multiple of 256; the op must hand back the
        # dense layout its fake implementation reports.
        return y.contiguous()

    def _a6w4_quant_gemm_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        return torch.empty(
            (x.shape[0], out_features), dtype=torch.bfloat16, device=x.device
        )

    direct_register_custom_op(
        op_name="a6w4_diffusion_quant_gemm",
        op_func=_a6w4_quant_gemm,
        mutates_args=[],
        fake_impl=_a6w4_quant_gemm_fake,
    )


class A6W4LinearMethod(PackedMXLinearMethod):
    format_name = "A6W4"

    def _require_kernels(self) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "aiter A6W4 kernels not available. Install aiter with A6W4 "
                f"support. Import failed with: {_IMPORT_ERROR}"
            )

    def _quantize_weight(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return quant_mxfp4_gemm(weight)

    def _packed_gemm(
        self,
        x_2d: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        return torch.ops.sglang.a6w4_diffusion_quant_gemm(
            x_2d, weight, weight_scale, out_features, in_features
        )


class A6W4Config(PackedMXConfig):
    """MXFP6 activations by MXFP4 weights on ROCm MI350+ (gfx95x)."""

    method_name = "a6w4"

    def linear_method_cls(self) -> type[PackedMXLinearMethod]:
        return A6W4LinearMethod


__all__ = ["A6W4Config", "A6W4LinearMethod"]
