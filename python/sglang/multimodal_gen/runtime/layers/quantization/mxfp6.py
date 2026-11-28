# SPDX-License-Identifier: Apache-2.0
"""Online MXFP6 for diffusion linears: ``--quantization mxfp6`` on a bf16 checkpoint.

Weights and activations both quantize to AITER's packed E2M3 blob with one E8M0
scale per 32 elements along K, and the product is the A6W6 ASM GEMM, which
computes ``A @ B.T`` in bf16. See :mod:`aiter_packed_mx` for the shared
contract.
"""

import torch

from sglang.multimodal_gen.runtime.layers.quantization.aiter_packed_mx import (
    PackedMXConfig,
    PackedMXLinearMethod,
    aiter_import_error,
)
from sglang.srt.utils.custom_op import direct_register_custom_op

_IMPORT_ERROR = aiter_import_error("aiter.ops.gemm_op_a6w6")

if _IMPORT_ERROR is None:
    from aiter.ops.gemm_op_a6w6 import gemm_a6w6, quant_mxfp6_gemm

    def _mxfp6_quant_gemm(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        x_fp6, x_scale = quant_mxfp6_gemm(x)
        y = gemm_a6w6(
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

    def _mxfp6_quant_gemm_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        return torch.empty(
            (x.shape[0], out_features), dtype=torch.bfloat16, device=x.device
        )

    # Wrapping the activation quant and the gemm in one functional op keeps the
    # layer torch.compile compatible.
    direct_register_custom_op(
        op_name="mxfp6_diffusion_quant_gemm",
        op_func=_mxfp6_quant_gemm,
        mutates_args=[],
        fake_impl=_mxfp6_quant_gemm_fake,
    )


class Mxfp6LinearMethod(PackedMXLinearMethod):
    format_name = "MXFP6"

    def _require_kernels(self) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "aiter MXFP6 kernels not available. Install aiter with MXFP6 "
                f"support. Import failed with: {_IMPORT_ERROR}"
            )

    def _quantize_weight(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return quant_mxfp6_gemm(weight)

    def _packed_gemm(
        self,
        x_2d: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        return torch.ops.sglang.mxfp6_diffusion_quant_gemm(
            x_2d, weight, weight_scale, out_features, in_features
        )


class Mxfp6Config(PackedMXConfig):
    """MXFP6 (E2M3 activations and weights) on ROCm MI350+ (gfx95x)."""

    method_name = "mxfp6"

    def linear_method_cls(self) -> type[PackedMXLinearMethod]:
        return Mxfp6LinearMethod


__all__ = ["Mxfp6Config", "Mxfp6LinearMethod"]
