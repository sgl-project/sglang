# Copyright 2023-2025 SGLang Team
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

from typing import Optional

import torch

from sglang.srt.utils import (
    direct_register_custom_op,
    is_flashinfer_rmsnorm_quant_kernels_available,
)


def register_flashinfer_fused_ops():
    if is_flashinfer_rmsnorm_quant_kernels_available():
        import flashinfer

        def _flashinfer_rms_norm_quant(
            out: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
            eps: float = 1e-6,
            enable_pdl: Optional[bool] = None,
        ) -> torch.Tensor:
            return flashinfer.norm.rmsnorm_quant(
                out, input, weight, scale, eps, enable_pdl
            )

        def _flashinfer_rms_norm_quant_fake(
            out: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
            eps: float,
            enable_pdl: Optional[bool],
        ) -> None:
            pass

        direct_register_custom_op(
            op_name="flashinfer_rmsnorm_quant",
            op_func=_flashinfer_rms_norm_quant,
            mutates_args=["out"],
            fake_impl=_flashinfer_rms_norm_quant_fake,
        )

        def _flashinfer_fused_add_rmsnorm_quant(
            out: torch.Tensor,
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
            eps: float = 1e-6,
            enable_pdl: Optional[bool] = None,
        ) -> torch.Tensor:
            return flashinfer.norm.fused_add_rmsnorm_quant(
                out, input, residual, weight, scale, eps, enable_pdl
            )

        def _flashinfer_fused_add_rmsnorm_quant_fake(
            out: torch.Tensor,
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
            eps: float = 1e-6,
            enable_pdl: Optional[bool] = None,
        ) -> None:
            pass

        direct_register_custom_op(
            op_name="flashinfer_fused_add_rmsnorm_quant",
            op_func=_flashinfer_fused_add_rmsnorm_quant,
            mutates_args=["out", "residual"],
            fake_impl=_flashinfer_fused_add_rmsnorm_quant_fake,
        )
