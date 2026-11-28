# Copyright 2026 SGLang Team
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
"""MXFP8 MoE experts on FlashInfer's CuTe-DSL MoE (``quant_mode="mxfp8"``).

The checkpoint's E4M3 expert weights are used as stored; their row-major UE8M0
block-32 scales are swizzled once into the kernel's MMA layout. Activations are
quantized to MXFP8 per forward. The GEMM1 epilogue fuses the activation, which
includes the soft-clamped squared ReLU (FlashInfer's ``ClampedRelu2``) that no
other MXFP8 MoE runner here implements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import Module, Parameter

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

_MXFP8_BLOCK = 32


class Mxfp8CuteDslMoEMethod(Fp8MoEMethod):
    """Serves an MXFP8 (block [1, 32], UE8M0 scales) checkpoint's experts with
    ``--moe-runner-backend flashinfer_cutedsl``."""

    def process_weights_after_loading(self, layer: Module) -> None:
        from flashinfer.cute_dsl.utils import convert_row_major_sf_to_mma_layout

        for name in ("w13", "w2"):
            weight = getattr(layer, f"{name}_weight")
            scale = getattr(layer, f"{name}_weight_scale_inv")
            num_experts, rows, cols = weight.shape
            if cols % _MXFP8_BLOCK != 0:
                raise ValueError(
                    f"MXFP8 {name} input dim {cols} is not a multiple of {_MXFP8_BLOCK}"
                )
            mma = convert_row_major_sf_to_mma_layout(
                scale.data.view(torch.uint8),
                m=rows,
                k=cols,
                num_groups=num_experts,
                sf_vec_size=_MXFP8_BLOCK,
            )
            # The kernel checks the MMA view's exact strides: keep the view.
            setattr(
                layer, f"{name}_weight_scale_inv", Parameter(mma, requires_grad=False)
            )
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def create_moe_runner(
        self, layer: Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        if moe_runner_config.is_gated:
            # Gated experts would need w13 interleaved into the kernel's
            # [up, gate] pairs, which this method does not do.
            raise NotImplementedError(
                "MXFP8 experts on --moe-runner-backend flashinfer_cutedsl support "
                f"non-gated activations only, got {moe_runner_config.activation!r}."
            )
        # Registers the ("none", "flashinfer_cutedsl") fused function.
        import sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl  # noqa: F401

        self.moe_runner_config = moe_runner_config
        self._owns_moe_runner = True
        layer._cutedsl_wrapper = None
        self.runner = MoeRunner(MoeRunnerBackend.FLASHINFER_CUTEDSL, moe_runner_config)

    def get_moe_routing_method_type(self, layer, requested_method):
        return requested_method

    def apply(
        self, layer: Module, dispatch_output: StandardDispatchOutput
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
            CuteDslFp4MoeQuantInfo,
            ensure_cutedsl_wrapper,
        )

        ensure_cutedsl_wrapper(layer, quant_mode="mxfp8")
        w1_alpha, _, w2_alpha = layer._cutedsl_scales
        quant_info = CuteDslFp4MoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_weight_sf=layer.w13_weight_scale_inv,
            w2_weight_sf=layer.w2_weight_scale_inv,
            w1_alpha=w1_alpha,
            w2_alpha=w2_alpha,
            a1_scale=None,
            a2_scale=None,
            wrapper=layer._cutedsl_wrapper,
            quant_mode="mxfp8",
        )
        return self.runner.run(dispatch_output, quant_info)
