# SPDX-License-Identifier: Apache-2.0
"""Step-based mixed activation precision for ModelOpt FP8 DiT linears.

A ModelOpt FP8 checkpoint runs every quantized linear as W8A8 (FP8 weights,
statically quantized FP8 activations). The first and last denoising steps are
the most sensitive to activation quantization error, so this module lets a
transformer run those edge steps as W8A16 instead: the same resident FP8
weights are dequantized to the activation dtype per call and fed to a plain
16-bit GEMM, and ``input_scale`` is simply unused. Middle steps keep the
checkpoint's W8A8 scaled-mm path. No second checkpoint and no extra persistent
weight memory are needed.

The precision is selected once per denoising step (before any transformer
call for that step), so conditional and unconditional CFG branches of the same
step always share one selection.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.linear import LinearBase, LinearMethodBase
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8 import (
    ModelOptFp8Config as FlatModelOptFp8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8 import (
    ModelOptFp8LinearMethod as FlatModelOptFp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8Config as HfModelOptFp8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8LinearMethod as HfModelOptFp8LinearMethod,
)

logger = logging.getLogger(__name__)

# Two ModelOpt FP8 static per-tensor implementations exist (flat
# `quant_method=modelopt` exports vs `modelopt_fp8` hf_quant_config ones).
# Both store the post-load weight as a column-major [in, out] FP8 view with a
# scalar-or-channelwise weight_scale, so one W8A16 path serves both.
MODELOPT_FP8_QUANT_CONFIGS = (
    FlatModelOptFp8Config,
    HfModelOptFp8Config,
)
MODELOPT_FP8_LINEAR_METHODS = (
    FlatModelOptFp8LinearMethod,
    HfModelOptFp8LinearMethod,
)


class StepMixedPrecisionController:
    """Holds the precision selected for the current denoising step."""

    def __init__(self, first_steps: int, last_steps: int) -> None:
        if first_steps < 0 or last_steps < 0:
            raise ValueError(
                f"first_steps/last_steps must be non-negative, got "
                f"{first_steps}/{last_steps}"
            )
        self.first_steps = first_steps
        self.last_steps = last_steps
        self.high_precision = False

    def set_step(self, step_index: int, num_steps: int) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if step_index < 0 or step_index >= num_steps:
            raise IndexError(
                f"step_index must be in [0, {num_steps}), got {step_index}"
            )
        # A one-step schedule is typically the engine warmup probe; keep it on
        # the base W8A8 path rather than treating it as all-edge.
        if num_steps == 1:
            self.high_precision = False
            return
        self.high_precision = (
            step_index < self.first_steps or step_index >= num_steps - self.last_steps
        )

    def reset(self) -> None:
        self.high_precision = False


def apply_fp8_w8a16_linear(
    layer: nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """16-bit GEMM against the layer's resident ModelOpt FP8 weight.

    ``ModelOptFp8LinearMethod.process_weights_after_loading`` stores the FP8
    weight as a column-major ``[in, out]`` view; ``.t()`` recovers the
    row-major ``[out, in]`` layout ``F.linear`` wants. ``weight_scale`` is
    either the per-tensor scalar or its channelwise expansion (equal values),
    so both broadcast correctly. ``input_scale`` is intentionally unused.
    """
    weight = layer.weight.t()
    scale = layer.weight_scale.to(x.dtype)
    if scale.numel() > 1:
        scale = scale.view(-1, 1)
    return F.linear(x, weight.to(x.dtype) * scale, bias)


class StepMixedPrecisionFp8LinearMethod(LinearMethodBase):
    """Routes each call to W8A8 (base method) or W8A16 per the controller."""

    def __init__(
        self,
        base_method: LinearMethodBase,
        controller: StepMixedPrecisionController,
    ) -> None:
        self.base_method = base_method
        self.controller = controller

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        self.base_method.create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self.base_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.controller.high_precision:
            return apply_fp8_w8a16_linear(layer, x, bias)
        return self.base_method.apply(layer, x, bias)


def install_step_mixed_precision(
    module_lists: Iterable[nn.Module],
    controller: StepMixedPrecisionController,
) -> int:
    """Wrap every ModelOpt FP8 linear under *module_lists* for step dispatch.

    Must run after the loader's ``process_weights_after_loading`` pass so the
    wrapped method only ever dispatches ``apply``. Returns the number of
    wrapped linears; 0 means the model is not a ModelOpt FP8 checkpoint.
    """
    wrapped = 0
    for root in module_lists:
        for module in root.modules():
            if not isinstance(module, LinearBase):
                continue
            if not isinstance(module.quant_method, MODELOPT_FP8_LINEAR_METHODS):
                continue
            module.quant_method = StepMixedPrecisionFp8LinearMethod(
                base_method=module.quant_method,
                controller=controller,
            )
            wrapped += 1
    return wrapped
