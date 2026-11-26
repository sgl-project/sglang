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

The checkpoint owns the step schedule
(``quantization_config.runtime.diffusion_step_policy`` in the component's
``config.json``, schema shared with vLLM-Omni): no policy means no mixed
precision. Explicitly-set env vars act as a manual override for experiments.
The precision is selected once per denoising step (before any
transformer call for that step), so conditional and unconditional CFG branches
of the same step always share one selection. The reasoner (UND) path uses a
static per-request mode from the policy instead of the step schedule.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping

import msgspec
import torch
import torch.nn as nn
import torch.nn.functional as F

import sglang.multimodal_gen.envs as envs
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

REASONER_PATH = "reasoner"
GENERATION_PATH = "generation"

# Checkpoint schema shared with vLLM-Omni; unknown or missing fields fail
# closed so a policy the runtime cannot honor never degrades silently.
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "type",
        "index_space",
        "scope",
        "default_mode",
        "first_steps",
        "last_steps",
        "overlap",
        "reasoner",
    }
)
_STEP_RANGE_FIELDS = frozenset({"count", "mode"})
_POLICY_COMPONENT = "transformer"


class StepPolicy(msgspec.Struct, frozen=True, kw_only=True):
    first_steps: int
    last_steps: int
    reasoner_a16: bool = True


class StepMixedPrecisionController:
    """Holds the precision selected for the current denoising step."""

    def __init__(
        self, first_steps: int, last_steps: int, reasoner_a16: bool = True
    ) -> None:
        if first_steps < 0 or last_steps < 0:
            raise ValueError(
                f"first_steps/last_steps must be non-negative, got "
                f"{first_steps}/{last_steps}"
            )
        self.first_steps = first_steps
        self.last_steps = last_steps
        self.reasoner_a16 = reasoner_a16
        self.high_precision = False

    def use_high_precision(self, path: str) -> bool:
        if path == REASONER_PATH:
            return self.reasoner_a16
        return self.high_precision

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


def read_checkpoint_step_policy(
    quantization_config: Mapping | None,
) -> StepPolicy | None:
    """Parse ``runtime.diffusion_step_policy`` from a checkpoint quant config.

    Missing metadata returns None (ordinary checkpoint behavior). Metadata
    that is present but malformed or unsupported raises, matching vLLM-Omni's
    fail-closed contract for this schema.
    """
    if not isinstance(quantization_config, Mapping):
        return None
    if "runtime" not in quantization_config:
        return None
    runtime = quantization_config["runtime"]
    if not isinstance(runtime, Mapping):
        raise TypeError("quantization_config.runtime must be a mapping")
    if "diffusion_step_policy" not in runtime:
        return None
    policy = runtime["diffusion_step_policy"]
    if not isinstance(policy, Mapping):
        raise TypeError(
            "quantization_config.runtime.diffusion_step_policy must be a mapping"
        )
    return _parse_step_policy(policy)


def _parse_step_policy(policy: Mapping) -> StepPolicy | None:
    unknown = set(policy) - _POLICY_FIELDS
    if unknown:
        raise ValueError(f"Unknown diffusion_step_policy fields: {sorted(unknown)}")
    missing = _POLICY_FIELDS - set(policy)
    if missing:
        raise ValueError(f"Missing diffusion_step_policy fields: {sorted(missing)}")

    schema_version = policy["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ValueError("diffusion_step_policy.schema_version must be the integer 1")
    if policy["type"] != "first_last_n":
        raise ValueError("diffusion_step_policy.type must be 'first_last_n'")
    if policy["index_space"] != "denoising_loop_iteration":
        raise ValueError(
            "diffusion_step_policy.index_space must be 'denoising_loop_iteration'"
        )
    if policy["default_mode"] != "native":
        raise ValueError("diffusion_step_policy.default_mode must be 'native'")
    if policy["overlap"] != "a16":
        raise ValueError("diffusion_step_policy.overlap must be 'a16'")

    scope = policy["scope"]
    if (
        not isinstance(scope, list)
        or not scope
        or not all(isinstance(item, str) for item in scope)
    ):
        raise TypeError(
            "diffusion_step_policy.scope must be a non-empty list of strings"
        )

    first_steps = _parse_step_range(policy["first_steps"], "first_steps")
    last_steps = _parse_step_range(policy["last_steps"], "last_steps")

    reasoner = policy["reasoner"]
    if reasoner not in ("native", "a16"):
        raise ValueError("diffusion_step_policy.reasoner must be 'native' or 'a16'")

    if _POLICY_COMPONENT not in scope:
        return None
    return StepPolicy(
        first_steps=first_steps,
        last_steps=last_steps,
        reasoner_a16=reasoner == "a16",
    )


def _parse_step_range(value: object, name: str) -> int:
    if not isinstance(value, Mapping):
        raise TypeError(f"diffusion_step_policy.{name} must be a mapping")
    unknown = set(value) - _STEP_RANGE_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown diffusion_step_policy.{name} fields: {sorted(unknown)}"
        )
    missing = _STEP_RANGE_FIELDS - set(value)
    if missing:
        raise ValueError(
            f"Missing diffusion_step_policy.{name} fields: {sorted(missing)}"
        )
    if value["mode"] != "a16":
        raise ValueError(f"diffusion_step_policy.{name}.mode must be 'a16'")
    count = value["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError(
            f"diffusion_step_policy.{name}.count must be a non-negative integer"
        )
    return count


def resolve_step_policy(
    quantization_config: Mapping | None,
) -> tuple[StepPolicy | None, str]:
    """Resolve the effective step policy and a human-readable source label.

    The checkpoint owns the behavior: mixed precision runs only when the
    checkpoint carries a diffusion_step_policy. The enable env var is a
    kill-switch, and explicitly-set FIRST/LAST env vars are a manual
    override — per field on top of a checkpoint policy, or standing alone
    to force-enable on a checkpoint without one.
    """
    if not envs.SGLANG_DIFFUSION_ENABLE_COSMOS3_STEP_MIXED_PRECISION:
        return (
            None,
            "disabled by SGLANG_DIFFUSION_ENABLE_COSMOS3_STEP_MIXED_PRECISION=0",
        )

    checkpoint_policy = read_checkpoint_step_policy(quantization_config)

    overridden = []
    if "SGLANG_DIFFUSION_COSMOS3_STEP_MIXED_PRECISION_FIRST_STEPS" in os.environ:
        overridden.append("first_steps")
    if "SGLANG_DIFFUSION_COSMOS3_STEP_MIXED_PRECISION_LAST_STEPS" in os.environ:
        overridden.append("last_steps")

    if checkpoint_policy is None and not overridden:
        return None, "checkpoint carries no diffusion_step_policy"

    base = checkpoint_policy or StepPolicy(
        first_steps=envs.SGLANG_DIFFUSION_COSMOS3_STEP_MIXED_PRECISION_FIRST_STEPS,
        last_steps=envs.SGLANG_DIFFUSION_COSMOS3_STEP_MIXED_PRECISION_LAST_STEPS,
        reasoner_a16=True,
    )
    if checkpoint_policy is None:
        source = f"env vars ({', '.join(overridden)} set)"
    elif overridden:
        source = f"checkpoint with env override of {', '.join(overridden)}"
    else:
        source = "checkpoint"

    first_steps = base.first_steps
    last_steps = base.last_steps
    if "first_steps" in overridden:
        first_steps = envs.SGLANG_DIFFUSION_COSMOS3_STEP_MIXED_PRECISION_FIRST_STEPS
    if "last_steps" in overridden:
        last_steps = envs.SGLANG_DIFFUSION_COSMOS3_STEP_MIXED_PRECISION_LAST_STEPS

    return (
        StepPolicy(
            first_steps=first_steps,
            last_steps=last_steps,
            reasoner_a16=base.reasoner_a16,
        ),
        source,
    )


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
        path: str = GENERATION_PATH,
    ) -> None:
        self.base_method = base_method
        self.controller = controller
        self.path = path

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
        if self.controller.use_high_precision(self.path):
            return apply_fp8_w8a16_linear(layer, x, bias)
        return self.base_method.apply(layer, x, bias)


def install_step_mixed_precision(
    reasoner_modules: Iterable[nn.Module],
    generation_modules: Iterable[nn.Module],
    controller: StepMixedPrecisionController,
) -> tuple[int, int]:
    """Wrap every ModelOpt FP8 linear for per-path precision dispatch.

    Must run after the loader's ``process_weights_after_loading`` pass so the
    wrapped method only ever dispatches ``apply``. Returns the wrapped counts
    per path; (0, 0) means the model is not a ModelOpt FP8 checkpoint.
    """
    return (
        _wrap_path(reasoner_modules, controller, REASONER_PATH),
        _wrap_path(generation_modules, controller, GENERATION_PATH),
    )


def _wrap_path(
    roots: Iterable[nn.Module],
    controller: StepMixedPrecisionController,
    path: str,
) -> int:
    wrapped = 0
    for root in roots:
        for module in root.modules():
            if not isinstance(module, LinearBase):
                continue
            if not isinstance(module.quant_method, MODELOPT_FP8_LINEAR_METHODS):
                continue
            module.quant_method = StepMixedPrecisionFp8LinearMethod(
                base_method=module.quant_method,
                controller=controller,
                path=path,
            )
            wrapped += 1
    return wrapped
