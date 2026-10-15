# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepare small derived consumer values without modifying live model state.

Weight and block-scale permutations belong to byte bindings. These updates only
handle NVFP4's complete scalar calibration values, never decoded XOR masks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace

import torch


@dataclass(frozen=True)
class PreparedCopy:
    destination: torch.Tensor
    source: torch.Tensor


@dataclass(frozen=True)
class DerivedUpdate:
    """An update whose inputs can include several independently changed scales.

    ``overrides`` contains complete tensors keyed by ``id(original_source)``.
    The caller assembles any partial canonical updates before preparation and
    applies the returned copies only after successful DE/apply completion.
    """

    sources: tuple[torch.Tensor, ...]
    destinations: tuple[torch.Tensor, ...]
    compute: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]
    checks: Callable | None = None

    def prepare(self, overrides: Mapping[int, torch.Tensor]):
        result, checks = self._prepare(overrides)
        _validate_checks(checks)
        return result

    def _prepare(self, overrides):
        # The batch preparer collects device predicates across layers. Public
        # prepare() still validates before returning any prepared copies.
        values = tuple(overrides.get(id(source), source) for source in self.sources)
        for original, value in zip(self.sources, values):
            _same_geometry(original, value, "source")
        with torch.no_grad():
            outputs = tuple(self.compute(values))
            if len(outputs) != len(self.destinations):
                raise ValueError("GPU delta derived output count changed")
            result, snapshots = [], {}
            for destination, value in zip(self.destinations, outputs):
                _same_geometry(destination, value, "destination")
                # Retain complete target values independently of live buffers;
                # later source copies must not alter a prepared update.
                if id(value) not in snapshots:
                    snapshots[id(value)] = value.detach().clone()
                result.append(PreparedCopy(destination, snapshots[id(value)]))
            checks = self.checks(outputs) if self.checks is not None else ()
        return tuple(result), checks


def _validate_checks(checks):
    if not checks:
        return
    # One small transfer/synchronization for the whole prepared update. Keep
    # individual results so failures retain their original useful diagnostics.
    passed = torch.stack([predicate for predicate, _ in checks]).cpu().tolist()
    for ok, (_, message) in zip(passed, checks):
        if not ok:
            raise ValueError(message)


def _same_geometry(original, value, role):
    if (
        not isinstance(value, torch.Tensor)
        or original.shape != value.shape
        or original.dtype != value.dtype
        or original.device != value.device
    ):
        raise ValueError(f"GPU delta derived {role} geometry changed")


def _storage_key(tensor):
    return tensor.device, tensor.untyped_storage().data_ptr()


def _element_offsets(tensor):
    return [
        tensor.storage_offset()
        + sum(i * stride for i, stride in zip(index, tensor.stride()))
        for index in product(*(range(size) for size in tensor.shape))
    ]


@dataclass(frozen=True)
class _SourcePlan:
    source: torch.Tensor
    targets: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _UpdatePlan:
    update: DerivedUpdate
    sources: tuple[_SourcePlan, ...]


def plan_updates(updates, targets):
    """Validate and cache scalar patch geometry once at admission."""
    by_storage = {}
    for target in targets:
        by_storage.setdefault(_storage_key(target), []).append(target)
    plans = []
    for update in updates:
        sources = []
        for original in update.sources:
            patches = by_storage.get(_storage_key(original), ())
            if not patches:
                continue
            if not original.is_contiguous():
                raise ValueError("GPU delta derived source must be contiguous")
            start, stop = (
                original.storage_offset(),
                original.storage_offset() + original.numel(),
            )
            written = set()
            for target in patches:
                if target.dtype != original.dtype:
                    raise ValueError("GPU delta scalar view dtype mismatch")
                offsets = set(_element_offsets(target))
                if any(offset < start or offset >= stop for offset in offsets):
                    raise ValueError("GPU delta scalar view is outside its source")
                if len(offsets) != target.numel() or offsets & written:
                    raise ValueError("GPU delta scalar updates overlap")
                written.update(offsets)
            sources.append(_SourcePlan(original, tuple(patches)))
        plans.append(_UpdatePlan(update, tuple(sources)))
    return tuple(plans)


def prepare_updates(plans, raw_copies):
    """Patch cached scalar views, then validate device predicates once per update."""
    incoming = {
        id(target): value
        for targets, values in raw_copies.values()
        for target, value in zip(targets, values)
    }
    prepared, checks = [], []
    for plan in plans:
        overrides = {}
        for source_plan in plan.sources:
            changed = [
                target for target in source_plan.targets if id(target) in incoming
            ]
            if not changed:
                continue
            original = source_plan.source
            complete = original.detach().clone()
            for target in changed:
                value = incoming[id(target)]
                _same_geometry(target, value, "raw source")
                offset = target.storage_offset() - original.storage_offset()
                complete.as_strided(target.shape, target.stride(), offset).copy_(value)
            overrides[id(original)] = complete
        if overrides:
            copies, predicates = plan.update._prepare(overrides)
            prepared.extend(copies)
            checks.extend(predicates)
    _validate_checks(checks)
    return tuple(prepared)


def _normal_scale_helpers():
    # Keep dependencies lazy: byte-layout planning need not load inference
    # kernels merely to describe an update.
    from sglang.srt.layers.quantization.modelopt_quant import (
        _compute_gemm1_alphas,
        _input_scale_to_local_experts,
    )

    return _compute_gemm1_alphas, _input_scale_to_local_experts


def _cutedsl_scale_helper():
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
        resolve_cutedsl_standard_scales,
    )

    return resolve_cutedsl_standard_scales


def _trtllm_scale_helper():
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import _compute_g1_scale_c

    return _compute_g1_scale_c


def nvfp4_derived_update(layer, backend):
    """Describe the selected FlashInfer consumer's small NVFP4 scale updates.

    This mirrors the ordinary loader's scale policy and reuses its pure helpers.
    It neither invokes weight postprocessing nor replaces captured tensors.
    MegaMOE's input normalization is a captured scalar, so a change to that
    scalar is rejected during preparation, before any live weight is modified.
    """
    backend = backend.removeprefix("flashinfer_")
    if backend == "trtllm_routed":
        backend = "trtllm"
    if backend not in {"trtllm", "cutedsl", "megamoe"}:
        raise ValueError(f"unsupported NVFP4 derived consumer: {backend}")
    cute = backend == "cutedsl"
    mega = backend == "megamoe"
    wrapper = layer._cutedsl_wrapper if cute else None
    if cute and (wrapper is None or wrapper.quant_mode not in {"w4a4", "w4a16"}):
        raise ValueError("CuTe DSL wrapper must be warmed before GPU delta admission")
    w4a16 = cute and wrapper.quant_mode == "w4a16"
    per_token = bool(layer.quant_config.use_per_token_activation)
    neutral_input = w4a16 or per_token
    config = layer.moe_runner_config
    gated, activation = config.is_gated, config.activation
    local_experts, experts, ep_rank = (
        layer.num_local_experts,
        layer.num_experts,
        layer.moe_ep_rank,
    )
    scalar_input13 = layer.w13_input_scale_quant.ndim == 0
    source_names = ["w13_weight_scale_2", "w2_weight_scale_2"]
    if not neutral_input:
        source_names.extend(("w13_input_scale", "w2_input_scale"))
    sources = tuple(getattr(layer, name) for name in source_names)
    destinations = [layer.g1_alphas, layer.g1_alphas_up, layer.g2_alphas]

    # W4A16 already has neutral activation constants. Retain its direct-copy
    # refresh rather than recomputing those constants every update.
    if not w4a16:
        destinations.extend((layer.w13_input_scale_quant, layer.w2_input_scale_quant))
    if backend == "trtllm":
        destinations.append(layer.g1_scale_c)
    extra_names = []
    if backend == "trtllm" and gated:
        if getattr(layer, "gemm1_clamp_limit", None) is not None:
            extra_names.append("gemm1_clamp_limit")
        if getattr(layer, "gemm1_beta", None) is not None:
            extra_names.append("gemm1_beta")
        destinations.extend(getattr(layer, name) for name in extra_names)
    if cute:
        scales = layer._cutedsl_scales
        input_scale = layer._cutedsl_input_scale
        if not isinstance(scales, tuple) or len(scales) != 3:
            raise ValueError("CuTe DSL scale metadata must be warmed before admission")
        destinations.extend((scales[0], scales[2]) if w4a16 else (*scales, input_scale))
    mega_scalar = None
    if mega:
        if per_token:
            raise ValueError("FlashInfer MegaMOE does not support per-token NVFP4")
        consumer = layer._flashinfer_megamoe_layer
        cached = layer._flashinfer_megamoe_input_norm_const
        if consumer is None or cached is None:
            raise ValueError("FlashInfer MegaMOE must be warmed before delta admission")
        mega_scalar = float(consumer._megakernel_config.input_norm_const)
        if float(cached) != mega_scalar:
            raise ValueError("FlashInfer MegaMOE input normalization cache disagrees")
    for tensor in (*sources, *destinations):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError("NVFP4 derived calibration requires FP32 scalar tensors")

    # Resolve pure inference helpers and configuration once at admission.
    if not w4a16:
        gemm1_alphas, local_input = _normal_scale_helpers()
    if cute and not w4a16:
        cutedsl_values = _cutedsl_scale_helper()
        fixed_input_scale = layer._cutedsl_input_scale
    if backend == "trtllm":
        trtllm_scale_c = _trtllm_scale_helper()
    clamp = config.gemm1_clamp_limit or config.swiglu_limit
    if "gemm1_clamp_limit" in extra_names and (clamp is None or activation == "situ"):
        raise ValueError("unsupported TRTLLM activation clamp metadata")
    situ_beta = config.gemm1_clamp_limit
    neutral13, neutral2 = layer.w13_input_scale_quant, layer.w2_input_scale_quant

    def compute(values):
        weight13, weight2 = values[:2]
        if w4a16:
            if gated and weight13.ndim == 2 and weight13.shape[1] >= 2:
                gate, up = weight13[:, 0], weight13[:, 1]
            else:
                gate = up = weight13.reshape(weight13.shape[0])
            return gate, up, weight2, gate, weight2

        shadow = SimpleNamespace(
            num_local_experts=local_experts,
            num_experts=experts,
            moe_ep_rank=ep_rank,
            w13_weight_scale_2=weight13,
            w2_weight_scale_2=weight2,
        )
        if neutral_input:
            # Per-token consumers already have neutral activation constants.
            # Reuse those captured tensors rather than allocating ones and
            # recomputing their reciprocals for every update.
            if gated and weight13.ndim == 2 and weight13.shape[1] >= 2:
                shadow.g1_alphas, shadow.g1_alphas_up = weight13[:, 0], weight13[:, 1]
            else:
                shadow.g1_alphas = shadow.g1_alphas_up = weight13.reshape(
                    weight13.shape[0]
                )
            shadow.g2_alphas = weight2
            shadow.w13_input_scale_quant = neutral13
            shadow.w2_input_scale_quant = neutral2
        else:
            raw13, raw2 = values[2:]
            input13 = raw13.max().to(torch.float32)
            if cute or mega:
                input2 = local_input(raw2, local_experts, experts, ep_rank)
                if cute:
                    input13 = input13.expand(local_experts)
                    if scalar_input13:
                        input13 = input13[0]
            else:
                input2 = raw2.max().to(torch.float32)
            shadow.g1_alphas, shadow.g1_alphas_up = gemm1_alphas(
                weight13, input13, gated
            )
            shadow.g2_alphas = (input2 * weight2).to(torch.float32)
            shadow.w13_input_scale_quant = (1 / input13).to(torch.float32)
            shadow.w2_input_scale_quant = (1 / input2).to(torch.float32)
        outputs = [
            shadow.g1_alphas,
            shadow.g1_alphas_up,
            shadow.g2_alphas,
            shadow.w13_input_scale_quant,
            shadow.w2_input_scale_quant,
        ]
        if backend == "trtllm":
            outputs.append(
                trtllm_scale_c(
                    shadow.w2_input_scale_quant,
                    shadow.g1_alphas,
                    shadow.g1_alphas_up,
                    gated,
                    activation,
                )
            )
        for name in extra_names:
            if name == "gemm1_clamp_limit":
                outputs.append((clamp / shadow.g1_alphas).to(torch.float32))
            elif activation == "situ":
                outputs.append(torch.full_like(shadow.g1_alphas, situ_beta))
            else:
                outputs.append((1 / shadow.g1_alphas).to(torch.float32))
        if cute:
            w1, fc2_input, w2, input_scale = cutedsl_values(shadow)
            if per_token:
                input_scale = fixed_input_scale
            outputs.extend((w1, fc2_input, w2, input_scale))
        return tuple(outputs)

    def mega_checks(outputs):
        checks = [
            (
                # Match the former Python-float comparison exactly, including a
                # captured scalar that is not representable as an FP32 value.
                outputs[3].max().to(torch.float64) == mega_scalar,
                "FlashInfer MegaMOE input normalization changed; CUDA graph recapture is required",
            )
        ]
        if gated:
            checks.append(
                (
                    torch.isclose(outputs[0], outputs[1]).all(),
                    "FlashInfer MegaMOE requires matching gate/up FC1 alphas",
                )
            )
        return tuple(checks)

    return DerivedUpdate(
        sources, tuple(destinations), compute, mega_checks if mega else None
    )
