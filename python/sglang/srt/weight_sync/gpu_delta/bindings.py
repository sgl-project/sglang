# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared canonical byte bindings and admitted FlashInfer storage layouts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from functools import cached_property

import torch

from sglang.srt.weight_sync.gpu_delta.flashinfer import (
    DestinationSegment,
    expert_destinations,
    expert_scale_target,
    identify_layout,
)

BACKEND_LAYOUTS = {
    "dense": "unquantized-tp",
    "moe": "flashinfer-native-byte-layouts",
}


_DTYPES = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int32: "I32",
    torch.int64: "I64",
}

_TORCH_DTYPES = {name: dtype for dtype, name in _DTYPES.items()}
_ITEMSIZES = {name: dtype.itemsize for name, dtype in _TORCH_DTYPES.items()}


def _dtype_name(dtype):
    return _DTYPES.get(dtype, str(dtype))


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _full_slices(shape):
    return [[0, size] for size in shape]


@dataclass
class TensorBinding:
    name: str
    dtype: str
    shape: tuple[int, ...]
    slices: list[list[int]]
    storage: tuple[torch.Tensor, ...]
    encoding: str = "xor_bytes"
    destinations: tuple[torch.Tensor, ...] = ()
    segments: tuple[DestinationSegment, ...] = ()
    format: str | None = None
    checkpoint_dtype: str | None = None

    def __post_init__(self):
        # The admitted binding is immutable for the session. Keep dtype and
        # slicing metadata off the paused per-tensor application path.
        self.torch_dtype = _TORCH_DTYPES[self.dtype]
        self.raw_ue8m0 = self.format == "fp8-block128-ue8m0" and self.dtype == "U8"
        if self.raw_ue8m0 and (
            self.encoding != "raw_bytes"
            or any(target.dtype != torch.float32 for target in self.storage)
        ):
            raise ValueError("block128 UE8M0 requires the admitted FP32 API image")
        self._selection = tuple(slice(start, stop) for start, stop in self.slices)
        layer = re.search(r"(?:^|\.)layers\.(\d+)\.", self.name)
        self.layer = int(layer[1]) if layer else None

    @cached_property
    def storage_pointers(self):
        return tuple(tensor.untyped_storage().data_ptr() for tensor in self.storage)

    @cached_property
    def view_id(self):
        return _digest({"name": self.name, "slices": self.slices})

    def describe(self):
        result = {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "encoding": self.encoding,
            "views": [{"id": self.view_id, "slices": self.slices}],
        }
        if self.format is not None:
            result["format"] = self.format
        if self.checkpoint_dtype is not None:
            result["checkpoint_dtype"] = self.checkpoint_dtype
        return result

    def apply_bindings(self):
        """Return byte-layout pieces without adding negotiated tensor views."""
        return self._apply_views

    @cached_property
    def _apply_views(self):
        if not self.segments:
            return (self,)
        return tuple(
            replace(
                self,
                slices=[list(axis) for axis in segment.slices],
                destinations=segment.destinations,
                segments=(),
            )
            for segment in self.segments
        )

    def raw_sources(self, payload, expanded=None):
        """Bind complete values; UE8M0 expansion is batched by the preparer."""
        if self.encoding != "raw_bytes":
            raise ValueError("raw sources require a complete-value binding")
        if self.raw_ue8m0:
            if expanded is None:
                raise ValueError("UE8M0 scales require a prepared FP32 payload")
            value = expanded.reshape(self.shape)[self._selection]
            return tuple(
                (target, value.reshape(target.shape)) for target in self.storage
            )
        value = self.selected_bytes(payload).view(self.torch_dtype)
        return tuple(
            (target, value.reshape(target.shape).to(target.dtype))
            for target in self.storage
        )

    def selected_bytes(self, canonical_bytes):
        value = canonical_bytes.view(self.torch_dtype).reshape(self.shape)
        # Keep column-sharded selections as views of reusable decoded scratch.
        # The decoder fills this storage after preparation; never cache a copy.
        return _byte_view(value[self._selection])


def validate_ue8m0_payload(payload):
    """Validate the complete packed scale prefix on CPU before its single H2D."""
    if payload.device.type != "cpu" or payload.dtype != torch.uint8:
        raise ValueError("UE8M0 admission requires a CPU byte payload")
    if payload.numpy().max(initial=0) == 255:
        raise ValueError("UE8M0 scale payload contains NaN")


def _byte_view(tensor):
    # A strided outer dimension is fine, but dtype reinterpretation requires
    # contiguous elements within each row. Never silently clone a destination.
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.stride(-1) != 1:
        raise ValueError("delta destination requires a contiguous innermost dimension")
    return tensor.detach().view(torch.uint8)


def _direct_binding(name, meta, target, slices=None):
    dtype = _TORCH_DTYPES[meta["dtype"]]
    slices = _full_slices(meta["shape"]) if slices is None else slices
    view_shape = tuple(stop - start for start, stop in slices)
    if target.dtype != dtype or target.numel() != math.prod(view_shape):
        raise ValueError(
            f"canonical/live dtype or shape mismatch for {name}: "
            f"canonical={meta['dtype']}{view_shape}, "
            f"live={_dtype_name(target.dtype)}{tuple(target.shape)}"
        )
    direct = len(meta["shape"]) <= 1
    return TensorBinding(
        name,
        meta["dtype"],
        tuple(meta["shape"]),
        slices,
        (target,),
        encoding="raw_bytes" if direct else "xor_bytes",
        destinations=() if direct else (_byte_view(target),),
        format=meta.get("format"),
        checkpoint_dtype=meta.get("checkpoint_dtype"),
    )


def _normalize_expert_meta(meta, layout, suffix):
    if layout.format != "fp8-block128-ue8m0":
        return
    if suffix == "weight":
        if meta["dtype"] != "F8_E4M3":
            raise ValueError("block128 FP8 requires canonical E4M3 weights")
    elif suffix == "weight_scale_inv":
        if meta["dtype"] == "F32":
            meta["checkpoint_dtype"] = "F32"
            meta["dtype"] = "U8"
        elif meta["dtype"] != "U8":
            raise ValueError("block128 FP8 requires exact UE8M0 scale values")
        meta["encoding"] = "raw_bytes"
    else:
        raise ValueError(f"unclassified block128 FP8 expert tensor: {suffix}")
    meta["format"] = layout.format


def _moe_binding(name, meta, layer, expert, projection, suffix, layout):
    _normalize_expert_meta(meta, layout, suffix)
    if not layout.gated and projection == "gate":
        raise ValueError("non-gated experts have no gate projection")
    stem = "w2" if projection == "down" else "w13"
    half = 0 if projection == "gate" else 1  # scalar metadata stays gate-first
    if suffix == "input_scale" and layout.format == "nvfp4":
        if layout.quant_mode == "w4a16" or layer.quant_config.use_per_token_activation:
            return None
        # Calibration is global even when this expert's weights belong to a
        # different EP rank: the ordinary consumer uses the shared maximum.
        target = getattr(layer, f"{stem}_input_scale")
        target = (
            target[expert, half]
            if projection != "down" and layout.gated
            else target[expert]
        )
        return _direct_binding(name, meta, target)
    local = layer._map_global_expert_id_to_local_expert_id(expert)
    if local < 0:
        return None
    if suffix == "weight_scale_2" and layout.format == "nvfp4":
        target = getattr(layer, f"{stem}_weight_scale_2")
        target = (
            target[local, half]
            if projection != "down" and layout.gated
            else target[local]
        )
        return _direct_binding(name, meta, target)
    if layout.format == "fp8-block128-ue8m0" and suffix == "weight_scale_inv":
        target = expert_scale_target(layer, layout, local, projection, meta["shape"])
        return TensorBinding(
            name,
            meta["dtype"],
            tuple(meta["shape"]),
            _full_slices(meta["shape"]),
            (target,),
            encoding="raw_bytes",
            format=meta["format"],
            checkpoint_dtype=meta.get("checkpoint_dtype"),
        )
    byte_suffix = suffix
    if layout.format == "mxfp8" and suffix == "weight_scale_inv":
        byte_suffix = "weight_scale"
    elif layout.format == "mxfp4" and suffix == "weight_packed":
        byte_suffix = "weight"
    segments = expert_destinations(
        layer,
        layout,
        local,
        projection,
        byte_suffix,
        tuple(meta["shape"]),
        meta["dtype"],
    )
    storage = {}
    for segment in segments:
        for target in segment.destinations:
            storage.setdefault(target.untyped_storage().data_ptr(), target)
    return TensorBinding(
        name,
        meta["dtype"],
        tuple(meta["shape"]),
        _full_slices(meta["shape"]),
        tuple(storage.values()),
        segments=segments,
        format=meta.get("format"),
        checkpoint_dtype=meta.get("checkpoint_dtype"),
    )


def _identify_moe_layout(layer):
    # All ordinary expert methods initialize runner. The direct TRTLLM wrapper
    # sets it to None and dispatches using the explicit runtime selection.
    runner = layer.quant_method.runner
    if runner is None:
        from sglang.srt.runtime_context import get_exec

        backend = get_exec().moe.moe_runner_backend
    else:
        backend = runner.runner_backend
    return identify_layout(layer, backend)


@dataclass
class DerivedImage:
    """An independent consumer refreshed from a live canonical storage view."""

    name: str
    destination: torch.Tensor
    source: torch.Tensor

    def __post_init__(self):
        if (
            self.source.shape != self.destination.shape
            or self.source.dtype != self.destination.dtype
        ):
            raise ValueError(f"unsupported derived delta buffer geometry: {self.name}")

    @cached_property
    def source_pointer(self):
        return self.source.untyped_storage().data_ptr()


def _tensor_identity(tensor):
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def dense_target(name, meta, module, target, slices):
    quant_name = type(getattr(module, "quant_method", None)).__name__
    if quant_name not in {
        "NoneType",
        "UnquantizedLinearMethod",
        "UnquantizedEmbeddingMethod",
    }:
        raise ValueError(f"unsupported dense delta quantization {quant_name}: {name}")
    if tuple(target.shape) != tuple(stop - start for start, stop in slices):
        if hasattr(module, "shard_indices"):
            indices = module.shard_indices
            if indices.num_added_elements:
                raise ValueError("added-vocabulary delta layout is not supported")
            start, stop = indices.org_vocab_start_index, indices.org_vocab_end_index
            slices[0] = [start, stop]
            target = target[: stop - start]
        else:
            mismatches = [
                i
                for i, (local, full) in enumerate(zip(target.shape, meta["shape"]))
                if local != full
            ]
            if len(mismatches) != 1 or len(target.shape) != len(meta["shape"]):
                raise ValueError(f"unclassified dense shape mapping: {name}")
            dim = mismatches[0]
            allowed_dim = (
                getattr(target, "input_dim", None)
                if type(module).__name__ == "RowParallelLinear"
                else getattr(target, "output_dim", None)
            )
            if dim != allowed_dim or not hasattr(module, "tp_rank"):
                raise ValueError(f"unclassified tensor-parallel mapping: {name}")
            size = target.shape[dim]
            slices[dim] = [module.tp_rank * size, (module.tp_rank + 1) * size]
    return target, slices


class ConsumerSnapshot:
    """Read live consumer fields without retaining a stale replacement tensor."""

    def __init__(self, read):
        self.read = read
        self.objects = read()
        self.tensors = [
            (index, _tensor_identity(value))
            for index, value in enumerate(self.objects)
            if isinstance(value, torch.Tensor)
        ]

    def check_objects(self):
        current = self.read()
        if len(current) != len(self.objects) or any(
            a is not b for a, b in zip(current, self.objects)
        ):
            raise RuntimeError(
                "GPU delta consumer storage changed; readmission required"
            )

    def check(self):
        self.check_objects()
        if any(
            _tensor_identity(self.objects[index]) != expected
            for index, expected in self.tensors
        ):
            raise RuntimeError(
                "GPU delta consumer storage changed; readmission required"
            )


def gate_up_target(target, module, shard, slices):
    dim = target.output_dim
    half = target.shape[dim] // 2
    target = target.narrow(dim, shard * half, half)
    slices[dim] = [module.tp_rank * half, (module.tp_rank + 1) * half]
    return target


class ParameterBindings:
    """Shared routed-expert and ordinary dense binding context at admission."""

    def __init__(self, model, stacked_params=()):
        self.modules = dict(model.named_modules())
        self.params = dict(model.named_parameters(remove_duplicate=False))
        self.gate_up_params = [
            (source, shard)
            for target, source, shard in stacked_params
            if target == "gate_up_proj"
        ]
        self.excluded = {}
        self.moe_layers = {}
        self.moe_layouts = {}

    def bind(self, name, meta, target_name, target=None):
        if meta["dtype"] not in _DTYPES.values():
            raise ValueError(f"unsupported canonical dtype for {name}: {meta['dtype']}")
        match = re.fullmatch(
            r"(.+\.experts)\.(\d+)\.(gate|up|down)_proj\.(.+)", target_name
        )
        if match:
            prefix, expert, projection, suffix = match.groups()
            layer = self.modules.get(prefix)
            if layer is None:
                raise ValueError(f"canonical expert has no runtime layer: {name}")
            if prefix not in self.moe_layers:
                if layer.moe_tp_size != 1 or layer.use_presharded_weights:
                    raise ValueError(
                        "GPU delta requires expert TP=1 and an unsharded canonical checkpoint"
                    )
                self.moe_layouts[prefix] = _identify_moe_layout(layer)
                self.moe_layers[prefix] = layer
            binding = _moe_binding(
                name,
                meta,
                layer,
                int(expert),
                projection,
                suffix,
                self.moe_layouts[prefix],
            )
            if binding is None:
                self.excluded[name] = (
                    "static activation calibration"
                    if suffix == "input_scale"
                    else "expert owned by another EP rank"
                )
            return binding
        if target is None:
            target = self.params.get(target_name)
        slices = _full_slices(meta["shape"])
        if target is None:
            for source_stem, shard in self.gate_up_params:
                if source_stem not in target_name or ".experts." in target_name:
                    continue
                candidate = target_name.replace(source_stem, "gate_up_proj")
                target = self.params.get(candidate)
                if target is None:
                    continue
                module = self.modules[candidate.rsplit(".", 1)[0]]
                target = gate_up_target(target, module, shard, slices)
                target_name = candidate
                break
        if target is None:
            raise ValueError(f"unclassified canonical tensor: {name}")
        module = self.modules[target_name.rsplit(".", 1)[0]]
        target, slices = dense_target(name, meta, module, target, slices)
        return _direct_binding(name, meta, target, slices)
