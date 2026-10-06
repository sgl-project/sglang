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
from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import torch

BACKEND_LAYOUTS = {
    "dense": "unquantized-tp-v1",
    "moe": "flashinfer-cutedsl-nvfp4-w4a16-v1",
}


def swizzle_scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    """NVFP4 128x4 block-scale permutation, with *zero* mask padding."""
    if scale.dtype != torch.uint8 or scale.ndim < 2:
        raise ValueError("block-scale layout requires uint8 matrices")
    rows, cols = scale.shape[-2:]
    padded_rows, padded_cols = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    out = scale.new_zeros((*scale.shape[:-2], padded_rows, padded_cols))
    out[..., :rows, :cols].copy_(scale)
    batch = math.prod(out.shape[:-2])
    return (
        out.reshape(batch, padded_rows // 128, 4, 32, padded_cols // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(out.shape)
    )


def interleave_gate_up_bytes(
    gate: torch.Tensor, up: torch.Tensor, group_rows: int, up_first: bool
) -> torch.Tensor:
    """Fuse projections using row groups without interpreting packed nibbles."""
    if gate.dtype != torch.uint8 or up.dtype != torch.uint8 or gate.shape != up.shape:
        raise ValueError("gate/up must be equal-shaped uint8 tensors")
    rows, cols = gate.shape[-2:]
    if rows % group_rows:
        raise ValueError("projection rows must be divisible by the interleave group")
    first, second = (up, gate) if up_first else (gate, up)
    return torch.stack(
        (
            first.reshape(*first.shape[:-2], rows // group_rows, group_rows, cols),
            second.reshape(*second.shape[:-2], rows // group_rows, group_rows, cols),
        ),
        dim=-3,
    ).reshape(*gate.shape[:-2], 2 * rows, cols)


def flashinfer_delta_layout(
    tensor: torch.Tensor,
    dtype: str,
    backend: str,
    kind: str,
    projection: str = "down",
) -> torch.Tensor:
    """Transform one complete projection's mask into its physical byte plane.

    ``nvfp4/cutedsl`` uses up-first 64-row groups; ``nvfp4/megamoe``
    describes gate-first 16-row groups. This helper does not admit a runtime
    backend: admission separately checks actual model tensors and aliases.
    BF16 is unchanged byte storage (including its byte axis), for ordinary
    FlashInfer dense GEMMs. Numerical alpha transforms are intentionally absent.
    """
    if tensor.dtype != torch.uint8:
        raise ValueError("delta layout takes uint8 bits, never floating-point masks")
    if backend not in {"cutedsl", "megamoe"}:
        raise ValueError(f"unsupported FlashInfer delta backend: {backend}")
    if dtype == "bf16":
        if kind != "weight" or projection != "down":
            raise ValueError("BF16 helper supports plain weight storage only")
        return tensor
    if dtype != "nvfp4" or kind not in {"weight", "scale"}:
        raise ValueError(f"unsupported delta plane: {dtype}/{kind}")
    if projection not in {"gate", "up", "down"}:
        raise ValueError(f"unsupported projection: {projection}")
    out = tensor
    if projection != "down":
        zero = torch.zeros_like(tensor)
        gate, up = (tensor, zero) if projection == "gate" else (zero, tensor)
        out = interleave_gate_up_bytes(
            gate,
            up,
            group_rows=64 if backend == "cutedsl" else 16,
            up_first=backend == "cutedsl",
        )
    if kind == "scale":
        out = swizzle_scale_bytes(out)
    # Transposed MegaMoE weight views do not move physical storage bytes.
    return out


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
    xor: Callable[[torch.Tensor], None] | None
    storage: tuple[torch.Tensor, ...]
    encoding: str = "xor_bytes"
    destinations: tuple[torch.Tensor, ...] = ()

    def __post_init__(self):
        # The admitted binding is immutable for the session. Keep dtype and
        # slicing metadata off the paused per-tensor application path.
        self.torch_dtype = _TORCH_DTYPES[self.dtype]
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
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "encoding": self.encoding,
            "views": [{"id": self.view_id, "slices": self.slices}],
        }

    def selected_bytes(self, canonical_bytes):
        value = canonical_bytes.view(self.torch_dtype).reshape(self.shape)
        # Keep column-sharded selections as views of reusable decoded scratch.
        # The decoder fills this storage after preparation; never cache a copy.
        return _byte_view(value[self._selection])


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
    xor = None
    if not direct:
        byte_target = _byte_view(target)

        def xor(mask):
            byte_target.bitwise_xor_(mask.reshape(byte_target.shape))

    return TensorBinding(
        name,
        meta["dtype"],
        tuple(meta["shape"]),
        slices,
        xor,
        (target,),
        encoding="raw_bytes" if direct else "xor_bytes",
        destinations=() if direct else (byte_target,),
    )


def _same_storage(a, b):
    return (
        a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        and a.storage_offset() * a.element_size()
        == b.storage_offset() * b.element_size()
        and a.numel() * a.element_size() == b.numel() * b.element_size()
    )


def _scale_images(layer, stem, expert):
    """Return each independent swizzled image, including the MMA view."""
    primary = getattr(layer, f"{stem}_blockscale_swizzled")[expert]
    images = [primary]
    mma = getattr(layer, f"{stem}_blockscale_mma", None)
    if mma is not None:
        # Public FlashInfer MMA logical axes: (32,4,mtiles,4,ktiles,E).
        # Permuting them back describes physical storage, without rebuilding it.
        physical = mma[..., expert].permute(2, 4, 0, 1, 3)
        if not physical.is_contiguous():
            raise ValueError("unsupported nonstandard CuTe DSL MMA physical strides")
        physical = physical.view(primary.shape).view(primary.dtype)
        if not any(_same_storage(physical, current) for current in images):
            images.append(physical)
    source = getattr(layer, f"{stem}_weight_scale")[expert]
    if not any(_same_storage(source, current) for current in images):
        # Ordinary process_weights_after_loading keeps this swizzled too when
        # aliasing succeeds. A distinct canonical reload buffer is not consumed
        # by the kernel and needs its own explicit mapping; reject for now.
        raise ValueError("NVFP4 source scale does not alias the prepared scale image")
    return images


def _moe_binding(name, meta, layer, expert, projection, suffix):
    if layer.moe_tp_size != 1:
        raise ValueError("direct GPU delta requires expert TP=1")
    quant = layer.quant_method
    if not getattr(quant, "_is_cutedsl_v2_standard", False):
        raise ValueError("runtime NVFP4 delta requires standard flashinfer_cutedsl")
    if not layer.moe_runner_config.is_gated:
        raise ValueError("runtime NVFP4 delta currently requires gated experts")
    if layer.use_presharded_weights:
        raise ValueError("presharded canonical checkpoints are not admitted")
    local = layer._map_global_expert_id_to_local_expert_id(expert)
    stem = "w2" if projection == "down" else "w13"
    half = 0 if projection == "gate" else 1  # scalar metadata stays gate-first
    if suffix == "input_scale":
        # ModelOpt's CuTe DSL W4A16 preparation neutralizes checkpoint activation
        # scales; BF16 activations never consume these calibration buffers.
        return None
    if local < 0:
        return None
    if suffix == "weight_scale_2":
        target = getattr(layer, f"{stem}_weight_scale_2")
        target = target[local] if projection == "down" else target[local, half]
        return _direct_binding(name, meta, target)
    if suffix == "weight":
        target = getattr(layer, f"{stem}_weight")[local]
        if projection != "down":
            rows, cols = target.shape
            if rows % 128:
                raise ValueError("CuTe DSL gate/up rows must be divisible by 128")
            # Inference weights are [up0:64,gate0:64,up64:128,gate64:128,...].
            target = target.reshape(rows // 128, 2, 64, cols)[:, 1 - half]
        return _direct_binding(name, meta, target)
    if suffix != "weight_scale":
        raise ValueError(f"unclassified NVFP4 canonical expert tensor: {name}")
    if meta["dtype"] != "F8_E4M3" or len(meta["shape"]) != 2:
        raise ValueError(f"invalid NVFP4 blockscale tensor: {name}")
    rows, cols = meta["shape"]
    full_rows = rows if projection == "down" else 2 * rows
    images = _scale_images(layer, stem, local)
    for image in images:
        if image.numel() != ((full_rows + 127) // 128 * 128) * ((cols + 3) // 4 * 4):
            raise ValueError(f"unexpected scale padding geometry: {name}")

    group = 128 if projection == "down" else 64
    destinations = []
    if (
        rows % group == 0
        and cols % 4 == 0
        and all(image.is_contiguous() for image in images)
    ):
        # Physical scale axes: row tile, column tile, row within 32,
        # row group within 128, column within 4. Gate/up own disjoint groups.
        for image in images:
            view = image.view(torch.uint8).view(rows // group, cols // 4, 32, 4, 4)
            if projection != "down":
                start = 2 if projection == "gate" else 0  # CuTe DSL is up-first.
                view = view[:, :, :, start : start + 2, :]
            destinations.append(view.permute(0, 3, 2, 1, 4))
        mask_shape = destinations[0].shape

        def xor(mask):
            mask = mask.reshape(mask_shape)
            for destination in destinations:
                destination.bitwise_xor_(mask)

    else:

        def xor(mask):
            transformed = flashinfer_delta_layout(
                mask.reshape(rows, cols),
                dtype="nvfp4",
                backend="cutedsl",
                kind="scale",
                projection=projection,
            )
            for image in images:
                image.view(torch.uint8).bitwise_xor_(transformed)

    return TensorBinding(
        name,
        meta["dtype"],
        tuple(meta["shape"]),
        _full_slices(meta["shape"]),
        xor,
        tuple(images),
        destinations=tuple(destinations),
    )


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


def moe_derived_images(prefix, layer):
    derived = []
    # W4A16 neutralizes activation scales, so no reciprocal/reduction of
    # arbitrary mask bit patterns appears here. ModelOpt's FP32 scale
    # parameters supply live views, never admission-time value copies.
    gate = layer.w13_weight_scale_2[:, 0]
    up = layer.w13_weight_scale_2[:, 1]
    down = layer.w2_weight_scale_2
    for name, destination, source in (
        ("g1_alphas", layer.g1_alphas, gate),
        ("g1_alphas_up", layer.g1_alphas_up, up),
        ("g2_alphas", layer.g2_alphas, down),
    ):
        derived.append(DerivedImage(f"{prefix}.{name}", destination, source))
    if layer._cutedsl_wrapper is None:
        raise ValueError(
            "CuTe DSL warmup must initialize its wrapper before delta admission"
        )
    if layer._cutedsl_wrapper.quant_mode != "w4a16":
        raise ValueError("prepared CuTe DSL wrapper is not W4A16")
    # Standard CuTe DSL uses the gate alpha for its fused GEMM1.
    derived.extend(
        (
            DerivedImage(
                f"{prefix}._cutedsl_scales.0",
                layer._cutedsl_scales[0],
                gate,
            ),
            DerivedImage(
                f"{prefix}._cutedsl_scales.2",
                layer._cutedsl_scales[2],
                down,
            ),
        )
    )
    return derived


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

    def check(self):
        current = self.read()
        if (
            len(current) != len(self.objects)
            or any(a is not b for a, b in zip(current, self.objects))
            or any(
                _tensor_identity(current[index]) != expected
                for index, expected in self.tensors
            )
        ):
            raise RuntimeError(
                "GPU delta consumer storage changed; readmission required"
            )


def gate_up_target(target, module, shard, slices):
    dim = getattr(target, "output_dim", 0)
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
            self.moe_layers[prefix] = layer
            binding = _moe_binding(name, meta, layer, int(expert), projection, suffix)
            if binding is None:
                self.excluded[name] = (
                    "static W4A16 activation calibration"
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
                if meta["dtype"] != _dtype_name(target.dtype):
                    raise ValueError(
                        f"numerical fused mapping requires an adapter: {name}"
                    )
                target = gate_up_target(target, module, shard, slices)
                target_name = candidate
                break
        if target is None:
            raise ValueError(f"unclassified canonical tensor: {name}")
        module = self.modules[target_name.rsplit(".", 1)[0]]
        target, slices = dense_target(name, meta, module, target, slices)
        return _direct_binding(name, meta, target, slices)
