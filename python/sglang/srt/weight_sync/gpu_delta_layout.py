# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Receiver-owned byte layouts for direct GPU weight deltas.

These functions operate on *bits*, including scale masks whose FP8 bit patterns
may be NaNs. They deliberately do not call the normal quantization/reload hooks.
The latter are value transforms and may reduce scales or rebind graph pointers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Callable

import orjson
import torch


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
    def view_id(self):
        return _digest({"name": self.name, "slices": self.slices})

    @cached_property
    def view_shape(self):
        return tuple(stop - start for start, stop in self.slices)

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


def _indexer_norm_binding(name, meta, target):
    """Match the CUDA indexer's ordinary BF16-checkpoint -> FP32 load.

    A numeric cast is not a byte permutation. These small parameters receive
    complete canonical target values, then use the normal loader's conversion
    while retaining the live FP32 parameter and graph address.
    """
    if meta["dtype"] != "BF16" or tuple(meta["shape"]) != tuple(target.shape):
        raise ValueError(f"unsupported canonical indexer norm: {name}")
    if target.dtype != torch.float32:
        raise ValueError(f"unsupported live indexer norm dtype: {name}")

    return TensorBinding(
        name,
        meta["dtype"],
        tuple(meta["shape"]),
        _full_slices(meta["shape"]),
        None,
        (target,),
        encoding="raw_bytes",
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


class GpuDeltaLayout:
    """Frozen, fail-closed map from startup checkpoint names to live buffers."""

    def __init__(self, model, inventory):
        from sglang.srt.weight_sync.gpu_delta_host import _natural_key

        self.model = model
        if not inventory:
            raise ValueError("canonical checkpoint metadata is unavailable")
        self.inventory = inventory
        self.bindings = []
        self.excluded = {}
        self.derived = []
        self._modules = dict(model.named_modules())
        self._params = dict(model.named_parameters(remove_duplicate=False))
        self._moe_layers = {}
        self._mla_layers = {}
        for name, meta in inventory.items():
            binding = self._bind(name, meta)
            if binding is not None:
                self.bindings.append(binding)
        # Layer and expert order matches the shared host arena. Each canonical
        # tensor still owns its full decoded byte extent, including TP shards.
        self.bindings.sort(key=lambda b: _natural_key(b.name))
        for prefix, layer in self._moe_layers.items():
            self._add_moe_derived(prefix, layer)
        for prefix, attn in self._mla_layers.items():
            self._add_mla_derived(prefix, attn)
        self._parameter_roots = [
            (module, name, tensor, _tensor_identity(tensor))
            for module in self._modules.values()
            for name, tensor in module._parameters.items()
            if tensor is not None
        ]
        self._moe_consumers = [
            (
                layer,
                layer._cutedsl_wrapper,
                layer._cutedsl_scales,
                [
                    (tensor, _tensor_identity(tensor))
                    for tensor in (layer._cutedsl_scales[0], layer._cutedsl_scales[2])
                ],
            )
            for layer in self._moe_layers.values()
        ]
        self._mla_consumers = [
            (
                attn,
                [
                    (tensor, _tensor_identity(tensor))
                    for tensor in (attn.w_kc, attn.w_vc)
                ],
            )
            for attn in self._mla_layers.values()
        ]
        self._reject_overlaps()
        self.rank_plan_digest = _digest(
            {
                "adapter": "deepseek-nvfp4-cutedsl-w4a16-v1",
                "tensors": [b.describe() for b in self.bindings],
                "excluded": self.excluded,
            }
        )

    def check_identity(self):
        for module, name, tensor, expected in self._parameter_roots:
            if (
                module._parameters.get(name) is not tensor
                or _tensor_identity(tensor) != expected
            ):
                raise RuntimeError(
                    "GPU delta parameter identity changed; readmission required"
                )
        for layer, wrapper, scales, tensors in self._moe_consumers:
            if (
                layer._cutedsl_wrapper is not wrapper
                or layer._cutedsl_scales is not scales
            ):
                raise RuntimeError(
                    "GPU delta CuTe DSL consumer changed; readmission required"
                )
            _check_consumer_tensors((scales[0], scales[2]), tensors)
        for attn, tensors in self._mla_consumers:
            _check_consumer_tensors((attn.w_kc, attn.w_vc), tensors)

    def _bind(self, name, meta):
        layer_id = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if layer_id and int(layer_id[1]) >= self.model.config.num_hidden_layers:
            self.excluded[name] = "static bundled draft"
            return None
        if name.endswith("rotary_emb.inv_freq"):
            self.excluded[name] = "loader ignores computed rotary frequencies"
            return None
        if meta["dtype"] not in _DTYPES.values():
            raise ValueError(f"unsupported canonical dtype for {name}: {meta['dtype']}")
        mapped_name = self.model.mutate_weight_preload(name)
        match = re.fullmatch(
            r"(.+\.experts)\.(\d+)\.(gate|up|down)_proj\.(.+)", mapped_name
        )
        if match:
            prefix, expert, projection, suffix = match.groups()
            layer = self._modules.get(prefix)
            if layer is None:
                raise ValueError(f"canonical expert has no runtime layer: {name}")
            self._moe_layers[prefix] = layer
            binding = _moe_binding(name, meta, layer, int(expert), projection, suffix)
            if binding is None:
                self.excluded[name] = (
                    "static W4A16 activation calibration"
                    if suffix == "input_scale"
                    else "expert owned by another EP rank"
                )
            return binding
        # Unfused BF16 shared experts use the same ordinary dense TP slicing
        # below. The expert-TP restriction applies to routed NVFP4 experts.
        target_name = mapped_name
        target = self._params.get(target_name)
        slices = _full_slices(meta["shape"])
        if target is None:
            for param_stem, source_stem, shard in self.model.stacked_params_mapping:
                if source_stem not in target_name or ".experts." in target_name:
                    continue
                candidate = target_name.replace(source_stem, param_stem)
                target = self._params.get(candidate)
                if target is None:
                    continue
                module = self._modules[candidate.rsplit(".", 1)[0]]
                if meta["dtype"] != _dtype_name(target.dtype):
                    raise ValueError(
                        f"numerical fused mapping requires an adapter: {name}"
                    )
                # GLM's fused Q/KV-A projection is replicated. Dense gate/up
                # uses the module's resolved TP rank, not global TP rank.
                if param_stem == "fused_qkv_a_proj_with_mqa":
                    sizes = [
                        self.model.config.q_lora_rank,
                        self.model.config.kv_lora_rank
                        + self.model.config.qk_rope_head_dim,
                    ]
                    if not meta["shape"]:
                        raise ValueError(
                            "fused scalar projection metadata is not supported"
                        )
                    target = target.narrow(0, sum(sizes[:shard]), sizes[shard])
                elif param_stem == "gate_up_proj":
                    dim = getattr(target, "output_dim", 0)
                    half = target.shape[dim] // 2
                    target = target.narrow(dim, shard * half, half)
                    slices[dim] = [module.tp_rank * half, (module.tp_rank + 1) * half]
                else:
                    raise ValueError(f"unclassified stacked parameter: {name}")
                target_name = candidate
                break
        if target is None and ".indexer." in target_name:
            for source, at_end in (("wk", False), ("weights_proj", True)):
                marker = f".indexer.{source}.weight"
                if target_name.endswith(marker):
                    candidate = (
                        target_name[: -len(marker)] + ".indexer.wk_weights_proj.weight"
                    )
                    target = self._params.get(candidate)
                    if target is not None and meta["dtype"] == "BF16":
                        rows = meta["shape"][0]
                        target = target[-rows:] if at_end else target[:rows]
                        target_name = candidate
                    break
        if target is None:
            raise ValueError(f"unclassified canonical tensor: {name}")
        module = self._modules[target_name.rsplit(".", 1)[0]]
        quant_name = type(getattr(module, "quant_method", None)).__name__
        if quant_name not in {
            "NoneType",
            "UnquantizedLinearMethod",
            "UnquantizedEmbeddingMethod",
        }:
            raise ValueError(
                f"unsupported dense delta quantization {quant_name}: {name}"
            )
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
        if ".kv_b_proj.weight" in name:
            prefix = target_name.rsplit(".kv_b_proj.weight", 1)[0]
            self._mla_layers[prefix] = self._modules[prefix]
        if (
            target.dtype == torch.float32
            and meta["dtype"] == "BF16"
            and re.fullmatch(
                r"model\.layers\.\d+\.self_attn\.indexer\.k_norm\.(weight|bias)",
                name,
            )
        ):
            return _indexer_norm_binding(name, meta, target)
        return _direct_binding(name, meta, target, slices)

    def _add_moe_derived(self, prefix, layer):
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
            self.derived.append(DerivedImage(f"{prefix}.{name}", destination, source))
        if layer._cutedsl_wrapper is None:
            raise ValueError(
                "CuTe DSL warmup must initialize its wrapper before delta admission"
            )
        if layer._cutedsl_wrapper.quant_mode != "w4a16":
            raise ValueError("prepared CuTe DSL wrapper is not W4A16")
        # Standard CuTe DSL uses the gate alpha for its fused GEMM1.
        self.derived.extend(
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

    def _add_mla_derived(self, prefix, attn):
        if attn.kv_b_proj.weight.dtype != torch.bfloat16:
            raise ValueError(
                "MLA delta cache refresh currently requires BF16 canonical weights"
            )

        key, value = attn.kv_b_proj.weight.unflatten(
            0, (-1, attn.qk_nope_head_dim + attn.v_head_dim)
        ).split([attn.qk_nope_head_dim, attn.v_head_dim], dim=1)

        self.derived.extend(
            (
                DerivedImage(f"{prefix}.w_kc", attn.w_kc, key),
                DerivedImage(
                    f"{prefix}.w_vc",
                    attn.w_vc,
                    value.transpose(1, 2),
                ),
            )
        )

    def _reject_overlaps(self):
        # Gate/up intentionally share one interleaved weight/scale allocation;
        # their maps are disjoint. Other shared storage needs an explicit alias
        # adapter so a tied parameter is never XORed twice.
        owners = {}
        for binding in self.bindings:
            for tensor in binding.storage:
                key = tensor.untyped_storage().data_ptr()
                prior = owners.get(key)
                if prior and prior != binding.name:
                    pair = (prior, binding.name)
                    if not all(
                        ".experts." in name for name in pair
                    ) and not _known_fused_pair(*pair):
                        raise ValueError(f"unclassified live weight alias: {pair}")
                owners[key] = binding.name


def _known_fused_pair(a, b):
    for left, right in (
        ("gate_proj", "up_proj"),
        ("q_a_proj", "kv_a_proj_with_mqa"),
        ("wk", "weights_proj"),
    ):
        if a.replace(left, right) == b or b.replace(left, right) == a:
            return True
    return False


def _tensor_identity(tensor):
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _check_consumer_tensors(current, admitted):
    for tensor, (original, expected) in zip(current, admitted):
        if tensor is not original or _tensor_identity(tensor) != expected:
            raise RuntimeError(
                "GPU delta consumer storage changed; readmission required"
            )


def _require_fixed_moe_topology(moe):
    # The canonical expert map is valid only while physical expert ownership is
    # the trivial EP partition. EPLB/custom/elastic maps can change ownership
    # without changing tensor pointers, so pointer checks alone cannot admit it.
    if (
        moe.enable_eplb
        or moe.elastic_ep_backend is not None
        or moe.init_expert_location != "trivial"
        or moe.ep_num_redundant_experts != 0
        or moe.moe_a2a_backend != "none"
        or moe.moe_runner_backend != "flashinfer_cutedsl"
    ):
        raise ValueError(
            "direct GPU deltas require fixed trivial EP, no redundant experts or EPLB, "
            "and flashinfer_cutedsl with moe_a2a_backend=none"
        )


class GpuDeltaBackend:
    """Scheduler-owned plan; preparation owns only immutable bytes and scratch."""

    def __init__(self, model_runner, identity):
        from sglang.srt.runtime_context import get_exec
        from sglang.srt.weight_sync.gpu_delta_checkpoint import (
            read_canonical_checkpoint_inventory,
        )
        from sglang.srt.weight_sync.gpu_delta_payload import (
            OuterZstdPool,
            configured_codec,
            configured_cpu_workers,
        )

        self.codec = configured_codec()
        _require_fixed_moe_topology(get_exec().moe)
        self.identity = dict(identity)
        self._canonical_plan = None
        self.batch_plan = None
        inventory = read_canonical_checkpoint_inventory(model_runner)
        self.layout = GpuDeltaLayout(model_runner.model, inventory)
        self.device = next(model_runner.model.parameters()).device
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("direct GPU deltas require an explicit CUDA device")
        from sglang.srt.weight_sync.gpu_delta_host import HostArena

        self.outer_pool = OuterZstdPool(configured_cpu_workers())
        self.host_arena = HostArena(identity["engine_id"])

    def describe(self):
        self.layout.check_identity()
        return {
            "adapter": "deepseek-nvfp4-cutedsl-w4a16-v1",
            "codec": self.codec,
            "rank_plan_digest": self.layout.rank_plan_digest,
            "tensors": [binding.describe() for binding in self.layout.bindings],
            "excluded": self.layout.excluded,
        }

    def prepare(self, manifest_path, manifest_sha256, metadata):
        # Called by the session's background worker: no weight reads, writes,
        # collectives, loader hooks or generation-stream synchronization here.
        with torch.cuda.device(self.device):
            prepared = PreparedDelta.__new__(PreparedDelta)
            try:
                prepared.__init__(self, manifest_path, manifest_sha256, metadata)
                return prepared
            except BaseException:
                prepared.close()
                raise

    def close(self):
        self.outer_pool.close()
        self.host_arena.close()


@dataclass
class _PreparedBatch:
    copies: list[tuple[torch.Tensor, torch.Tensor]]
    decoder: object
    apply: Callable[[], None] | None
    transformed: list[tuple[Callable, torch.Tensor]]
    zero_ranges: list[torch.Tensor]
    check_status: Callable[[], None]


def _plan_layers(backend, bindings, entries):
    """Cache membership, canonical offsets and apply geometry, never wire frames."""
    key = tuple(binding.name for binding in bindings)
    if backend.batch_plan is None or backend.batch_plan[0] != key:
        if not bindings:
            backend.batch_plan = (key, [])
            return []
        from sglang.srt.weight_sync.gpu_delta_apply import plan_apply

        layers = {}
        for binding in bindings:
            layers.setdefault(binding.layer, []).append(binding)
        plans = []
        for group in layers.values():
            outputs, size = [], 0
            for binding in group:
                offset = (size + 15) // 16 * 16
                nbytes = entries[binding.name]["nbytes"]
                outputs.append((binding, offset, nbytes))
                size = offset + nbytes
            apply, transformed = plan_apply(outputs)
            plans.append((outputs, size, apply, transformed))
        backend.batch_plan = (key, plans)
    return backend.batch_plan[1]


def _plan_batch(outputs, entries, records):
    """Map adjacent registered host spans and canonical outputs for one layer."""
    ranges, spans = [], []
    encoded_size = 0
    for binding, decoded_offset, _ in outputs:
        entry, record = entries[binding.name], records[binding.name]
        source, size = record["offset"], record["nbytes"]
        if spans and source == (spans[-1][0] + spans[-1][2] + 15) // 16 * 16:
            start, destination, _ = spans[-1]
            encoded_offset = destination + source - start
            spans[-1] = (start, destination, source + size - start)
        else:
            encoded_offset = (encoded_size + 15) // 16 * 16
            spans.append((source, encoded_offset, size))
        encoded_size = encoded_offset + size
        ranges.append((entry["frames"], encoded_offset, decoded_offset))
    return ranges, spans, encoded_size


def _decoded_gaps(outputs, entries):
    """Publication-specific omitted bytes; alignment padding is never consumed."""
    gaps = []
    for binding, offset, size in outputs:
        cursor = 0
        for frame in entries[binding.name]["frames"]:
            start = frame["decoded_offset"]
            if cursor < start:
                gaps.append((offset + cursor, start - cursor))
            cursor = start + frame["decoded_bytes"]
        if cursor < size:
            gaps.append((offset + cursor, size - cursor))
    return gaps


def _canonical_views(views):
    return [
        {"id": view_id, "slices": [list(pair) for pair in slices]}
        for view_id, slices in sorted(
            (view["id"], tuple(tuple(pair) for pair in view["slices"]))
            for view in views
        )
    ]


def _qualify_canonical_plan(backend, manifest):
    """Cache only qualified static definitions; payload geometry stays per-publication."""
    cached = backend._canonical_plan
    if cached is not None and manifest["plan_digest"] != cached[0]:
        raise ValueError("negotiated canonical delta plan changed")
    entries, signatures, definitions = {}, {}, []
    for entry in manifest["tensors"]:
        name = entry["name"]
        if name in entries:
            raise ValueError("duplicate canonical delta tensor")
        entries[name] = entry
        if cached is not None:
            definition = cached[1].get(name)
            if definition is None or (
                entry["dtype"] != definition[0]
                or entry["shape"] != definition[1]
                or entry["encoding"] != definition[2]
                or entry["nbytes"] != definition[3]
                or entry.get("byte_order") != definition[4]
                or (
                    entry["views"] != definition[5]
                    and _canonical_views(entry["views"]) != definition[5]
                )
            ):
                raise ValueError(f"canonical delta definition changed: {name}")
            continue
        if name not in backend.layout.inventory or backend.layout.excluded.get(
            name
        ) not in {None, "expert owned by another EP rank"}:
            raise ValueError(f"delta publication has an unadmitted tensor: {name}")
        canonical = backend.layout.inventory[name]
        if (
            entry["shape"] != canonical["shape"]
            or entry["dtype"] != canonical["dtype"]
            or entry.get("byte_order") != "little"
        ):
            raise ValueError(f"canonical tensor metadata mismatch: {name}")
        if entry["nbytes"] != math.prod(entry["shape"]) * _ITEMSIZES[
            entry["dtype"]
        ] or entry["encoding"] != (
            "raw_bytes" if len(entry["shape"]) <= 1 else "xor_bytes"
        ):
            raise ValueError(f"unsupported canonical tensor size/encoding: {name}")
        signature = (
            entry["dtype"],
            list(entry["shape"]),
            entry["encoding"],
            entry["nbytes"],
            entry["byte_order"],
            _canonical_views(entry["views"]),
        )
        signatures[name] = signature
        definitions.append(
            {
                "name": name,
                "dtype": signature[0],
                "shape": signature[1],
                "encoding": signature[2],
                "views": signature[5],
            }
        )
    if not {binding.name for binding in backend.layout.bindings} <= entries.keys():
        raise ValueError("publication omits an admitted mutable tensor")
    if cached is not None:
        if entries.keys() != cached[1].keys():
            raise ValueError("publication omits a negotiated canonical tensor")
    else:
        if (
            _digest(sorted(definitions, key=lambda entry: entry["name"]))
            != manifest["plan_digest"]
        ):
            raise ValueError(
                "publication does not match its negotiated canonical view plan"
            )
        # Detached static lists permit direct warm equality without rebuilding
        # nested signatures. No frames, payloads or manifest objects are retained.
        backend._canonical_plan = manifest["plan_digest"], signatures
    return entries, cached is not None


class PreparedDelta:
    def __init__(self, backend, manifest_path, manifest_sha256, metadata):
        self.stream = None
        self.copy_stream = None
        self.host_snapshot = None
        self.batches = []
        self.raw_copies = {}

        from pathlib import Path

        preparation_started = time.perf_counter()
        self.timing_enabled = os.environ.get("GPU_DELTA_TIMING", "0") == "1"
        self.h2d_stages = int(os.environ.get("GPU_DELTA_H2D_STAGES", "2"))
        if self.h2d_stages < 2:
            raise ValueError("GPU_DELTA_H2D_STAGES requires at least two slots")
        self.events = {}
        self.timings = {}
        from sglang.srt.weight_sync.gpu_delta_codec import DecodeFrame, NvcompDecoder
        from sglang.srt.weight_sync.gpu_delta_payload import validate_codec

        self.backend = backend
        self.device = backend.device
        self.stream = torch.cuda.Stream(device=self.device)
        manifest_started = time.perf_counter()
        path = Path(manifest_path).resolve(strict=True)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest_sha256:
            raise ValueError("immutable delta manifest SHA256 mismatch")
        manifest = orjson.loads(content)
        self.timings["host_manifest_read_parse_s"] = (
            time.perf_counter() - manifest_started
        )
        plan_started = time.perf_counter()
        validate_codec(manifest, backend.codec)
        if (
            type(manifest["base_version"]) is not int
            or manifest["target_version"] != manifest["base_version"] + 1
        ):
            raise ValueError("direct deltas require one consecutive version transition")
        self.target_version = manifest["target_version"]
        entries, reused_plan = _qualify_canonical_plan(backend, manifest)
        self.timings["host_plan_validate_s"] = time.perf_counter() - plan_started
        self.timings["host_plan_cache_reused"] = int(reused_plan)
        host_names = metadata["host_tensor_names"][backend.identity["host_cache_id"]]
        if (
            not {binding.name for binding in backend.layout.bindings} <= set(host_names)
            or not set(host_names) <= entries.keys()
        ):
            raise ValueError(
                "host tensor union does not cover the admitted local tensors"
            )
        payload_started = time.perf_counter()
        self.host_snapshot = backend.host_arena.prepare(
            path,
            manifest_sha256,
            manifest,
            host_names,
            backend.outer_pool,
            self.timings,
            metadata,
        )
        self.timings["host_payload_read_sha256_s"] = (
            self.timings["host_payload_read_s"] + self.timings["host_payload_sha256_s"]
        )
        self.timings["host_shared_prepare_s"] = time.perf_counter() - payload_started
        backend.host_arena.register(self.device, self.timings)

        tensors_started = time.perf_counter()
        compressed, direct = [], []
        for binding in backend.layout.bindings:
            entry = entries[binding.name]
            matching = [v for v in entry["views"] if v["id"] == binding.view_id]
            if len(matching) != 1 or matching[0]["slices"] != binding.slices:
                raise ValueError(f"missing or conflicting rank view for {binding.name}")
            if binding.encoding == "raw_bytes":
                if entry["changed_bytes"]:
                    direct.append((binding, entry))
                continue
            if not entry["frames"]:
                continue  # Omitted XOR masks need no work.
            compressed.append(binding)
        previous_plan = backend.batch_plan
        static_plans = _plan_layers(backend, compressed, entries)
        self.timings["host_batch_plan_reused"] = int(
            previous_plan is not None and backend.batch_plan is previous_plan
        )
        plans = [
            _plan_batch(outputs, entries, self.host_snapshot.index["tensors"])
            for outputs, _, _, _ in static_plans
        ]
        max_encoded = max((plan[2] for plan in plans), default=0)
        self.encoded_slot_bytes = (max_encoded + 15) // 16 * 16
        frames = [
            [
                DecodeFrame(
                    (index % self.h2d_stages) * self.encoded_slot_bytes
                    + encoded_offset
                    + frame["encoded_offset"],
                    frame["encoded_bytes"],
                    decoded_offset + frame["decoded_offset"],
                    frame["decoded_bytes"],
                )
                for ranges, encoded_offset, decoded_offset in plan[0]
                for frame in ranges
            ]
            for index, plan in enumerate(plans)
        ]
        if plans:
            self.copy_stream = torch.cuda.Stream(device=self.device)
            self.copy_ready = [torch.cuda.Event() for _ in range(self.h2d_stages)]
            self.copy_free = [torch.cuda.Event() for _ in range(self.h2d_stages)]
        max_decoded = max((plan[1] for plan in static_plans), default=0)
        self.matrix_tensor_count = len(compressed)
        # Rank-local tensor metadata; host-shared decompression is measured above.
        self.timings["host_tensor_prepare_s"] = time.perf_counter() - tensors_started
        # Scalars and vectors are complete target values, never delta masks.
        # Pack once on the preparation worker and upload the small arena before
        # pause, avoiding a pinned allocation/H2D/decode per tiny tensor.
        raw_started = time.perf_counter()
        self.raw_tensor_count = len(direct)
        raw_bytes = sum(entry["nbytes"] for _, entry in direct)
        raw_offsets, raw_h2d_bytes = [], 0
        for _, entry in direct:
            # Every supported canonical dtype can be viewed at an 8-byte
            # boundary, even when an odd-length U8 vector precedes FP32.
            position = (raw_h2d_bytes + 7) // 8 * 8
            raw_offsets.append(position)
            raw_h2d_bytes = position + entry["nbytes"]
        self.raw_pinned = torch.empty(
            raw_h2d_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        raw_view = memoryview(self.raw_pinned.numpy())
        for (_, entry), position in zip(direct, raw_offsets):
            source = memoryview(self.host_snapshot.get(entry["name"]).numpy())
            size = entry["nbytes"]
            raw_view[position : position + size] = source
        self.timings.update(
            host_raw_pack_s=time.perf_counter() - raw_started,
            raw_tensors=len(direct),
            raw_bytes=raw_bytes,
            raw_h2d_bytes=raw_h2d_bytes,
        )
        decoder_started = time.perf_counter()
        self.decoder = NvcompDecoder(self.device) if plans else None
        self.workspace = self.decoder.allocate_workspace(frames) if plans else None
        with torch.cuda.stream(self.stream):
            self.raw_device = self.raw_pinned.to(self.device, non_blocking=True)
            for (binding, entry), position in zip(direct, raw_offsets):
                size = entry["nbytes"]
                payload = self.raw_device[position : position + size]
                target = binding.storage[0]
                source = (
                    binding.selected_bytes(payload)
                    .view(binding.torch_dtype)
                    .reshape(target.shape)
                )
                # Match numeric loader conversion during preparation, before
                # pause; the common same-dtype case is only an arena view.
                source = source.to(target.dtype)
                targets, sources = self.raw_copies.setdefault(target.dtype, ([], []))
                targets.append(target)
                sources.append(source)
            self.encoded = torch.empty(
                self.h2d_stages * self.encoded_slot_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
            self.decoded = torch.empty(
                max_decoded, dtype=torch.uint8, device=self.device
            )
            self.error = torch.zeros(1, dtype=torch.int32, device=self.device)
            decoders = (
                self.decoder.prepare_batches(
                    frames, self.encoded, self.decoded, self.workspace, self.stream
                )
                if plans
                else []
            )
            if decoders:
                from sglang.srt.weight_sync.gpu_delta_apply import prepare_status_check

            host = self.backend.host_arena.tensor
            pointer_rows = []
            tuned_batches, tune_bytes, tune_s, tune_cache_hits, tune_skipped = (
                0,
                0,
                0.0,
                0,
                0,
            )
            apply_groups = [
                group for _, _, group, _ in static_plans if group is not None
            ]
            for group in apply_groups:
                tuned, footprint, elapsed, reused, skipped = group.prepare(
                    self.decoded, self.error
                )
                tuned_batches += tuned
                tune_bytes += footprint
                tune_s += elapsed
                tune_cache_hits += reused
                tune_skipped += skipped
                pointer_rows.extend(group.pointer_rows(self.decoded.data_ptr()))
            self.apply_host_metadata = torch.empty(
                len(pointer_rows), dtype=torch.int64, pin_memory=True
            )
            self.apply_host_metadata.numpy()[:] = pointer_rows
            self.apply_metadata = self.apply_host_metadata.to(
                self.device, non_blocking=True
            )
            position = 0
            for index, (
                (_, spans, _),
                (outputs, _, group, transformed),
                decode,
            ) in enumerate(zip(plans, static_plans, decoders)):
                slot_offset = (index % self.h2d_stages) * self.encoded_slot_bytes
                apply = None
                if group is not None:
                    count = 2 * len(group.sources)
                    pointers = self.apply_metadata[position : position + count]
                    apply = partial(group.launch, pointers, self.error)
                    position += count
                self.batches.append(
                    _PreparedBatch(
                        [
                            (
                                self.encoded[
                                    slot_offset + target : slot_offset + target + size
                                ],
                                host[source : source + size],
                            )
                            for source, target, size in spans
                        ],
                        decode,
                        apply,
                        [
                            (
                                binding.xor,
                                binding.selected_bytes(
                                    self.decoded[offset : offset + size]
                                ),
                            )
                            for binding, offset, size in transformed
                        ],
                        [
                            self.decoded[offset : offset + size]
                            for offset, size in _decoded_gaps(outputs, entries)
                        ],
                        prepare_status_check(decode, self.error),
                    )
                )
            changed_bindings = compressed + [binding for binding, _ in direct]
            changed_storages = {
                tensor.untyped_storage().data_ptr()
                for binding in changed_bindings
                for tensor in binding.storage
            }
            self.derived = [
                image
                for image in backend.layout.derived
                if image.source.untyped_storage().data_ptr() in changed_storages
            ]
            ready = torch.cuda.Event()
            ready.record(self.stream)
        self.timings.update(
            host_decoder_prepare_s=time.perf_counter() - decoder_started,
            host_apply_tune_s=tune_s,
            apply_tuned_batches=tuned_batches,
            apply_tune_skipped_batches=tune_skipped,
            apply_tune_bytes=tune_bytes,
            apply_tune_cache_hits=tune_cache_hits,
            decoder_metadata_uploads=int(bool(plans)),
            decoder_metadata_h2d_bytes=4 * 8 * sum(map(len, frames)),
            apply_metadata_h2d_bytes=self.apply_metadata.numel() * 8,
            apply_groups=len(apply_groups),
            apply_grid_ctas=sum(group.grid[0] for group in apply_groups),
            apply_descriptor_h2d_bytes=0,
            apply_contracts=sum(len(group.contracts) for group in apply_groups),
            apply_word32_contracts=sum(
                group.word32_contracts for group in apply_groups
            ),
            apply_static_groups=sum(group.config[2] for group in apply_groups),
            transformed_tensors=sum(len(batch.transformed) for batch in self.batches),
            compressed_batches=len(plans),
            compressed_tensors=self.matrix_tensor_count,
            compressed_h2d_spans=sum(len(batch.copies) for batch in self.batches),
            decoded_zero_ranges=sum(len(batch.zero_ranges) for batch in self.batches),
            decoded_zero_bytes=sum(
                span.numel() for batch in self.batches for span in batch.zero_ranges
            ),
            encoded_scratch_bytes=self.encoded.numel(),
            encoded_slot_bytes=self.encoded_slot_bytes,
            encoded_buffers=self.h2d_stages if plans else 0,
            decoded_scratch_bytes=max_decoded,
            decoder_workspace_bytes=(
                self.workspace.temporary.numel() if self.workspace else 0
            ),
        )
        # This constructor runs on the preparation worker. PREPARED means
        # immutable pinned inputs, reusable arenas and decoder metadata are ready.
        # Encoded bytes are uploaded only as each model layer is applied.
        ready_started = time.perf_counter()
        ready.synchronize()
        self.timings["host_ready_wait_s"] = time.perf_counter() - ready_started
        self.timings["host_prepare_s"] = time.perf_counter() - preparation_started
        self.h2d_bytes = raw_h2d_bytes + sum(
            source.numel() for batch in self.batches for _, source in batch.copies
        )

    @contextmanager
    def _phase(self, name, stream=None):
        if not self.timing_enabled:
            yield
            return
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        stream = self.stream if stream is None else stream
        start.record(stream)
        yield
        end.record(stream)
        self.events.setdefault(name, []).append((start, end))

    def apply(self):
        apply_started = time.perf_counter()
        backend = self.backend
        backend.layout.check_identity()
        self.timings["host_apply_identity_s"] = time.perf_counter() - apply_started
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self.stream),
            torch.no_grad(),
        ):
            self.stream.wait_stream(torch.cuda.default_stream(self.device))
            with self._phase("paused_gpu_total"):
                raw_started = time.perf_counter()
                # Immutable raw targets are already uploaded and validated.
                # Their copies cannot introduce a decoder failure and run
                # before compressed units, whose errors poison the session.
                with self._phase("raw_apply"):
                    for targets, sources in self.raw_copies.values():
                        torch._foreach_copy_(targets, sources)
                self.timings["host_raw_enqueue_s"] = time.perf_counter() - raw_started
                matrices_started = time.perf_counter()
                if self.batches:
                    self._copy_batch(self.batches[0], 0)
                prefetched = 1
                for index, batch in enumerate(self.batches):
                    with self._phase("paused_copy_wait"):
                        self.stream.wait_event(self.copy_ready[index % self.h2d_stages])
                    self._decode_batch(batch)
                    self.copy_free[index % self.h2d_stages].record(self.stream)
                    # No new prefetch follows a synchronous decoder launch error.
                    # Device status errors keep the existing sticky apply gate.
                    while (
                        prefetched < len(self.batches)
                        and prefetched < index + self.h2d_stages
                    ):
                        self._copy_batch(self.batches[prefetched], prefetched)
                        prefetched += 1
                    self._apply_batch(batch)
                self.timings["host_matrix_enqueue_s"] = (
                    time.perf_counter() - matrices_started
                )
                derived_started = time.perf_counter()
                with self._phase("derived_refresh"):
                    # All decoder status updates are complete on this stream.
                    # Reuse one scalar predicate across derived consumers.
                    if self.derived:
                        decode_succeeded = self.error.view(()) == 0
                    for derived in self.derived:
                        # Device predicate: no per-tensor host synchronization.
                        torch.where(
                            decode_succeeded,
                            derived.source,
                            derived.destination,
                            out=derived.destination,
                        )
                self.timings["host_derived_enqueue_s"] = (
                    time.perf_counter() - derived_started
                )
            done = torch.cuda.Event()
            done.record(self.stream)
        completion_started = time.perf_counter()
        done.synchronize()
        self.timings["host_apply_completion_wait_s"] = (
            time.perf_counter() - completion_started
        )
        final_started = time.perf_counter()
        if self.error.item() != 0:
            raise RuntimeError(
                "direct GPU delta decompression failed; session is poisoned"
            )
        self.timings["host_apply_status_s"] = time.perf_counter() - final_started
        self.timings["paused_apply_host_wall_s"] = time.perf_counter() - apply_started
        if self.timing_enabled:
            self.timings["cuda_event_ms"] = {
                name: sum(start.elapsed_time(end) for start, end in pairs)
                for name, pairs in self.events.items()
            }
        return {
            "applied": True,
            "verification": "artifact-sha256-and-decoder-status",
            "target_version": self.target_version,
            "tensors": self.matrix_tensor_count + self.raw_tensor_count,
            "timing_enabled": self.timing_enabled,
            "timings": self.timings,
            "h2d_bytes": self.h2d_bytes,
        }

    def _copy_batch(self, batch, index):
        with torch.cuda.stream(self.copy_stream):
            if index >= self.h2d_stages:
                self.copy_stream.wait_event(self.copy_free[index % self.h2d_stages])
            with self._phase("paused_layer_h2d", self.copy_stream):
                for destination, source in batch.copies:
                    destination.copy_(source, non_blocking=True)
            self.copy_ready[index % self.h2d_stages].record(self.copy_stream)

    def _decode_batch(self, batch):
        with self._phase("decode"):
            # Raw targets have a separate prepared arena and foreach-copy
            # pass. Every tensor here is XOR; absent frames are zero deltas.
            if batch.zero_ranges:
                torch._foreach_zero_(batch.zero_ranges)
            batch.decoder.enqueue()
            batch.check_status()

    def _apply_batch(self, batch):
        # No gather of current weights, canonical reconstruction or weight hash.
        with self._phase("layout_apply"):
            if batch.apply is not None:
                batch.apply()
            for apply, payload in batch.transformed:
                apply(torch.where(self.error == 0, payload, 0))

    def release_and_close(self):
        # Miles queues resume only after every original engine rank applied.
        # Ordinary abort/error close must never authorize a shared overwrite.
        try:
            self.host_snapshot.mark_reusable()
        finally:
            self.close()

    def close(self):
        # Cancellation may race a background upload, but never frees storage
        # while either stream is using it. A failed decode can leave its already
        # queued copy in flight; attempt both drains before releasing any views.
        # No device-wide synchronization is performed here.
        try:
            if self.copy_stream is not None:
                self.copy_stream.synchronize()
        finally:
            if self.stream is not None:
                self.stream.synchronize()
        self.batches.clear()
        self.raw_copies.clear()
        self.apply_metadata = self.apply_host_metadata = None
        if self.host_snapshot is not None:
            self.host_snapshot.close()
            self.host_snapshot = None
        self.encoded = self.decoded = self.raw_device = self.raw_pinned = None
        self.workspace = self.decoder = None
