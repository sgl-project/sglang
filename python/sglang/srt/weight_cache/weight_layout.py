# SPDX-License-Identifier: Apache-2.0
"""Explicit, bounded descriptions of immutable runtime weight storage."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import torch

from .weight_runtime_manifest import (
    LogicalWeightTensor,
    WeightManifestError,
    WeightPartition,
    _storage_key,
)


@dataclass(frozen=True)
class WeightRegion:
    tensor_id: str
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    parameter: torch.Tensor
    parameter_names: tuple[str, ...]
    byte_offset: int
    partition: WeightPartition
    expert_id: int | None = None
    layout_fingerprint: str = "sglang:row-major"

    @property
    def nbytes(self) -> int:
        return prod(self.local_shape) * self.parameter.element_size()


@dataclass(frozen=True)
class WeightInitialization:
    parameter: torch.Tensor
    parameter_names: tuple[str, ...]
    byte_offset: int
    nbytes: int
    value: bool | int | float


def registered_tensors(model):
    return dict(
        (
            *model.named_parameters(remove_duplicate=False),
            *model.named_buffers(remove_duplicate=False),
        )
    )


@dataclass(frozen=True)
class WeightLayoutPlan:
    tensors: tuple[LogicalWeightTensor, ...]
    views: tuple[WeightRegion, ...]
    initializations: tuple[WeightInitialization, ...]
    local_buffers: frozenset[str]
    derived_state: tuple[str, ...]
    bindings: dict[str, tuple]

    def validate_bindings(self, model) -> None:
        tensors = registered_tensors(model)
        if set(tensors) != self.bindings.keys() | self.local_buffers:
            raise WeightManifestError(
                "registered tensor names changed after layout construction"
            )
        for name, signature in self.bindings.items():
            if _storage_key(tensors[name]) != signature:
                raise WeightManifestError(
                    f"weight storage changed after layout construction: {name}"
                )
        for region in (*self.views, *self.initializations):
            if (
                _storage_key(region.parameter)
                != self.bindings[region.parameter_names[0]]
            ):
                raise WeightManifestError(
                    f"declared tensor storage changed: {region.parameter_names[0]}"
                )

    def initialize_local_regions(self) -> None:
        with torch.no_grad():
            for region in self.initializations:
                itemsize = region.parameter.element_size()
                region.parameter.view(-1).narrow(
                    0, region.byte_offset // itemsize, region.nbytes // itemsize
                ).fill_(region.value)

    def validate_derived_state(self, model) -> None:
        for name in self.derived_state:
            module_name, _, attr = name.rpartition(".")
            module = model.get_submodule(module_name) if module_name else model
            if not isinstance(
                getattr(module, attr, None), (torch.Tensor, bool, int, float)
            ):
                raise WeightManifestError(
                    f"derived weight state was not rebuilt: {name}"
                )


class WeightLayoutBuilder:
    """Account for every parameter/buffer without inspecting tensor values.

    Each declared region is a logical box stored as one contiguous byte range.
    Helpers emit a fixed number of boxes per fusion segment or expert; arbitrary
    tensor transforms and per-element fragmentation are deliberately unsupported.
    """

    def __init__(self, model: Any):
        self.model = model
        self.tensors = registered_tensors(model)
        self._names = {}
        for name, tensor in self.tensors.items():
            self._names.setdefault(_storage_key(tensor), []).append(name)
        self._views = []
        self._initializations = []
        self._local_buffers = set()
        self._derived_state = []
        self._view_index = {}
        self._initialization_index = set()
        self._declarations = {}

    def _parameter_names(self, tensor):
        names = self._names.get(_storage_key(tensor))
        if names is None:
            raise WeightManifestError(
                "layout region does not reference registered storage"
            )
        if (
            tensor.is_sparse
            or not tensor.is_contiguous()
            or tensor.device.type == "meta"
        ):
            raise WeightManifestError(f"unsupported runtime storage: {names[0]}")
        return tuple(sorted(names))

    def region(
        self,
        tensor,
        tensor_id,
        *,
        global_shape,
        global_offset=None,
        local_shape=None,
        byte_offset=0,
        expert_id=None,
        partition=WeightPartition(),
        layout_fingerprint="sglang:row-major",
    ):
        global_shape = tuple(global_shape)
        local_shape = tuple(local_shape if local_shape is not None else tensor.shape)
        global_offset = tuple(
            global_offset if global_offset is not None else (0,) * len(global_shape)
        )
        if (
            not tensor_id
            or not layout_fingerprint
            or not (len(global_shape) == len(global_offset) == len(local_shape))
            or any(
                g <= 0 or n <= 0 or o < 0 or o + n > g
                for g, o, n in zip(global_shape, global_offset, local_shape)
            )
        ):
            raise WeightManifestError(f"invalid logical region: {tensor_id}")
        names = self._parameter_names(tensor)
        self.declare(
            tensor,
            tensor_id,
            global_shape=global_shape,
            partition=partition,
            expert_id=expert_id,
            layout_fingerprint=layout_fingerprint,
        )
        region = WeightRegion(
            tensor_id,
            global_shape,
            global_offset,
            local_shape,
            tensor,
            names,
            byte_offset,
            partition,
            expert_id,
            layout_fingerprint,
        )
        self._check_range(region)
        # The same exact storage can be exposed by multiple names (tied heads).
        key = (names, byte_offset, region.nbytes)
        for previous in self._view_index.get(key, ()):
            if (
                previous.tensor_id,
                previous.global_shape,
                previous.global_offset,
                previous.local_shape,
                previous.layout_fingerprint,
                previous.expert_id,
            ) == (
                tensor_id,
                global_shape,
                global_offset,
                local_shape,
                layout_fingerprint,
                expert_id,
            ):
                return
        self._views.append(region)
        self._view_index.setdefault(key, []).append(region)

    def declare(
        self,
        tensor,
        tensor_id,
        *,
        global_shape,
        partition=WeightPartition(),
        expert_id=None,
        layout_fingerprint="sglang:row-major",
    ):
        """Keep ownership even when this rank has only vocabulary padding."""
        self._parameter_names(tensor)
        shape = tuple(global_shape) or (1,)
        if (
            not tensor_id
            or any(n <= 0 for n in shape)
            or (partition.dim is not None and partition.dim >= len(shape))
            or (expert_id is not None and partition.kind != "expert")
        ):
            raise WeightManifestError(f"invalid tensor declaration: {tensor_id}")
        declaration = LogicalWeightTensor(
            tensor_id=tensor_id,
            global_shape=shape,
            dtype=str(tensor.dtype).removeprefix("torch."),
            itemsize=tensor.element_size(),
            partition=partition,
            layout_fingerprint=layout_fingerprint,
            expert_id=expert_id,
        )
        if self._declarations.setdefault(tensor_id, declaration) != declaration:
            raise WeightManifestError(f"conflicting tensor declaration: {tensor_id}")

    def initialize(self, tensor, *, byte_offset, nbytes, value=0):
        if nbytes == 0:
            return
        region = WeightInitialization(
            tensor, self._parameter_names(tensor), byte_offset, nbytes, value
        )
        self._check_range(region)
        key = (region.parameter_names, byte_offset, nbytes, value)
        if key not in self._initialization_index:
            self._initializations.append(region)
            self._initialization_index.add(key)

    @staticmethod
    def _check_range(region):
        itemsize = region.parameter.element_size()
        if (
            region.byte_offset < 0
            or region.nbytes <= 0
            or region.byte_offset % itemsize
            or region.nbytes % itemsize
            or region.byte_offset + region.nbytes > region.parameter.numel() * itemsize
        ):
            raise WeightManifestError(
                f"layout byte range is out of bounds: {region.parameter_names[0]}"
            )

    def local_buffer(self, name):
        if name not in dict(self.model.named_buffers(remove_duplicate=False)):
            raise WeightManifestError(
                f"local initialization must name a buffer: {name}"
            )
        self._local_buffers.add(name)

    def derived(self, *names):
        for name in names:
            if name in self.tensors:
                raise WeightManifestError(
                    f"derived state must not replace registered storage: {name}"
                )
            self._derived_state.append(name)

    def build(self):
        # Exact aliases were grouped above. Different overlapping views cannot
        # independently own the same bytes, even if both declarations are full.
        intervals = sorted(
            (
                tensor.data_ptr(),
                tensor.data_ptr() + tensor.numel() * tensor.element_size(),
                names[0],
            )
            for names in self._names.values()
            if (tensor := self.tensors[names[0]]).numel()
        )
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise WeightManifestError(
                    f"overlapping registered tensor storage: {previous[2]}, {current[2]}"
                )
        grouped = {}
        for region in (*self._views, *self._initializations):
            grouped.setdefault(_storage_key(region.parameter), []).append(region)
        for key, names in self._names.items():
            local = set(names) & self._local_buffers
            if local:
                if local != set(names) or key in grouped:
                    raise WeightManifestError(
                        f"conflicting local buffer declaration: {names[0]}"
                    )
                continue
            tensor = self.tensors[names[0]]
            cursor = 0
            for region in sorted(grouped.get(key, ()), key=lambda r: r.byte_offset):
                if region.byte_offset != cursor:
                    raise WeightManifestError(
                        f"layout regions overlap or leave a gap: {names[0]}"
                    )
                cursor += region.nbytes
            if cursor != tensor.numel() * tensor.element_size():
                raise WeightManifestError(
                    f"layout does not cover complete tensor: {names[0]}"
                )
        plan = WeightLayoutPlan(
            tuple(self._declarations[name] for name in sorted(self._declarations)),
            tuple(self._views),
            tuple(self._initializations),
            frozenset(self._local_buffers),
            tuple(self._derived_state),
            {
                n: _storage_key(t)
                for n, t in self.tensors.items()
                if n not in self._local_buffers
            },
        )
        plan.validate_bindings(self.model)
        return plan
