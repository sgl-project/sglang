# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepared groups of byte-affine XOR destinations; imported only at admission."""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _xor_group(
    pointers,
    error,
    COUNT: tl.constexpr,
    SIZE: tl.constexpr,
    SOURCE_SHAPE: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    TARGET_SHAPE: tl.constexpr,
    TARGET_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(error) == 0:
        item = tl.program_id(1)
        index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        source = tl.load(pointers + item).to(tl.pointer_type(tl.uint8))
        target = tl.load(pointers + COUNT + item).to(tl.pointer_type(tl.uint8))
        source_offset = tl.full((BLOCK,), 0, tl.int64)
        target_offset = tl.full((BLOCK,), 0, tl.int64)
        rest = index
        for axis in tl.static_range(len(SOURCE_SHAPE) - 1, -1, -1):
            source_offset += (rest % SOURCE_SHAPE[axis]) * SOURCE_STRIDE[axis]
            rest //= SOURCE_SHAPE[axis]
        rest = index
        for axis in tl.static_range(len(TARGET_SHAPE) - 1, -1, -1):
            target_offset += (rest % TARGET_SHAPE[axis]) * TARGET_STRIDE[axis]
            rest //= TARGET_SHAPE[axis]
        mask = index < SIZE
        value = tl.load(source + source_offset, mask=mask, other=0)
        previous = tl.load(target + target_offset, mask=mask, other=0)
        tl.store(target + target_offset, previous ^ value, mask=mask)


@dataclass
class ByteApplyGroup:
    """Static geometry and live destinations; version scratch pointers stay separate."""

    geometry: tuple
    sources: list[int]
    targets: list[int]
    kernel: object = None

    def compile(self, pointers, error):
        if self.kernel is None:
            source_shape, source_stride, target_shape, target_stride = self.geometry
            self.grid = (triton.cdiv(math.prod(source_shape), 4096), len(self.sources))
            self.kernel = _xor_group.warmup(
                pointers,
                error,
                len(self.sources),
                math.prod(source_shape),
                source_shape,
                source_stride,
                target_shape,
                target_stride,
                4096,
                grid=self.grid,
                num_warps=4,
            )
            # Resolving the compiled launcher loads its CUDA module without
            # executing a kernel or touching the live weights.
            _ = self.kernel.run
            self.launch = self.kernel[self.grid]

    def enqueue(self, pointers, error):
        self.launch(pointers, error)


def plan_groups(outputs):
    """Build once per active tensor set, using only CPU metadata/meta tensors."""
    groups, transformed = {}, []
    for binding, offset, size in outputs:
        if not binding.destinations:
            transformed.append((binding, offset, size))
            continue
        canonical = torch.empty(size, dtype=torch.uint8, device="meta")
        source = binding.selected_bytes(canonical)
        for target in binding.destinations:
            geometry = (
                tuple(source.shape),
                source.stride(),
                tuple(target.shape),
                target.stride(),
            )
            group = groups.setdefault(geometry, ByteApplyGroup(geometry, [], []))
            group.sources.append(offset + source.storage_offset())
            group.targets.append(target.data_ptr())
    return list(groups.values()), transformed
