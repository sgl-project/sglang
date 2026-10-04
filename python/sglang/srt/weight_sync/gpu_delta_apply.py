# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepared groups of byte-affine XOR destinations; imported only at admission."""

import math
from dataclasses import dataclass
from functools import partial
from itertools import accumulate

import torch
import triton
import triton.language as tl


@triton.jit
def _check_decode(
    statuses, actual_sizes, expected_sizes, error, count, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < count
    status = tl.load(statuses + index, active, other=0)
    actual = tl.load(actual_sizes + index, active, other=0)
    expected = tl.load(expected_sizes + index, active, other=0)
    if tl.sum(((status != 0) | (actual != expected)).to(tl.int32), 0) != 0:
        tl.atomic_or(error, 1)


def prepare_status_check(decoder, error):
    """Compile/load before PREPARED; the returned call only enqueues validation."""
    count = decoder.statuses.numel()
    arguments = (
        decoder.statuses,
        decoder.actual_sizes,
        decoder.expected_sizes,
        error,
        count,
    )
    grid = (triton.cdiv(count, 1024), 1, 1)
    kernel = _check_decode.warmup(*arguments, 1024, grid=grid, num_warps=4)
    _ = kernel.run
    return partial(kernel[grid], *arguments)


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


@triton.jit
def _xor_linear_group(
    pointers,
    error,
    COUNT: tl.constexpr,
    LOOKUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(error) == 0:
        tile = tl.program_id(0).to(tl.int64)
        lookup = tl.arange(0, LOOKUP)
        # Pointer rows are followed by cumulative tile ends and byte lengths.
        ends = tl.load(pointers + 2 * COUNT + lookup, lookup < COUNT, other=0)
        item = tl.sum(((lookup < COUNT) & (tile >= ends)).to(tl.int32), 0)
        first = tl.load(pointers + 2 * COUNT + item - 1, item > 0, other=0)
        size = tl.load(pointers + 3 * COUNT + item)
        index = (tile - first) * BLOCK + tl.arange(0, BLOCK)
        source = tl.load(pointers + item).to(tl.pointer_type(tl.uint8))
        target = tl.load(pointers + COUNT + item).to(tl.pointer_type(tl.uint8))
        mask = index < size
        value = tl.load(source + index, mask, other=0)
        previous = tl.load(target + index, mask, other=0)
        tl.store(target + index, previous ^ value, mask)


@dataclass
class ByteApplyGroup:
    """Static geometry and live destinations; version scratch pointers stay separate."""

    geometry: tuple | None
    sources: list[int]
    targets: list[int]
    static_metadata: tuple[int, ...] = ()
    kernel: object = None

    def compile(self, pointers, error):
        if self.kernel is None:
            count = len(self.sources)
            if self.geometry is None:
                self.grid = (self.static_metadata[count - 1], 1, 1)
                self.kernel = _xor_linear_group.warmup(
                    pointers,
                    error,
                    count,
                    triton.next_power_of_2(count),
                    4096,
                    grid=self.grid,
                    num_warps=4,
                )
            else:
                source_shape, source_stride, target_shape, target_stride = self.geometry
                self.grid = (
                    triton.cdiv(math.prod(source_shape), 4096),
                    count,
                    1,
                )
                self.kernel = _xor_group.warmup(
                    pointers,
                    error,
                    count,
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


def _normalize(shape, stride):
    """Keep the same flat byte mapping with fewer affine axes."""
    axes = []
    for size, step in reversed(tuple(zip(shape, stride))):
        if size == 1:
            continue
        if axes and step == axes[-1][0] * axes[-1][1]:
            axes[-1] = (size * axes[-1][0], axes[-1][1])
        else:
            axes.append((size, step))
    if not axes:
        return (1,), (1,)
    shape, stride = zip(*reversed(axes))
    return shape, stride


def plan_groups(outputs, expert_contracts):
    """Build once per active tensor set, using only CPU metadata/meta tensors."""
    groups, transformed, linear_sizes = {}, [], []
    for binding, offset, size in outputs:
        if not binding.destinations:
            transformed.append((binding, offset, size))
            continue
        # The admitted expert family has one layout contract. Its layer and
        # projection/suffix remain distinct; only actual pointers vary by expert.
        family = binding.expert_family
        contract = expert_contracts.get(family) if family is not None else None
        if contract is None:
            canonical = torch.empty(size, dtype=torch.uint8, device="meta")
            source = binding.selected_bytes(canonical)
            contract = (
                _normalize(source.shape, source.stride()),
                source.storage_offset(),
                tuple(_normalize(t.shape, t.stride()) for t in binding.destinations),
            )
            if family is not None:
                expert_contracts[family] = contract
        (source_shape, source_stride), source_offset, destinations = contract
        for index, target in enumerate(binding.destinations):
            target_shape, target_stride = destinations[index]
            geometry = source_shape, source_stride, target_shape, target_stride
            if family is None and source_stride == target_stride == (1,):
                geometry = None
                linear_sizes.append(math.prod(source_shape))
            group = groups.setdefault(geometry, ByteApplyGroup(geometry, [], []))
            group.sources.append(offset + source_offset)
            group.targets.append(target.data_ptr())
    if linear_sizes:
        group = groups[None]
        if all(size == linear_sizes[0] for size in linear_sizes):
            shape = (linear_sizes[0],)
            group.geometry = shape, (1,), shape, (1,)
        else:
            group.static_metadata = (
                *accumulate((size + 4095) // 4096 for size in linear_sizes),
                *linear_sizes,
            )
    return list(groups.values()), transformed
