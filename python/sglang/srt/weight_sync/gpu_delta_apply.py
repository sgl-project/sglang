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
def _affine_offset(index, shape, stride):
    offset = tl.full(index.shape, 0, tl.int64)
    rest = index
    for axis in tl.static_range(len(shape) - 1, 0, -1):
        if shape[axis] != 1:
            offset += (rest % shape[axis]) * stride[axis]
            rest //= shape[axis]
    return offset + rest * stride[0]


@triton.jit
def _xor_persistent(
    pointers,
    error,
    COUNT: tl.constexpr,
    AXES: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(error) == 0:
        tile = tl.program_id(0).to(tl.int64)
        item = 0
        first = tl.full((), 0, tl.int64)
        metadata = pointers + 2 * COUNT
        end = tl.load(metadata)
        # A CTA retains its descriptor over consecutive grid-stride tiles.
        # Only tensor boundaries walk the compact prefix; no per-tile table.
        while tile < TILES:
            while tile >= end:
                first = end
                item += 1
                end = tl.load(metadata + item * (4 + 4 * AXES))
            row = metadata + item * (4 + 4 * AXES)
            size = tl.load(row + 1)
            source_mode = tl.load(row + 2)
            target_mode = tl.load(row + 3)
            source = tl.load(pointers + item).to(tl.pointer_type(tl.uint8))
            target = tl.load(pointers + COUNT + item).to(tl.pointer_type(tl.uint8))
            source_shape, source_stride, target_shape, target_stride = (), (), (), ()
            for axis in tl.static_range(AXES):
                source_shape += (tl.load(row + 4 + axis),)
                source_stride += (tl.load(row + 4 + AXES + axis),)
                target_shape += (tl.load(row + 4 + 2 * AXES + axis),)
                target_stride += (tl.load(row + 4 + 3 * AXES + axis),)
            while tile < end:
                base = (tile - first) * BLOCK
                lanes = tl.arange(0, BLOCK)
                index = base + lanes
                if source_mode == 1:
                    source_offset = index
                elif source_mode == 2:
                    source_offset = (
                        _affine_offset(base, source_shape, source_stride) + lanes
                    )
                else:
                    source_offset = _affine_offset(index, source_shape, source_stride)
                if target_mode == 1:
                    target_offset = index
                elif target_mode == 2:
                    target_offset = (
                        _affine_offset(base, target_shape, target_stride) + lanes
                    )
                else:
                    target_offset = _affine_offset(index, target_shape, target_stride)
                mask = index < size
                value = tl.load(source + source_offset, mask, other=0)
                previous = tl.load(target + target_offset, mask, other=0)
                tl.store(target + target_offset, previous ^ value, mask)
                tile += tl.num_programs(0)


@dataclass
class ByteApplyGroup:
    """One cached affine descriptor plan; scratch pointers stay publication-local."""

    sources: list[int]
    targets: list[int]
    static_metadata: tuple[int, ...]
    axes: int
    tiles: int
    kernel: object = None

    def compile(self, pointers, error):
        if self.kernel is None:
            sms = torch.cuda.get_device_properties(
                pointers.device
            ).multi_processor_count
            self.grid = (min(self.tiles, 4 * sms), 1, 1)
            self.kernel = _xor_persistent.warmup(
                pointers,
                error,
                len(self.sources),
                self.axes,
                self.tiles,
                4096,
                grid=self.grid,
                num_warps=4,
            )
            # Load the module/launcher before PREPARED, without live writes.
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


def _access_mode(shape, stride):
    if stride == (1,):
        return 1
    # A tile cannot cross an affine row boundary in this geometry. Compute its
    # base address once per CTA iteration instead of dividing every byte index.
    return 2 if stride[-1] == 1 and shape[-1] % 4096 == 0 else 0


def plan_groups(outputs):
    """Cache one generic affine plan per layer; no model-name classification."""
    sources, targets, contracts, transformed = [], [], [], []
    for binding, offset, size in outputs:
        if not binding.destinations:
            transformed.append((binding, offset, size))
            continue
        canonical = torch.empty(size, dtype=torch.uint8, device="meta")
        source = binding.selected_bytes(canonical)
        source_shape, source_stride = _normalize(source.shape, source.stride())
        for target in binding.destinations:
            target_shape, target_stride = _normalize(target.shape, target.stride())
            sources.append(offset + source.storage_offset())
            targets.append(target.data_ptr())
            contracts.append((source_shape, source_stride, target_shape, target_stride))
    if not contracts:
        return [], transformed
    axes = max(max(len(c[0]), len(c[2])) for c in contracts)
    metadata, end = [], 0
    for source_shape, source_stride, target_shape, target_stride in contracts:
        size = math.prod(source_shape)
        end += (size + 4095) // 4096
        metadata.extend(
            (
                end,
                size,
                _access_mode(source_shape, source_stride),
                _access_mode(target_shape, target_stride),
            )
        )
        for shape, stride in (
            (source_shape, source_stride),
            (target_shape, target_stride),
        ):
            metadata.extend((*shape, *((1,) * (axes - len(shape)))))
            metadata.extend((*stride, *((0,) * (axes - len(stride)))))
    return [ByteApplyGroup(sources, targets, tuple(metadata), axes, end)], transformed
