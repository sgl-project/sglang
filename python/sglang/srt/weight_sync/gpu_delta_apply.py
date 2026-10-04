# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepared groups of byte-affine XOR destinations; imported only at admission."""

import ctypes
import math
import statistics
import time
from dataclasses import dataclass
from functools import cache, partial

import torch
import triton
import triton.language as tl


@cache
def _occupancy_driver():
    driver = ctypes.CDLL("libcuda.so.1")
    query = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor
    query.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    return driver


@cache
def _resident_ctas(kernel, device_index):
    """One resource-residency wave, cached per loaded kernel and CUDA device."""
    with torch.cuda.device(device_index):
        blocks = ctypes.c_int()
        status = _occupancy_driver().cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(blocks),
            ctypes.c_void_p(kernel.function),
            kernel.metadata.num_warps * 32,
            kernel.metadata.shared,
        )
        if status or blocks.value <= 0:
            raise RuntimeError(
                f"CUDA occupancy query failed: status={status}, blocks={blocks.value}"
            )
        return (
            blocks.value
            * torch.cuda.get_device_properties(device_index).multi_processor_count
        )


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
def _affine_offset_flat(
    index,
    CONTRACTS: tl.constexpr,
    SHAPE_START: tl.constexpr,
    STRIDE_START: tl.constexpr,
    AXES: tl.constexpr,
):
    offset = tl.full(index.shape, 0, tl.int64)
    rest = index
    for axis in tl.static_range(AXES - 1, 0, -1):
        if CONTRACTS[SHAPE_START + axis] != 1:
            offset += (rest % CONTRACTS[SHAPE_START + axis]) * CONTRACTS[
                STRIDE_START + axis
            ]
            rest //= CONTRACTS[SHAPE_START + axis]
    return offset + rest * CONTRACTS[STRIDE_START]


@triton.jit
def _apply_contract(
    pointers,
    tile,
    CONTRACTS: tl.constexpr,
    GROUP: tl.constexpr,
    AXES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    START: tl.constexpr = GROUP * (11 + 4 * AXES)
    SIZE: tl.constexpr = CONTRACTS[START]
    SOURCE_MODE: tl.constexpr = CONTRACTS[START + 1]
    TARGET_MODE: tl.constexpr = CONTRACTS[START + 2]
    SOURCE_ALIGNMENT: tl.constexpr = CONTRACTS[START + 3]
    TARGET_ALIGNMENT: tl.constexpr = CONTRACTS[START + 4]
    WORD: tl.constexpr = CONTRACTS[START + 5]
    COUNT: tl.constexpr = CONTRACTS[START + 6]
    POSITION: tl.constexpr = CONTRACTS[START + 7]
    FIRST: tl.constexpr = CONTRACTS[START + 8]
    DTYPE: tl.constexpr = tl.uint32 if WORD == 4 else tl.uint8
    PER_ITEM: tl.constexpr = triton.cdiv(SIZE, BLOCK)
    local = tile - FIRST
    item = local // PER_ITEM
    base = (local % PER_ITEM) * BLOCK
    source = tl.multiple_of(tl.load(pointers + POSITION + item), SOURCE_ALIGNMENT).to(
        tl.pointer_type(DTYPE)
    )
    target = tl.multiple_of(
        tl.load(pointers + POSITION + COUNT + item), TARGET_ALIGNMENT
    ).to(tl.pointer_type(DTYPE))
    lanes = tl.arange(0, BLOCK // WORD) * WORD
    index = base + lanes
    if SOURCE_MODE == 1:
        so = index
    elif SOURCE_MODE == 2:
        so = (
            _affine_offset_flat(base, CONTRACTS, START + 11, START + 11 + AXES, AXES)
            + lanes
        )
    else:
        so = _affine_offset_flat(index, CONTRACTS, START + 11, START + 11 + AXES, AXES)
    if TARGET_MODE == 1:
        to = index
    elif TARGET_MODE == 2:
        to = (
            _affine_offset_flat(
                base, CONTRACTS, START + 11 + 2 * AXES, START + 11 + 3 * AXES, AXES
            )
            + lanes
        )
    else:
        to = _affine_offset_flat(
            index, CONTRACTS, START + 11 + 2 * AXES, START + 11 + 3 * AXES, AXES
        )
    mask = index < SIZE
    value = tl.load(source + so // WORD, mask, other=0)
    previous = tl.load(target + to // WORD, mask, other=0)
    tl.store(target + to // WORD, value ^ previous, mask)


@triton.jit
def _xor_batch(
    pointers,
    error,
    CONTRACTS: tl.constexpr,
    GROUPS: tl.constexpr,
    AXES: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    PERSISTENT: tl.constexpr,
):
    if tl.load(error) == 0:
        tile = tl.program_id(0).to(tl.int64)
        while tile < TILES:
            for group in tl.static_range(GROUPS):
                if (tile >= CONTRACTS[group * (11 + 4 * AXES) + 8]) & (
                    tile < CONTRACTS[group * (11 + 4 * AXES) + 9]
                ):
                    _apply_contract(pointers, tile, CONTRACTS, group, AXES, BLOCK)
            if PERSISTENT:
                tile += tl.num_programs(0)
            else:
                tile = tl.full((), TILES, tl.int64)


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


def _access_mode(shape, stride, block):
    if stride == (1,):
        return 1
    return 2 if stride[-1] == 1 and shape[-1] % block == 0 else 0


def _word_bytes(contract, alignment):
    size, ss, st, ts, tt = contract
    aligned_rows = all(
        stride[-1] == 1
        and shape[-1] % 4 == 0
        and all(n == 1 or step % 4 == 0 for n, step in zip(shape[:-1], stride[:-1]))
        for shape, stride in ((ss, st), (ts, tt))
    )
    return 4 if size % 4 == 0 and min(alignment) >= 4 and aligned_rows else 1


@cache
def _parameters(contracts, alignments, block):
    """Flat constexpr records; no per-tensor geometry is uploaded to the GPU."""
    axes = max(max(len(c[1]), len(c[3])) for c, _ in contracts)
    values, first, position = [], 0, 0
    for (contract, count), alignment in zip(contracts, alignments):
        size, ss, st, ts, tt = contract
        end = first + count * triton.cdiv(size, block)
        values.extend(
            (
                size,
                _access_mode(ss, st, block),
                _access_mode(ts, tt, block),
                *alignment,
                _word_bytes(contract, alignment),
                count,
                position,
                first,
                end,
                axes,
            )
        )
        for shape, stride in ((ss, st), (ts, tt)):
            values.extend((*shape, *((1,) * (axes - len(shape)))))
            values.extend((*stride, *((0,) * (axes - len(stride)))))
        first, position = end, position + 2 * count
    return tuple(values), axes, first


@cache
def _compiled(contracts, alignments, device_index, config):
    block, warps, persistent = config
    values, axes, tiles = _parameters(contracts, alignments, block)
    with torch.cuda.device(device_index):
        kernel = _xor_batch.warmup(
            torch.int64,
            torch.int32,
            values,
            len(contracts),
            axes,
            tiles,
            block,
            persistent,
            grid=(1, 1, 1),
            num_warps=warps,
        )
        _ = kernel.run
        _resident_ctas(kernel, device_index)
        return kernel


def _grid(tiles, kernel, device_index, config):
    wave = _resident_ctas(kernel, device_index)
    return (min(tiles, 4 * wave) if config[2] else tiles, 1, 1)


# The measured naive policy also covers batches that cannot be tuned entirely
# within private decoded scratch. Apply has no configuration selection.
_CONFIGS = ((2048, 4, False), (2048, 4, True))
_choices = {}
_tune_bytes = {}


def _choose(contracts, alignments, scratch, error):
    device = scratch.device.index
    key = (contracts, alignments, device)
    if key in _choices:
        return _choices[key], 0, 0, 0.0, 1, 0
    spans = [
        tuple(
            (sum((n - 1) * s for n, s in zip(shape, stride)) + 16) // 16 * 16
            for shape, stride in ((c[1], c[2]), (c[3], c[4]))
        )
        for c, _ in contracts
    ]
    footprint = sum(
        count * (source + target)
        for (_, count), (source, target) in zip(contracts, spans)
    )
    if footprint > scratch.numel() or _tune_bytes.get(device, 0) + footprint > 4 << 30:
        _choices[key] = _CONFIGS[0]
        return _CONFIGS[0], 0, 0, 0.0, 0, 1
    started = time.perf_counter()
    _tune_bytes[device] = _tune_bytes.get(device, 0) + footprint
    # All counts and geometry are the real batch's. Only addresses change:
    # disjoint synthetic source/target spans borrow unused decoded scratch.
    scratch[:footprint].zero_()
    origin, rows = scratch.data_ptr(), []
    for (_, count), (source, target) in zip(contracts, spans):
        rows.extend(origin + i * source for i in range(count))
        origin += count * source
        rows.extend(origin + i * target for i in range(count))
        origin += count * target
    pointers = torch.tensor(rows, dtype=torch.int64, device=scratch.device)
    scores = []
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for config in _CONFIGS:
        kernel = _compiled(contracts, alignments, device, config)
        _, _, tiles = _parameters(contracts, alignments, config[0])
        launch = kernel[_grid(tiles, kernel, device, config)]
        timings = []
        for trial in range(4):
            start.record()
            launch(pointers, error)
            end.record()
            end.synchronize()
            if trial:
                timings.append(start.elapsed_time(end))
        scores.append(statistics.median(timings))
    chosen = _CONFIGS[min(range(len(scores)), key=scores.__getitem__)]
    _choices[key] = chosen
    return chosen, 1, footprint, time.perf_counter() - started, 0, 0


@dataclass
class ByteApplyGroup:
    """One layer launch, with exact uniform contracts resolved at compile time."""

    sources: list[int]
    targets: list[int]
    contracts: tuple
    kernel: object = None

    def pointer_rows(self, origin):
        position = 0
        for _, count in self.contracts:
            end = position + count
            yield from (origin + offset for offset in self.sources[position:end])
            yield from self.targets[position:end]
            position = end

    def prepare(self, scratch, error):
        if self.kernel is not None:
            return 0, 0, 0.0, 1, 0
        alignments, position = [], 0
        for _, count in self.contracts:
            end = position + count
            common = (
                math.gcd(*(scratch.data_ptr() + i for i in self.sources[position:end])),
                math.gcd(*self.targets[position:end]),
            )
            alignments.append(tuple(min(16, value & -value) for value in common))
            position = end
        self.alignments = tuple(alignments)
        self.config, tuned, footprint, elapsed, reused, skipped = _choose(
            self.contracts, self.alignments, scratch, error
        )
        self.kernel = _compiled(
            self.contracts, self.alignments, scratch.device.index, self.config
        )
        _, _, self.tiles = _parameters(self.contracts, self.alignments, self.config[0])
        self.grid = _grid(self.tiles, self.kernel, scratch.device.index, self.config)
        self.word32_contracts = sum(
            _word_bytes(contract, alignment) == 4
            for (contract, _), alignment in zip(self.contracts, self.alignments)
        )
        self.launch = self.kernel[self.grid]
        return tuned, footprint, elapsed, reused, skipped


def plan_apply(outputs):
    """One affine batch with shared constexpr contracts, without model names."""
    groups, transformed = {}, []
    for binding, offset, size in outputs:
        if not binding.destinations:
            transformed.append((binding, offset, size))
            continue
        canonical = torch.empty(size, dtype=torch.uint8, device="meta")
        source = binding.selected_bytes(canonical)
        ss, st = _normalize(source.shape, source.stride())
        for target in binding.destinations:
            ts, tt = _normalize(target.shape, target.stride())
            contract = (math.prod(ss), ss, st, ts, tt)
            sources, targets = groups.setdefault(contract, ([], []))
            sources.append(offset + source.storage_offset())
            targets.append(target.data_ptr())
    if not groups:
        return None, transformed
    sources, targets, contracts = [], [], []
    for contract, (group_sources, group_targets) in groups.items():
        contracts.append((contract, len(group_sources)))
        sources.extend(group_sources)
        targets.extend(group_targets)
    return ByteApplyGroup(sources, targets, tuple(contracts)), transformed
