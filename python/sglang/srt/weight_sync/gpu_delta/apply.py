# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepared groups of byte-affine XOR destinations; imported only at admission."""

import math
from dataclasses import dataclass, field
from functools import lru_cache, partial

import torch
import triton
import triton.language as tl

BLOCK = 2048
ELEMENTS = 512
WARPS = 4
RESIDENT_WAVES = 4
MAX_AXES = 64
_INT64_MAX = (1 << 63) - 1
_MAX_TILES = (1 << 31) - 1


@lru_cache(maxsize=None, typed=True)
def _resident_ctas(kernel, device_index):
    """One resource-residency wave, cached per loaded kernel and CUDA device."""
    from cuda.bindings import driver

    with torch.cuda.device(device_index):
        status, blocks = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            driver.CUfunction(int(kernel.function)),
            kernel.metadata.num_warps * 32,
            kernel.metadata.shared,
        )
        if status != driver.CUresult.CUDA_SUCCESS or blocks <= 0:
            raise RuntimeError(
                f"CUDA occupancy query failed: status={status}, blocks={blocks}"
            )
        return (
            blocks
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
    """Prepare metadata-only validation; the returned call enqueues on apply."""
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
def _scalar_quotient(numerator, denominator):
    # Runtime scalar row/base division: never narrow a large coordinate or axis.
    if numerator <= 0xFFFFFFFF:
        if denominator > 0xFFFFFFFF:
            quotient = tl.full((), 0, tl.int64)
        else:
            quotient = (numerator.to(tl.uint32) // denominator.to(tl.uint32)).to(
                tl.int64
            )
    else:
        quotient = numerator // denominator
    return quotient


@triton.jit
def _affine_offset(index, metadata, shape, stride, axes):
    offset = tl.full(index.shape, 0, tl.int64)
    rest = index
    axis = axes - 1
    while axis > 0:
        length = tl.load(metadata + shape + axis)
        step = tl.load(metadata + stride + axis)
        quotient = _scalar_quotient(rest, length)
        offset += (rest - quotient * length) * step
        rest = quotient
        axis -= 1
    return offset + rest * tl.load(metadata + stride)


@triton.jit
def _affine_lanes(base, delta, metadata, shape, stride, axes, MAX_DELTA: tl.constexpr):
    offset = tl.full(delta.shape, 0, tl.int64)
    axis = axes - 1
    while axis > 0:
        length = tl.load(metadata + shape + axis)
        step = tl.load(metadata + stride + axis)
        quotient = _scalar_quotient(base, length)
        remainder = base - quotient * length
        if length > MAX_DELTA:
            lane_quotient = tl.full(delta.shape, 0, tl.uint32)
            lane_remainder = delta
        else:
            lane_quotient = delta // length.to(tl.uint32)
            lane_remainder = delta - lane_quotient * length.to(tl.uint32)
        # b=qL+r, d=aL+t => coordinate=r+t-cL, next flat=q+a+c.
        # 0<=t<L gives a single carry; the new delta a+c never exceeds d.
        # Group subtraction before addition to avoid an overflowing r+t when
        # a valid axis approaches INT64_MAX. Only bounded deltas use uint32.
        carry = (lane_remainder.to(tl.int64) >= length - remainder).to(tl.uint32)
        coordinate = remainder + (
            lane_remainder.to(tl.int64) - carry.to(tl.int64) * length
        )
        offset += coordinate * step
        base = quotient
        delta = lane_quotient + carry
        axis -= 1
    return offset + (base + delta.to(tl.int64)) * tl.load(metadata + stride)


@triton.jit
def _load_source(
    source,
    metadata,
    shape,
    axes,
    mode,
    base,
    size,
    WORD: tl.constexpr,
    ELEMENTS: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    PACKED: tl.constexpr,
):
    lanes = tl.arange(0, ELEMENTS)
    index = base + lanes * WORD
    mask = index < size
    # Keep the actual memory operations in the uniform access-mode branches.
    # Merging an arbitrary affine offset before a common load loses the
    # contiguous branch's vector-address proof. Only base has a fixed multiple;
    # arbitrary outer strides in row/affine mappings keep their true alignment.
    if mode == 1:
        offset = tl.multiple_of(base // WORD, ELEMENTS) + lanes
        value = tl.load(source + offset, mask, other=0)
    elif PACKED or mode == 2:
        if PACKED:
            start = _packed_row_start(base, metadata, shape, ELEMENTS * WORD)
        else:
            start = _affine_offset(base, metadata, shape, shape + axes, axes)
        start = tl.multiple_of(start // WORD, ALIGNMENT // WORD)
        value = tl.load(source + start + lanes, mask, other=0)
    else:
        offset = _affine_lanes(
            base,
            (lanes * WORD).to(tl.uint32),
            metadata,
            shape,
            shape + axes,
            axes,
            (ELEMENTS - 1) * WORD,
        )
        value = tl.load(source + offset // WORD, mask, other=0)
    return value


@triton.jit
def _xor_target(
    target,
    value,
    metadata,
    shape,
    axes,
    mode,
    base,
    size,
    WORD: tl.constexpr,
    ELEMENTS: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    PACKED: tl.constexpr,
):
    lanes = tl.arange(0, ELEMENTS)
    index = base + lanes * WORD
    mask = index < size
    if mode == 1:
        offset = tl.multiple_of(base // WORD, ELEMENTS) + lanes
        previous = tl.load(target + offset, mask, other=0)
        tl.store(target + offset, value ^ previous, mask)
    elif PACKED or mode == 2:
        if PACKED:
            start = _packed_row_start(base, metadata, shape, ELEMENTS * WORD)
        else:
            start = _affine_offset(base, metadata, shape, shape + axes, axes)
        start = tl.multiple_of(start // WORD, ALIGNMENT // WORD)
        offset = start + lanes
        previous = tl.load(target + offset, mask, other=0)
        tl.store(target + offset, value ^ previous, mask)
    else:
        offset = (
            _affine_lanes(
                base,
                (lanes * WORD).to(tl.uint32),
                metadata,
                shape,
                shape + axes,
                axes,
                (ELEMENTS - 1) * WORD,
            )
            // WORD
        )
        previous = tl.load(target + offset, mask, other=0)
        tl.store(target + offset, value ^ previous, mask)


@triton.jit
def _packed_row_start(base, metadata, shape, BLOCK: tl.constexpr):
    # Packed row contracts are exactly two axes; contiguous sides bypass this.
    # Active total tiles <= INT32_MAX and the inner row is BLOCK-divisible.
    # Keep only tile division uint32; byte coordinates/strides remain int64.
    inner = tl.load(metadata + shape + 1)
    outer_stride = tl.load(metadata + shape + 2)
    tile = (base // BLOCK).to(tl.uint32)
    row_tiles = (inner // BLOCK).to(tl.uint32)
    row = (tile // row_tiles).to(tl.int64)
    return row * outer_stride + (base - row * inner)


@triton.jit
def _apply_record(
    pointers,
    metadata,
    record,
    tile,
    tile_stride,
    WORD: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    BLOCK: tl.constexpr,
    FAMILY: tl.constexpr,
):
    size = tl.load(metadata + record)
    source_mode = tl.load(metadata + record + 1)
    target_mode = tl.load(metadata + record + 2)
    count = tl.load(metadata + record + 6)
    position = tl.load(metadata + record + 7)
    first = tl.load(metadata + record + 8)
    end = tl.load(metadata + record + 9)
    axes = tl.load(metadata + record + 10)
    # Active contracts have at most INT32_MAX tiles by admission. Only byte
    # offsets/affine coordinates need int64; variable tile division uses uint32.
    per_item = tl.cdiv(size, BLOCK).to(tl.uint32)
    DTYPE: tl.constexpr = tl.uint32 if WORD == 4 else tl.uint8
    ELEMENTS: tl.constexpr = 512
    # A CTA may visit several tiles of this group. Keep record selection and
    # scalar geometry outside that loop, then search again only across END.
    while tile < end:
        local = (tile - first).to(tl.uint32)
        item = local // per_item
        base = (local - item * per_item).to(tl.int64) * BLOCK
        source = tl.multiple_of(tl.load(pointers + position + item), ALIGNMENT).to(
            tl.pointer_type(DTYPE)
        )
        target = tl.multiple_of(
            tl.load(pointers + position + count + item), ALIGNMENT
        ).to(tl.pointer_type(DTYPE))
        shape = record + 11
        if FAMILY == 1:
            # This family's coordinate descriptors are uint32-word based.
            # CPU preparation validates/classifies the original byte contract;
            # size/tile accounting stays byte-based. The source is contiguous.
            lanes = tl.arange(0, ELEMENTS)
            index = tl.multiple_of(base // 4, ELEMENTS) + lanes
            target_shape = shape + 2 * axes
            offset = _affine_lanes(
                base // 4,
                lanes.to(tl.uint32),
                metadata,
                target_shape,
                target_shape + axes,
                axes,
                ELEMENTS - 1,
            )
            mask = base + lanes * 4 < size
            value = tl.load(source + index, mask, other=0)
            previous = tl.load(target + offset, mask, other=0)
            tl.store(target + offset, value ^ previous, mask)
        else:
            # Keep loads/stores inside their uniform access-mode branches:
            # merging row starts first loses the contiguous address proof.
            # Packed WORD4 uses one subtile; general byte fallback uses four.
            part = 0
            while part < 4 // WORD:
                subbase = base + part * ELEMENTS * WORD
                value = _load_source(
                    source,
                    metadata,
                    shape,
                    axes,
                    source_mode,
                    subbase,
                    size,
                    WORD,
                    ELEMENTS,
                    ALIGNMENT,
                    FAMILY == 0,
                )
                _xor_target(
                    target,
                    value,
                    metadata,
                    shape + 2 * axes,
                    axes,
                    target_mode,
                    subbase,
                    size,
                    WORD,
                    ELEMENTS,
                    ALIGNMENT,
                    FAMILY == 0,
                )
                part += 1
        tile += tile_stride
    return tile


@triton.jit
def _xor_batch(pointers, metadata, error, BLOCK: tl.constexpr, FAMILY: tl.constexpr):
    # No pointer or model access may precede the sticky decode-status gate.
    if tl.load(error) == 0:
        tile = tl.program_id(0).to(tl.int64)
        tile_stride = tl.num_programs(0).to(tl.int64)
        groups = tl.load(metadata)
        total = tl.load(metadata + groups)  # final END; metadata[0] is0 if empty
        while tile < total:
            low = tl.full((), 0, tl.int64)
            high = groups
            # First cumulative END greater than tile. Empty groups are skipped.
            while low < high:
                middle = (low + high) // 2
                end = tl.load(metadata + 1 + middle)
                if tile < end:
                    high = middle
                else:
                    low = middle + 1
            record = tl.load(metadata + 1 + groups + low)
            word = tl.load(metadata + record + 5)
            if FAMILY != 2 or word == 4:
                source_alignment = tl.load(metadata + record + 3)
                target_alignment = tl.load(metadata + record + 4)
                if (source_alignment >= 16) & (target_alignment >= 16):
                    tile = _apply_record(
                        pointers,
                        metadata,
                        record,
                        tile,
                        tile_stride,
                        4,
                        16,
                        BLOCK,
                        FAMILY,
                    )
                else:
                    tile = _apply_record(
                        pointers,
                        metadata,
                        record,
                        tile,
                        tile_stride,
                        4,
                        4,
                        BLOCK,
                        FAMILY,
                    )
            else:
                tile = _apply_record(
                    pointers, metadata, record, tile, tile_stride, 1, 1, BLOCK, FAMILY
                )


@lru_cache(maxsize=None, typed=True)
def _compiled(device_index, family):
    """Three fixed access families per device, all loaded before pause."""
    if (
        type(device_index) is not int
        or not 0 <= device_index < torch.cuda.device_count()
    ):
        raise ValueError("invalid GPU delta apply device")
    if type(family) is not int or family not in (0, 1, 2):
        raise ValueError("invalid GPU delta apply family")
    with torch.cuda.device(device_index):
        kernel = _xor_batch.warmup(
            torch.int64,
            torch.int64,
            torch.int32,
            BLOCK,
            family,
            grid=(1, 1, 1),
            num_warps=WARPS,
        )
        _ = kernel.run
        _resident_ctas(kernel, device_index)
        return kernel


def _normalize(shape, stride):
    """Keep the same flat element mapping with fewer affine axes."""
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
    """Classify normalized byte geometry without model or format dimensions.

    0: arbitrary affine coordinates; each lane computes its runtime byte offset.
    1: contiguous bytes; flat logical and physical offsets are identical.
    2: contiguous innermost rows whose lengths are block-divisible, so a tile
       cannot cross a row boundary; only the tile's row start needs remapping.
    """
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


def _row_alignment(shape, stride, alignment):
    if _access_mode(shape, stride, BLOCK) != 2:
        return alignment
    # The inner row coordinate is BLOCK aligned. Include every active outer
    # stride so a16B base pointer never overstates a4B-aligned row start.
    return math.gcd(
        alignment, *(step for n, step in zip(shape[:-1], stride[:-1]) if n > 1)
    )


def _validate_contract(contract, count):
    size, ss, st, ts, tt = contract
    if (
        type(count) is not int
        or not 0 <= count <= _INT64_MAX
        or type(size) is not int
        or not 0 < size <= _INT64_MAX
    ):
        raise ValueError("invalid GPU delta apply size/count")
    for shape, stride in ((ss, st), (ts, tt)):
        if not 1 <= len(shape) <= MAX_AXES or len(shape) != len(stride):
            raise ValueError("unsupported GPU delta affine axis count")
        if any(type(n) is not int or not 0 < n <= _INT64_MAX for n in shape) or any(
            type(step) is not int or not 0 <= step <= _INT64_MAX for step in stride
        ):
            raise ValueError("invalid GPU delta affine geometry")
        if (
            math.prod(shape) != size
            or sum((n - 1) * step for n, step in zip(shape, stride)) > _INT64_MAX
        ):
            raise ValueError("invalid GPU delta affine extent")
    if count * triton.cdiv(size, BLOCK) > _INT64_MAX:
        raise ValueError("GPU delta apply tile count overflow")


def _record(contract, count, alignment, position):
    """Prepare one record from its original byte contract and pointer position.

    Classify once in byte units. Only word-affine coordinate descriptors are then
    normalized into uint32-word units; size/count/tile fields remain byte-based.
    """
    _validate_contract(contract, count)
    if len(alignment) != 2 or any(a not in (1, 2, 4, 8, 16) for a in alignment):
        raise ValueError("invalid GPU delta apply pointer alignment")
    if type(position) is not int or not 0 <= position <= _INT64_MAX - 2 * count:
        raise ValueError("GPU delta apply pointer position overflow")
    size, ss, st, ts, tt = contract
    source_mode = _access_mode(ss, st, BLOCK)
    target_mode = _access_mode(ts, tt, BLOCK)
    alignment = (
        _row_alignment(ss, st, alignment[0]),
        _row_alignment(ts, tt, alignment[1]),
    )
    word = _word_bytes(contract, alignment)
    if (
        word == 4
        and source_mode in (1, 2)
        and target_mode in (1, 2)
        and (source_mode == 1 or len(ss) == 2)
        and (target_mode == 1 or len(ts) == 2)
    ):
        family = 0
    elif word == 4 and source_mode == 1:
        family = 1
        # Discard inactive axes before dividing potentially arbitrary strides.
        # WORD4 admission proves divisibility of every active outer stride and
        # the contiguous inner length. A four-byte inner axis becomes unit.
        ts, tt = _normalize(ts, tt)
        ts, tt = _normalize(
            (*ts[:-1], ts[-1] // 4),
            (*(step // 4 for step in tt[:-1]), 1),
        )
        ss, st = (size // 4,), (1,)
    else:
        family = 2
    axes = max(len(ss), len(ts))
    values = [
        size,
        source_mode,
        target_mode,
        *alignment,
        word,
        count,
        position,
        0,
        0,
        axes,
    ]
    for shape, stride in ((ss, st), (ts, tt)):
        values.extend((*shape, *((1,) * (axes - len(shape)))))
        values.extend((*stride, *((0,) * (axes - len(stride)))))
    return family, tuple(values)


def _metadata(records):
    """Assemble one family's END/index header and records, padded to 16 bytes."""
    values = [len(records)] + [0] * (2 * len(records))
    first = 0
    for index, record in enumerate(records):
        end = first + record[6] * triton.cdiv(record[0], BLOCK)
        if end > _MAX_TILES:
            raise ValueError("GPU delta apply metadata overflow")
        values[1 + index] = end
        values[1 + len(records) + index] = len(values)
        values.extend((*record[:8], first, end, *record[10:]))
        first = end
    # Preserve the precompiled pointer signature for the next slab slice.
    if len(values) % 2:
        values.append(0)
    return values, first


@dataclass
class ApplyProgram:
    family: str
    kernel: object
    metadata_start: int
    metadata_stop: int
    tiles: int
    grid: tuple
    resident_ctas: int


def _launch_prebound(calls):
    for call in calls:
        call()


@dataclass
class ByteApplyGroup:
    """A runtime batch partitioned into three fixed, prewarmed access families."""

    sources: list[int]
    targets: list[int]
    contracts: tuple
    kernels: tuple = ()
    _prepared_device: int | None = field(default=None, init=False, repr=False)

    def prepare_metadata(self, device_index):
        """Prepare immutable device metadata once; return whether it was cached."""
        if type(device_index) is int and device_index == self._prepared_device:
            self._restore_aligned_binding()
            return True
        self._prepared_device = None
        for contract, count in self.contracts:
            _validate_contract(contract, count)
        if sum(count for _, count in self.contracts) != len(self.sources) or len(
            self.sources
        ) != len(self.targets):
            raise ValueError("GPU delta apply pointer count mismatch")
        if any(
            type(value) is not int or not 0 <= value <= _INT64_MAX
            for value in self.sources
        ) or any(
            type(value) is not int or not 0 < value <= _INT64_MAX
            for value in self.targets
        ):
            raise ValueError("invalid GPU delta apply pointer/offset")
        if (
            sum(count * triton.cdiv(c[0], BLOCK) for c, count in self.contracts)
            > _MAX_TILES
        ):
            raise ValueError("GPU delta apply launch grid overflow")
        # Capacity belongs to the validated immutable original byte geometry.
        self.contracts = tuple(
            ((c[0], *(tuple(part) for part in c[1:])), count)
            for c, count in self.contracts
        )
        words = (
            1
            + 2 * len(self.contracts)
            + sum(11 + 4 * max(len(c[1]), len(c[3])) for c, _ in self.contracts)
        )
        self.metadata_size = words + words % 2 + 4
        self.kernels = tuple(_compiled(device_index, family) for family in range(3))
        self.resident_ctas = tuple(
            _resident_ctas(kernel, device_index) for kernel in self.kernels
        )
        # Snapshot geometry: no caller-owned mutable pointer lists enter the
        # preplanned descriptor template. Actual scratch addresses remain unknown.
        self.sources, self.targets = tuple(self.sources), tuple(self.targets)
        self.source_limit = self.source_max_offset = 0
        alignments, indices, position, pointer_position = [], [], 0, 0
        for contract, count in self.contracts:
            end = position + count
            offsets, targets = self.sources[position:end], self.targets[position:end]
            span = 1 + sum((n - 1) * step for n, step in zip(contract[1], contract[2]))
            if offsets:
                maximum = max(offsets)
                self.source_limit = max(self.source_limit, maximum + span)
                self.source_max_offset = max(self.source_max_offset, maximum)
            # For origin %16 ==0, capped gcd(origin+offsets) equals gcd(16,offsets).
            # Empty contracts retain the existing alignment1 convention.
            alignments.append(
                (math.gcd(16, *offsets), math.gcd(16, *targets)) if count else (1, 1)
            )
            indices.extend(range(pointer_position, pointer_position + count))
            pointer_position += 2 * count
            position = end
        self.source_indices = tuple(indices)
        self._plan_binding(alignments)
        self._aligned_binding = (
            tuple(self.metadata_rows),
            tuple(self.programs),
            self.tiles,
            self.word32_contracts,
            self.grid_ctas,
        )
        self._binding_aligned = True
        self._prepared_device = device_index
        return False

    def _restore_aligned_binding(self):
        if not self._binding_aligned:
            rows, programs, tiles, words, grid = self._aligned_binding
            # A fallback may replace the current descriptors; retain the
            # immutable aligned template across preparation and binding.
            self.metadata_rows, self.programs = list(rows), list(programs)
            self.tiles, self.word32_contracts, self.grid_ctas = tiles, words, grid
            self._binding_aligned = True

    def pointer_rows(self, origin):
        position = 0
        for _, count in self.contracts:
            end = position + count
            yield from (origin + offset for offset in self.sources[position:end])
            yield from self.targets[position:end]
            position = end

    def bind(self, scratch):
        """Validate actual scratch, reusing prepared geometry when 16B aligned.

        Allocation alignment is checked, never assumed. The uncommon misaligned
        caller still uses the same conservative runtime descriptor construction.
        """
        if self._prepared_device is None:
            raise RuntimeError("GPU delta apply kernel must be prepared before pause")
        origin = scratch.data_ptr()
        if scratch.numel() < self.source_limit:
            raise ValueError("GPU delta apply source exceeds decoded scratch")
        if (
            type(origin) is not int
            or not 0 <= origin <= _INT64_MAX
            or (
                self.sources
                and (origin == 0 or origin + self.source_max_offset > _INT64_MAX)
            )
        ):
            raise ValueError("invalid GPU delta apply source pointer")
        if origin % 16 == 0:
            self._restore_aligned_binding()
            return True
        alignments, position = [], 0
        for _, count in self.contracts:
            end = position + count
            addresses = [origin + i for i in self.sources[position:end]]
            common = (math.gcd(*addresses), math.gcd(*self.targets[position:end]))
            alignments.append(tuple(min(16, v & -v) if v else 1 for v in common))
            position = end
        self._plan_binding(alignments)
        self._binding_aligned = False
        return False

    def _plan_binding(self, alignments):
        """Build descriptors/launches before pause, or for a misaligned fallback."""
        # Keep the single global pointer table in its original order. Only
        # descriptor records are partitioned, with original pointer positions.
        if len(alignments) != len(self.contracts):
            raise ValueError("GPU delta apply alignment count mismatch")
        position, families, self.word32_contracts = 0, [[], [], []], 0
        for (contract, count), alignment in zip(self.contracts, alignments):
            family, record = _record(contract, count, alignment, position)
            families[family].append(record)
            self.word32_contracts += record[5] == 4
            position += 2 * count
        self.metadata_rows, self.programs, self.tiles = [], [], 0
        for family, records in enumerate(families):
            rows, tiles = _metadata(records)
            start = len(self.metadata_rows)
            self.metadata_rows.extend(rows)
            self.tiles += tiles
            if tiles:
                resident = self.resident_ctas[family]
                # Keep the fixed family policy: word-affine uses a full grid;
                # packed/general reuse descriptors over four occupancy waves.
                # Broader word-affine admission requires fresh measurements.
                grid_ctas = (
                    tiles if family == 1 else min(tiles, RESIDENT_WAVES * resident)
                )
                self.programs.append(
                    ApplyProgram(
                        ("packed", "word_affine", "general")[family],
                        self.kernels[family],
                        start,
                        len(self.metadata_rows),
                        tiles,
                        (grid_ctas, 1, 1),
                        resident,
                    )
                )
        self.metadata_rows.extend([0] * (self.metadata_size - len(self.metadata_rows)))
        if len(self.metadata_rows) != self.metadata_size:
            raise RuntimeError("GPU delta family metadata capacity exceeded")
        self.grid_ctas = sum(program.grid[0] for program in self.programs)

    def bind_launch(self, pointers, metadata, error, grids=None):
        """Bind retained metadata views once; optional grids support validation."""
        if grids is None:
            grids = tuple(program.grid for program in self.programs)
        if len(grids) != len(self.programs) or any(
            len(grid) != 3
            or grid[1:] != (1, 1)
            or type(grid[0]) is not int
            or not 0 < grid[0] <= _MAX_TILES
            for grid in grids
        ):
            raise ValueError("invalid GPU delta family launch grids")
        calls = tuple(
            partial(
                program.kernel[grid],
                pointers,
                metadata[program.metadata_start : program.metadata_stop],
                error,
            )
            for program, grid in zip(self.programs, grids)
        )
        return partial(_launch_prebound, calls) if calls else None


def plan_apply(outputs):
    """One affine batch with runtime contracts, without model names."""
    groups = {}
    for binding, offset, size in outputs:
        if not binding.destinations:
            raise ValueError("GPU delta apply requires affine byte destinations")
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
        return None
    sources, targets, contracts = [], [], []
    for contract, (group_sources, group_targets) in groups.items():
        contracts.append((contract, len(group_sources)))
        sources.extend(group_sources)
        targets.extend(group_targets)
    return ByteApplyGroup(sources, targets, tuple(contracts))
