"""Encode BF16 rows in the signed, 16-scale DSv4.1 cache format on Ascend."""

import torch
import triton
import triton.language as tl


@triton.jit
def _positive_bf16(value):
    """Round nonnegative FP32 products exactly as the BF16 cache consumer does."""
    bits = value.to(tl.int32, bitcast=True)
    rounded = (bits + 32767 + ((bits >> 16) & 1)) & -65536
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _scales(peak, denom):
    raw = tl.where(peak > 0.0, peak, 1.0) / denom.to(tl.float32)
    wide = (raw < 0.00006103515625) | (raw > 65504.0)
    # Keep the arithmetic scale identical to the bits placed on the wire.
    # Normal half values use ten fraction bits and an exponent bias of 15.
    fp32 = raw.to(tl.int32, bitcast=True)
    half = ((fp32 + 4095 + ((fp32 >> 13) & 1)) >> 13) - (112 << 10)
    bfloat = (fp32 + 32767 + ((fp32 >> 16) & 1)) >> 16
    bits = tl.where(wide, bfloat | 32768, half)
    restored = tl.where(wide, bfloat << 16, (half + (112 << 10)) << 13)
    scale = restored.to(tl.float32, bitcast=True)
    min_bfloat16 = tl.full((), 65536, tl.int32).to(tl.float32, bitcast=True)
    return tl.maximum(scale, min_bfloat16), bits


@triton.jit
def _levels(magnitude, scale):
    # The wire format uses signed, symmetric INT8 with ties rounded to even.
    value = tl.minimum(magnitude / scale, 127.0)
    lower = tl.floor(value)
    fraction = value - lower
    up = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    return lower + up.to(tl.float32)


@triton.jit
def _fit(
    Input,
    MetaWords,
    Exact,
    size,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    STRIDE: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    FAST: tl.constexpr,
    LANES: tl.constexpr,
):
    positions = tl.arange(0, BLOCK)
    row_positions = tl.arange(0, ROW_BLOCK)
    candidate = tl.arange(0, LANES)
    groups: tl.constexpr = WIDTH // GROUP
    for work in range(tl.program_id(0), size, tl.num_programs(0)):
        take_group = True
        if not FAST:
            take_group = tl.load(Exact + work) == 0
        if take_group:
            row = work // groups
            group = work % groups
            source = Input + row.to(tl.int64) * STRIDE
            full = tl.load(source + row_positions, row_positions < WIDTH, 0).to(
                tl.float32
            )
            maximum = tl.max(tl.abs(full), 0)
            # Extract the same power-of-two BF16 row factor as the torch encoder.
            exponent = (
                maximum.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.int32) & 32640
            )
            exponent = tl.maximum(exponent, 128)
            factor = exponent.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            original = tl.load(
                source + group * GROUP + positions, positions < GROUP, 0
            ).to(tl.float32)
            magnitude = tl.abs(original) / factor
            norm = tl.maximum(tl.max(magnitude, 0), 5.960464477539063e-8)
            error = tl.where(positions < GROUP, float("inf"), 0.0)
            for slot in range(16):
                # FP16 0.125 is the first reference candidate for an exact group.
                scale_word = 12288
                if tl.max(error, 0) > 0.0:
                    peak = tl.max(tl.where(error > 0.0, magnitude, 0.0), 0)
                    best_score = float("inf")
                    best_denom = 8
                    passes: tl.constexpr = 1 if FAST else triton.cdiv(120, LANES)
                    for index in range(passes):
                        if FAST:
                            denoms = tl.minimum(8 + candidate * (128 // LANES), 127)
                        else:
                            denoms = 8 + index * LANES + candidate
                        trial, unused = _scales(peak, denoms)
                        levels = _levels(magnitude[None, :], trial[:, None])
                        restored = _positive_bf16(levels * trial[:, None])
                        combined = tl.minimum(
                            tl.abs(restored - magnitude[None, :]) / norm, error[None, :]
                        )
                        square = tl.sum(combined * combined, 1)
                        count = tl.sum((combined != 0.0).to(tl.float32), 1)
                        score = tl.where(
                            slot == 15, square, count + tl.minimum(square, 0.99)
                        )
                        score = tl.where(denoms <= 127, score, float("inf"))
                        smallest = tl.min(score, 0)
                        denom = tl.min(tl.where(score == smallest, denoms, 128), 0)
                        best_denom = tl.where(smallest < best_score, denom, best_denom)
                        best_score = tl.minimum(best_score, smallest)
                    scale, bits = _scales(peak, best_denom)
                    scale_word = bits
                    level = _levels(magnitude, scale)
                    restored = _positive_bf16(level * scale)
                    diff = tl.abs(restored - magnitude) / norm
                    error = tl.minimum(error, diff)
                tl.store(
                    MetaWords + row.to(tl.int64) * WORDS + group * 16 + slot,
                    scale_word.to(tl.int16),
                )
            if FAST:
                tl.store(Exact + work, (tl.max(error, 0) == 0.0).to(tl.int32))
            if group == 0:
                tl.store(
                    MetaWords + row.to(tl.int64) * WORDS + WORDS - 1,
                    exponent.to(tl.int16),
                )


@triton.jit
def _apply_palette(
    Input,
    ScaleWords,
    Values,
    Selectors,
    size,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    STRIDE: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Encode from the stored (rounded) palette. Every lane selects the first
    # scale with minimum error after reconstruction in the consumer's BF16.
    slots = tl.arange(0, 16)
    groups: tl.constexpr = WIDTH // GROUP
    tiles: tl.constexpr = triton.cdiv(GROUP, BLOCK)
    for job in range(tl.program_id(0), size, tl.num_programs(0)):
        record = job // tiles
        row = record // groups
        group = record % groups
        columns = (job % tiles) * BLOCK + tl.arange(0, BLOCK)
        offset = group * GROUP + columns
        source = tl.load(
            Input + row.to(tl.int64) * STRIDE + offset, columns < GROUP, 0
        ).to(tl.float32)
        words = ScaleWords + row.to(tl.int64) * WORDS
        factor = tl.load(words + WORDS - 1).to(tl.bfloat16, bitcast=True).to(tl.float32)
        word = tl.load(words + group * 16 + slots)
        magnitude = (word.to(tl.int32) & 32767).to(tl.uint16)
        palette = tl.where(
            word < 0,
            magnitude.to(tl.bfloat16, bitcast=True).to(tl.float32),
            magnitude.to(tl.float16, bitcast=True).to(tl.float32),
        )
        normalized = tl.abs(source) / factor
        tiny = tl.full((), 65536, tl.int32).to(tl.float32, bitcast=True)
        levels = _levels(normalized[None, :], tl.maximum(palette[:, None], tiny))
        restored = _positive_bf16(levels * palette[:, None])
        error = tl.abs(restored - normalized[None, :])
        minimum = tl.min(error, 0)
        select = tl.min(tl.where(error == minimum[None, :], slots[:, None], 16), 0)
        chosen = tl.sum(tl.where(slots[:, None] == select[None, :], levels, 0.0), 0)
        signed = tl.where(source < 0.0, -chosen, chosen)
        dst = row.to(tl.int64) * WIDTH + offset
        tl.store(Values + dst, signed.to(tl.int8), columns < GROUP)
        tl.store(Selectors + dst, select.to(tl.uint8), columns < GROUP)


def encode_rows(rows, group_size):
    """Return signed codes and packed scales/selectors without a host synchronization.

    A narrow candidate search is accepted only when it reconstructs every BF16
    value in the group exactly. Remaining groups run the full 120-candidate
    search. Both passes keep their shapes fixed for ACL graph replay.
    """
    width = rows.shape[-1]
    if rows.dtype != torch.bfloat16 or (width, group_size) not in (
        (128, 128),
        (448, 224),
    ):
        raise ValueError("Expected DSv4.1 BF16 index or non-rotary KV rows")
    source = rows.reshape(-1, width)
    if source.stride(1) != 1:
        source = source.contiguous()
    groups = width // group_size
    words = groups * 16 + width // 4 + 1
    values = torch.empty(rows.shape, dtype=torch.int8, device=rows.device)
    meta = torch.empty(
        (*rows.shape[:-1], words * 2), dtype=torch.uint8, device=rows.device
    )
    count = source.shape[0] * groups
    if count:
        exact = torch.empty(count, dtype=torch.int32, device=rows.device)
        padded_words = triton.cdiv(groups * 16 + 1, 16) * 16
        scale_words = torch.empty(
            (source.shape[0], padded_words), dtype=torch.int16, device=rows.device
        )
        args = (
            source,
            scale_words,
            exact,
            count,
            width,
            group_size,
            source.stride(0),
            padded_words,
            triton.next_power_of_2(group_size),
            triton.next_power_of_2(width),
        )
        _fit[(min(count, 48),)](*args, True, 16, enable_fp_fusion=False)
        _fit[(min(count, 48),)](*args, False, 8, enable_fp_fusion=False)
        target = meta.view(source.shape[0], -1)
        scale_bytes = groups * 32
        target[:, :scale_bytes].copy_(scale_words[:, : groups * 16].view(torch.uint8))
        target[:, -2:].view(torch.int16).copy_(scale_words[:, -1:])
        selectors = torch.empty(
            (source.shape[0], width), dtype=torch.uint8, device=rows.device
        )
        block = 32
        tiles = count * triton.cdiv(group_size, block)
        _apply_palette[(min(tiles, 48),)](
            source,
            scale_words,
            values,
            selectors,
            tiles,
            width,
            group_size,
            source.stride(0),
            padded_words,
            block,
            enable_fp_fusion=False,
        )
        pairs = selectors.reshape(source.shape[0], width // 2, 2)
        target[:, scale_bytes:-2].copy_(pairs[..., 0] + 16 * pairs[..., 1])
    return values, meta
