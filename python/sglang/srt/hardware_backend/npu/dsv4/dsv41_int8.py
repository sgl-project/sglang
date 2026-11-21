"""Grouped signed INT8 storage and active-page translation for DeepSeek-V4.1."""

import torch

SCALES_PER_GROUP = 16
# Limit temporary candidate tensors to about seven MiB of FP32 per tile.
_TILE_COMPONENTS = 14336


def metadata_bytes(width: int, group_size: int) -> int:
    """Two-byte group scales, two selectors per byte, and one BF16 row factor."""
    return 2 * (width // group_size) * SCALES_PER_GROUP + width // 2 + 2


def _fit_shared_scales(groups, denominators):
    magnitude = groups.abs()
    norm = magnitude.amax(-1, keepdim=True).clamp_min(2**-24)
    error = torch.full_like(groups, torch.inf)
    values = torch.zeros_like(groups, dtype=torch.int8)
    selectors = torch.zeros_like(groups, dtype=torch.int32)
    scales = []
    for slot in range(SCALES_PER_GROUP):
        # Anchor each scale to the largest value still reconstructed inexactly.
        peak = torch.where(error > 0, magnitude, 0).amax(-1, keepdim=True)
        raw = torch.where(peak > 0, peak, 1.0) / denominators
        wide = (raw < 2**-14) | (raw > 65504)
        half, bfloat = raw.half(), raw.bfloat16()
        # Positive scales need no sign bit. Use it as a format flag for
        # magnitudes where FP16 would have fewer significand bits than BF16.
        bits = torch.where(
            wide, bfloat.view(torch.int16).int() - 32768, half.view(torch.int16).int()
        ).short()
        choices = torch.where(wide, bfloat.float(), half.float()).clamp_min(2**-133)
        candidate_scale = choices.unsqueeze(-1)
        levels = (groups.unsqueeze(1) / candidate_scale).round().clamp_(-127, 127)
        restored = (levels * candidate_scale).bfloat16().float()
        candidate_error = (restored - groups.unsqueeze(1)).abs() / norm.unsqueeze(1)
        combined = torch.minimum(candidate_error, error.unsqueeze(1))
        loss = combined.square().sum(-1)
        if slot < SCALES_PER_GROUP - 1:
            # Cover exact BF16 values first. At the last scale minimize the
            # remaining error under the BF16 precision consumed by sparse MLA.
            loss = (combined != 0).sum(-1).float() + loss.clamp_max_(0.99)
        which = loss.argmin(-1, keepdim=True)
        scale = choices.gather(1, which)
        levels = (groups / scale.float()).round().clamp_(-127, 127).to(torch.int8)
        restored = (levels.float() * scale.float()).bfloat16().float()
        candidate_error = (restored - groups).abs() / norm
        take = candidate_error < error
        values = torch.where(take, levels, values)
        selectors = torch.where(take, slot, selectors)
        error = torch.minimum(error, candidate_error)
        scales.append(bits.gather(1, which))
    return values, torch.cat(scales, dim=-1), selectors


def encode_int8(
    rows: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    width = rows.shape[-1]
    if (
        rows.dtype != torch.bfloat16
        or group_size < 1
        or width % group_size
        or width % 2
    ):
        raise ValueError("Expected BF16 rows divisible by the group size and two")
    values = rows.new_empty(rows.shape, dtype=torch.int8)
    metadata = rows.new_empty(
        (*rows.shape[:-1], metadata_bytes(width, group_size)), dtype=torch.uint8
    )
    if not rows.numel():
        return values, metadata
    inputs = rows.reshape(-1, width)
    output = values.view(-1, width)
    meta = metadata.view(-1, metadata.shape[-1])
    count = max(1, _TILE_COMPONENTS // width)
    scale_bytes = 2 * (width // group_size) * SCALES_PER_GROUP
    denominators = torch.arange(8, 128, device=rows.device, dtype=torch.float32)
    for start in range(0, inputs.shape[0], count):
        tile = inputs[start : start + count]
        peak = tile.abs().amax(-1, keepdim=True)
        # A BF16 exponent supplies an exact power of two. This keeps finite
        # model activations within the range of the per-group FP16 scales.
        factor = (peak.view(torch.int16) & 0x7F80).view(torch.bfloat16)
        factor = factor.clamp_min(torch.finfo(torch.bfloat16).tiny)
        groups = (tile.float() / factor.float()).reshape(-1, group_size)
        quantized, scales, selectors = _fit_shared_scales(groups, denominators)
        target = meta[start : start + count]
        output[start : start + count].copy_(quantized.reshape_as(tile))
        target[:, :scale_bytes].view(torch.int16).copy_(
            scales.reshape(tile.shape[0], -1)
        )
        pairs = selectors.reshape(tile.shape[0], width // 2, 2)
        target[:, scale_bytes:-2].copy_((pairs[..., 0] + 16 * pairs[..., 1]).byte())
        target[:, -2:].view(torch.bfloat16).copy_(factor)
    return values, metadata


def decode_int8(
    values: torch.Tensor, metadata: torch.Tensor, group_size: int
) -> torch.Tensor:
    width = values.shape[-1]
    groups = width // group_size
    if (
        values.dtype != torch.int8
        or metadata.dtype != torch.uint8
        or metadata.shape != (*values.shape[:-1], metadata_bytes(width, group_size))
    ):
        raise ValueError(
            "The signed INT8 values and group metadata have incompatible layouts"
        )
    scale_bytes = groups * SCALES_PER_GROUP * 2
    bits = (
        metadata[..., :scale_bytes]
        .view(torch.int16)
        .unflatten(-1, (groups, SCALES_PER_GROUP))
    )
    positive = (bits.int() & 0x7FFF).short()
    scales = torch.where(
        bits < 0,
        positive.view(torch.bfloat16).float(),
        positive.view(torch.float16).float(),
    )
    payload = metadata[..., scale_bytes:-2].float()
    # Binary fractions are exact for UINT8 and avoid scalar shift host fallback on 910B.
    upper = (payload * 0.0625).floor()
    selectors = torch.stack((payload - 16 * upper, upper), dim=-1).reshape(
        *values.shape[:-1], groups, group_size
    )
    selected = scales.gather(-1, selectors.long()).flatten(-2).float()
    factor = metadata[..., -2:].view(torch.bfloat16).float()
    limit = torch.finfo(torch.bfloat16).max
    return ((values.float() * selected) * factor).clamp(-limit, limit).bfloat16()


def physical_page_map(block_table, page_size, lengths, *, fixed_shape):
    """Return referenced physical IDs and their page table for the operator."""
    page_pos = torch.arange(
        block_table.shape[1], device=block_table.device, dtype=torch.int32
    )
    valid = (page_pos * page_size < lengths.reshape(-1, 1)) & (block_table > 0)
    ids = torch.where(valid, block_table, 0).flatten()
    ids = torch.cat((ids.new_zeros(1), ids))
    if fixed_shape:
        # Graph replay can change physical page IDs and live sequence lengths.
        # Keep one scratch page per table entry so the allocation shape is static.
        physical = ids
        inverse = torch.arange(ids.numel(), device=ids.device, dtype=torch.int32)
    else:
        physical, inverse = ids.unique(sorted=True, return_inverse=True)
    remapped = inverse[1:].reshape_as(block_table).to(torch.int32)
    remapped = torch.where(valid, remapped, block_table.to(torch.int32).clamp_max(0))
    return physical.long(), remapped.contiguous()
