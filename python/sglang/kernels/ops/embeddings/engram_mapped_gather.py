"""A3 mapped INT8/FP8 lookup with packed E8M0 scales."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mapped_engram_kernel(
    bases,
    ids,
    out,
    stride_t,
    stride_h,
    ROWS_PER_OWNER: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    FP8: tl.constexpr,
):
    row = tl.program_id(0)
    source = tl.load(ids + row // HEADS * stride_t + row % HEADS * stride_h).to(
        tl.int64
    )
    valid = (source >= 0) & (source < NUM_ROWS)
    source = tl.where(valid, source, 0).to(tl.uint32)
    owner = source // ROWS_PER_OWNER
    local = source % ROWS_PER_OWNER
    row_bytes = DIM + DIM // GROUP
    address = tl.load(bases + owner).to(tl.uint64) + local.to(tl.uint64) * row_bytes
    col = tl.arange(0, DIM)
    if FP8:
        # A3 has no native FP8 conversion for this mapped-host path. Decode
        # E4M3FN bytes explicitly, including subnormals and NaN encodings.
        bits = tl.load(address.to(tl.pointer_type(tl.uint8)) + col).to(tl.int32) & 255
        exponent = (bits >> 3) & 15
        mantissa = bits & 7
        sign = (bits & 128) << 24
        normal = sign | ((exponent + 120) << 23) | (mantissa << 20)
        subnormal = (mantissa.to(tl.float32) * (2.0**-9)).to(
            tl.int32, bitcast=True
        ) | sign
        value = tl.where(exponent == 0, subnormal, normal).to(tl.float32, bitcast=True)
        value = tl.where((bits & 127) == 127, float("nan"), value)
    else:
        value = tl.load(address.to(tl.pointer_type(tl.int8)) + col).to(tl.float32)
    groups = tl.arange(0, DIM // GROUP)
    scale = (
        tl.load((address + DIM).to(tl.pointer_type(tl.uint8)) + groups).to(tl.int32)
        & 255
    )
    scale_bits = scale << 23
    scale_bits = tl.where(scale == 0, 0x00400000, scale_bits)
    scale_bits = tl.where(scale == 255, 0x7FC00000, scale_bits)
    scale = scale_bits.to(tl.float32, bitcast=True)
    value = tl.reshape(value, (DIM // GROUP, GROUP)) * scale[:, None]
    value = tl.reshape(value, (DIM,))
    tl.store(
        out + row.to(tl.int64) * DIM + col, tl.where(valid, value, 0).to(tl.bfloat16)
    )


def mapped_engram_gather(bases, ids, rows_per_owner, num_rows, dim, group, fp8):
    out = torch.empty((*ids.shape, dim), dtype=torch.bfloat16, device=ids.device)
    if ids.numel():
        _mapped_engram_kernel[(ids.numel(),)](
            bases,
            ids,
            out,
            ids.stride(0),
            ids.stride(1),
            ROWS_PER_OWNER=rows_per_owner,
            NUM_ROWS=num_rows,
            HEADS=ids.shape[1],
            DIM=dim,
            GROUP=group,
            FP8=fp8,
            num_warps=4,
        )
    return out
