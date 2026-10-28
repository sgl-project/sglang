# SPDX-License-Identifier: Apache-2.0
"""Non-online R8 reduction and ring: read history/current inputs, publish
compressed index keys, then write ring.

The shared BF16/FP8 rounding-contract reduction
(_compressed_index_key_bytes) lives here with its sole consumer; no
separate module.
"""

import dataclasses

import torch
import triton
import triton.language as tl


@triton.jit
def _fresh_product(weight, value):
    return tl.inline_asm_elementwise(
        "{ .reg .f32 p; mul.rn.f32 p, $1, $2; add.rn.f32 $0, p, 0f00000000; }",
        constraints="=f,f,f",
        args=[weight, value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _compressed_index_key_bytes(index_keys, compression_logits):
    # raw payloads retain their BF16 bits, including unused NaN slots, until
    # all eight tokens are ready for the numerical reduction.
    maximum = tl.full((256,), -float("inf"), tl.float32)
    for token in tl.static_range(8):
        indices = tl.full((1, 256), token, tl.int32)
        logit = tl.gather(compression_logits, indices, axis=0).reshape((256,))
        logit = logit.to(tl.bfloat16, bitcast=True).to(tl.float32)
        maximum = tl.maximum(maximum, logit, propagate_nan=tl.PropagateNan.NONE)
    region_max = tl.max(maximum, axis=0)
    numerator = tl.full((256,), 0.0, tl.float32)
    denominator = tl.full((256,), 0.0, tl.float32)
    for token in tl.static_range(8):
        indices = tl.full((1, 256), token, tl.int32)
        logit = tl.gather(compression_logits, indices, axis=0).reshape((256,))
        value = tl.gather(index_keys, indices, axis=0).reshape((256,))
        logit = logit.to(tl.bfloat16, bitcast=True).to(tl.float32)
        value = value.to(tl.bfloat16, bitcast=True).to(tl.float32)
        weight = tl.exp2((logit - region_max) * 1.4426950408889634)
        if token == 0:
            numerator = _fresh_product(weight, value)
        else:
            numerator = tl.fma(weight, value, numerator)
        denominator = denominator + weight
    mean = tl.div_rn(
        numerator,
        tl.maximum(denominator, 1.0e-20, propagate_nan=tl.PropagateNan.NONE),
    )
    mean = tl.where(denominator > 0.0, mean, 0.0)
    rounded = mean.to(tl.bfloat16).to(tl.float32)
    packed = tl.inline_asm_elementwise(
        """{
            .reg .b16 pair;
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
            cvt.rn.satfinite.e4m3x2.f32 pair, zero, $1;
            cvt.u32.u16 $0, pair;
        }""",
        constraints="=r,f",
        args=[rounded],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return packed.to(tl.uint8)


@triton.jit
def _admit_row(
    row,
    slots,
    bases,
    active,
    admitted,
    count_valid,
    tags,
    enabled,
    success,
    CAPACITY: tl.constexpr,
    ASSERT_SUCCESS: tl.constexpr,
):
    live = tl.load(active + row) != 0
    ok = tl.full((), False, tl.int1)
    if live & (tl.load(admitted + row) != 0) & (tl.load(count_valid) != 0):
        slot = tl.load(slots + row).to(tl.int64)
        base = tl.load(bases + row)
        pos = base // 8 * 8 + tl.arange(0, 8)
        actual = tl.load(tags + slot * CAPACITY + pos % CAPACITY)
        history_ok = tl.sum(((pos < base) & (actual != pos)).to(tl.int32), 0) == 0
        ok = history_ok
    tl.store(enabled + row, ok)
    row_ok = (tl.load(count_valid) != 0) & (~live | ok)
    tl.store(success + row, row_ok)
    if ASSERT_SUCCESS:
        tl.device_assert(row_ok, "Step5 CSA ring history failed")
    return ok


@triton.jit
def _admit(
    slots,
    bases,
    active,
    admitted,
    count_valid,
    tags,
    enabled,
    success,
    CAPACITY: tl.constexpr,
    ASSERT_SUCCESS: tl.constexpr,
):
    _admit_row(
        tl.program_id(0),
        slots,
        bases,
        active,
        admitted,
        count_valid,
        tags,
        enabled,
        success,
        CAPACITY,
        ASSERT_SUCCESS,
    )


@triton.jit
def _compress(
    index_key,
    compression_logits,
    slots,
    bases,
    sources,
    lengths,
    physical,
    enabled,
    index_key_ring,
    compression_logits_ring,
    paged,
    active,
    admitted,
    count_valid,
    tags,
    success,
    ADMIT: tl.constexpr,
    ASSERT_SUCCESS: tl.constexpr,
    CAPACITY: tl.constexpr,
    ROWS: tl.constexpr,
    FUSE_WRITE: tl.constexpr,
    Z_ROW: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    region_offset, row = tl.program_id(0), tl.program_id(1)
    if ADMIT:
        # Exactly one CTA per request: admission and compression share its
        # register result. The fused path finishes all old-ring reads before
        # writing in this same CTA. Multi-region prefill admits and writes in
        # separate launches instead of racing/repeating them per region.
        ok = _admit_row(
            row,
            slots,
            bases,
            active,
            admitted,
            count_valid,
            tags,
            enabled,
            success,
            CAPACITY,
            ASSERT_SUCCESS,
        )
    else:
        ok = tl.load(enabled + row)
    if ok:
        base = tl.load(bases + row)
        end = base + tl.load(lengths + row)
        first = base // 8 * 8
        region = first + region_offset * 8
        if region + 8 <= end:
            slot = tl.load(slots + row).to(tl.int64)
            source = tl.load(sources + row).to(tl.int64)
            pos = region + tl.arange(0, 8)
            ch = tl.arange(0, 256)
            history = pos < base
            ring_offset = (slot * CAPACITY + pos % CAPACITY)[:, None] * 256 + ch[
                None, :
            ]
            input_offset = (source + pos - base)[:, None] * 256 + ch[None, :]
            rk = tl.load(
                index_key_ring.to(tl.pointer_type(tl.uint16)) + ring_offset,
                history[:, None],
                0,
            )
            rz = tl.load(
                compression_logits_ring.to(tl.pointer_type(tl.uint16)) + ring_offset,
                history[:, None],
                0,
            )
            if USE_PDL:
                iz = tl.load(
                    compression_logits.to(tl.pointer_type(tl.uint16))
                    + (source + pos - base)[:, None] * Z_ROW
                    + ch[None, :],
                    ~history[:, None],
                    0,
                )
                # Only the current K rows depend on the preceding norm/RoPE.
                tl.extra.cuda.gdc_wait()
            ik = tl.load(
                index_key.to(tl.pointer_type(tl.uint16)) + input_offset,
                ~history[:, None],
                0,
            )
            if not USE_PDL:
                iz = tl.load(
                    compression_logits.to(tl.pointer_type(tl.uint16))
                    + (source + pos - base)[:, None] * Z_ROW
                    + ch[None, :],
                    ~history[:, None],
                    0,
                )
            packed = _compressed_index_key_bytes(
                tl.where(history[:, None], rk, ik), tl.where(history[:, None], rz, iz)
            )
            dest = tl.load(physical + region_offset * ROWS + row).to(tl.int64)
            within = dest % 8
            sw128 = (
                (ch // 128) * 1024
                + within * 128
                + (((ch % 128) // 16) ^ within) * 16
                + ch % 16
            )
            tl.store(paged + dest // 8 * 2048 + sw128, packed)
    if FUSE_WRITE:
        tl.static_assert(ADMIT)
        if USE_PDL:
            # A forward without a completed region still writes current K.
            tl.extra.cuda.gdc_wait()
        # A single CTA owns this request. Finish every old-ring read before
        # publishing pending tokens, including rejected speculative positions.
        tl.debug_barrier()
        _write_row(
            row,
            ok,
            index_key,
            compression_logits,
            slots,
            bases,
            sources,
            lengths,
            index_key_ring,
            compression_logits_ring,
            tags,
            CAPACITY,
            CAPACITY,
            Z_ROW,
        )


@triton.jit
def _write_row(
    row,
    enabled,
    index_key,
    compression_logits,
    slots,
    bases,
    sources,
    lengths,
    index_key_ring,
    compression_logits_ring,
    tags,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    Z_ROW: tl.constexpr,
):
    if enabled:
        slot = tl.load(slots + row).to(tl.int64)
        base = tl.load(bases + row)
        end = base + tl.load(lengths + row)
        source = tl.load(sources + row).to(tl.int64)
        # One lane per ring cell, selecting the newest input with that residue.
        cell = tl.arange(0, BLOCK)
        pos = end - 1 - (end - 1 + CAPACITY - cell) % CAPACITY
        write = (cell < CAPACITY) & (pos >= base)
        ch = tl.arange(0, 256)
        src = (source + pos - base)[:, None] * 256 + ch[None, :]
        dst = (slot * CAPACITY + cell)[:, None] * 256 + ch[None, :]
        ik = tl.load(index_key.to(tl.pointer_type(tl.uint16)) + src, write[:, None], 0)
        iz = tl.load(
            compression_logits.to(tl.pointer_type(tl.uint16))
            + (source + pos - base)[:, None] * Z_ROW
            + ch[None, :],
            write[:, None],
            0,
        )
        tl.store(
            index_key_ring.to(tl.pointer_type(tl.uint16)) + dst, ik, write[:, None]
        )
        tl.store(
            compression_logits_ring.to(tl.pointer_type(tl.uint16)) + dst,
            iz,
            write[:, None],
        )
        previous = tl.load(tags + slot * CAPACITY + cell, cell < CAPACITY, -1)
        # At a compression boundary no earlier input can be needed again.
        # Invalidate unwritten cells, including stale tags from a reused slot.
        tag = tl.where(write, pos, tl.where(base % 8 == 0, -1, previous))
        tl.store(tags + slot * CAPACITY + cell, tag, cell < CAPACITY)


@triton.jit
def _write(
    index_key,
    compression_logits,
    slots,
    bases,
    sources,
    lengths,
    enabled,
    index_key_ring,
    compression_logits_ring,
    tags,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    Z_ROW: tl.constexpr,
):
    row = tl.program_id(0)
    _write_row(
        row,
        tl.load(enabled + row),
        index_key,
        compression_logits,
        slots,
        bases,
        sources,
        lengths,
        index_key_ring,
        compression_logits_ring,
        tags,
        CAPACITY,
        BLOCK,
        Z_ROW,
    )


@dataclasses.dataclass(frozen=True)
class AppendRingPlan:
    """Execution plan for one batched append: one row per request span."""

    slots: torch.Tensor
    bases: torch.Tensor
    sources: torch.Tensor
    lengths: torch.Tensor
    active: torch.Tensor
    admitted: torch.Tensor
    count_valid: torch.Tensor
    physical_regions: torch.Tensor
    max_regions: int
    enabled: torch.Tensor
    success: torch.Tensor


def append_ring(
    *,
    plan: AppendRingPlan,
    index_key,
    compression_logits,
    index_key_ring,
    compression_logits_ring,
    positions,
    compressed_index_keys,
    assert_success=False,
    use_pdl=False,
):
    slots, bases, sources, lengths = (
        plan.slots,
        plan.bases,
        plan.sources,
        plan.lengths,
    )
    active, admitted, count_valid = plan.active, plan.admitted, plan.count_valid
    physical_regions, max_regions = plan.physical_regions, plan.max_regions
    enabled, success = plan.enabled, plan.success
    """Read history/current inputs before overwriting ring cells.

    The native page64 prefix cache resumes fresh requests at an R8 boundary;
    later chunks/verify read only [floor(base/8)*8, base). With C >= 7 + W,
    a candidate at p <= base + W - 1 overwrites p-C < floor(base/8)*8.
    Thus any legal accepted prefix retains every historical input it can need.
    Slot reuse is ordered by the scheduler's forward stream / WAR barrier;
    position tags diagnose missing inputs, not a second request identity.
    """
    rows = slots.numel()
    capacity = index_key_ring.shape[1]
    if rows:
        fuse_write = (
            max_regions == 1
            and capacity == 16
            and rows <= 64
            and torch.version.hip is None
            and index_key.is_cuda
            and torch.cuda.get_device_capability(index_key.device) == (10, 3)
        )
        if max_regions != 1:
            _admit[(rows,)](
                slots,
                bases,
                active,
                admitted,
                count_valid,
                positions,
                enabled,
                success,
                capacity,
                assert_success,
                debug=True,
            )
        if max_regions:
            _compress[(max_regions, rows)](
                index_key,
                compression_logits,
                slots,
                bases,
                sources,
                lengths,
                physical_regions,
                enabled,
                index_key_ring,
                compression_logits_ring,
                compressed_index_keys,
                active,
                admitted,
                count_valid,
                positions,
                success,
                max_regions == 1,
                assert_success,
                capacity,
                rows,
                fuse_write,
                compression_logits.stride(0),
                USE_PDL=use_pdl and fuse_write,
                launch_pdl=use_pdl and fuse_write,
                num_warps=8,
                enable_fp_fusion=True,
                debug=True,
            )
        if not fuse_write:
            _write[(rows,)](
                index_key,
                compression_logits,
                slots,
                bases,
                sources,
                lengths,
                enabled,
                index_key_ring,
                compression_logits_ring,
                positions,
                capacity,
                triton.next_power_of_2(capacity),
                compression_logits.stride(0),
                num_warps=8,
            )
    return success
