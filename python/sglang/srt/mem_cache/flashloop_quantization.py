# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FlashLoop contributors
"""Packed INT4 anchor / recurrent residuals with a rolling BF16 tail.

K groups span 64 tokens; V groups span 64 channels. Readers reconstruct only
requested tiles, never a persistent or temporary full-length BF16 cache.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def load_prefix(
    P,
    MIN,
    SCALE,
    FLAGS,
    slots,
    head,
    dims,
    valid,
    CAP: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    LOOP: tl.constexpr,
    IS_K: tl.constexpr,
):
    result = tl.full((slots.shape[0], dims.shape[0]), 0, tl.float32)
    for stream in tl.static_range(LOOP + 1):
        ix = ((stream * CAP + slots[:, None]) * H + head) * (D // 2) + dims[
            None, :
        ] // 2
        byte = tl.load(P + ix, valid[:, None], 0).to(tl.int32)
        code = (byte >> ((dims[None, :] % 2) * 4)) & 15
        if IS_K:
            mi = ((stream * (CAP // 64) + slots[:, None] // 64) * H + head) * D + dims[
                None, :
            ]
        else:
            mi = ((stream * CAP + slots[:, None]) * H + head) * (D // 64) + dims[
                None, :
            ] // 64
        lo = tl.load(MIN + mi, valid[:, None], 0).to(tl.float32)
        scale = tl.load(SCALE + mi, valid[:, None], 0).to(tl.float32)
        active = tl.load(FLAGS + stream * CAP + slots, valid, 0) != 0
        result += tl.where(active[:, None], code.to(tl.float32) * scale + lo, 0.0)
    return result


@triton.jit
def load_rows(
    P,
    MIN,
    SCALE,
    TAIL,
    FLAGS,
    slots,
    logical,
    request,
    length,
    head,
    dims,
    valid,
    CAP: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    LOOP: tl.constexpr,
    IS_K: tl.constexpr,
):
    boundary = tl.maximum(0, (length - 64) // 64 * 64)
    prefix = valid & (logical < boundary)
    result = load_prefix(
        P, MIN, SCALE, FLAGS, slots, head, dims, prefix, CAP, H, D, LOOP, IS_K
    )
    ti = (((LOOP * R + request) * 128 + logical[:, None] % 128) * H + head) * D + dims[
        None, :
    ]
    tail = tl.load(TAIL + ti, (valid & ~prefix)[:, None], 0).to(tl.float32)
    return result + tail


@triton.jit
def store_group(
    X,
    P,
    MIN,
    SCALE,
    FLAGS,
    active,
    slots,
    head,
    dims,
    CAP: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    LOOP: tl.constexpr,
    IS_K: tl.constexpr,
):
    if head == 0:
        tl.store(FLAGS + LOOP * CAP + slots, active.to(tl.uint8))
    if IS_K:
        lo = tl.min(X, 0)
        hi = tl.max(X, 0)
        span = hi - lo
        scale = tl.where(span > 0, span * (1.0 / 15.0), 1.0)
        codes = tl.minimum(
            15,
            tl.maximum(
                0, libdevice.nearbyint(tl.div_rn(X - lo[None, :], scale[None, :]))
            ),
        )
        page = tl.min(slots, 0) // 64
        mi = ((LOOP * (CAP // 64) + page) * H + head) * D + dims
    else:
        groups = tl.reshape(X, (64, D // 64, 64))
        lo = tl.min(groups, 2)
        hi = tl.max(groups, 2)
        span = hi - lo
        scale = tl.where(span > 0, span * (1.0 / 15.0), 1.0)
        codes = tl.reshape(
            tl.minimum(
                15,
                tl.maximum(
                    0,
                    libdevice.nearbyint(
                        tl.div_rn(groups - lo[:, :, None], scale[:, :, None])
                    ),
                ),
            ),
            (64, D),
        )
        mi = ((LOOP * CAP + slots[:, None]) * H + head) * (D // 64) + tl.arange(
            0, D // 64
        )[None, :]
    tl.store(MIN + mi, lo)
    tl.store(SCALE + mi, scale)
    pair = tl.reshape(codes.to(tl.int32), (64, D // 2, 2))
    packed = tl.sum(
        pair
        * tl.reshape(tl.full((2,), 1, tl.int32) << (tl.arange(0, 2) * 4), (1, 1, 2)),
        2,
    )
    pi = ((LOOP * CAP + slots[:, None]) * H + head) * (D // 2) + tl.arange(0, D // 2)[
        None, :
    ]
    tl.store(P + pi, packed.to(tl.uint8))


@triton.jit
def _prefill_write(
    X,
    ROWS,
    START,
    MAP,
    REQ,
    LENS,
    P,
    MIN,
    SCALE,
    TAIL,
    FLAGS,
    XS: tl.constexpr,
    XH: tl.constexpr,
    MS: tl.constexpr,
    CAP: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    LOOP: tl.constexpr,
    IS_K: tl.constexpr,
):
    group, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    logical = group * 64 + tl.arange(0, 64)
    dims = tl.arange(0, D)
    length = tl.load(LENS + batch)
    request = tl.load(REQ + batch)
    if group * 64 < length and request > 0:
        valid = logical < length
        slots = tl.load(MAP + request * MS + logical, valid, 0)
        start = tl.load(START + batch)
        rank = tl.load(ROWS + start + logical, valid, -1)
        raw = tl.load(
            X + rank[:, None] * XS + head * XH + dims[None, :],
            (valid & (rank >= 0))[:, None],
            0,
        ).to(tl.float32)
        if LOOP > 0:
            previous = load_rows(
                P,
                MIN,
                SCALE,
                TAIL,
                FLAGS,
                slots,
                logical,
                request,
                length,
                head,
                dims,
                valid,
                CAP,
                H,
                D,
                R,
                LOOP - 1,
                IS_K,
            )
        else:
            previous = tl.full((64, D), 0, tl.float32)
        target = tl.where((rank >= 0)[:, None], raw, previous)
        boundary = tl.maximum(0, (length - 64) // 64 * 64)
        if group * 64 < boundary:
            store_group(
                target - previous,
                P,
                MIN,
                SCALE,
                FLAGS,
                rank >= 0,
                slots,
                head,
                dims,
                CAP,
                H,
                D,
                LOOP,
                IS_K,
            )
        else:
            ti = (
                ((LOOP * R + request) * 128 + logical[:, None] % 128) * H + head
            ) * D + dims[None, :]
            tl.store(TAIL + ti, target, valid[:, None])


@triton.jit
def _decode_tail(
    X,
    REQ,
    LENS,
    TAIL,
    XS: tl.constexpr,
    XH: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    LOOP: tl.constexpr,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    request = tl.load(REQ + batch)
    position = tl.load(LENS + batch) - 1
    dims = tl.arange(0, D)
    x = tl.load(X + batch * XS + head * XH + dims)
    ti = (((LOOP * R + request) * 128 + position % 128) * H + head) * D + dims
    tl.store(TAIL + ti, x, (request > 0) & (position >= 0))


@triton.jit
def _decode_flush(
    MAP,
    REQ,
    LENS,
    P,
    MIN,
    SCALE,
    TAIL,
    FLAGS,
    MS: tl.constexpr,
    CAP: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    LOOP: tl.constexpr,
    IS_K: tl.constexpr,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    request = tl.load(REQ + batch)
    length = tl.load(LENS + batch)
    if request > 0 and length >= 128 and length % 64 == 0:
        logical = length - 128 + tl.arange(0, 64)
        dims = tl.arange(0, D)
        slots = tl.load(MAP + request * MS + logical)
        ti = (
            ((LOOP * R + request) * 128 + logical[:, None] % 128) * H + head
        ) * D + dims[None, :]
        target = tl.load(TAIL + ti).to(tl.float32)
        previous = load_prefix(
            P,
            MIN,
            SCALE,
            FLAGS,
            slots,
            head,
            dims,
            logical >= 0,
            CAP,
            H,
            D,
            LOOP - 1,
            IS_K,
        )
        store_group(
            target - previous,
            P,
            MIN,
            SCALE,
            FLAGS,
            logical >= 0,
            slots,
            head,
            dims,
            CAP,
            H,
            D,
            LOOP,
            IS_K,
        )


class QuantizedView:
    def __init__(self, storage, loop, is_key):
        self.storage, self.loop, self.is_key = storage, loop, is_key
        self.packed, self.minimum, self.scale, self.tail, self.flags = (
            storage.keys if is_key else storage.values
        )
        self.shape = (storage.capacity, storage.heads, storage.dims)
        self.device, self.dtype = self.tail.device, self.tail.dtype

    def args(self):
        return (
            self.packed,
            self.minimum,
            self.scale,
            self.tail,
            self.flags,
            self.storage.capacity,
            self.storage.requests,
            self.loop,
            self.is_key,
        )


class QuantizedStorage:
    def __init__(self, capacity, heads, dims, requests, device):
        if capacity % 64 or dims % 64 or dims & (dims - 1):
            raise ValueError(
                "INT4 storage requires aligned 64-token pages and power-of-two head dims divisible by 64"
            )
        self.capacity, self.heads, self.dims, self.requests = (
            capacity,
            heads,
            dims,
            requests,
        )

        def alloc(is_key):
            packed = torch.zeros(
                (4, capacity, heads, dims // 2), device=device, dtype=torch.uint8
            )
            shape = (
                (4, capacity // 64, heads, dims)
                if is_key
                else (4, capacity, heads, dims // 64)
            )
            minimum = torch.zeros(shape, device=device, dtype=torch.float16)
            scale = torch.ones(shape, device=device, dtype=torch.float16)
            tail = torch.zeros(
                (4, requests, 128, heads, dims), device=device, dtype=torch.bfloat16
            )
            flags = torch.zeros((4, capacity), device=device, dtype=torch.uint8)
            return packed, minimum, scale, tail, flags

        self.keys, self.values = alloc(True), alloc(False)

    def view(self, loop, is_key):
        return QuantizedView(self, loop, is_key)

    def nbytes(self):
        return sum(
            t.numel() * t.element_size()
            for buffers in (self.keys, self.values)
            for t in buffers
        )

    def write_prefill(
        self, loop, k, v, row_map, starts, req_map, req_ids, lengths, max_length
    ):
        for raw, buffers, is_key in ((k, self.keys, True), (v, self.values, False)):
            _prefill_write[(triton.cdiv(max_length, 64), self.heads, req_ids.numel())](
                raw,
                row_map,
                starts,
                req_map,
                req_ids,
                lengths,
                *buffers,
                raw.stride(0),
                raw.stride(1),
                req_map.stride(0),
                self.capacity,
                self.heads,
                self.dims,
                self.requests,
                loop,
                is_key,
                enable_fp_fusion=False,
            )

    def write_decode(self, loop, k, v, req_map, req_ids, lengths):
        for raw, buffers, is_key in ((k, self.keys, True), (v, self.values, False)):
            _decode_tail[(req_ids.numel(), self.heads)](
                raw,
                req_ids,
                lengths,
                buffers[3],
                raw.stride(0),
                raw.stride(1),
                self.heads,
                self.dims,
                self.requests,
                loop,
            )
            _decode_flush[(req_ids.numel(), self.heads)](
                req_map,
                req_ids,
                lengths,
                *buffers,
                req_map.stride(0),
                self.capacity,
                self.heads,
                self.dims,
                self.requests,
                loop,
                is_key,
                enable_fp_fusion=False,
            )
