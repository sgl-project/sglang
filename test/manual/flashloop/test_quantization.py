# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FlashLoop contributors
"""Independent Torch oracle for packed bytes, recurrence and paged readers."""

import pytest
import torch

from sglang.srt.layers.attention.flashloop.kernels import (
    dense_paged_attention,
    paged_scores,
)
from sglang.srt.layers.attention.flashloop.prefill import sparse_paged_prefill
from sglang.srt.mem_cache.flashloop_quantization import QuantizedStorage

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def unpack(storage, loop, key, slots, request, length):
    p, lo, scale, tail, flags = storage.keys if key else storage.values
    boundary = max(0, (length - 64) // 64 * 64)
    prefix = torch.zeros((boundary, storage.heads, storage.dims), device="cuda")
    for stream in range(loop + 1):
        byte = p[stream, slots[:boundary]]
        code = torch.stack((byte & 15, byte >> 4), -1).flatten(-2).float()
        if key:
            minimum, step = (
                lo[stream, slots[:boundary] // 64],
                scale[stream, slots[:boundary] // 64],
            )
        else:
            minimum = lo[stream, slots[:boundary]].repeat_interleave(64, -1)
            step = scale[stream, slots[:boundary]].repeat_interleave(64, -1)
        prefix += torch.where(
            flags[stream, slots[:boundary], None, None].bool(),
            code * step.float() + minimum.float(),
            0.0,
        )
    pos = torch.arange(boundary, length, device="cuda")
    return torch.cat((prefix, tail[loop, request, pos % 128].float()), 0)


def oracle_pack(x, key):
    if key:
        lo, hi = x.amin(0, keepdim=True), x.amax(0, keepdim=True)
        scale = torch.where(hi > lo, (hi - lo) / 15, 1.0)
        codes = ((x - lo) / scale).round().clamp(0, 15).byte()
        minimum, step = lo.squeeze(0), scale.squeeze(0)
    else:
        grouped = x.reshape(64, x.shape[1], -1, 64)
        lo, hi = grouped.amin(-1, keepdim=True), grouped.amax(-1, keepdim=True)
        scale = torch.where(hi > lo, (hi - lo) / 15, 1.0)
        codes = ((grouped - lo) / scale).round().clamp(0, 15).byte().reshape_as(x)
        minimum, step = lo.squeeze(-1), scale.squeeze(-1)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed, minimum.half(), step.half()


def setup(lengths=(256, 191), dims=64):
    torch.manual_seed(314)
    cap, h, r = 2048, 2, 4
    storage = QuantizedStorage(cap, h, dims, r, "cuda")
    # Permute whole pages: no assumption of contiguous request storage.
    pages = torch.randperm(cap // 64 - 1, device="cuda") + 1
    mapping = torch.zeros((r, 512), device="cuda", dtype=torch.int32)
    for i in range(1, 3):
        mapping[i] = (
            (pages[(i - 1) * 8 : i * 8, None] * 64 + torch.arange(64, device="cuda"))
            .flatten()
            .int()
        )
    ids = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, lengths[0]], device="cuda", dtype=torch.int32)
    rows = torch.arange(sum(lengths), device="cuda", dtype=torch.int32)
    k = torch.randn((sum(lengths), h, dims), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    return storage, mapping, ids, lens, starts, rows, k, v


@pytest.mark.parametrize(
    "length,dims", [(n, 64) for n in (63, 64, 127, 128, 191, 192, 256)] + [(192, 128)]
)
def test_prefill_packing_and_readers(length, dims):
    s, m, ids, lens, starts, rows, k, v = setup((length, length - 1), dims)
    q = torch.randn((2, 2, dims), device="cuda", dtype=torch.bfloat16)
    for loop in range(4):
        kk = (k.float() + loop * 0.03125).bfloat16()
        vv = (v.float() - loop * 0.015625).bfloat16()
        previous = []
        for key in (True, False):
            previous.append(
                [
                    unpack(s, loop - 1, key, m[r, :n].long(), r, n)
                    if loop
                    else torch.zeros((n, 2, dims), device="cuda")
                    for r, n in ((1, length), (2, length - 1))
                ]
            )
        s.write_prefill(loop, kk, vv, rows, starts, m, ids, lens, length)
        for ki, (raw, key, buffers) in enumerate(
            ((kk, True, s.keys), (vv, False, s.values))
        ):
            for batch, n in enumerate((length, length - 1)):
                boundary = max(0, (n - 64) // 64 * 64)
                offset = 0 if batch == 0 else length
                slots = m[batch + 1, :n].long()
                for begin in range(0, boundary, 64):
                    delta = (
                        raw[offset + begin : offset + begin + 64].float()
                        - previous[ki][batch][begin : begin + 64]
                    )
                    packed, minimum, step = oracle_pack(delta, key)
                    torch.testing.assert_close(
                        buffers[0][loop, slots[begin : begin + 64]],
                        packed,
                        rtol=0,
                        atol=0,
                        msg=lambda x: f"loop={loop} key={key} batch={batch}: {x}",
                    )
                    actual_min = (
                        buffers[1][loop, slots[begin] // 64]
                        if key
                        else buffers[1][loop, slots[begin : begin + 64]]
                    )
                    actual_step = (
                        buffers[2][loop, slots[begin] // 64]
                        if key
                        else buffers[2][loop, slots[begin : begin + 64]]
                    )
                    torch.testing.assert_close(actual_min, minimum, rtol=0, atol=0)
                    torch.testing.assert_close(actual_step, step, rtol=0, atol=0)
                rec = unpack(s, loop, key, slots, batch + 1, n)
                torch.testing.assert_close(
                    rec[boundary:],
                    raw[offset + boundary : offset + n].float(),
                    rtol=0,
                    atol=0,
                )
        scores = paged_scores(q, s.view(loop, True), m, ids, lens, 320)
        out = dense_paged_attention(
            q, s.view(loop, True), s.view(loop, False), m, ids, lens, 320
        )
        for b, n in enumerate((length, length - 1)):
            key = unpack(s, loop, True, m[b + 1, :n].long(), b + 1, n)
            value = unpack(s, loop, False, m[b + 1, :n].long(), b + 1, n)
            ref = torch.einsum("hd,thd->ht", q[b].float(), key) / (dims**0.5)
            torch.testing.assert_close(scores[b, :, :n], ref, rtol=2e-5, atol=2e-5)
            expected = torch.einsum("ht,thd->hd", ref.softmax(-1), value).bfloat16()
            torch.testing.assert_close(out[b], expected, rtol=0.008, atol=0.008)
            assert torch.isneginf(scores[b, :, n:]).all()


def test_decode_flush_graph_and_request_reuse():
    s, m, ids, lens, starts, rows, k, v = setup((127, 191))
    for loop in range(4):
        s.write_prefill(loop, k, v, rows, starts, m, ids, lens, 191)
    stepk = torch.randn((2, 2, 64), device="cuda", dtype=torch.bfloat16)
    stepv = torch.randn_like(stepk)
    q = torch.randn_like(stepk)
    lens.add_(1)

    def forward():
        for loop in range(4):
            s.write_decode(loop, stepk, stepv, m, ids, lens)
        return dense_paged_attention(
            q, s.view(3, True), s.view(3, False), m, ids, lens, 512
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    for tick in range(130):
        graph.replay()
        if tick in (0, 1, 63, 64, 127, 129):
            for b, n in enumerate(lens.tolist()):
                actual = unpack(s, 3, True, m[b + 1, :n].long(), b + 1, n)
                torch.testing.assert_close(actual[-1], stepk[b].float(), rtol=0, atol=0)
                values = unpack(s, 3, False, m[b + 1, :n].long(), b + 1, n)
                ref = (torch.einsum("hd,thd->ht", q[b].float(), actual) / 8).softmax(-1)
                expected = torch.einsum("ht,thd->hd", ref, values).bfloat16()
                torch.testing.assert_close(output[b], expected, rtol=0.008, atol=0.008)
                initial = 127 if b == 0 else 191
                start = 0 if b == 0 else 127
                truth = torch.cat(
                    (
                        k[start : start + initial].float(),
                        stepk[b : b + 1].float().expand(n - initial, -1, -1),
                    ),
                    0,
                )
                torch.testing.assert_close(actual, truth, rtol=0.001, atol=0.003)
                # Flushed values across all previous positions remain finite.
                assert torch.isfinite(actual).all()
        lens.add_(1)
    # Reusing the request IDs must replace old packed pages and BF16 tails.
    lens.copy_(torch.tensor([127, 191], device="cuda"))
    for loop in range(4):
        s.write_prefill(loop, k * 0, v * 0, rows, starts, m, ids, lens, 191)
    for b, n in enumerate((127, 191)):
        assert not unpack(s, 3, True, m[b + 1, :n].long(), b + 1, n).count_nonzero()


@pytest.mark.parametrize("dims", [64, 128])
def test_sparse_prefill_and_physical_bytes(dims):
    s, m, ids, lens, starts, rows, k, v = setup(dims=dims)
    for loop in range(2):
        s.write_prefill(loop, k, v, rows, starts, m, ids, lens, 256)
    selected = torch.tensor([0, 127, 255, 256, 400, 446], device="cuda")
    row_map = torch.full_like(rows, -1)
    row_map[selected] = torch.arange(6, device="cuda", dtype=torch.int32)
    previous = [
        unpack(s, 1, True, m[b + 1, :n].long(), b + 1, n)
        for b, n in enumerate((256, 191))
    ]
    s.write_prefill(2, k[selected], v[selected], row_map, starts, m, ids, lens, 256)
    s.write_prefill(3, k[selected], v[selected], row_map, starts, m, ids, lens, 256)
    for b, n in enumerate((256, 191)):
        boundary = max(0, (n - 64) // 64 * 64)
        current = unpack(s, 2, True, m[b + 1, :n].long(), b + 1, n)
        inactive = torch.ones(n, device="cuda", dtype=torch.bool)
        local = selected[
            (selected >= (0 if b == 0 else 256)) & (selected < (256 if b == 0 else 447))
        ] - (0 if b == 0 else 256)
        inactive[local] = False
        torch.testing.assert_close(
            current[inactive], previous[b][inactive], rtol=0, atol=0
        )
    pos = torch.tensor([0, 127, 255, 0, 144, 190], device="cuda", dtype=torch.int64)
    st = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    count = torch.tensor([3, 3], device="cuda", dtype=torch.int32)
    q = torch.randn((6, 2, dims), device="cuda", dtype=torch.bfloat16)
    out = sparse_paged_prefill(
        q, s.view(3, True), s.view(3, False), pos, m, ids, lens, st, count, 3
    )
    for b, n in enumerate((256, 191)):
        key = unpack(s, 3, True, m[b + 1, :n].long(), b + 1, n).bfloat16().float()
        val = unpack(s, 3, False, m[b + 1, :n].long(), b + 1, n).bfloat16().float()
        for i in range(3):
            row = b * 3 + i
            end = int(pos[row]) + 1
            probs = (
                torch.einsum("hd,thd->ht", q[row].float(), key[:end]) / (dims**0.5)
            ).softmax(-1)
            ref = torch.einsum("ht,thd->hd", probs, val[:end]).bfloat16()
            torch.testing.assert_close(out[row], ref, rtol=0.025, atol=0.025)
    dense = s.capacity * s.heads * s.dims * 4 * 4
    expected_prefix = s.capacity * s.heads * s.dims * 4 * 2 * 9 // 16
    expected_tail = 4 * s.requests * 128 * s.heads * s.dims * 4
    assert s.nbytes() == expected_prefix + expected_tail + s.capacity * 4 * 2
    assert s.nbytes() < dense
