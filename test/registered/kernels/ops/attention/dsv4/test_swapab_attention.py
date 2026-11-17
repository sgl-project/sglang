"""V4 packed-page addressing, softmax/sink and dynamic graph replay checks, plus the
gfx950 kernel's buffer gather, dequant and combine RoPE-tile checks."""

import unittest

import torch
import triton

from sglang.kernels.ops.attention.dsv4.decode_attention_sm100 import _combine
from sglang.kernels.ops.attention.dsv4.norm_rope import fused_rope_inplace
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
# backend-specific: the HIP swap-AB Gluon kernel runs on gfx950.
register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

if is_hip():
    from sglang.kernels.ops.attention.dsv4.swapab_gluon_hip import (
        MAX_BATCH,
        combine_gfx950,
        swapab_attention,
    )
else:
    from sglang.kernels.ops.attention.dsv4.decode_attention_sm100 import (
        swapab_attention,
    )

if is_gfx95_supported():
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    from sglang.kernels.ops.attention.dsv4.decode_attention_sm100_gluon import _load_v4
    from sglang.kernels.ops.attention.dsv4.swapab_gluon_hip import _load_v4_hip

    @gluon.jit
    def load_both(
        K,
        IDS,
        VALID,
        BASE,
        FUSED,
        PAGE: gl.constexpr,
        STRIDE: gl.constexpr,
        BT: gl.constexpr,
    ):
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 4], [1, 0])
        n = gl.arange(0, BT, gl.SliceLayout(1, layout))
        ids = gl.load(IDS + n)
        valid = gl.load(VALID + n) != 0
        d = gl.arange(0, 512, gl.SliceLayout(0, layout))
        out = n[:, None] * 512 + d[None, :]
        gl.store(BASE + out, _load_v4(K, ids, valid, PAGE, STRIDE, layout, 576, 8, 64))
        gl.store(
            FUSED + out, _load_v4_hip(K, ids, valid, PAGE, STRIDE, layout, 576, 8, 64)
        )


# fp8 e4m3 0xFF and bf16 0xFFFF are NaN, so a read of a poisoned byte shows as NaN
POISON = 0xFF


def make_cache(page, pages=11):
    # Page padding and poisoned unused bytes catch confusion between a view's
    # logical token stride and the physically packed payload/scales planes.
    stride = page * 584 + 256
    raw = torch.full((pages, stride), 0x7F, dtype=torch.uint8, device="cuda")
    fp8 = torch.randn(pages, page, 448, device="cuda").to(torch.float8_e4m3fn)
    exponents = torch.randint(
        124, 130, (pages, page, 7), device="cuda", dtype=torch.uint8
    )
    rope = torch.randn(pages, page, 64, device="cuda").to(torch.bfloat16)
    raw[:, : page * 576].view(pages, page, 576)[..., :448].copy_(fp8.view(torch.uint8))
    raw[:, : page * 576].view(pages, page, 576)[..., 448:].copy_(rope.view(torch.uint8))
    raw[:, page * 576 : page * 584].view(pages, page, 8)[..., :7].copy_(exponents)
    values = torch.cat(
        (
            (
                fp8.float().view(pages, page, 7, 64)
                * torch.exp2(exponents.float() - 127)[..., None]
            ).flatten(-2),
            rope.float(),
        ),
        -1,
    ).reshape(-1, 512)
    cache = raw.as_strided((pages, page, 1, 584), (stride, 584, 584, 1))
    return raw, cache, values


def reference(q, values, ids, lengths, sink, extra=None):
    heads = q.shape[-2]

    def gather(vals, idx, lens):
        idx = idx[:, 0]
        valid = (idx >= 0) & (idx < vals.shape[0])
        valid &= torch.arange(idx.shape[-1], device=idx.device)[None, :] < lens[:, None]
        data = vals[idx.clamp(0, vals.shape[0] - 1).long()].double()
        return torch.where(valid[..., None], data, 0), valid

    kv, valid = gather(values, ids, lengths)
    if extra is not None:
        ev, em = gather(*extra)
        kv, valid = torch.cat((kv, ev), 1), torch.cat((valid, em), 1)
    scores = q[:, 0].double() @ kv.transpose(1, 2) * 512**-0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    scores = torch.cat(
        (scores, sink[None, :heads, None].double().expand(q.shape[0], -1, -1)), -1
    )
    prob = torch.nan_to_num(torch.softmax(scores, -1)[..., :-1])
    return prob @ kv


@unittest.skipUnless(
    torch.cuda.is_available()
    and (is_gfx95_supported() or torch.cuda.get_device_capability()[0] == 10),
    "requires Blackwell SM100/SM103 or AMD gfx950",
)
class TestSwapABAttention(CustomTestCase):
    def check_case(self, b, page, nk, ne, graph=False, heads=16, q_scale=0.5):
        torch.manual_seed(20260915 + b + nk + ne)
        storage = torch.full(
            (b, 1, 64, 512), float("nan"), device="cuda", dtype=torch.bfloat16
        )
        q = storage[:, :, :heads]
        q.copy_(torch.randn_like(q) * q_scale)
        raw, kv, values = make_cache(page)
        extra_raw, extra_kv, extra_values = make_cache(16)
        # Strided rows exercise source views made by metadata alignment.
        ids_storage = torch.randint(
            -3, values.shape[0] + 4, (b, 1, nk + 19), device="cuda", dtype=torch.int32
        )
        ids = ids_storage[..., :nk]
        ei_storage = torch.randint(
            -3,
            extra_values.shape[0] + 4,
            (b, 1, ne + 17),
            device="cuda",
            dtype=torch.int32,
        )
        ei = ei_storage[..., :ne]
        lengths = torch.tensor(
            [0, 1, 31, 63, 64, nk - 1, nk, nk + 3] * 5, device="cuda", dtype=torch.int32
        )[:b]
        el = torch.tensor(
            [ne, ne // 2, 1, 0, 63, ne - 1, 65, ne] * 5,
            device="cuda",
            dtype=torch.int32,
        )[:b]
        sink = torch.randn(64, device="cuda")
        for h0 in range(0, heads, 16):
            sink[h0], sink[h0 + 1], sink[h0 + 2] = -torch.inf, torch.inf, 20.0
        originals = [x.clone() for x in (raw, extra_raw, ids, ei, q, lengths, el, sink)]

        def candidate():
            return swapab_attention(
                q,
                kv,
                ids,
                lengths,
                sink,
                extra_kv if ne else None,
                ei if ne else None,
                el if ne else None,
            )

        def check(out):
            gold = reference(
                q, values, ids, lengths, sink, (extra_values, ei, el) if ne else None
            )
            floor_error = (gold.bfloat16().double() - gold).square().mean().sqrt()
            error = (out.double() - gold).square().mean().sqrt()
            self.assertLessEqual(error.item(), floor_error.item() * 1.01 + 1e-6)
            # Absolute error is bounded by a BF16 output ULP plus a small
            # accumulation allowance, including cancellation near zero.
            torch.testing.assert_close(out.float(), gold.float(), atol=2e-5, rtol=0.004)

        out = candidate()
        check(out)
        for before, after in zip(
            originals, (raw, extra_raw, ids, ei, q, lengths, el, sink)
        ):
            torch.testing.assert_close(before, after, rtol=0, atol=0, equal_nan=True)
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                candidate()
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                replay_out = candidate()
            for iteration in range(12):
                # Changing device metadata and query values must affect replay;
                # none of the lengths or sparse selections are host constants.
                q.copy_(torch.randn_like(q) * (0.2 + iteration * 0.01))
                ids.copy_(
                    torch.randint(
                        -1, values.shape[0], ids.shape, device="cuda", dtype=torch.int32
                    )
                )
                lengths.copy_(
                    torch.randint(
                        0, nk + 1, lengths.shape, device="cuda", dtype=torch.int32
                    )
                )
                if ne:
                    ei.copy_(
                        torch.randint(
                            -1,
                            extra_values.shape[0],
                            ei.shape,
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                    el.copy_(
                        torch.randint(
                            0, ne + 1, el.shape, device="cuda", dtype=torch.int32
                        )
                    )
                g.replay()
                check(replay_out)

    @unittest.skipUnless(is_gfx95_supported(), "HIP fused inverse RoPE")
    def test_fused_inverse_rope(self):
        torch.manual_seed(103)
        q = torch.randn(4, 16, 512, device="cuda", dtype=torch.bfloat16) * 0.25
        _, cache, _ = make_cache(32)
        indices = torch.randint(352, (4, 128), device="cuda", dtype=torch.int32)
        lengths = torch.full((4,), 128, device="cuda", dtype=torch.int32)
        sink = torch.randn(16, device="cuda")
        angles = torch.randn(16, 32, device="cuda")
        freqs = torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(1)
        positions = torch.arange(4, device="cuda") + 7
        for extra in (False, True):
            kwargs = (
                dict(extra_kv=cache, extra_indices=indices, extra_lengths=lengths)
                if extra
                else {}
            )
            expected = swapab_attention(q, cache, indices, lengths, sink, **kwargs)
            fused_rope_inplace(
                expected[..., -64:],
                None,
                torch.view_as_complex(freqs.view(16, 32, 2)),
                positions,
                inverse=True,
            )
            actual = swapab_attention(
                q, cache, indices, lengths, sink, inv_rope=(freqs, positions), **kwargs
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_packed_pages_and_sink(self):
        for case in (
            (1, 1, 1, 0),
            (5, 16, 133, 0),
            (8, 128, 192, 1024),
            (6, 16, 192, 65),
        ):
            with self.subTest(case=case):
                self.check_case(*case)

    @unittest.skipUnless(is_hip(), "not run on SM100 yet")
    def test_ragged_lengths_and_tile_tails(self):
        for case in ((4, 64, 100, 333), (3, 64, 129, 1000), (8, 64, 33, 65)):
            with self.subTest(case=case):
                self.check_case(*case)

    @unittest.skipUnless(is_hip(), "not run on SM100 yet")
    def test_sharp_softmax(self):
        for q_scale in (2.0, 6.0):
            with self.subTest(q_scale=q_scale):
                self.check_case(4, 64, 256, 512, q_scale=q_scale)

    def test_dynamic_graph_replay(self):
        self.check_case(6, 64, 192, 512, graph=True)

    @unittest.skipIf(is_hip(), "the gfx950 kernel keeps the 16-head contract")
    def test_tp2_head_groups(self):
        """TP2's 32 local heads run as two 16-head groups; each group must read
        its own query heads and sinks and write its own output rows."""
        for case in ((1, 1, 1, 0), (5, 16, 133, 0), (8, 128, 192, 1024)):
            with self.subTest(case=case):
                self.check_case(*case, heads=32)
        self.check_case(6, 64, 192, 512, graph=True, heads=32)

    @unittest.skipUnless(is_hip(), "the gfx950 router cap")
    def test_rows_up_to_gfx950_cap(self):
        self.check_case(MAX_BATCH, 64, 192, 512)
        self.check_case(MAX_BATCH, 64, 192, 512, graph=True)

    @unittest.skipIf(is_hip(), "the gfx950 router has its own cap")
    def test_rows_up_to_cap(self):
        """SM100 admits up to 40 rows: DSpark verify at TP4 bs <= 8."""
        for case in ((20, 16, 133, 0), (40, 128, 192, 1024)):
            with self.subTest(case=case):
                self.check_case(*case)
        self.check_case(40, 64, 192, 512, graph=True)


def poisoned_cache(page, pages=11):
    raw, _, values = make_cache(page, pages)
    stride = raw.shape[1]
    storage = torch.full((pages + 2, stride), POISON, dtype=torch.uint8, device="cuda")
    storage[:pages] = raw
    storage[0, :576] = POISON
    storage[0, page * 576 : page * 576 + 8] = POISON
    values[0] = float("nan")
    cache = storage[:pages].as_strided((pages, page, 1, 584), (stride, 584, 584, 1))
    return cache, values


def cache_at_top(raw, page, pages_total):
    pages, stride = raw.shape
    big = torch.empty((pages_total, stride), dtype=torch.uint8, device="cuda")
    big[-pages:] = raw
    cache = big.as_strided((pages_total, page, 1, 584), (stride, 584, 584, 1))
    return cache, (pages_total - pages) * page


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(), "requires gfx950"
)
class TestSwapABBufferGather(CustomTestCase):
    def test_poisoned_invalid_rows(self):
        torch.manual_seed(7)
        b, nk, ne = 4, 200, 100
        kv, values = poisoned_cache(64)
        extra_kv, extra_values = poisoned_cache(16)
        q = torch.randn(b, 1, 16, 512, device="cuda").bfloat16() * 0.5
        ids = torch.randint(1, values.shape[0], (b, 1, nk), device="cuda").int()
        ei = torch.randint(1, extra_values.shape[0], (b, 1, ne), device="cuda").int()
        ids[..., ::5] = -1
        ids[..., 1::7] = values.shape[0] + 2
        ei[..., ::3] = -2
        ei[..., 1::4] = extra_values.shape[0] + 17
        lengths = torch.tensor([nk, nk - 37, 64, 0], device="cuda", dtype=torch.int32)
        el = torch.tensor([ne, 5, 0, ne], device="cuda", dtype=torch.int32)
        sink = torch.randn(64, device="cuda")
        out = swapab_attention(q, kv, ids, lengths, sink, extra_kv, ei, el)
        gold = reference(q, values, ids, lengths, sink, (extra_values, ei, el))
        self.assertFalse(out.isnan().any().item())
        torch.testing.assert_close(out.float(), gold.float(), atol=2e-5, rtol=0.004)

    def check_top_pages(self, pages_total, past_buffer_range):
        torch.manual_seed(11)
        b, nk, page = 2, 96, 64
        raw, _, values = make_cache(page)
        kv, shift = cache_at_top(raw, page, pages_total)
        self.assertEqual(shift // page * raw.shape[1] > 2**31, past_buffer_range)
        q = torch.randn(b, 1, 16, 512, device="cuda").bfloat16() * 0.5
        ids = torch.randint(0, values.shape[0], (b, 1, nk), device="cuda").int()
        ids[..., ::9] = -1
        lengths = torch.tensor([nk, nk - 21], device="cuda", dtype=torch.int32)
        sink = torch.randn(64, device="cuda")
        shifted = torch.where(ids >= 0, ids + shift, ids)
        out = swapab_attention(q, kv, shifted, lengths, sink)
        gold = reference(q, values, ids, lengths, sink)
        torch.testing.assert_close(out.float(), gold.float(), atol=2e-5, rtol=0.004)

    def test_odd_page_stride(self):
        torch.manual_seed(13)
        b, nk, page, pages = 2, 96, 64, 11
        raw, _, values = make_cache(page, pages)
        stride = raw.shape[1] + 1
        storage = torch.zeros((pages, stride), dtype=torch.uint8, device="cuda")
        storage[:, :-1] = raw
        kv = storage.as_strided((pages, page, 1, 584), (stride, 584, 584, 1))
        q = torch.randn(b, 1, 16, 512, device="cuda").bfloat16() * 0.5
        ids = torch.randint(0, values.shape[0], (b, 1, nk), device="cuda").int()
        ids[..., ::9] = -1
        lengths = torch.tensor([nk, nk - 21], device="cuda", dtype=torch.int32)
        sink = torch.randn(64, device="cuda")
        out = swapab_attention(q, kv, ids, lengths, sink)
        gold = reference(q, values, ids, lengths, sink)
        torch.testing.assert_close(out.float(), gold.float(), atol=2e-5, rtol=0.004)

    def test_top_of_buffer_range(self):
        stride = 64 * 584 + 256
        self.check_top_pages((2**31 - 2) // stride, past_buffer_range=False)

    def test_past_buffer_range(self):
        stride = 64 * 584 + 256
        self.check_top_pages(2**31 // stride + 12, past_buffer_range=True)


@unittest.skipUnless(is_gfx95_supported(), "requires AMD gfx950")
class TestSwapABDequantHip(CustomTestCase):
    def check(self, lo, hi, seed):
        page, pages, bt = 64, 9, 64
        stride = page * 584 + 256
        gen = torch.Generator(device="cuda").manual_seed(seed)
        raw = torch.randint(
            0, 256, (pages, stride), dtype=torch.uint8, device="cuda", generator=gen
        )
        scales = raw[:, page * 576 : page * 584].view(pages, page, 8)
        scales[:] = torch.randint(
            lo, hi, scales.shape, dtype=torch.uint8, device="cuda", generator=gen
        )
        ids = torch.randint(
            0, pages * page, (bt,), dtype=torch.int32, device="cuda", generator=gen
        )
        valid = torch.rand(bt, device="cuda", generator=gen) > 0.2
        base = torch.empty(bt, 512, dtype=torch.bfloat16, device="cuda")
        fused = torch.empty_like(base)
        load_both[(1,)](
            raw, ids, valid.to(torch.int8), base, fused, page, stride, bt, num_warps=4
        )

        a, b = base.view(torch.int16), fused.view(torch.int16)
        both_nan = base.isnan() & fused.isnan()
        bad = (a != b) & ~both_nan
        self.assertEqual(int(bad.sum()), 0, f"{int(bad.sum())} of {bad.numel()} differ")
        self.assertTrue(bool(valid.any()) and bool((~valid).any()))
        self.assertTrue(bool(base[~valid].eq(0).all()))

    def test_serving_exponents(self):
        self.check(100, 150, 1)

    def test_wide_exponents(self):
        self.check(1, 255, 2)

    def test_zero_exponent(self):
        self.check(0, 1, 3)


def launch_combine(combine, b, nt, bd, rope_dim, inputs):
    part, mx, den, sink, freqs, positions = inputs
    out = torch.empty((b, 16, 512), dtype=torch.bfloat16, device="cuda")
    combine[(b, 16, 512 // bd)](
        part,
        mx,
        den,
        sink,
        out,
        freqs,
        positions,
        FREQ_STRIDE=freqs.stride(0),
        ROPE_DIM=rope_dim,
        NT=nt,
        ST=triton.next_power_of_2(nt),
        H=16,
        BD=bd,
        num_warps=4,
    )
    return out


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(), "requires gfx950"
)
class TestSwapABCombineRopeTiles(CustomTestCase):
    def check(self, b, nt, bd, rope_dim, sink_values=None, seed=0):
        self.assertEqual((512 - rope_dim) % bd, 0)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        part = torch.randn(b, nt, 16, 512, device="cuda", generator=gen) * 8
        part[torch.rand(part.shape, device="cuda", generator=gen) < 0.1] = 0
        mx = torch.randn(b, nt, 16, device="cuda", generator=gen) * 3
        den = torch.rand(b, nt, 16, device="cuda", generator=gen)
        den[torch.rand(den.shape, device="cuda", generator=gen) < 0.2] = 0
        sink = torch.randn(16, device="cuda", generator=gen)
        if sink_values is not None:
            sink[: len(sink_values)] = torch.tensor(sink_values, device="cuda")
        angles = torch.randn(64, rope_dim // 2, device="cuda", generator=gen)
        freqs = torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(1)
        freqs[:, :4] = torch.tensor([1.0, 0.0, 0.0, -1.0], device="cuda")
        positions = torch.randint(0, 64, (b,), device="cuda", generator=gen)
        inputs = (part, mx, den, sink, freqs, positions)
        tiled = launch_combine(combine_gfx950, b, nt, bd, rope_dim, inputs)
        generic = launch_combine(_combine, b, nt, bd, rope_dim, inputs)
        self.assertTrue(
            torch.equal(tiled.view(torch.int16), generic.view(torch.int16)),
            f"b={b} nt={nt} bd={bd} rope_dim={rope_dim}",
        )

    def test_bit_identical(self):
        for rope_dim in (64, 128):
            for b, nt, bd in [(1, 4, 64), (4, 20, 64), (3, 17, 64), (4, 5, 128)]:
                if (512 - rope_dim) % bd == 0:
                    self.check(b, nt, bd, rope_dim, seed=b + nt)

    def test_inf_sinks(self):
        self.check(
            2, 9, 64, 64, sink_values=[float("inf"), float("-inf"), 0.0, 1e4], seed=5
        )


if __name__ == "__main__":
    unittest.main()
