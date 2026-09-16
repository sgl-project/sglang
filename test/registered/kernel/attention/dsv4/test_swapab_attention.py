"""V4 packed-page addressing, softmax/sink and dynamic graph replay checks."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.swapab_attention import swapab_attention
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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
        (scores, sink[None, :16, None].double().expand(q.shape[0], -1, -1)), -1
    )
    prob = torch.nan_to_num(torch.softmax(scores, -1)[..., :-1])
    return prob @ kv


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "requires Blackwell SM100/SM103 tcgen05",
)
class TestSwapABAttention(CustomTestCase):
    def check_case(self, b, page, nk, ne, graph=False):
        torch.manual_seed(20260915 + b + nk + ne)
        storage = torch.full(
            (b, 1, 64, 512), float("nan"), device="cuda", dtype=torch.bfloat16
        )
        q = storage[:, :, :16]
        q.copy_(torch.randn_like(q) * 0.5)
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
            [0, 1, 31, 63, 64, nk - 1, nk, nk + 3][:b], device="cuda", dtype=torch.int32
        )
        el = torch.tensor(
            [ne, ne // 2, 1, 0, 63, ne - 1, 65, ne][:b],
            device="cuda",
            dtype=torch.int32,
        )
        sink = torch.randn(64, device="cuda")
        sink[0], sink[1], sink[2] = -torch.inf, torch.inf, 20.0
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
            self.assertTrue(torch.isfinite(out).all().item())
            self.assertEqual(out.shape, (b, 16, 512))
            self.assertTrue(out.is_contiguous())
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

    def test_packed_pages_and_sink(self):
        for case in (
            (1, 1, 1, 0),
            (5, 16, 133, 0),
            (6, 64, 192, 512),
            (8, 128, 192, 1024),
            (6, 16, 192, 65),
        ):
            with self.subTest(case=case):
                self.check_case(*case)

    def test_dynamic_graph_replay(self):
        self.check_case(6, 64, 192, 512, graph=True)


if __name__ == "__main__":
    unittest.main()
