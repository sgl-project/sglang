"""TOP split decode and spec verify attention, planned once per forward: every row bit for bit its per-row split.

``split_decode_paged`` is the reference program (one entry per row and piece, masked empty pieces, FP32 combine input).
Serving runs ``split_decode`` (piece-major entries, BF16 combine input) for decode, and ``split_verify`` for chain
verify, where row j of a request is the decode of position cache_seqlens - d + j and the request's d rows share each
full piece's read. Both are compared with the reference row by row over the same paged cache, eager and with the plan
built inside a captured CUDA graph (as serving builds it). Qwen3.5 attention shapes: 16 query heads, 2 KV heads, head
dim 256, page 128, a 40960-token page table."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")

HQ, HKV, D, PAGE, SPLIT, CTX = 16, 2, 256, 128, 4096, 40960
dev = "cuda"


def _bits(x):
    return x.contiguous().view(torch.int16)


def _lens(d):
    """cache_seqlens (draft rows included) for d rows: rows inside one piece, on and off the split grid, rows crossing
    a piece boundary at every offset, the last row ending a piece, deep prefixes, and rows with no key (graph padding)."""
    s = SPLIT
    lens = [d, d + 1, 700, s, s + d - 1, 2 * s + 1, 3 * s + 5, 9 * s + 17, CTX - 3, 1]
    lens += [
        s + j for j in range(1, d)
    ]  # row j - 1 is the last of piece 0, row j the first of piece 1
    lens += [2 * s - d + 1 + j for j in range(d)]
    return lens


def _cache(lens, width):
    n_pages = sum(-(-L // PAGE) for L in lens) + 1
    kc = torch.randn(n_pages, PAGE, HKV, D, device=dev).to(torch.bfloat16)
    vc = torch.randn(n_pages, PAGE, HKV, D, device=dev).to(torch.bfloat16)
    perm = torch.randperm(n_pages - 1, device=dev).to(torch.int32) + 1
    pt = torch.zeros(len(lens), width, device=dev, dtype=torch.int32)
    used = 0
    for i, L in enumerate(lens):
        k = -(-L // PAGE)
        pt[i, :k] = perm[used : used + k]
        used += k
    return kc, vc, pt


def _reference(q, kc, vc, pt, seqlens, d, pieces):
    from sglang.srt.true_on_policy.attention import split_decode_paged

    b = seqlens.shape[0]
    steps = torch.arange(1 - d, 1, device=dev, dtype=torch.int32)
    return split_decode_paged(
        q,
        kc,
        vc,
        pt.repeat_interleave(d, dim=0),
        (seqlens.view(b, 1) + steps).reshape(-1),
        SPLIT,
        pieces,
        D**-0.5,
    )


def _bad_rows(a, b, d):
    rows = (_bits(a) != _bits(b)).flatten(1).any(1)
    return [(int(r) // d, int(r) % d) for r in rows.nonzero().flatten()]


def _verify(q, kc, vc, pt, seqlens, d, pieces):
    from sglang.srt.true_on_policy.attention import SplitVerifyPlan, split_verify

    plan = SplitVerifyPlan(
        page_table=pt,
        cache_seqlens=seqlens,
        rows_per_request=d,
        split=SPLIT,
        pieces=pieces,
        page_size=PAGE,
    )
    return split_verify(q, kc, vc, plan, D**-0.5)


def _decode(q, kc, vc, pt, seqlens, pieces):
    from sglang.srt.true_on_policy.attention import SplitDecodePlan, split_decode

    plan = SplitDecodePlan(
        page_table=pt, cache_seqlens=seqlens, split=SPLIT, pieces=pieces, page_size=PAGE
    )
    return split_decode(q, kc, vc, plan, D**-0.5)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestSplitVerify(unittest.TestCase):
    width = CTX // PAGE
    pieces = -(-width // (SPLIT // PAGE))

    def test_verify_rows_equal_decode_rows(self):
        for d in range(2, 7):
            with self.subTest(d=d):
                torch.manual_seed(d)
                lens = _lens(d)
                kc, vc, pt = _cache(lens, self.width)
                seqlens = torch.tensor(lens, device=dev, dtype=torch.int32)
                q = torch.randn(len(lens) * d, HQ, D, device=dev).to(torch.bfloat16)
                out = _verify(q, kc, vc, pt, seqlens, d, self.pieces)
                ref = _reference(q, kc, vc, pt, seqlens, d, self.pieces)
                self.assertEqual(out.shape, ref.shape)
                self.assertEqual(
                    _bad_rows(out, ref, d), [], f"(request, row) differ at d={d}"
                )

    def test_decode_rows_equal_reference(self):
        from sglang.srt.true_on_policy.attention import split_decode_paged

        torch.manual_seed(1)
        lens = _lens(4) + [0]
        kc, vc, pt = _cache(lens, self.width)
        seqlens = torch.tensor(lens, device=dev, dtype=torch.int32)
        q = torch.randn(len(lens), HQ, D, device=dev).to(torch.bfloat16)
        out = _decode(q, kc, vc, pt, seqlens, self.pieces)
        ref = split_decode_paged(q, kc, vc, pt, seqlens, SPLIT, self.pieces, D**-0.5)
        self.assertEqual(_bad_rows(out, ref, 1), [])

    def _graph(self, fn, q, seqlens, d):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()  # compile outside the capture
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        gen = torch.Generator(device="cpu").manual_seed(11)
        for _ in range(3):
            seqlens.copy_(torch.randint(1, CTX + 1, seqlens.shape, generator=gen))
            q.copy_(torch.randn(q.shape, generator=gen).to(torch.bfloat16))
            graph.replay()
            yield out

    def test_cuda_graph_replay(self):
        """Plans built inside the captured graph and replayed with other lengths: no host sync, the reference bytes."""
        from sglang.srt.true_on_policy.attention import split_decode_paged

        torch.manual_seed(7)
        d, n = 4, len(_lens(4))
        kc, vc, pt = _cache([CTX] * n, self.width)
        seqlens = torch.ones(n, device=dev, dtype=torch.int32)
        q = torch.randn(n * d, HQ, D, device=dev).to(torch.bfloat16)
        for out in self._graph(
            lambda: _verify(q, kc, vc, pt, seqlens, d, self.pieces), q, seqlens, d
        ):
            ref = _reference(q, kc, vc, pt, seqlens, d, self.pieces)
            self.assertEqual(_bad_rows(out, ref, d), [])
        q1 = q[:n].clone()
        for out in self._graph(
            lambda: _decode(q1, kc, vc, pt, seqlens, self.pieces), q1, seqlens, 1
        ):
            ref = split_decode_paged(
                q1, kc, vc, pt, seqlens, SPLIT, self.pieces, D**-0.5
            )
            self.assertEqual(_bad_rows(out, ref, 1), [])


if __name__ == "__main__":
    unittest.main()
