"""Explicit production BLOCK8/16 precision diagnostics against CPU FP64.

No model/checkpoint/route changes. Run directly on real SM80. Uses the EXISTING
oracle tolerances (.015/.004 output, 2e-5 LSE), never bitwise agreement between
FP32 reduction geometries. Metrics are diagnostic, not replacement thresholds.
"""
import importlib.util
import json
import pathlib
import unittest

import torch

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "_geometry_precision_tiled_tests", HERE / "test_dsv41_sm80_sparse_tiled.py"
)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
base = original.base
candidate = base.leaf("sparse_attention_sm80_headshared").sparse_attention_sm80_headshared
KVLayout = base.KVLayout


def realistic_cache(layout, page):
    """Finite moderate scales: no E4M3 NaN or overflow hiding comparisons.

    FP8 payload comes from CPU rounded normal values; scale bytes 124..128
    represent 1/8..2. FP4 payload has all nibble codes, scale bytes 48..64
    represent .5..2, so decoded magnitudes <=12. No native GPU FP8 arithmetic.
    Nonzero storage offset and padded physical stride are retained.
    """
    pitch = layout.page_bytes(page)
    backing = torch.zeros((3, pitch + 192), device="cuda", dtype=torch.uint8)
    raw = backing[:, 64:64 + pitch]
    if layout == KVLayout.V41:
        values = (torch.randn(3, page * 512) * .5).clamp(-2, 2)
        payload = values.to(torch.float8_e4m3fn).view(torch.uint8)
        scale = torch.randint(124, 129, (3, page * 16), dtype=torch.uint8)
    else:
        payload = torch.randint(0, 256, (3, page * 256), dtype=torch.uint8)
        scale = torch.randint(48, 65, (3, page * 32), dtype=torch.uint8)
    raw[:, :page * layout.data_bytes].copy_(payload)
    raw[:, layout.scale_offset(page):pitch].copy_(scale)
    return raw


def metrics(actual, reference):
    a, r = actual.detach().cpu(), reference.detach().cpu()
    delta = a.float() - r.float()
    bits_a = a.contiguous().view(torch.int16).int()
    bits_r = r.contiguous().view(torch.int16).int()
    ordered_a = torch.where(bits_a < 0, -32768 - bits_a, bits_a)
    ordered_r = torch.where(bits_r < 0, -32768 - bits_r, bits_r)
    return {
        "max_abs": delta.abs().max().item(),
        "relative_rms": (delta.square().mean().sqrt()
                         / r.float().square().mean().sqrt().clamp_min(1e-30)).item(),
        "bf16_max_ulp": (ordered_a - ordered_r).abs().max().item(),
        "bf16_different": (a != r).sum().item(),
    }


@unittest.skipUnless(original.SM80, "real SM80 required")
class TestGeometryPrecision(unittest.TestCase):
    def check(self, got, expected, baseline, tag):
        # Exactly the original test standard; do not loosen for a geometry.
        torch.testing.assert_close(got[0].cpu(), expected[0], rtol=.015, atol=.004)
        torch.testing.assert_close(got[1].cpu(), expected[1], rtol=2e-5, atol=2e-5)
        self.assertTrue(torch.isfinite(got[0]).all().item())
        finite = torch.isfinite(expected[1])
        lse_error = (got[1].cpu()[finite] - expected[1][finite]).abs()
        report = {"case": tag, "oracle": metrics(got[0], expected[0]),
                  "block32": metrics(got[0], baseline[0]),
                  "lse_max_abs": lse_error.max().item() if lse_error.numel() else 0.}
        print(json.dumps(report), flush=True)

    def fixture(self, k, extra_page):
        q = torch.empty(1, 8, 512, device="cuda", dtype=torch.bfloat16)
        raw = realistic_cache(KVLayout.V41, 256)
        extra = realistic_cache(KVLayout.V41_FP4, extra_page)
        ids = torch.randint(0, 768, (1, k), device="cuda", dtype=torch.int32)
        ei = torch.randint(0, extra_page * 3, (1, 65), device="cuda", dtype=torch.int64)
        ids[:, ::11] = -1
        ei[:, ::13] = -3
        # Duplicates cross BLOCK8,16,32 boundaries and pools remain independent.
        for at in (7, 8, 15, 16, 31, 32):
            if at < k:
                ids[:, at] = ids[:, 1]
        lengths = torch.tensor([k - 3], device="cuda", dtype=torch.int32)
        el = torch.tensor([63], device="cuda", dtype=torch.int32)
        sink = torch.linspace(-2, 2, 8, device="cuda")
        kw = dict(topk_length=lengths, attn_sink=sink, extra_cache=extra,
                  extra_indices=ei, extra_page_size=extra_page,
                  extra_layout=KVLayout.V41_FP4, extra_topk_length=el)
        return q, raw, ids, extra, ei, lengths, el, sink, kw

    def test_explicit_geometry_random_tiny_and_sharp_logits(self):
        torch.manual_seed(416031)
        for k, page in ((65, 128), (128, 256), (513, 128)):
            q, raw, ids, ex, ei, lengths, el, sink, kw = self.fixture(k, page)
            # Tiny/tied logits, actual normal range, and sharper but bounded
            # finite logits. This is not a model-level logit-margin test.
            for qscale in (0., 1e-4, .15, 2.):
                q.copy_((torch.randn_like(q.float()) * qscale).bfloat16())
                expected = base.reference(q, raw, ids, 256, KVLayout.V41,
                                          512 ** -.5, lengths, sink,
                                          (ex, ei, page, KVLayout.V41_FP4, el))
                baseline = original.tiled(q, raw, ids, 256, KVLayout.V41,
                                          512 ** -.5, **kw)
                for block in (8, 16):
                    with self.subTest(k=k, page=page, qscale=qscale, block=block):
                        got = candidate(q, raw, ids, 256, KVLayout.V41,
                                        512 ** -.5, block=block, warps=4, **kw)
                        self.check(got, expected, baseline,
                                   f"eager/K{k}/P{page}/q{qscale}/BLOCK{block}")

    def test_graph_replay_mutates_queries_cache_bytes_and_lengths(self):
        torch.manual_seed(416032)
        for block, k, page in ((8, 128, 128), (16, 512, 256)):
            q, raw, ids, ex, ei, lengths, el, sink, kw = self.fixture(k, page)
            q.normal_(0, .15)
            out = torch.empty_like(q)
            lse = torch.empty((1, 8), device="cuda")

            def call():
                return candidate(q, raw, ids, 256, KVLayout.V41, 512 ** -.5,
                                 block=block, warps=4, out=out, lse=lse, **kw)

            for _ in range(3):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            for replay in range(4):
                # All mutation happens outside capture; kernels must observe
                # current bytes, not stale decoded values or static lengths.
                if replay:
                    q.normal_(0, .15 if replay != 2 else 1e-4)
                    raw.copy_(realistic_cache(KVLayout.V41, 256))
                    ex.copy_(realistic_cache(KVLayout.V41_FP4, page))
                    ei.copy_(torch.randint(0, page * 3, ei.shape, device="cuda"))
                    ids[:, ::5] = -1
                    lengths.fill_(0 if replay == 2 else k - replay)
                    el.fill_(0 if replay == 2 else 65 - replay)
                    sink.copy_(torch.linspace(-20, 20, 8, device="cuda"))
                graph.replay()
                expected = base.reference(q, raw, ids, 256, KVLayout.V41,
                                          512 ** -.5, lengths, sink,
                                          (ex, ei, page, KVLayout.V41_FP4, el))
                baseline = original.tiled(q, raw, ids, 256, KVLayout.V41,
                                          512 ** -.5, **kw)
                self.check((out, lse), expected, baseline,
                           f"graph/BLOCK{block}/replay{replay}")
                before = out.clone(), lse.clone()
                graph.replay()
                # Identical input replay is deterministic; distinct geometries
                # need NOT be bitwise identical to one another.
                torch.testing.assert_close(out, before[0], rtol=0, atol=0)
                torch.testing.assert_close(lse, before[1], rtol=0, atol=0)

    def test_each_production_geometry_large_finite_cardinality(self):
        saved = original.tiled
        import functools
        try:
            for block in (8, 16):
                original.tiled = functools.partial(candidate, block=block, warps=4)
                original.TestSparseTiledSM80.test_large_finite_scores_preserve_chunk_cardinality(self)
        finally:
            original.tiled = saved

    # Reused cardinality method only requires this existing comparison helper.
    assert_attention = base.TestSparseSM80.assert_attention


if __name__ == "__main__":
    unittest.main(verbosity=2)
