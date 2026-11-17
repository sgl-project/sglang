"""The Triton DSv4 dual-scope sparse decode splits K when few query heads share a rank, and the
split result matches the non-split kernel and a float64 reference on the packed fp8 KV layout."""

import unittest
from unittest import mock

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd")

D, NOPE, ROPE = 512, 448, 64
DATA_BYTES, SCALE_BYTES = 576, 8
SWA, TOPK = 128, 512


class TestSmallHqSplitkDecision(CustomTestCase):
    def test_small_hq_always_splits(self):
        from sglang.kernels.ops.attention.dsv4.triton_sparse_decode import (
            triton_mla_kernels_decode_fused as fused,
        )

        decide = fused._decide_splitk_dual_scope
        total_topk = SWA + TOPK
        for tokens, expected in ((1, 8), (6, 8), (48, 8), (49, 4), (144, 4), (256, 4)):
            for h_q in (8, 16):
                with self.subTest(tokens=tokens, h_q=h_q):
                    self.assertEqual(decide(tokens, h_q, total_topk), expected)
        self.assertEqual(decide(144, 8, SWA), 4)
        # Larger head counts keep their thresholds.
        self.assertEqual(decide(144, 32, total_topk), 0)
        self.assertEqual(decide(144, 64, total_topk), 0)
        self.assertEqual(decide(144, 128, total_topk), 4)
        self.assertEqual(decide(4, 128, total_topk), 8)


def _pack_pool(num_pages, page, fp8_dtype, gen):
    """Random keys in the V4 packed layout (page = page * 576 data bytes, then page * 8
    ue8m0 scale bytes); returns the 4-D cache view and the dequantized keys [slots, D]."""
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

    page_bytes = KVLayout.V4.page_bytes(page)
    raw = torch.zeros(num_pages, page_bytes, dtype=torch.uint8, device="cuda")
    nope = (torch.randn(num_pages, page, NOPE, generator=gen) * 0.5).to(fp8_dtype)
    rope = (torch.randn(num_pages, page, ROPE, generator=gen) * 0.5).to(torch.bfloat16)
    exps = torch.randint(124, 130, (num_pages, page, 7), generator=gen).to(torch.uint8)
    data = raw[:, : page * DATA_BYTES].view(num_pages, page, DATA_BYTES)
    data[..., :NOPE] = nope.view(torch.uint8).cuda()
    data[..., NOPE:] = rope.view(torch.uint8).cuda()
    scales = raw[:, page * DATA_BYTES : page * (DATA_BYTES + SCALE_BYTES)]
    scales.view(num_pages, page, SCALE_BYTES)[..., :7] = exps.cuda()
    scale = torch.exp2(exps.float() - 127.0).repeat_interleave(64, dim=-1)
    deq = torch.cat((nope.float() * scale, rope.float()), dim=-1).view(-1, D)
    cache = raw[:, : page * (DATA_BYTES + SCALE_BYTES)]
    return cache.view(num_pages, page, 1, DATA_BYTES + SCALE_BYTES), deq.cuda()


def _reference(q, sink, scopes, sm_scale):
    """float64 attention with a sink over the concatenated (keys, indices [n, 1, w]) scopes."""
    kv = torch.cat([deq[idx[:, 0].long()] for deq, idx in scopes], dim=1).double()
    scores = torch.einsum("nhd,nwd->nhw", q[:, 0].double(), kv) * sm_scale
    sink = sink.double()[None, :, None].expand(q.shape[0], -1, 1)
    prob = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
    return torch.einsum("nhw,nwd->nhd", prob, kv)


@unittest.skipUnless(
    is_hip(), "the Triton DSv4 sparse decode is the ROCm FlashMLA path"
)
class TestSmallHqSplitkNumerics(CustomTestCase):
    def test_splitk_matches_unsplit_and_reference(self):
        import triton.language as tl

        from sglang.kernels.ops.attention.dsv4.triton_sparse_decode import (
            triton_fp8_attention_fwd,
        )
        from sglang.kernels.ops.attention.dsv4.triton_sparse_decode import (
            triton_mla_kernels_decode_fused as fused,
        )

        fp8_dtype = (
            torch.float8_e4m3fnuz
            if fused._KV_FP8_TY == tl.float8e4b8
            else torch.float8_e4m3fn
        )
        decide = fused._decide_splitk_dual_scope
        decisions = []

        def record_decision(*args):
            decisions.append(decide(*args))
            return decisions[-1]

        gen = torch.Generator().manual_seed(7)
        main_cache, main_deq = _pack_pool(4, 256, fp8_dtype, gen)
        extra_cache, extra_deq = _pack_pool(16, 64, fp8_dtype, gen)
        heads, sm_scale = 8, D**-0.5
        sink = torch.randn(heads, generator=gen).cuda()
        for rows in (6, 144):
            with self.subTest(rows=rows):
                q = (torch.randn(rows, 1, heads, D, generator=gen) * 0.5).to(
                    torch.bfloat16
                )
                main_idx = torch.stack(
                    [
                        torch.randperm(main_deq.shape[0], generator=gen)[:SWA]
                        for _ in range(rows)
                    ]
                )
                extra_idx = torch.stack(
                    [
                        torch.randperm(extra_deq.shape[0], generator=gen)[:TOPK]
                        for _ in range(rows)
                    ]
                )
                main_idx = main_idx.to(torch.int32).view(rows, 1, SWA).cuda()
                extra_idx = extra_idx.to(torch.int32).view(rows, 1, TOPK).cuda()
                kwargs = dict(
                    q=q.cuda(),
                    k_cache=main_cache,
                    head_dim_v=D,
                    softmax_scale=sm_scale,
                    indices=main_idx,
                    attn_sink=sink,
                    extra_k_cache=extra_cache,
                    extra_indices_in_kvcache=extra_idx,
                    topk_length=torch.full(
                        (rows,), SWA, dtype=torch.int32, device="cuda"
                    ),
                    extra_topk_length=torch.full(
                        (rows,), TOPK, dtype=torch.int32, device="cuda"
                    ),
                )
                decisions.clear()
                with mock.patch.object(
                    fused, "_decide_splitk_dual_scope", side_effect=record_decision
                ):
                    split = triton_fp8_attention_fwd(**kwargs)[0]
                self.assertTrue(decisions and all(d > 0 for d in decisions))
                with mock.patch.object(
                    fused, "_decide_splitk_dual_scope", return_value=0
                ):
                    unsplit = triton_fp8_attention_fwd(**kwargs)[0]
                ref = _reference(
                    kwargs["q"],
                    sink,
                    [(main_deq, main_idx), (extra_deq, extra_idx)],
                    sm_scale,
                )
                bound = 2e-2 * max(1.0, ref.abs().max().item())
                for name, out in (("split", split), ("unsplit", unsplit)):
                    err = (out.view(rows, heads, D).double() - ref).abs().max().item()
                    self.assertLess(err, bound, f"{name} vs reference {err:.3e}")
                torch.testing.assert_close(split, unsplit, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
