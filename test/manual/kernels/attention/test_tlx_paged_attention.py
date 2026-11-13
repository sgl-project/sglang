"""Independent FP64 references for the actual paged uTLX entry point."""

import unittest

import torch

from sglang.kernels.ops.attention.tlx import paged
from sglang.kernels.ops.attention.tlx.paged import unified_attention
from sglang.test.kernels.tlx_attention import make_paged_case, paged_reference
from sglang.test.test_utils import CustomTestCase


class TestPagedUTLX(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if (
            torch.version.hip is None
            or not torch.cuda.is_available()
            or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250")
        ):
            raise unittest.SkipTest("requires gfx1250 and the uTLX TDM extension")

    def check_output(self, args, reference):
        actual = unified_attention(**args)
        self.assertTrue(torch.isfinite(actual).all().item())
        error = (actual.double() - reference).square().mean().sqrt()
        norm = reference.square().mean().sqrt()
        self.assertLess(
            (error / norm).item(), 0.02 if actual.dtype == torch.bfloat16 else 0.004
        )
        # Empty attention rows must be exact zeros, independent of RMS tolerance.
        empty = reference.abs().sum(-1) == 0
        self.assertTrue((actual[empty] == 0).all().item())

    def test_prefill_and_prefix(self):
        for dim in (64, 128):
            for dtype in (torch.bfloat16, torch.float16):
                for window in (-1, 0, 2047):
                    with self.subTest(dim=dim, dtype=dtype, window=window):
                        args, ref = make_paged_case(
                            dim,
                            dtype,
                            qlens=(129, 17, 5),
                            klens=(129, 2063, 3),
                            window=window,
                        )
                        self.check_output(args, ref)

    def test_fp8_decode_and_verify(self):
        for dim in (64, 128):
            for page in (16, 128):
                for window in (-1, 2047):
                    with self.subTest(dim=dim, page=page, window=window):
                        args, ref = make_paged_case(
                            dim,
                            fp8=True,
                            page=page,
                            qlens=(0, 8, 1),
                            klens=(0, 4097, 2051),
                            window=window,
                        )
                        self.check_output(args, ref)

    def test_noncausal_mha(self):
        args, ref = make_paged_case(group=1, causal=False)
        self.check_output(args, ref)

    def test_long_noncausal_gqa(self):
        args, ref = make_paged_case(fp8=True, qlens=(1025,), klens=(513,), causal=False)
        self.check_output(args, ref)

    def test_long_prefill_with_shuffled_page_pairs(self):
        # Exercise the two-page load path through the public interface. The
        # partial final page and graph replay must use current page IDs.
        for dtype, fp8 in (
            (torch.bfloat16, True),
            (torch.bfloat16, False),
            (torch.float16, False),
        ):
            for window in (-1, 0, 2047):
                with self.subTest(dtype=dtype, fp8=fp8, window=window):
                    args, ref = make_paged_case(
                        dtype=dtype,
                        fp8=fp8,
                        qlens=(1025,),
                        klens=(2051,),
                        window=window,
                    )
                    last_page = args["block_table"][0, 2051 // 16].item()
                    for cache in (args["k"], args["v"]):
                        cache[last_page, 2051 % 16 :] = float("nan")
                    self.check_output(args, ref)
                    if fp8 and window == -1:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            unified_attention(**args)
                        args["block_table"][0, :2] = args["block_table"][0, :2].flip(0)
                        updated = paged_reference(args, (1025,), (2051,))
                        for _ in range(30):
                            graph.replay()
                            actual = args["out"].double()
                            self.assertTrue(torch.isfinite(actual).all().item())
                            relative = (
                                (actual - updated).square().mean()
                                / updated.square().mean()
                            ).sqrt()
                            self.assertLess(relative.item(), 0.02)

    def test_tall_prefill_tiles(self):
        # Long full-attention prefill takes 256-row tiles once it fills the GPU.
        # Report one compute unit so test-sized prompts take that path anywhere.
        qlens, klens = (1100, 2050), (1100, 4100)
        compute_units = paged._compute_units
        paged._compute_units = lambda device: 1
        try:
            for page in (16, 64, 128):
                for fp8 in (False, True):
                    with self.subTest(page=page, fp8=fp8):
                        args, ref = make_paged_case(
                            fp8=fp8, page=page, qlens=qlens, klens=klens
                        )
                        block_m, block_n, _, num_warps = paged._launch_config(
                            args["q"],
                            args["k"],
                            len(qlens),
                            args["max_seqlen_q"],
                            args["max_seqlen_k"],
                            -1,
                        )
                        self.assertEqual((block_m, num_warps), (256, 8))
                        self.assertEqual(block_n, max(page, 64))
                        for seq, length in enumerate(klens):
                            physical = args["block_table"][seq, length // page].item()
                            for cache in (args["k"], args["v"]):
                                cache[physical, length % page :] = float("nan")
                        self.check_output(args, ref)
        finally:
            paged._compute_units = compute_units

    def test_gqa_head_mapping_and_page_sizes(self):
        for group in (2, 4, 16):
            for page in (16, 64, 128):
                with self.subTest(group=group, page=page):
                    args, ref = make_paged_case(
                        group=group, page=page, fp8=True, qlens=(1, 1), klens=(513, 257)
                    )
                    self.check_output(args, ref)

    def test_uninitialized_page_tail_does_not_poison_output(self):
        # The last physical page can contain NaNs beyond the logical KV length.
        # A zero probability alone cannot suppress them in the PV dot product.
        qlens, klens = (17, 7), (31, 513)
        args, ref = make_paged_case(qlens=qlens, klens=klens, causal=False)
        page = args["k"].shape[1]
        for seq, length in enumerate(klens):
            physical = args["block_table"][seq, length // page].item()
            for cache in (args["k"], args["v"]):
                cache[physical, length % page :] = float("nan")
        self.check_output(args, ref)

    def test_fp8_query_decode(self):
        for dim in (64, 128):
            args, _ = make_paged_case(dim, fp8=True, qlens=(1, 1, 1))
            args["q_descale"] = torch.tensor(0.63, device="cuda")
            args["q"] = (args["q"].float() / args["q_descale"]).to(torch.float8_e4m3fn)
            ref = paged_reference(args, (1, 1, 1), (31, 513, 2063))
            self.check_output(args, ref)

    def test_zero_kv_and_masked_rows(self):
        args, ref = make_paged_case(qlens=(8, 3, 1), klens=(2, 0, 0))
        self.check_output(args, ref)

    def test_graph_replay_uses_updated_metadata(self):
        qlens, klens = (8, 1, 3), (64, 65, 129)
        args, ref = make_paged_case(fp8=True, qlens=qlens, klens=klens, window=31)
        self.check_output(args, ref)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            unified_attention(**args)
        args["seqused_k"].copy_(torch.tensor((17, 33, 128), device="cuda"))
        args["block_table"][0, :2] = args["block_table"][0, :2].flip(0)
        updated_ref = paged_reference(args, qlens, (17, 33, 128))
        for _ in range(30):
            graph.replay()
            actual = args["out"].double()
            relative = (
                actual - updated_ref
            ).square().mean().sqrt() / updated_ref.square().mean().sqrt()
            self.assertLess(relative.item(), 0.02)

    def test_capability_rejects_before_launch(self):
        args, _ = make_paged_case()
        for update in (
            {"softcap": 10.0},
            {"window_size": (4, 2)},
            {"q_descale": torch.ones(2, device="cuda")},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                unified_attention(**(args | update))


if __name__ == "__main__":
    unittest.main()
