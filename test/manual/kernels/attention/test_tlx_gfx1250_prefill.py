"""FP64 references for the gfx1250 uTLX paged prefill kernel."""

import unittest

import torch

from sglang.test.kernels.tlx_attention import make_paged_case, paged_reference
from sglang.test.test_utils import CustomTestCase


class TestGfx1250Prefill(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if (
            torch.version.hip is None
            or not torch.cuda.is_available()
            or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250")
        ):
            raise unittest.SkipTest("requires gfx1250 and the uTLX TDM extension")

    def run_case(self, page, fp8, window, qlens, klens, q_gain=1.0):
        from sglang.kernels.ops.attention.tlx import gfx1250_prefill as prefill

        args, _ = make_paged_case(
            page=page,
            fp8=fp8,
            qlens=qlens,
            klens=klens,
            window=window,
            softmax_scale=0.125,
        )
        # Large logits push the fixed-center pass into its online-softmax replay.
        args["q"].mul_(q_gain)
        reference = paged_reference(args, qlens, klens)
        window = window + 1 if window >= 0 else 0
        scales = dict(k_scale=args["k_descale"], v_scale=args["v_descale"])
        self.assertIsNone(
            prefill.unsupported_reason(
                args["q"],
                args["k"],
                args["v"],
                args["out"],
                args["block_table"],
                args["max_seqlen_k"],
                0.125,
                window,
                **scales,
            )
        )
        actual = prefill.paged_prefill(
            args["q"],
            args["k"],
            args["v"],
            args["out"],
            args["cu_seqlens_q"],
            args["seqused_k"],
            args["block_table"],
            args["max_seqlen_q"],
            args["max_seqlen_k"],
            window,
            **scales,
        )
        self.assertTrue(torch.isfinite(actual).all().item())
        error = (actual.double() - reference).square().mean().sqrt()
        norm = reference.square().mean().sqrt()
        self.assertLess((error / norm).item(), 0.02)

    def test_prefix_and_ragged_lengths(self):
        for page in (16, 64):
            for fp8 in (False, True):
                for window in (-1, 2046, 2047):
                    with self.subTest(page=page, fp8=fp8, window=window):
                        self.run_case(
                            page,
                            fp8,
                            window,
                            qlens=(129, 17, 1000),
                            klens=(129, 2063, 3100),
                        )

    def test_long_chunk_over_prefix(self):
        for page in (16, 64):
            for fp8 in (False, True):
                for window in (-1, 2046, 2047):
                    with self.subTest(page=page, fp8=fp8, window=window):
                        self.run_case(page, fp8, window, qlens=(2048,), klens=(6001,))

    def test_large_logits_replay(self):
        for page in (16, 64):
            for fp8 in (False, True):
                with self.subTest(page=page, fp8=fp8):
                    self.run_case(
                        page, fp8, -1, qlens=(900, 300), klens=(900, 4000), q_gain=40.0
                    )


if __name__ == "__main__":
    unittest.main()
