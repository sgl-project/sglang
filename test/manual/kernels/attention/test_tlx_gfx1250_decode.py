"""FP64 references for the gfx1250 uTLX paged decode and verify kernel."""

import unittest

import torch
import triton

from sglang.test.kernels.tlx_attention import make_paged_case, paged_reference
from sglang.test.test_utils import CustomTestCase


class TestGfx1250Decode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if (
            torch.version.hip is None
            or not torch.cuda.is_available()
            or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250")
        ):
            raise unittest.SkipTest("requires gfx1250 and the uTLX TDM extension")

    def run_case(self, page, fp8, window, query_length, klens):
        from sglang.kernels.ops.attention.tlx import gfx1250_decode as decode

        qlens = (query_length,) * len(klens)
        args, _ = make_paged_case(
            page=page,
            fp8=fp8,
            qlens=qlens,
            klens=klens,
            window=window,
            softmax_scale=0.125,
        )
        reference = paged_reference(args, qlens, klens)
        # The table need not cover whole 64-key tiles; the kernel clamps page indices.
        table = args["block_table"][:, : triton.cdiv(max(klens), page)].contiguous()
        window = window + 1 if window >= 0 else 0
        call = (
            args["q"],
            args["k"],
            args["v"],
            args["out"],
            table,
            args["seqused_k"],
            query_length,
            max(klens),
            window,
        )
        scales = dict(k_scale=args["k_descale"], v_scale=args["v_descale"])
        self.assertIsNone(decode.unsupported_reason(*call, **scales))
        actual = decode.paged_decode(
            args["q"],
            args["k"],
            args["v"],
            args["out"],
            args["seqused_k"],
            table,
            query_length,
            max(klens),
            0.125,
            window,
            **scales,
        )
        self.assertTrue(torch.isfinite(actual).all().item())
        error = (actual.double() - reference).square().mean().sqrt()
        norm = reference.square().mean().sqrt()
        self.assertLess((error / norm).item(), 0.02)

    def test_verify_and_decode(self):
        for page in (16, 64):
            for fp8 in (False, True):
                for window in (-1, 2046):
                    for query_length, klens in (
                        (8, (8, 300, 2063, 5000)),
                        (1, (1, 777, 4096)),
                    ):
                        with self.subTest(
                            page=page, fp8=fp8, window=window, query_length=query_length
                        ):
                            self.run_case(page, fp8, window, query_length, klens)

    def test_long_context_splits(self):
        for page in (16, 64):
            for fp8 in (False, True):
                with self.subTest(page=page, fp8=fp8):
                    self.run_case(page, fp8, -1, 8, (40000, 33000))


if __name__ == "__main__":
    unittest.main()
