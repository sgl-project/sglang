"""Unit tests for NPU speculative decoding fixes, platform hooks, and FP8 LUT."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=10, suite="extra-a-test-npu")


class TestPlatformDefaultDraftAttentionBackend(unittest.TestCase):
    def test_platform_overrides_return_expected_backends(self):
        from sglang.srt.platforms.npu import NPUSRTPlatform
        from sglang.srt.platforms.rocm import RocmSRTPlatform
        from sglang.srt.platforms.xpu import XpuSRTPlatform

        self.assertEqual(
            NPUSRTPlatform().get_default_speculative_draft_attention_backend("DSPARK"),
            "ascend",
        )
        self.assertEqual(
            RocmSRTPlatform().get_default_speculative_draft_attention_backend("DSPARK"),
            "triton",
        )
        self.assertEqual(
            XpuSRTPlatform().get_default_speculative_draft_attention_backend("DFLASH"),
            "triton",
        )


@unittest.skipUnless(
    is_npu() and torch.npu.is_available(),
    "Requires Ascend NPU",
)
class TestNpuSpecDecodeFixes(unittest.TestCase):
    def test_dequant_e4m3fn_lut_bit_exact_all_256_bytes(self):
        from sglang.srt.hardware_backend.npu.quantization.dtype_utils import (
            _dequant_e4m3fn_to_float32,
        )

        u8_cpu = torch.arange(256, dtype=torch.uint8)
        u8_npu = u8_cpu.to("npu").view(torch.float8_e4m3fn)
        got = _dequant_e4m3fn_to_float32(u8_npu).cpu()
        ref = u8_cpu.view(torch.float8_e4m3fn).to(torch.float32)
        nan_mask = torch.isnan(got) & torch.isnan(ref)
        self.assertTrue(
            ((got.view(torch.int32) == ref.view(torch.int32)) | nan_mask).all()
        )

    def test_softcap_and_cast_to_float32_logits_npu(self):
        from sglang.kernels.ops.activation.softcap import (
            cast_to_float32_logits,
            softcap_to_float32_logits,
        )

        torch.manual_seed(44)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(8, 2048, dtype=dtype, device="npu") * 5.0
            out = torch.empty_like(x, dtype=torch.float32)
            ret = cast_to_float32_logits(x, out)
            self.assertIs(ret, out)
            self.assertTrue(torch.equal(out, x.float()))

            sc = 30.0
            out_sc = torch.empty_like(x, dtype=torch.float32)
            softcap_to_float32_logits(x, sc, out_sc)
            ref_sc = torch.tanh(x.float() / sc) * sc
            self.assertLess((out_sc - ref_sc).abs().max().item(), 1e-5)

    def test_markov_greedy_prefers_over_sample_on_npu(self):
        from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
            MarkovGreedyStep,
        )

        for bs in (1, 4, 16, 32):
            self.assertTrue(MarkovGreedyStep.prefers_over_sample(bs))

    def test_markov_sample_step_npu_softcap_vocab_limit_and_fallback(self):
        from sglang.kernels.ops.speculative.dspark import markov_sample as ms

        torch.manual_seed(33)
        B, V = 8, 2048
        base = torch.randn(B, V, dtype=torch.bfloat16, device="npu") * 10.0
        base[:, 1024:] = 1e4
        bias = torch.randn(B, V, dtype=torch.bfloat16, device="npu") * 2.0
        temperatures = torch.zeros(B, dtype=torch.float32, device="npu")
        offset = torch.zeros(1, dtype=torch.int64, device="npu")

        for softcap, vocab_limit in (
            (None, None),
            (30.0, None),
            (None, 1024),
            (25.0, 1024),
        ):
            with self.subTest(softcap=softcap, vocab_limit=vocab_limit):
                kwargs = dict(
                    base_logits=base,
                    bias=bias,
                    temperatures=temperatures,
                    key=12345,
                    offset=offset,
                    step=0,
                    step_dtype=torch.bfloat16,
                    softcap=softcap,
                    vocab_limit=vocab_limit,
                )
                tok_exec = ms.MarkovSampleStep.execute(**kwargs)
                tok_ref = ms.MarkovSampleStep.torch(**kwargs)
                self.assertTrue(torch.equal(tok_exec, tok_ref))
                if vocab_limit is not None:
                    self.assertTrue((tok_exec < vocab_limit).all())

        # Simulate an older sgl_kernel_npu wheel lacking softcap/vocab_limit params:
        # MarkovSampleStep.execute must fall back to cls.torch instead of ignoring them.
        called = []

        def _legacy_kernel(**kw):
            called.append(kw)
            raise AssertionError(
                "Legacy kernel should not be called with softcap/vocab_limit"
            )

        with patch.object(
            ms, "_npu_markov_sample_kernel", return_value=(_legacy_kernel, frozenset())
        ):
            kwargs = dict(
                base_logits=base,
                bias=bias,
                temperatures=temperatures,
                key=12345,
                offset=offset,
                step=0,
                step_dtype=torch.bfloat16,
                softcap=30.0,
                vocab_limit=1024,
            )
            tok_fb = ms.MarkovSampleStep.execute(**kwargs)
            tok_ref = ms.MarkovSampleStep.torch(**kwargs)
            self.assertEqual(len(called), 0)
            self.assertTrue(torch.equal(tok_fb, tok_ref))
            self.assertTrue((tok_fb < 1024).all())


if __name__ == "__main__":
    unittest.main()
