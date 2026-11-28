"""The BF16-output LM head takes the SM12x GEMV when it applies, honouring `out`."""

import unittest
from unittest import mock

import torch
from torch import nn

from sglang.kernels.ops.gemm.sm12x_bf16_gemv import use_sm12x_bf16_gemv
from sglang.srt.configs.inkling import InklingMMConfig
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import unquant
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

K = 256


def _is_sm12x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


class TestSm12xBf16LMHead(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")
        config = InklingMMConfig(
            text_config={"vocab_size": 4096, "enable_lm_head_output_fp32": False}
        )
        self.processor = LogitsProcessor(config.text_config, skip_all_gather=True)
        self.assertFalse(self.processor.use_fp32_lm_head)
        self.assertFalse(self.processor.use_fp32_lm_head_output)

    def _head(self, vocab):
        torch.manual_seed(0)
        head = nn.Linear(K, vocab, bias=False, device="cuda", dtype=torch.bfloat16)
        head.requires_grad_(False)  # inference weights
        hidden = torch.randn((3, K), device="cuda", dtype=torch.bfloat16)
        return head, hidden

    def _stub(self):
        """Stand in for the SM12x GEMV on any GPU: record calls, compute with cuBLAS."""
        calls = []

        def gemv(x, weight, bias=None, out_dtype=None, out=None):
            calls.append(out)
            y = torch.matmul(x, weight.T)
            if out is None:
                return y
            out.copy_(y)
            return out

        patches = [
            mock.patch.object(unquant, "_sm12x_bf16_gemv", gemv),
            mock.patch.object(unquant, "_use_sm12x_bf16_gemv", use_sm12x_bf16_gemv),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return calls

    def test_routes_bf16_lm_head(self):
        calls = self._stub()
        head, hidden = self._head(4096)
        ref = torch.matmul(hidden, head.weight.T)

        logits = self.processor._compute_lm_head(hidden, head)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[-1])
        self.assertEqual(logits.dtype, torch.bfloat16)
        torch.testing.assert_close(logits, ref, rtol=0, atol=0)

        # A BF16 logits buffer is written in place.
        out = torch.empty((3, 4096), device="cuda", dtype=torch.bfloat16)
        logits = self.processor._compute_lm_head(hidden, head, out=out)
        self.assertIs(calls[-1], out)
        self.assertEqual(logits.data_ptr(), out.data_ptr())
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

        # An FP32 buffer keeps BF16 logits (same as the cuBLAS path), unbuffered.
        out32 = torch.empty((3, 4096), device="cuda", dtype=torch.float32)
        logits = self.processor._compute_lm_head(hidden, head, out=out32)
        self.assertIsNone(calls[-1])
        self.assertEqual(logits.dtype, torch.bfloat16)
        self.assertEqual(len(calls), 3)

    def test_skips_narrow_vocab_and_unset_backend(self):
        calls = self._stub()
        head, hidden = self._head(1024)  # below MIN_N
        self.processor._compute_lm_head(hidden, head)
        self.assertEqual(calls, [])

        with mock.patch.object(unquant, "_sm12x_bf16_gemv", None):
            head, hidden = self._head(4096)
            logits = self.processor._compute_lm_head(hidden, head)
        self.assertEqual(calls, [])
        self.assertEqual(logits.dtype, torch.bfloat16)

    @unittest.skipUnless(_is_sm12x(), "SM12x GEMV")
    def test_sm12x_kernel_matches_cublas(self):
        names = ("_sm12x_bf16_gemv", "_use_sm12x_bf16_gemv", "_b12x_bf16_gemv")
        saved = {n: getattr(unquant, n) for n in names}
        self.addCleanup(lambda: [setattr(unquant, n, v) for n, v in saved.items()])
        unquant.initialize_bf16_gemm_config()
        self.assertIsNotNone(unquant._sm12x_bf16_gemv)
        head = nn.Linear(2560, 8192, bias=False, device="cuda", dtype=torch.bfloat16)
        head.requires_grad_(False)
        for m in (1, 3, 16, 32):
            hidden = torch.randn((m, 2560), device="cuda", dtype=torch.bfloat16)
            ref = torch.matmul(hidden.float(), head.weight.float().T)
            for out in (
                None,
                torch.empty((m, 8192), device="cuda", dtype=torch.bfloat16),
            ):
                with self.subTest(m=m, out=out is not None):
                    logits = self.processor._compute_lm_head(hidden, head, out=out)
                    self.assertEqual(logits.dtype, torch.bfloat16)
                    if out is not None:
                        self.assertEqual(logits.data_ptr(), out.data_ptr())
                    torch.testing.assert_close(
                        logits.float(), ref, rtol=1.6e-2, atol=1e-2
                    )


if __name__ == "__main__":
    unittest.main()
