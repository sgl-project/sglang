"""Exact logits conversion/softcap and persistent graph-buffer behavior."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.activation.softcap import (
    softcap_copy_logits,
    softcap_inplace_logits,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSoftcapCopy(CustomTestCase):
    def test_full_vocabulary_and_strided_graph(self):
        for rows, vocab, padding in [(256, 262144, 0), (33, 65537, 64)]:
            with self.subTest(rows=rows, vocab=vocab, padding=padding):
                x = torch.randn(
                    rows, vocab + padding, device="cuda", dtype=torch.bfloat16
                )[:, :vocab]
                backing = torch.full((rows, vocab + padding), 123.0, device="cuda")
                output = backing[:, :vocab]
                softcap_copy_logits(x, output, 30.0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = softcap_copy_logits(x, output, 30.0)
                for scale in (1.0, -10.0, 100.0):
                    x.mul_(scale)
                    graph.replay()
                    expected = softcap_inplace_logits(x.float(), 30.0)
                    self.assertEqual(actual.data_ptr(), output.data_ptr())
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if padding:
                        self.assertTrue(bool((backing[:, vocab:] == 123).all()))

    def test_special_values_match_existing_softcap(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.tensor(
                [
                    0.0,
                    -0.0,
                    1e-4,
                    -1e-4,
                    1.0,
                    -1.0,
                    1e4,
                    -1e4,
                    float("inf"),
                    -float("inf"),
                    float("nan"),
                ],
                device="cuda",
                dtype=dtype,
            ).repeat(33, 1)
            for cap in (1.0, 30.0, 50.0):
                expected = softcap_inplace_logits(x.float().clone(), cap)
                actual = softcap_copy_logits(
                    x, torch.empty_like(x, dtype=torch.float32), cap
                )
                torch.testing.assert_close(
                    actual, expected, rtol=0, atol=0, equal_nan=True
                )

    def test_logits_buffer_reuse_truncation_and_fallback(self):
        processor = LogitsProcessor.__new__(LogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.vocab_size = 65537
        x = torch.randn(33, 65600, device="cuda", dtype=torch.bfloat16)
        for use_buffer, buffer_rows in [(True, 33), (True, 1), (False, 33)]:
            with self.subTest(use_buffer=use_buffer, buffer_rows=buffer_rows):
                buffer = torch.full((buffer_rows, 65537), 123.0, device="cuda")
                metadata = SimpleNamespace(next_token_logits_buffer=buffer)
                actual = processor._copy_logits_to_buffer(
                    x, metadata, use_buffer=use_buffer, softcap=30.0
                )
                expected = softcap_inplace_logits(x[:, :65537].float(), 30.0)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if use_buffer and buffer_rows == 33:
                    self.assertEqual(actual.data_ptr(), buffer.data_ptr())
                else:
                    self.assertTrue(bool((buffer == 123).all()))
                uncapped = processor._copy_logits_to_buffer(
                    x, metadata, use_buffer=use_buffer
                )
                torch.testing.assert_close(
                    uncapped, x[:, :65537].float(), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
