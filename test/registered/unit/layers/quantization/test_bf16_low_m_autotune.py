"""
Unit tests for the autotuned BF16 low-M route in unquant.

M <= 16 runs FlashInfer's CuTe DSL low-M kernels; a larger M falls through,
and inside FlashInfer autotune profiles the M <= 16 buckets on a row slice.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.unquant as unquant
from sglang.test.test_utils import CustomTestCase


def _tuner(tuning, file_configs=()):
    state = SimpleNamespace(is_tuning_mode=tuning, _file_configs=dict(file_configs))
    return SimpleNamespace(get=lambda: state)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestBf16LowMAutotune(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.weight = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda") / 16
        self.calls = []

    def _route(self, tuning=False, file_configs=()):
        def fake_mm_bf16(a, b, pdl=False, out=None, backend=None):
            self.calls.append(a.shape[0])
            result = a @ b
            return result if out is None else out.copy_(result)

        stack = ExitStack()
        stack.enter_context(patch.object(unquant, "_enable_bf16_low_m_autotune", True))
        stack.enter_context(patch.object(unquant, "_mm_bf16", fake_mm_bf16))
        stack.enter_context(
            patch.object(unquant, "_flashinfer_autotuner", _tuner(tuning, file_configs))
        )
        return stack

    @torch.inference_mode()
    def test_low_m_taken(self):
        x = torch.randn(2, 4, 512, dtype=torch.bfloat16, device="cuda")
        with self._route():
            out = unquant._bf16_low_m_autotune(x, self.weight, None)
        self.assertEqual(self.calls, [8])
        self.assertEqual(out.shape, (2, 4, 256))
        torch.testing.assert_close(out, F.linear(x, self.weight))

    @torch.inference_mode()
    def test_into_caller_output(self):
        x = torch.randn(3, 512, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(3, 256, dtype=torch.bfloat16, device="cuda")
        with self._route():
            got = unquant._bf16_low_m_autotune(x, self.weight, None, out=out)
        self.assertEqual(got.data_ptr(), out.data_ptr())
        torch.testing.assert_close(out, F.linear(x, self.weight))

    @torch.inference_mode()
    def test_declined(self):
        x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
        bias = torch.zeros(256, dtype=torch.bfloat16, device="cuda")
        with self._route():
            self.assertIsNone(unquant._bf16_low_m_autotune(x, self.weight, bias))
            self.assertIsNone(
                unquant._bf16_low_m_autotune(x.float(), self.weight, None)
            )
            self.assertIsNone(
                unquant._bf16_low_m_autotune(
                    torch.randn(17, 512, dtype=torch.bfloat16, device="cuda"),
                    self.weight,
                    None,
                )
            )
        self.assertEqual(self.calls, [])

    @torch.inference_mode()
    def test_large_m_tunes_low_m_slice(self):
        x = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
        with self._route(tuning=True):
            self.assertIsNone(unquant._bf16_low_m_autotune(x, self.weight, None))
        self.assertEqual(self.calls, [unquant._BF16_LOW_M_AUTOTUNE_MAX_M])

    @torch.inference_mode()
    def test_large_m_skips_shape_in_file_cache(self):
        x = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
        cached = {"('bf16_gemm', 'CuteDSLDirectBf16Runner', ((2, 512), (512, 256)))": 0}
        with self._route(tuning=True, file_configs=cached):
            self.assertIsNone(unquant._bf16_low_m_autotune(x, self.weight, None))
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
