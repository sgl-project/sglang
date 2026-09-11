"""Unit tests for the narrow BS1 BF16 TGV dispatch table."""

import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.quantization import fp8  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBS1BF16TGVDispatch(CustomTestCase):
    @patch.object(fp8, "is_sm100_supported", return_value=True)
    @patch.object(fp8.envs.SGLANG_BS1_BF16_TGV, "get", return_value=True)
    def test_exact_profiled_shapes_are_selected(self, _env, _sm):
        self.assertEqual(
            fp8._BS1_BF16_TGV_SHAPES,
            frozenset(
                {
                    (1, 2048, 2048),
                    (6, 2048, 2048),
                    (6, 4096, 2048),
                }
            ),
        )
        for m, n, k in fp8._BS1_BF16_TGV_SHAPES:
            with self.subTest(m=m, n=n, k=k):
                self.assertTrue(
                    fp8._should_use_bs1_bf16_tgv(
                        m, n, k, bias=None, dtype=torch.bfloat16
                    )
                )

    @patch.object(fp8, "is_sm100_supported", return_value=True)
    @patch.object(fp8.envs.SGLANG_BS1_BF16_TGV, "get", return_value=True)
    def test_unprofiled_or_incompatible_calls_fall_back(self, _env, _sm):
        cases = (
            (2, 2048, 2048, None, torch.bfloat16),
            (1, 2049, 2048, None, torch.bfloat16),
            (1, 2048, 2048, torch.empty(2048), torch.bfloat16),
            (1, 2048, 2048, None, torch.float16),
            (1, 4096, 2048, None, torch.bfloat16),
            (1, 2624, 6144, None, torch.bfloat16),
            (6, 2624, 6144, None, torch.bfloat16),
            (1, 6144, 2048, None, torch.bfloat16),
            (6, 6144, 2048, None, torch.bfloat16),
            (1, 6144, 256, None, torch.bfloat16),
            (6, 6144, 256, None, torch.bfloat16),
            (1, 3072, 6144, None, torch.bfloat16),
            (6, 3072, 6144, None, torch.bfloat16),
            (1, 6144, 1536, None, torch.bfloat16),
            (6, 6144, 1536, None, torch.bfloat16),
        )
        for m, n, k, bias, dtype in cases:
            with self.subTest(m=m, n=n, k=k, bias=bias, dtype=dtype):
                self.assertFalse(fp8._should_use_bs1_bf16_tgv(m, n, k, bias, dtype))

    @patch.object(fp8.envs.SGLANG_BS1_BF16_TGV, "get", return_value=False)
    def test_default_off(self, _env):
        self.assertFalse(
            fp8._should_use_bs1_bf16_tgv(1, 2048, 2048, bias=None, dtype=torch.bfloat16)
        )

    @patch.object(fp8, "is_sm100_supported", return_value=False)
    @patch.object(fp8.envs.SGLANG_BS1_BF16_TGV, "get", return_value=True)
    def test_non_blackwell_falls_back(self, _env, _sm):
        self.assertFalse(
            fp8._should_use_bs1_bf16_tgv(1, 2048, 2048, bias=None, dtype=torch.bfloat16)
        )


if __name__ == "__main__":
    unittest.main()
