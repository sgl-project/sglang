"""Unit tests for DeepSeek-V4 sparse-MLA query-head selection."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import flash_mla_sm120
from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekV4SparseMlaHeads(CustomTestCase):
    @staticmethod
    def _layer(*, local_heads: int = 16, total_heads: int = 64, tp_size: int = 4):
        layer = deepseek_v4.MqaAttentionBase.__new__(deepseek_v4.MqaAttentionBase)
        torch.nn.Module.__init__(layer)
        layer.n_local_heads = local_heads
        layer.n_heads = total_heads
        layer.attn_tp_size = tp_size
        layer.attn_tp_rank = 1
        layer.attn_sink = torch.nn.Parameter(
            torch.arange(total_heads, dtype=torch.float32)
        )
        layer._attn_sink_local = None
        return layer

    def test_tp4_sm120_flashinfer_decode_uses_16_real_heads(self):
        layer = self._layer()
        with (
            patch.object(deepseek_v4, "is_sm120_supported", return_value=True),
            envs.SGLANG_SM120_FLASHMLA_BACKEND.override("flashinfer"),
            patch(
                "sglang.kernels.ops.attention.flash_mla_sm120."
                "flashinfer_dsv4_decode_supports_num_heads",
                return_value=True,
            ),
        ):
            self.assertEqual(layer._kernel_num_heads(num_tokens=64), 16)

    def test_tp8_sm120_flashinfer_prefill_keeps_64_head_padding(self):
        layer = self._layer(local_heads=8, tp_size=8)
        with (
            patch.object(deepseek_v4, "is_sm120_supported", return_value=True),
            envs.SGLANG_SM120_FLASHMLA_BACKEND.override("flashinfer"),
            patch(
                "sglang.kernels.ops.attention.flash_mla_sm120."
                "flashinfer_dsv4_decode_supports_num_heads",
                side_effect=lambda heads, tokens: heads == 8 and tokens <= 64,
            ),
        ):
            self.assertEqual(layer._kernel_num_heads(num_tokens=64), 8)
            self.assertEqual(layer._kernel_num_heads(num_tokens=65), 64)

    def test_flashinfer_decode_capability_rejects_prefill_token_count(self):
        with patch.object(
            flash_mla_sm120,
            "_flashinfer_dsv4_decode_capabilities",
            return_value=(64, frozenset({8, 16, 32, 64, 128})),
        ):
            self.assertTrue(
                flash_mla_sm120.flashinfer_dsv4_decode_supports_num_heads(8, 64)
            )
            self.assertFalse(
                flash_mla_sm120.flashinfer_dsv4_decode_supports_num_heads(8, 65)
            )

    def test_fallback_backends_keep_64_head_padding(self):
        layer = self._layer()
        with patch.object(deepseek_v4, "is_sm120_supported", return_value=True):
            for backend in ("triton", "torch"):
                with (
                    self.subTest(backend=backend),
                    envs.SGLANG_SM120_FLASHMLA_BACKEND.override(backend),
                ):
                    self.assertEqual(layer._kernel_num_heads(num_tokens=1), 64)

        with (
            patch.object(deepseek_v4, "is_sm120_supported", return_value=False),
            envs.SGLANG_SM120_FLASHMLA_BACKEND.override("flashinfer"),
        ):
            self.assertEqual(layer._kernel_num_heads(num_tokens=1), 64)

    def test_missing_flashinfer_specialization_keeps_padding(self):
        layer = self._layer()
        with (
            patch.object(deepseek_v4, "is_sm120_supported", return_value=True),
            envs.SGLANG_SM120_FLASHMLA_BACKEND.override("flashinfer"),
            patch(
                "sglang.kernels.ops.attention.flash_mla_sm120."
                "flashinfer_dsv4_decode_supports_num_heads",
                return_value=False,
            ),
        ):
            self.assertEqual(layer._kernel_num_heads(num_tokens=1), 64)

    def test_sink_width_matches_exact_and_fallback_paths(self):
        layer = self._layer()

        exact = layer._local_attn_sink(16)
        torch.testing.assert_close(exact, torch.arange(16, 32, dtype=torch.float32))

        padded = layer._local_attn_sink(64)
        torch.testing.assert_close(padded[:16], exact)
        torch.testing.assert_close(padded[16:], torch.zeros(48))
        self.assertEqual(exact.data_ptr(), padded.data_ptr())

    def test_sink_no_arg_preserves_dspark_padded_contract(self):
        layer = self._layer()
        sink = layer._local_attn_sink()

        torch.testing.assert_close(sink[:16], torch.arange(16, 32, dtype=torch.float32))
        torch.testing.assert_close(sink[16:], torch.zeros(48))


if __name__ == "__main__":
    unittest.main()
