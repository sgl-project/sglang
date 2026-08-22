import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.dsa import dsa_indexer

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAiterBlockQDispatch(CustomTestCase):
    def setUp(self):
        self.q = torch.empty((3, 32, 128), dtype=torch.uint8)
        self.kv = torch.empty((513, 128), dtype=torch.uint8)
        self.kv_scales = torch.empty(513, dtype=torch.float32)
        self.weights = torch.empty((3, 32), dtype=torch.float32)
        self.cu_starts = torch.zeros(3, dtype=torch.int32)
        self.cu_ends = torch.full((3,), 513, dtype=torch.int32)
        self.fallback_result = object()
        self.fp8_mqa_logits = MagicMock(return_value=self.fallback_result)

    def _dispatch(self):
        return dsa_indexer._aiter_fp8_mqa_logits_with_optional_blockq(
            self.q,
            self.kv,
            self.kv_scales,
            self.weights,
            self.cu_starts,
            self.cu_ends,
            self.fp8_mqa_logits,
        )

    def test_unavailable_kernel_falls_back(self):
        with (
            patch.object(dsa_indexer, "get_blockq_config", None),
            patch.object(dsa_indexer, "fp8_mqa_logits_blockq", None),
        ):
            result = self._dispatch()

        self.assertIs(result, self.fallback_result)
        self.fp8_mqa_logits.assert_called_once_with(
            self.q,
            self.kv,
            self.kv_scales,
            self.weights,
            self.cu_starts,
            self.cu_ends,
            clean_logits=False,
        )

    def test_unsupported_shape_falls_back(self):
        get_config = MagicMock(return_value=None)
        blockq = MagicMock()

        with (
            patch.object(dsa_indexer, "get_blockq_config", get_config),
            patch.object(dsa_indexer, "fp8_mqa_logits_blockq", blockq),
        ):
            result = self._dispatch()

        self.assertIs(result, self.fallback_result)
        get_config.assert_called_once_with(32, 128)
        blockq.assert_not_called()
        self.fp8_mqa_logits.assert_called_once_with(
            self.q,
            self.kv,
            self.kv_scales,
            self.weights,
            self.cu_starts,
            self.cu_ends,
            clean_logits=False,
        )

    def test_supported_shape_allocates_aligned_logits_and_calls_directly(self):
        get_config = MagicMock(return_value={"BLOCK_KV": 64})
        blockq = MagicMock(side_effect=lambda *args: args[-1])

        with (
            patch.object(dsa_indexer, "get_blockq_config", get_config),
            patch.object(dsa_indexer, "fp8_mqa_logits_blockq", blockq),
        ):
            logits = self._dispatch()

        self.assertEqual(logits.shape, (3, 513))
        self.assertEqual(logits.stride(), (768, 1))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(logits.device, self.q.device)
        get_config.assert_called_once_with(32, 128)
        blockq.assert_called_once()
        self.fp8_mqa_logits.assert_not_called()
        call = blockq.call_args
        self.assertEqual(call.kwargs, {})
        expected_args = (
            self.q,
            self.kv,
            self.kv_scales,
            self.weights,
            self.cu_starts,
            self.cu_ends,
        )
        for actual, expected in zip(call.args[:-1], expected_args):
            self.assertIs(actual, expected)
        self.assertIs(call.args[-1], logits)

    def test_supported_kernel_errors_propagate(self):
        with (
            patch.object(dsa_indexer, "get_blockq_config", return_value={}),
            patch.object(
                dsa_indexer,
                "fp8_mqa_logits_blockq",
                side_effect=RuntimeError("compile failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                self._dispatch()
        self.fp8_mqa_logits.assert_not_called()


if __name__ == "__main__":
    unittest.main()
