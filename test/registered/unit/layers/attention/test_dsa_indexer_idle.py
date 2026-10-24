import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import (
    Indexer,
    _make_eager_idle_topk_result,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.state_capturer.base import BaseDeviceCache
from sglang.srt.state_capturer.indexer_topk import (
    IndexerTopkCapturer,
    maybe_capture_shared_indexer_topk,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsa.dsa_indexer"


class TestDSAIndexerIdle(unittest.TestCase):
    def test_builds_invalid_topk_rows_for_padded_idle_tokens(self):
        result = _make_eager_idle_topk_result(
            torch.empty((8, 16)),
            index_topk=4,
            return_indices=True,
        )

        self.assertEqual(result.shape, (8, 4))
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(torch.all(result == -1))

    def test_returns_none_when_indices_are_not_consumed(self):
        self.assertIsNone(
            _make_eager_idle_topk_result(
                torch.empty((8, 16)),
                index_topk=4,
                return_indices=False,
            )
        )

    def test_eager_idle_short_circuits_before_paged_mqa(self):
        indexer = SimpleNamespace(index_topk=4)
        batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)

        with (
            patch(f"{_INDEXER}._is_cuda", True),
            patch(f"{_INDEXER}.get_is_capture_mode", return_value=False),
            patch(
                f"{_INDEXER}._broadcast_indexer_topk_from_rank0",
                side_effect=lambda result: result,
            ),
            patch(
                f"{_INDEXER}.maybe_capture_indexer_topk",
                side_effect=lambda _layer_id, result: result,
            ),
        ):
            result = Indexer.forward_cuda(
                indexer,
                x=torch.empty((8, 16)),
                q_lora=torch.empty((8, 16)),
                positions=torch.empty((8,), dtype=torch.int64),
                forward_batch=batch,
                layer_id=3,
            )

        self.assertEqual(result.shape, (8, 4))
        self.assertTrue(torch.all(result == -1))


class TestDSAIndexerCapture(unittest.TestCase):
    def test_shared_layers_capture_token_positions(self):
        capturer = object.__new__(IndexerTopkCapturer)
        capturer.device_cache = BaseDeviceCache(3, 4, 3, "cpu", "indexer_topk")
        capturer.device_cache.buffer.fill_(-2)
        raw = torch.tensor([[0, 1, -1], [2, 0, -1]], dtype=torch.int32)
        physical = torch.tensor([[64, 65, -1], [130, 128, -1]], dtype=torch.int32)

        with patch(
            "sglang.srt.state_capturer.indexer_topk.get_global_indexer_capturer",
            return_value=capturer,
        ):
            result = Indexer._capture_and_return(None, 1, physical, raw)
            self.assertIs(result, physical)
            for layer_id in (2, 3):
                result = maybe_capture_shared_indexer_topk(layer_id, result)
                self.assertIs(result, physical)
                torch.testing.assert_close(
                    capturer.device_cache.buffer[:2, layer_id], raw
                )

        self.assertTrue(torch.all(capturer.device_cache.buffer[2] == -2))

    def test_shared_capture_disabled_preserves_attention_indices(self):
        physical = torch.tensor([[64, 65, -1]], dtype=torch.int32)
        with patch(
            "sglang.srt.state_capturer.indexer_topk.get_global_indexer_capturer",
            return_value=None,
        ):
            self.assertIs(maybe_capture_shared_indexer_topk(2, physical), physical)
            self.assertIsNone(maybe_capture_shared_indexer_topk(2, None))


if __name__ == "__main__":
    unittest.main()
