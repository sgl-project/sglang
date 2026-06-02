import unittest
from types import SimpleNamespace

import torch

import sglang.srt.state_capturer.indexer_topk as indexer_topk
from sglang.srt.state_capturer.indexer_topk import IndexerTopkCapturer


class TestIndexerTopkCapturer(unittest.TestCase):
    def _make_capturer(self):
        capturer = object.__new__(IndexerTopkCapturer)
        capturer.topk_size = 2
        capturer.device_cache = SimpleNamespace(
            buffer=torch.arange(6 * 1 * 3, dtype=torch.int32).reshape(6, 1, 3)
        )
        return capturer

    def test_non_dp_attention_uses_forward_batch_length(self):
        capturer = self._make_capturer()
        forward_batch = SimpleNamespace(out_cache_loc=torch.empty(4, dtype=torch.int64))

        old_is_dp_attention_enabled = indexer_topk.is_dp_attention_enabled
        try:
            indexer_topk.is_dp_attention_enabled = lambda: False

            actual = capturer._get_local_slice(
                forward_batch, can_run_graph=False, cuda_graph_batch=None
            )

            expected = capturer.device_cache.buffer[:4, :, :2]
            self.assertTrue(torch.equal(actual, expected))
        finally:
            indexer_topk.is_dp_attention_enabled = old_is_dp_attention_enabled

    def test_dp_attention_uses_dp_local_slice(self):
        capturer = self._make_capturer()
        forward_batch = SimpleNamespace(out_cache_loc=torch.empty(4, dtype=torch.int64))

        old_is_dp_attention_enabled = indexer_topk.is_dp_attention_enabled
        old_get_dp_local_slice_cpu = indexer_topk.get_dp_local_slice_cpu
        try:
            indexer_topk.is_dp_attention_enabled = lambda: True
            indexer_topk.get_dp_local_slice_cpu = (
                lambda forward_batch, can_run_graph, cuda_graph_batch: (2, 3)
            )

            actual = capturer._get_local_slice(
                forward_batch, can_run_graph=True, cuda_graph_batch=8
            )

            expected = capturer.device_cache.buffer[2:5, :, :2]
            self.assertTrue(torch.equal(actual, expected))
        finally:
            indexer_topk.is_dp_attention_enabled = old_is_dp_attention_enabled
            indexer_topk.get_dp_local_slice_cpu = old_get_dp_local_slice_cpu


if __name__ == "__main__":
    unittest.main()
