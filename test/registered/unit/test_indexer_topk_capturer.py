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
        capturer.host_cache = SimpleNamespace(
            buffer=torch.zeros((32, 1, 2), dtype=torch.int32)
        )
        capturer._capture_num_tokens = None
        capturer._captured_layer_ids = set()
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
            capturer._capture_num_tokens = 5

            actual = capturer._get_local_slice(
                forward_batch, can_run_graph=True, cuda_graph_batch=8
            )

            expected = capturer.device_cache.buffer[2:5, :, :2]
            self.assertTrue(torch.equal(actual, expected))
        finally:
            indexer_topk.is_dp_attention_enabled = old_is_dp_attention_enabled
            indexer_topk.get_dp_local_slice_cpu = old_get_dp_local_slice_cpu

    def test_maybe_capture_keeps_first_capture_for_layer(self):
        class FakeCapturer:
            def __init__(self):
                self.calls = []
                self.captured_layer_ids = set()

            def has_capture_for_layer(self, layer_id):
                return layer_id in self.captured_layer_ids

            def capture(self, layer_id, topk_indices):
                self.captured_layer_ids.add(layer_id)
                self.calls.append((layer_id, topk_indices.clone()))

        capturer = FakeCapturer()
        raw_topk = torch.tensor([[0, 1]], dtype=torch.int32)
        transformed_topk = torch.tensor([[100, 101]], dtype=torch.int32)

        old_capturer = indexer_topk.get_global_indexer_capturer()
        try:
            indexer_topk.set_global_indexer_capturer(capturer)

            indexer_topk.maybe_capture_indexer_topk(0, raw_topk)
            indexer_topk.maybe_capture_indexer_topk(0, transformed_topk)

            self.assertEqual(len(capturer.calls), 1)
            self.assertEqual(capturer.calls[0][0], 0)
            self.assertTrue(torch.equal(capturer.calls[0][1], raw_topk))
        finally:
            indexer_topk.set_global_indexer_capturer(old_capturer)

    def test_dp_attention_slices_cache_locations_with_topk(self):
        capturer = self._make_capturer()
        forward_batch = SimpleNamespace(
            out_cache_loc=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64)
        )

        old_is_dp_attention_enabled = indexer_topk.is_dp_attention_enabled
        old_get_dp_local_slice_cpu = indexer_topk.get_dp_local_slice_cpu
        try:
            indexer_topk.is_dp_attention_enabled = lambda: True
            indexer_topk.get_dp_local_slice_cpu = (
                lambda forward_batch, can_run_graph, cuda_graph_batch: (2, 3)
            )
            capturer._capture_num_tokens = 5

            output = capturer.on_forward_end(
                forward_batch,
                can_run_graph=False,
                cuda_graph_batch=None,
                no_copy_to_cpu=True,
            )

            self.assertEqual(output.out_cache_loc.tolist(), [12, 13, 14])
            self.assertTrue(
                torch.equal(output.topk, capturer.device_cache.buffer[2:5, :, :2])
            )

            output.finalize()
            self.assertTrue(
                torch.equal(
                    capturer.host_cache.buffer[12:15],
                    capturer.device_cache.buffer[2:5, :, :2],
                )
            )
        finally:
            indexer_topk.is_dp_attention_enabled = old_is_dp_attention_enabled
            indexer_topk.get_dp_local_slice_cpu = old_get_dp_local_slice_cpu

    def test_dp_attention_uses_local_cache_locations_when_not_global(self):
        capturer = self._make_capturer()
        forward_batch = SimpleNamespace(
            out_cache_loc=torch.tensor([20, 21], dtype=torch.int64)
        )

        old_is_dp_attention_enabled = indexer_topk.is_dp_attention_enabled
        old_get_dp_local_slice_cpu = indexer_topk.get_dp_local_slice_cpu
        try:
            indexer_topk.is_dp_attention_enabled = lambda: True
            indexer_topk.get_dp_local_slice_cpu = (
                lambda forward_batch, can_run_graph, cuda_graph_batch: (4, 2)
            )
            capturer._capture_num_tokens = 2

            output = capturer.on_forward_end(
                forward_batch,
                can_run_graph=False,
                cuda_graph_batch=None,
                no_copy_to_cpu=True,
            )

            self.assertEqual(output.out_cache_loc.tolist(), [20, 21])
            self.assertTrue(
                torch.equal(output.topk, capturer.device_cache.buffer[:2, :, :2])
            )

            output.finalize()
            self.assertTrue(
                torch.equal(
                    capturer.host_cache.buffer[20:22],
                    capturer.device_cache.buffer[:2, :, :2],
                )
            )
        finally:
            indexer_topk.is_dp_attention_enabled = old_is_dp_attention_enabled
            indexer_topk.get_dp_local_slice_cpu = old_get_dp_local_slice_cpu

    def test_dp_attention_truncates_local_locations_to_captured_topk(self):
        capturer = self._make_capturer()
        forward_batch = SimpleNamespace(
            out_cache_loc=torch.tensor([20, 21, 22, 23, 24], dtype=torch.int64)
        )

        old_is_dp_attention_enabled = indexer_topk.is_dp_attention_enabled
        old_get_dp_local_slice_cpu = indexer_topk.get_dp_local_slice_cpu
        try:
            indexer_topk.is_dp_attention_enabled = lambda: True
            indexer_topk.get_dp_local_slice_cpu = (
                lambda forward_batch, can_run_graph, cuda_graph_batch: (4, 5)
            )
            capturer._capture_num_tokens = 3

            output = capturer.on_forward_end(
                forward_batch,
                can_run_graph=False,
                cuda_graph_batch=None,
                no_copy_to_cpu=True,
            )

            self.assertEqual(output.out_cache_loc.tolist(), [20, 21, 22])
            self.assertTrue(
                torch.equal(output.topk, capturer.device_cache.buffer[:3, :, :2])
            )

            output.finalize()
            self.assertTrue(
                torch.equal(
                    capturer.host_cache.buffer[20:23],
                    capturer.device_cache.buffer[:3, :, :2],
                )
            )
        finally:
            indexer_topk.is_dp_attention_enabled = old_is_dp_attention_enabled
            indexer_topk.get_dp_local_slice_cpu = old_get_dp_local_slice_cpu


if __name__ == "__main__":
    unittest.main()
