"""Regression tests for pending multimodal feature offload completion."""

import threading
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers import mm_schedule
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DelayedCompletion:
    def __init__(self, completed: threading.Event):
        self.completed = completed

    def synchronize(self):
        if not self.completed.wait(timeout=2.0):
            raise TimeoutError("pending multimodal feature offload did not finish")


class _RecordingCache:
    def __init__(self):
        self.stored = []

    def get_single(self, _key):
        return None

    def set(self, key, value):
        self.stored.append((key, value))


class TestMultimodalFeatureOffloadRace(CustomTestCase):
    def setUp(self):
        self.previous_event = getattr(mm_schedule, "host_offload_event", None)
        self.previous_cache = mm_schedule.embedding_cache

    def tearDown(self):
        mm_schedule.host_offload_event = self.previous_event
        mm_schedule.embedding_cache = self.previous_cache

    def _start_pending_write(self, feature: torch.Tensor):
        completed = threading.Event()

        def complete_offload():
            feature.fill_(7.0)
            completed.set()

        timer = threading.Timer(0.2, complete_offload)
        timer.start()
        mm_schedule.host_offload_event = _DelayedCompletion(completed)
        return timer

    def test_precomputed_embedding_waits_for_pending_cpu_offload(self):
        embedding = torch.zeros((2, 1), dtype=torch.float32)
        timer = self._start_pending_write(embedding)
        try:
            result = mm_schedule._get_precomputed_embedding(
                items=[SimpleNamespace(precomputed_embeddings=embedding)],
                items_size=[0, 1],
                prefix_length=[0],
                extend_length=[2],
                items_offset_list=[[(0, 1)]],
            )
        finally:
            timer.join()

        torch.testing.assert_close(result, torch.full_like(embedding, 7.0))

    def test_encoder_read_waits_for_pending_cpu_feature_offload(self):
        feature = torch.zeros((2, 1), dtype=torch.float32)
        item = SimpleNamespace(hash=1, feature=feature)
        mm_schedule.embedding_cache = _RecordingCache()

        def encode(items):
            return items[0].feature.clone()

        timer = self._start_pending_write(feature)
        try:
            result = mm_schedule._get_chunked_embedding_by_item(
                data_embedding_func=encode,
                embedding_items_per_req=[item],
                items_offset=[(0, 1)],
                extend_prefix_len=0,
                extend_seq_len=2,
                device=torch.device("cpu"),
            )
        finally:
            timer.join()

        torch.testing.assert_close(result, torch.full_like(feature, 7.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
