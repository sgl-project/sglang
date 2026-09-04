"""Regression test for UNO's base-only LoRA routing fast path."""

import unittest
from types import SimpleNamespace

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _InactiveSkippingBackend:
    skip_inactive_lora_batches = True

    def __init__(self):
        self.batch_info = object()
        self.prepare_called = False

    def reset_batch_state(self):
        self.batch_info = None

    def prepare_lora_batch(self, *args, **kwargs):
        self.prepare_called = True


class TestUnoInactiveLoRABatch(CustomTestCase):
    def test_all_base_batch_clears_stale_routing_before_graph_metadata(self):
        backend = _InactiveSkippingBackend()
        manager = LoRAManager.__new__(LoRAManager)
        manager.lora_backend = backend
        forward_batch = SimpleNamespace(lora_ids=[None], batch_size=1)

        manager.prepare_lora_batch(forward_batch)

        self.assertIsNone(backend.batch_info)
        self.assertFalse(backend.prepare_called)


if __name__ == "__main__":
    unittest.main()
