"""Unit tests for the process-wide CUDA graph capture resources (``pool.py``).

The capture *stream* is shared for the same reason the memory pool is: the
caching allocator partitions the graph pool's segments by stream, so a fresh
stream per capture pass re-reserves that pool's MoE / DeepEP scratch instead of
reusing what an earlier pass left inactive. With adaptive speculative decoding
that is one pass per candidate step times three runners — tens of GB of
segments that all read back ``inactive``.

CPU-only: ``get_stream`` allocates a ``torch.cuda.Stream`` handle, which does
not require a live CUDA context to construct in these assertions, and the
memory-pool tests use a fake device module.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.model_executor.runner_utils.pool import (
    _CAPTURE_STREAM_NAME,
    get_global_graph_memory_pool,
    get_or_create_global_graph_capture_stream,
    get_or_create_global_graph_memory_pool,
    set_global_graph_memory_pool,
)
from sglang.srt.runtime_context import get_resources, reset_context
from sglang.test.test_utils import CustomTestCase


class _FakeDeviceModule:
    """Stands in for ``torch.cuda``: counts pool-handle creations."""

    def __init__(self):
        self.handle_ct = 0

    def graph_pool_handle(self):
        self.handle_ct += 1
        return f"pool-{self.handle_ct}"


class TestGraphMemoryPool(CustomTestCase):
    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_created_once_and_reused(self):
        device_module = _FakeDeviceModule()
        first = get_or_create_global_graph_memory_pool(device_module)
        second = get_or_create_global_graph_memory_pool(device_module)
        # One handle for every later backend — that is what makes the captured
        # graphs share physical memory.
        self.assertIs(first, second)
        self.assertEqual(device_module.handle_ct, 1)
        self.assertIs(get_global_graph_memory_pool(), first)

    def test_reset_context_drops_the_pool(self):
        device_module = _FakeDeviceModule()
        get_or_create_global_graph_memory_pool(device_module)
        reset_context()
        self.assertIsNone(get_global_graph_memory_pool())
        get_or_create_global_graph_memory_pool(device_module)
        self.assertEqual(device_module.handle_ct, 2)

    def test_explicit_set_wins(self):
        set_global_graph_memory_pool("injected")
        device_module = _FakeDeviceModule()
        self.assertEqual(
            get_or_create_global_graph_memory_pool(device_module), "injected"
        )
        self.assertEqual(device_module.handle_ct, 0)


class TestGraphCaptureStream(CustomTestCase):
    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_shared_across_capture_passes(self):
        # Every capture pass must land on the same stream, or each one gets its
        # own segments in the shared graph pool.
        first = get_or_create_global_graph_capture_stream()
        second = get_or_create_global_graph_capture_stream()
        self.assertIs(first, second)

    def test_leased_from_the_named_stream_slot(self):
        stream = get_or_create_global_graph_capture_stream()
        # Held in the Resources tier's named-stream registry, so reset_context()
        # clears it along with the rest of the process-level handles.
        self.assertIs(get_resources().streams[_CAPTURE_STREAM_NAME], stream)

    def test_reset_context_drops_the_stream(self):
        first = get_or_create_global_graph_capture_stream()
        reset_context()
        self.assertIsNot(get_or_create_global_graph_capture_stream(), first)


if __name__ == "__main__":
    unittest.main()
