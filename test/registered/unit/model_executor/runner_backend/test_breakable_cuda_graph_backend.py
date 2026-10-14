# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for Breakable CUDA graph cleanup and recapture lifecycle on CPU."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
    set_global_graph_memory_pool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestBreakableCUDAGraphReset(CustomTestCase):
    def test_reset_clears_segments_and_resets_underlying_graphs(self):
        graph = BreakableCUDAGraph()
        seg1 = mock.Mock()
        seg2 = mock.Mock(spec=["original_graph"])
        seg2.original_graph = mock.Mock()

        graph._segments = [seg1, seg2]
        graph._break_fns = [lambda: None]

        graph.reset()

        seg1.reset.assert_called_once_with()
        seg2.original_graph.reset.assert_called_once_with()
        self.assertEqual(graph._segments, [])
        self.assertEqual(graph._break_fns, [])


class TestBreakableBackendCleanup(CustomTestCase):
    def test_cleanup_resets_graphs_and_global_memory_pool(self):
        backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        mock_graph1 = mock.Mock()
        mock_graph2 = mock.Mock()
        backend._graphs = {
            ShapeKey(size=4): mock_graph1,
            ShapeKey(size=8): mock_graph2,
        }
        backend._outputs = {ShapeKey(size=4): object()}
        backend._capture_inputs = {ShapeKey(size=4): object()}
        backend._pool = (1234, 5678)
        backend._shared_output_buffer = object()
        backend.close = mock.Mock()

        with mock.patch(
            "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend.set_global_graph_memory_pool"
        ) as mock_set_pool:
            backend.cleanup()

            backend.close.assert_called_once()
            mock_graph1.reset.assert_called_once()
            mock_graph2.reset.assert_called_once()
            self.assertEqual(backend._graphs, {})
            self.assertEqual(backend._outputs, {})
            self.assertEqual(backend._capture_inputs, {})
            self.assertIsNone(backend._pool)
            self.assertIsNone(backend._shared_output_buffer)
            mock_set_pool.assert_called_once_with(None)

    def test_recapture_mints_fresh_pool_after_cleanup(self):
        set_global_graph_memory_pool(None)

        device_module = SimpleNamespace(
            graph_pool_handle=mock.Mock(side_effect=[(1, 1), (2, 2)])
        )

        pool1 = get_or_create_global_graph_memory_pool(device_module)
        self.assertEqual(pool1, (1, 1))

        # Cleanup resets pool
        set_global_graph_memory_pool(None)

        pool2 = get_or_create_global_graph_memory_pool(device_module)
        self.assertEqual(pool2, (2, 2))
        self.assertNotEqual(pool1, pool2)

        set_global_graph_memory_pool(None)


class TestBreakableCaptureHygiene(CustomTestCase):
    def test_capture_enter_calls_empty_cache_and_host_empty_cache(self):
        graph = BreakableCUDAGraph()
        mock_device = SimpleNamespace(
            empty_cache=mock.Mock(),
            stream=mock.Mock(return_value=mock.MagicMock()),
            current_stream=mock.Mock(),
            Stream=SimpleNamespace(wait_stream=mock.Mock()),
        )

        with (
            mock.patch(
                "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph.get_device_module",
                return_value=mock_device,
            ),
            mock.patch(
                "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph.torch._C",
                create=True,
            ) as mock_torch_c,
        ):
            mock_torch_c._host_emptyCache = mock.Mock()

            capture = BreakableCUDAGraphCapture(graph, pool=(1, 1))
            capture._begin_new_segment = mock.Mock()
            capture._end_current_segment = mock.Mock()

            with capture:
                mock_device.empty_cache.assert_called_once()
                mock_torch_c._host_emptyCache.assert_called_once()
                capture._begin_new_segment.assert_called_once()
            capture._end_current_segment.assert_called_once()


if __name__ == "__main__":
    unittest.main()
