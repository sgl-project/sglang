import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend import breakable_cuda_graph_backend
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_backend():
    backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
    backend._cuda_graph_runner = SimpleNamespace(
        cuda_graph_output_rows=lambda output: None,
        cuda_graph_output_capacity_rows=lambda output: None,
    )
    backend._device_module = SimpleNamespace(synchronize=lambda: None)
    backend._tp_group = SimpleNamespace(barrier=lambda: None)
    backend._precarve = SimpleNamespace(
        measure=contextlib.nullcontext, mint=lambda: None
    )
    backend._hicache_graph_events = None
    backend._enable_hicache_event_capture = False
    backend._shared_output_buffer = None
    backend._capture_stream = None
    backend._segment_capture = None
    backend._pool = None
    backend._debug_eager = False
    backend.deduped_cuda_graph = None
    backend._graphs = {}
    backend._outputs = {}
    backend._capture_inputs = {}
    return backend


class TestBreakableCudaGraphBackendOutputs(CustomTestCase):
    def test_replay_keeps_the_rows_of_each_output(self):
        backend = _make_backend()
        hidden = torch.arange(8.0).view(2, 4)
        aux = torch.arange(48.0).view(8, 6)
        with mock.patch.multiple(
            breakable_cuda_graph_backend,
            BreakableCUDAGraph=mock.Mock(),
            BreakableCUDAGraphCapture=lambda **kwargs: contextlib.nullcontext(),
            graph_pool_capture_scope=contextlib.nullcontext,
            graph_pool_replay_scope=contextlib.nullcontext,
        ):
            backend.capture_one(ShapeKey(size=8), lambda: (hidden, aux))
            replayed = backend.replay(ShapeKey(size=8), static_forward_batch=None)
        torch.testing.assert_close(replayed[0], hidden)
        torch.testing.assert_close(replayed[1], aux)


if __name__ == "__main__":
    unittest.main()
