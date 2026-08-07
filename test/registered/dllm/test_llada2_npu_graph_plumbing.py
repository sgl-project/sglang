from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.hardware_backend.npu.graph_runner import npu_cudagraph_backend
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend import utils as backend_utils


def _runner(*, dllm_algorithm, backend):
    server_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=backend),
        ),
        dllm_algorithm=dllm_algorithm,
        enable_memory_saver=False,
    )
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            device="npu",
            server_args=server_args,
        )
    )


class TestLLaDA2NPUGraphPlumbing(unittest.TestCase):
    def test_prefill_backend_override_is_scoped_to_dllm(self):
        generic_backend = object()
        with patch.object(
            backend_utils,
            "TcPiecewiseCudaGraphBackend",
            return_value=generic_backend,
        ):
            resolved = backend_utils.resolve_prefill_backend(
                _runner(dllm_algorithm=None, backend=Backend.TC_PIECEWISE)
            )
        self.assertIs(resolved, generic_backend)

        dllm_backend = object()
        with patch.object(
            npu_cudagraph_backend,
            "NPUCudaGraphBackend",
            return_value=dllm_backend,
        ):
            resolved = backend_utils.resolve_prefill_backend(
                _runner(dllm_algorithm="JointThresholdInDel", backend=Backend.FULL)
            )
        self.assertIs(resolved, dllm_backend)

    def test_cleanup_keeps_update_worker_reusable(self):
        backend = npu_cudagraph_backend.NPUCudaGraphBackend.__new__(
            npu_cudagraph_backend.NPUCudaGraphBackend
        )
        backend._graphs = {"shape": object()}
        backend._outputs = {"shape": object()}
        backend._bound_update_signatures = {"shape": object()}
        backend._pool = object()
        backend._update_queue = Mock()
        backend._update_thread = Mock()

        backend.cleanup()

        self.assertEqual(backend._graphs, {})
        self.assertEqual(backend._outputs, {})
        self.assertEqual(backend._bound_update_signatures, {})
        self.assertIsNone(backend._pool)
        backend._update_queue.put.assert_not_called()
        backend._update_thread.join.assert_not_called()


if __name__ == "__main__":
    unittest.main()
