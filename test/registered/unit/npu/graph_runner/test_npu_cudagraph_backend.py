"""CPU unit tests for the NPU CUDA-graph backend's Python orchestration."""

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.npu.graph_runner import npu_cudagraph_backend as mod
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _GraphContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_backend():
    device_module = SimpleNamespace(
        current_device=mock.Mock(return_value=3),
        synchronize=mock.Mock(),
        set_device=mock.Mock(),
    )
    runner = SimpleNamespace(
        device_module=device_module,
        model_runner=SimpleNamespace(tp_group=SimpleNamespace(barrier=mock.Mock())),
        enable_torch_compile=False,
    )
    with mock.patch.object(mod.TorchMemorySaverAdapter, "create", return_value=None):
        return mod.NPUCudaGraphBackend(runner), runner


class TestNPUCudaGraphBackend(unittest.TestCase):
    def test_capture_warms_up_then_records_graph_and_output(self):
        backend, runner = _make_backend()
        graph = mock.Mock()
        graph_context = mock.Mock(return_value=_GraphContext())
        forward = mock.Mock(return_value="captured-output")
        post_warmup_hook = mock.Mock()
        shape_key = ShapeKey(size=4)

        with (
            mock.patch.dict(sys.modules, {"torch_npu": SimpleNamespace()}),
            mock.patch.object(
                torch,
                "npu",
                SimpleNamespace(
                    NPUGraph=mock.Mock(return_value=graph), graph=graph_context
                ),
                create=True,
            ),
        ):
            backend.capture_one(shape_key, forward, post_warmup_hook=post_warmup_hook)

        self.assertEqual(forward.call_count, 3)
        self.assertEqual(post_warmup_hook.call_count, 2)
        self.assertEqual(runner.device_module.synchronize.call_count, 2)
        self.assertEqual(runner.model_runner.tp_group.barrier.call_count, 2)
        graph_context.assert_called_once_with(
            graph, pool=None, stream=None, auto_dispatch_capture=True
        )
        self.assertIs(backend._graphs[shape_key], graph)
        self.assertEqual(backend._outputs[shape_key], "captured-output")

    def test_replay_reuses_captured_output_for_shape(self):
        backend, _ = _make_backend()
        graph = mock.Mock()
        shape_key = ShapeKey(size=2)
        backend._graphs[shape_key] = graph
        backend._outputs[shape_key] = "output"

        output = backend.replay(shape_key, static_forward_batch=None)

        graph.replay.assert_called_once_with()
        self.assertEqual(output, "output")

    def test_replay_update_converts_legacy_sequence_lengths_to_int32_tensor(self):
        backend, runner = _make_backend()
        graph = mock.Mock()
        shape_key = ShapeKey(size=2)
        backend._graphs[shape_key] = graph
        backend._outputs[shape_key] = "output"

        output = backend.replay_with_input_update(
            shape_key,
            seq_lens=[7, 9],
            attr_name="seq_lens",
            attr_type=torch.empty(0),
        )

        runner.device_module.set_device.assert_called_once_with(3)
        graph.replay.assert_called_once_with()
        update_input = graph.update.call_args.kwargs["cpu_update_input"]
        self.assertEqual(len(update_input), 1)
        self.assertEqual(set(update_input[0]), {"seq_lens"})
        self.assertEqual(update_input[0]["seq_lens"].dtype, torch.int32)
        self.assertTrue(torch.equal(update_input[0]["seq_lens"], torch.tensor([7, 9])))
        self.assertEqual(output, "output")

    def test_capture_session_reuses_pool_and_resets_stream(self):
        backend, runner = _make_backend()
        runner.device_module.graph_pool_handle = mock.Mock(return_value="pool")

        with (
            mock.patch.object(mod, "set_graph_pool_id") as set_pool,
            backend.capture_session("stream"),
        ):
            self.assertEqual(backend._capture_stream, "stream")

        self.assertIsNone(backend._capture_stream)
        runner.device_module.graph_pool_handle.assert_called_once_with()
        set_pool.assert_called_once_with("pool")


if __name__ == "__main__":
    unittest.main()
