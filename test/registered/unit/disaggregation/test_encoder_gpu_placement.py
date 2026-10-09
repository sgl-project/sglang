import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.encoder import runtime, server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DistributedBoundaryReached(Exception):
    pass


class TestEncoderGpuPlacement(unittest.TestCase):
    def test_tensor_parallel_device_matches_distributed_device(self):
        for base, step, rank, explicit, expected in (
            (0, 2, 1, None, 2),
            (3, 2, 1, None, 5),
            (0, 1, 1, None, 1),
            (3, 2, 0, None, 3),
            (3, 2, 0, 7, 7),
            (3, 2, 0, 0, 0),
        ):
            with self.subTest(base=base, step=step, rank=rank, explicit=explicit):
                device = SimpleNamespace(
                    device="cuda", base_gpu_id=base, gpu_id_step=step
                )
                device_module = Mock()
                distributed_init = Mock(side_effect=_DistributedBoundaryReached)
                with contextlib.ExitStack() as stack:
                    for name in (
                        "assert_published",
                        "configure_media_url_security",
                        "EncoderProfiler",
                        "ModelConfig",
                        "LoadConfig",
                        "DeviceConfig",
                        "get_model",
                        "get_mm",
                        "get_disagg",
                        "get_default_distributed_backend",
                    ):
                        stack.enter_context(patch.object(server, name))
                    stack.enter_context(
                        patch.object(server, "get_device", return_value=device)
                    )
                    stack.enter_context(
                        patch.object(
                            server,
                            "get_parallel",
                            return_value=SimpleNamespace(tp_size=2),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            server.torch,
                            "get_device_module",
                            return_value=device_module,
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            server, "init_distributed_environment", distributed_init
                        )
                    )
                    encoder = server.MMEncoder.__new__(server.MMEncoder)
                    # Stop at the distributed boundary before creating process groups
                    # or loading weights, after the actual constructor chooses a device.
                    with self.assertRaises(_DistributedBoundaryReached):
                        encoder.__init__(
                            Mock(),
                            rank=rank,
                            gpu_id=explicit,
                            dist_init_method="tcp://test",
                        )
                    self.assertEqual(encoder.gpu_id, expected)
                    device_module.set_device.assert_called_once_with(expected)
                    self.assertEqual(
                        server.DeviceConfig.call_args.kwargs["gpu_id"], expected
                    )
                    self.assertEqual(distributed_init.call_args.kwargs["rank"], rank)
                    self.assertEqual(
                        distributed_init.call_args.kwargs["local_rank"], expected
                    )

    def test_data_parallel_launcher_applies_stride_before_reindex(self):
        for base, step, reindex in (
            (0, 2, False),
            (3, 3, False),
            (0, 1, False),
            (3, 2, True),
        ):
            with self.subTest(base=base, step=step, reindex=reindex):
                selected_devices = []

                @contextlib.contextmanager
                def select_device(gpu_id):
                    selected_devices.append(gpu_id)
                    yield 0 if reindex else gpu_id

                process_context = Mock()
                with contextlib.ExitStack() as stack:
                    for name in ("get_zmq_socket", "DPDispatcher"):
                        stack.enter_context(patch.object(runtime, name))
                    stack.enter_context(patch.object(runtime.atexit, "register"))
                    stack.enter_context(patch.object(runtime.zmq.asyncio, "Context"))
                    stack.enter_context(
                        patch.object(
                            runtime.mp, "get_context", return_value=process_context
                        )
                    )
                    stack.enter_context(
                        patch.object(runtime, "maybe_reindex_device_id", select_device)
                    )
                    stack.enter_context(
                        patch.object(
                            runtime,
                            "get_device",
                            return_value=SimpleNamespace(
                                base_gpu_id=base, gpu_id_step=step
                            ),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            runtime,
                            "get_parallel",
                            return_value=SimpleNamespace(
                                dp_size=3, tp_size=1, num_dp_ranks=3
                            ),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            runtime,
                            "get_observability",
                            return_value=SimpleNamespace(
                                enable_metrics=False, extra_metric_labels=None
                            ),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            runtime,
                            "get_serving",
                            return_value=SimpleNamespace(served_model_name="test"),
                        )
                    )
                    runtime.launch_dp_runtime(Mock())

                expected = [base + rank * step for rank in range(3)]
                self.assertEqual(selected_devices, expected)
                calls = process_context.Process.call_args_list
                self.assertEqual([call.kwargs["args"][1] for call in calls], [0, 1, 2])
                self.assertEqual(
                    [call.kwargs["args"][2] for call in calls],
                    [0] * 3 if reindex else expected,
                )


if __name__ == "__main__":
    unittest.main()
