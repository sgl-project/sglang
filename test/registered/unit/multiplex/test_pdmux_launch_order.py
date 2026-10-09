import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed import bootstrap
from sglang.srt.multiplex import launch_order
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPDMuxLaunchOrder(unittest.TestCase):
    def _configure(
        self, *, torch_nccl=(2, 27, 3), pynccl=22703, cuda=(12, 8), driver=12080
    ):
        with (
            patch.object(launch_order.torch.version, "cuda", "12.8"),
            patch.object(
                launch_order.torch.cuda.nccl, "version", return_value=torch_nccl
            ),
            patch.object(
                launch_order,
                "NCCLLibrary",
                return_value=SimpleNamespace(ncclGetRawVersion=lambda: pynccl),
            ),
            patch.object(launch_order, "get_cuda_version", return_value=cuda),
            patch.object(
                launch_order,
                "get_cuda_driver_bindings",
                return_value=SimpleNamespace(cuDriverGetVersion=lambda: (0, driver)),
            ),
        ):
            launch_order.configure_pdmux_nccl_launch_order()

    def test_default_enables_implicit_order(self):
        with patch.dict(os.environ, {}, clear=True):
            self._configure()
            self.assertEqual(os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"], "1")

    def test_explicit_disable_is_rejected(self):
        with patch.dict(os.environ, {"NCCL_LAUNCH_ORDER_IMPLICIT": "0"}):
            with self.assertRaisesRegex(RuntimeError, "explicitly sets"):
                self._configure()

    def test_existing_enable_keeps_graph_mixing_policy(self):
        with patch.dict(
            os.environ,
            {"NCCL_LAUNCH_ORDER_IMPLICIT": "1", "NCCL_GRAPH_MIXING_SUPPORT": "1"},
            clear=True,
        ):
            self._configure()
            self.assertEqual(os.environ["NCCL_GRAPH_MIXING_SUPPORT"], "1")

    def test_non_cuda_does_not_impose_nccl_policy(self):
        with (
            patch.object(launch_order.torch.version, "cuda", None),
            patch.dict(os.environ, {}, clear=True),
            patch.object(launch_order, "NCCLLibrary") as library,
        ):
            launch_order.configure_pdmux_nccl_launch_order()
            library.assert_not_called()
            self.assertNotIn("NCCL_LAUNCH_ORDER_IMPLICIT", os.environ)

    def test_both_libraries_and_cuda_versions_must_support_overlap(self):
        for versions in (
            {"torch_nccl": (2, 25, 1)},
            {"pynccl": 22501},
            {"cuda": (12, 2)},
            {"driver": 12020},
        ):
            with (
                self.subTest(versions=versions),
                patch.dict(os.environ, {}, clear=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "requires NCCL"):
                    self._configure(**versions)

    def test_order_is_configured_before_first_process_group(self):
        class StopAtInitialization(Exception):
            pass

        for backend, pdmux in (("nccl", True), ("nccl", False), ("gloo", True)):
            events = []

            def initialize(**kwargs):
                events.append("communicator")
                raise StopAtInitialization

            with (
                self.subTest(backend=backend, pdmux=pdmux),
                patch.object(
                    bootstrap,
                    "get_parallel",
                    return_value=SimpleNamespace(
                        tp_size=8,
                        pp_size=1,
                        tp_rank=0,
                        pp_rank=0,
                        max_ep_size=8,
                        dist_timeout=10,
                    ),
                ),
                patch.object(
                    bootstrap,
                    "get_exec",
                    return_value=SimpleNamespace(
                        moe=SimpleNamespace(
                            is_ep_joiner=False,
                            is_ep_scale_joiner=False,
                            moe_a2a_backend=None,
                        ),
                    ),
                ),
                patch.object(
                    bootstrap,
                    "get_disagg",
                    return_value=SimpleNamespace(enable_pdmux=pdmux),
                ),
                patch.object(
                    launch_order,
                    "configure_pdmux_nccl_launch_order",
                    side_effect=lambda: events.append("configure"),
                ),
                patch.object(
                    bootstrap, "init_distributed_environment", side_effect=initialize
                ),
            ):
                with self.assertRaises(StopAtInitialization):
                    bootstrap._init_parallel_groups(
                        backend=backend,
                        dist_init_method="test",
                        server_args=None,
                        gpu_id=0,
                    )
                self.assertEqual(
                    events,
                    ["configure", "communicator"]
                    if backend == "nccl" and pdmux
                    else ["communicator"],
                )


if __name__ == "__main__":
    unittest.main()
