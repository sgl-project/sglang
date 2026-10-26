"""NPU EPLB uses a cached, independently configured WORLD communicator."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.hardware_backend.npu import eplb as comm_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNpuEplbGroup(unittest.TestCase):
    def setUp(self):
        self.world = object()
        self.group = object()
        self.options = SimpleNamespace()
        self.options_factory = Mock(return_value=self.options)
        self.new_group = Mock(return_value=self.group)
        fake_npu = SimpleNamespace(
            _C=SimpleNamespace(
                _distributed_c10d=SimpleNamespace(
                    ProcessGroupHCCL=SimpleNamespace(Options=self.options_factory)
                )
            )
        )
        for item in (
            patch.dict("sys.modules", {"torch_npu": fake_npu}),
            patch.object(comm_utils, "_group_world", None),
            patch.object(comm_utils, "_device_group", None),
            patch.object(
                comm_utils, "get_device", return_value=SimpleNamespace(device="npu")
            ),
            patch.object(comm_utils.dist, "is_initialized", return_value=True),
            patch.object(comm_utils.dist, "group", SimpleNamespace(WORLD=self.world)),
            patch.object(comm_utils.dist, "get_world_size", return_value=4),
            patch.object(comm_utils.dist, "new_group", self.new_group),
        ):
            item.start()
            self.addCleanup(item.stop)

    def test_full_world_membership_and_host_expansion(self):
        self.assertIs(comm_utils.get_eplb_device_group(), self.group)
        self.assertEqual(
            self.options.hccl_config,
            {"hccl_buffer_size": 32, "hccl_op_expansion_mode": 1},
        )
        # No ranks override: new_group uses all WORLD members in WORLD order.
        self.new_group.assert_called_once_with(backend="hccl", pg_options=self.options)

    def test_repeated_collectives_reuse_the_same_group(self):
        self.assertIs(comm_utils.get_eplb_device_group(), self.group)
        self.assertIs(comm_utils.get_eplb_device_group(), self.group)
        self.new_group.assert_called_once()

    def test_reinitialized_world_does_not_reuse_stale_group(self):
        comm_utils.get_eplb_device_group()
        replacement = object()
        self.new_group.return_value = replacement
        with patch.object(comm_utils.dist.group, "WORLD", object()):
            self.assertIs(comm_utils.get_eplb_device_group(), replacement)
        self.assertEqual(self.new_group.call_count, 2)

    def test_other_devices_and_uninitialized_dist_do_not_create_group(self):
        for device in ("cpu", "cuda"):
            with patch.object(
                comm_utils, "get_device", return_value=SimpleNamespace(device=device)
            ):
                self.assertIsNone(comm_utils.get_eplb_device_group())
        with patch.object(comm_utils.dist, "is_initialized", return_value=False):
            self.assertIsNone(comm_utils.get_eplb_device_group())
        self.new_group.assert_not_called()
        self.options_factory.assert_not_called()

    def test_setup_errors_propagate_without_caching_failed_world(self):
        self.new_group.side_effect = RuntimeError("group setup failed")
        with self.assertRaisesRegex(RuntimeError, "group setup failed"):
            comm_utils.get_eplb_device_group()
        self.assertIsNone(comm_utils._group_world)
        self.new_group.side_effect = None
        self.assertIs(comm_utils.get_eplb_device_group(), self.group)


if __name__ == "__main__":
    unittest.main()
