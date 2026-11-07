"""Unit tests for PyNccl communicator state transitions."""

import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.device_communicators import pynccl
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.utils import StatelessProcessGroup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPyNcclCommunicatorState(CustomTestCase):
    @staticmethod
    def make_unavailable_communicator():
        group = StatelessProcessGroup(rank=0, world_size=2, store=Mock())
        with patch.object(
            pynccl, "NCCLLibrary", side_effect=OSError("NCCL unavailable")
        ):
            return PyNcclCommunicator(group=group, device="cpu")

    @staticmethod
    def make_available_communicator(*, disabled: bool):
        communicator = PyNcclCommunicator.__new__(PyNcclCommunicator)
        communicator.available = True
        communicator.disabled = disabled
        return communicator

    def test_unavailable_communicator_cannot_be_enabled_explicitly(self):
        """Graph capture cannot activate an unavailable communicator."""
        communicator = self.make_unavailable_communicator()

        with self.assertRaisesRegex(RuntimeError, "unavailable PyNccl communicator"):
            with communicator.change_state(enable=True):
                self.fail("An unavailable communicator must not be enabled")

        self.assertFalse(communicator.available)
        self.assertTrue(communicator.disabled)

    def test_available_communicator_is_temporarily_enabled(self):
        communicator = self.make_available_communicator(disabled=True)

        with communicator.change_state(enable=True):
            self.assertTrue(communicator.available)
            self.assertFalse(communicator.disabled)

        self.assertTrue(communicator.available)
        self.assertTrue(communicator.disabled)

    def test_state_is_restored_after_exception(self):
        communicator = self.make_available_communicator(disabled=True)

        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            with communicator.change_state(enable=True):
                self.assertTrue(communicator.available)
                self.assertFalse(communicator.disabled)
                raise RuntimeError("capture failed")

        self.assertTrue(communicator.available)
        self.assertTrue(communicator.disabled)

    def test_group_coordinator_exposes_only_available_pynccl(self):
        coordinator = parallel_state.GroupCoordinator.__new__(
            parallel_state.GroupCoordinator
        )
        unavailable = self.make_unavailable_communicator()
        available = self.make_available_communicator(disabled=True)

        coordinator.pynccl_comm = None
        self.assertIsNone(coordinator.get_available_pynccl_comm())

        coordinator.pynccl_comm = unavailable
        self.assertIsNone(coordinator.get_available_pynccl_comm())
        with self.assertRaisesRegex(
            RuntimeError, "test operation requires an available PyNccl communicator"
        ):
            coordinator.require_pynccl_comm("test operation")

        coordinator.pynccl_comm = available
        self.assertIs(coordinator.get_available_pynccl_comm(), available)
        self.assertIs(coordinator.require_pynccl_comm("test operation"), available)

    def test_graph_capture_skips_unavailable_pynccl(self):
        coordinator = parallel_state.GroupCoordinator.__new__(
            parallel_state.GroupCoordinator
        )
        coordinator.pynccl_comm = self.make_unavailable_communicator()
        coordinator.pymscclpp_comm = None
        coordinator.ca_comm = None
        coordinator.device_module = Mock()
        coordinator.device_module.stream.return_value = nullcontext()
        stream = Mock()
        graph_capture_context = parallel_state.GraphCaptureContext(stream=stream)

        with patch.object(
            parallel_state, "get_current_device_stream_fast", return_value=stream
        ):
            with coordinator.graph_capture(graph_capture_context) as captured_context:
                self.assertIs(captured_context, graph_capture_context)

        self.assertFalse(coordinator.pynccl_comm.available)
        self.assertTrue(coordinator.pynccl_comm.disabled)

    def test_eager_all_reduce_keeps_existing_fallback(self):
        coordinator = parallel_state.GroupCoordinator.__new__(
            parallel_state.GroupCoordinator
        )
        coordinator.pynccl_comm = self.make_unavailable_communicator()
        coordinator.torch_symm_mem_comm = None
        coordinator.device_group = Mock()
        tensor = Mock()

        with patch.object(parallel_state.torch.distributed, "all_reduce") as all_reduce:
            coordinator._all_reduce_in_place(tensor)

        all_reduce.assert_called_once_with(tensor, group=coordinator.device_group)
        self.assertFalse(coordinator.pynccl_comm.available)
        self.assertTrue(coordinator.pynccl_comm.disabled)


if __name__ == "__main__":
    unittest.main()
