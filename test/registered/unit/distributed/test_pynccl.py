"""Unit tests for PyNccl communicator state transitions."""

import unittest
from unittest.mock import Mock, patch

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

        with communicator.change_state(enable=True):
            self.assertFalse(communicator.available)
            self.assertTrue(communicator.disabled)

        self.assertFalse(communicator.available)
        self.assertTrue(communicator.disabled)

    def test_unavailable_communicator_stays_disabled_by_default(self):
        communicator = self.make_unavailable_communicator()

        with communicator.change_state():
            self.assertFalse(communicator.available)
            self.assertTrue(communicator.disabled)

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

    def test_available_communicator_is_temporarily_disabled(self):
        communicator = self.make_available_communicator(disabled=False)

        with communicator.change_state(enable=False):
            self.assertTrue(communicator.available)
            self.assertTrue(communicator.disabled)

        self.assertTrue(communicator.available)
        self.assertFalse(communicator.disabled)


if __name__ == "__main__":
    unittest.main()
