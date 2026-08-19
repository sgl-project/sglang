"""Unit tests for IPC spin-timeout reporting in ipc_a2a.py.

The Ulysses CUDA-IPC transport's ``spin_wait`` kernel gives up after a budget
and flags ``timed_out`` instead of hanging the stream forever. Callers must
then discard the (incomplete) staging data and fall back to NCCL; previously
they consumed the staging view blindly because the flag was only read at the
next request boundary. These tests pin ``wait_and_check`` / ``exchange`` to the
expected behavior: report the timeout, skip the device read inside CUDA graph
capture, and return None from ``exchange`` so callers fall back.
"""

import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    IpcA2AState,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIpcA2aTimeoutReporting(CustomTestCase):
    def _fresh_state(self):
        state = IpcA2AState()
        state.ops = mock.Mock()
        return state

    def test_wait_and_check_reports_no_timeout(self):
        state = self._fresh_state()
        state.timed_out = torch.zeros(1, dtype=torch.int32)
        self.assertFalse(state.wait_and_check())
        state.ops.spin_wait.assert_called_once()

    def test_wait_and_check_reports_timeout(self):
        state = self._fresh_state()
        state.timed_out = torch.ones(1, dtype=torch.int32)
        self.assertTrue(state.wait_and_check())
        state.ops.spin_wait.assert_called_once()

    def test_wait_and_check_skips_device_read_during_capture(self):
        """Inside a CUDA graph capture the spin kernel is only recorded, so a
        device read must not happen; the transport reports no timeout."""
        state = self._fresh_state()
        state.timed_out = torch.ones(1, dtype=torch.int32)
        with mock.patch(
            "torch.cuda.is_current_stream_capturing", return_value=True
        ):
            self.assertFalse(state.wait_and_check())
        state.ops.spin_wait.assert_called_once()

    def test_exchange_returns_none_on_timeout(self):
        """A timed-out exchange must return None so the caller falls back to
        NCCL instead of consuming incomplete staging data."""
        state = self._fresh_state()
        local = torch.zeros(2, 8, dtype=torch.float32)
        peer = torch.zeros(2, 8, dtype=torch.float32)
        state.get_staging = mock.Mock(return_value=(local, peer))
        state.next_slot = mock.Mock(return_value=0)
        state.signal = mock.Mock()
        state.wait_and_check = mock.Mock(return_value=True)
        send = torch.zeros(8, dtype=torch.float32)
        self.assertIsNone(state.exchange(mock.Mock(), send, (2, 4)))
        state.signal.assert_called_once()

    def test_exchange_returns_data_on_success(self):
        state = self._fresh_state()
        local = torch.zeros(2, 8, dtype=torch.float32)
        peer = torch.zeros(2, 8, dtype=torch.float32)
        state.get_staging = mock.Mock(return_value=(local, peer))
        state.next_slot = mock.Mock(return_value=0)
        state.signal = mock.Mock()
        state.wait_and_check = mock.Mock(return_value=False)
        send = torch.zeros(8, dtype=torch.float32)
        result = state.exchange(mock.Mock(), send, (2, 4))
        self.assertIsNotNone(result)
        self.assertEqual(tuple(result.shape), (2, 4))
        state.signal.assert_called_once()

    def test_check_timeout_still_retires_transport(self):
        """check_timeout keeps raising at the request boundary so the transport
        is retired on every rank together."""
        state = self._fresh_state()
        state.inited = True
        state.failed = False
        state.timed_out = torch.ones(1, dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            state.check_timeout()
        self.assertTrue(state.failed)


if __name__ == "__main__":
    unittest.main()
