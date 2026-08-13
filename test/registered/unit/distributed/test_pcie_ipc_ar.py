"""Unit tests for the FlashInfer PCIe-IPC all-reduce adapter.

These cover the parts that decide whether a reduction reaches the kernels at
all -- world-size gating, workspace sizing, and the shape guard -- because each
of them fails *silently* in production: an unsupported shape is not an error,
it is a fall back to NCCL, which looks exactly like the kernels being slow.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.distributed.device_communicators import pcie_ipc_ar
from sglang.srt.distributed.device_communicators.pcie_ipc_ar import (
    PcieIpcCommunicator,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

HIDDEN = 6144


def _make_comm(world_size=8, decode_width=64, workspace_cls=None):
    """Build a communicator with the collaborators stubbed out.

    The constructor is bypassed: it needs a live process group and a CUDA
    device, neither of which this test has, and neither of which the logic
    under test depends on.
    """
    comm = PcieIpcCommunicator.__new__(PcieIpcCommunicator)
    comm.disabled = False
    comm.max_numel = 0
    comm._workspace = None
    comm._bound_stream = None
    comm._group = MagicMock()
    comm._device = torch.device("cpu")
    comm._world_size = world_size
    comm._workspace_cls = workspace_cls or MagicMock()
    comm._build_failed = False
    return comm


class TestWorldSizeGate(CustomTestCase):
    def test_unsupported_world_size_disables(self):
        """A world size the kernels have no IPC channels for must disable cleanly."""
        for world_size in (1, 3, 5, 6, 7, 16):
            with self.subTest(world_size=world_size):
                with patch.object(
                    pcie_ipc_ar.dist, "get_world_size", return_value=world_size
                ):
                    comm = PcieIpcCommunicator(group=MagicMock(), device=0)
                self.assertTrue(comm.disabled)

    def test_missing_flashinfer_disables_instead_of_raising(self):
        """A build without pcie_ipc_comm must degrade to NCCL, not crash the server."""
        with patch.object(pcie_ipc_ar.dist, "get_world_size", return_value=8), patch(
            "builtins.__import__", side_effect=ImportError("no pcie_ipc_comm")
        ):
            comm = PcieIpcCommunicator(group=MagicMock(), device=0)
        self.assertTrue(comm.disabled)


class TestWorkspaceSizing(CustomTestCase):
    def test_sized_for_decode_not_prefill(self):
        """The default bound is the decode width, so prefill chunks stay on NCCL.

        Sizing for a prefill chunk was measured 66% worse on TTFT and bought
        nothing on TPOT, so this is the behaviour that must not regress.
        """
        comm = _make_comm(decode_width=64)
        with patch.object(pcie_ipc_ar, "_decode_width", return_value=64):
            self.assertTrue(comm._ensure_workspace(torch.empty(1, HIDDEN)))
        self.assertEqual(comm.max_numel, 64 * HIDDEN)

        prefill = torch.empty(16384, HIDDEN)
        self.assertGreater(prefill.numel(), comm.max_numel)
        self.assertFalse(comm.should_pcie_ipc_ar(prefill))

    def test_falls_back_when_server_args_absent(self):
        """Embedded/unit use has no server args; the bound must still be finite."""
        comm = _make_comm()
        with patch.object(pcie_ipc_ar, "_decode_width", return_value=None):
            comm._ensure_workspace(torch.empty(1, HIDDEN))
        self.assertEqual(
            comm.max_numel, pcie_ipc_ar._FALLBACK_DECODE_WIDTH * HIDDEN
        )

    def test_env_override_wins(self):
        comm = _make_comm()
        with envs.SGLANG_PCIE_IPC_MAX_NUMEL.override(123456):
            with patch.object(pcie_ipc_ar, "_decode_width", return_value=64):
                comm._ensure_workspace(torch.empty(1, HIDDEN))
        self.assertEqual(comm.max_numel, 123456)

    def test_build_failure_disables_once(self):
        """A workspace that cannot be allocated must not be retried every layer."""
        cls = MagicMock(side_effect=RuntimeError("out of IPC handles"))
        comm = _make_comm(workspace_cls=cls)
        with patch.object(pcie_ipc_ar, "_decode_width", return_value=64):
            self.assertFalse(comm._ensure_workspace(torch.empty(1, HIDDEN)))
            self.assertFalse(comm._ensure_workspace(torch.empty(1, HIDDEN)))
        self.assertTrue(comm.disabled)
        self.assertEqual(cls.call_count, 1)


class TestShapeGuard(CustomTestCase):
    def _ready_comm(self):
        comm = _make_comm()
        with patch.object(pcie_ipc_ar, "_decode_width", return_value=64):
            comm._ensure_workspace(torch.empty(1, HIDDEN))
        comm._workspace.supports.return_value = True
        return comm

    def test_delegates_to_flashinfer_supports(self):
        """Coverage is FlashInfer's decision; a rejected shape keeps the NCCL path."""
        comm = self._ready_comm()
        inp = torch.empty(4, HIDDEN)
        self.assertTrue(comm.should_pcie_ipc_ar(inp))

        comm._workspace.supports.return_value = False
        self.assertFalse(comm.should_pcie_ipc_ar(inp))

    def test_rejects_noncontiguous_and_1d(self):
        comm = self._ready_comm()
        self.assertFalse(comm.should_pcie_ipc_ar(torch.empty(4, HIDDEN).t()))
        self.assertFalse(comm.should_pcie_ipc_ar(torch.empty(HIDDEN)))

    def test_disabled_communicator_never_claims_a_tensor(self):
        comm = self._ready_comm()
        comm.disabled = True
        self.assertFalse(comm.should_pcie_ipc_ar(torch.empty(4, HIDDEN)))


if __name__ == "__main__":
    unittest.main()
