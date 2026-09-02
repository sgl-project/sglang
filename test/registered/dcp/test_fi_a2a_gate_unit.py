"""CPU unit test for the ``fi_a2a`` MNNVL capability gate.

``init_fi_a2a_workspace`` used to hard-reject every system whose
``is_mnnvl_fabric_supported()`` is False, i.e. every Blackwell box without an
IMEX/NVL72 fabric (x86 B200/B300). FlashInfer 0.6.16 (flashinfer #3701) exports
the MNNVL workspace with POSIX file-descriptor handles and exchanges them over
AF_UNIX ``SCM_RIGHTS`` on such systems, which works inside a default-capability
container (the pidfd_getfd exchange it replaced needed CAP_SYS_PTRACE). The
remaining requirement is that all DCP ranks share one node.

This pins the replacement gate, ``_check_fi_a2a_intra_node_fallback``.

Usage:
    python -m pytest test_fi_a2a_gate_unit.py -v
    python test_fi_a2a_gate_unit.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.dcp.comm import _check_fi_a2a_intra_node_fallback
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _cp_group():
    return SimpleNamespace(device_group=object(), world_size=4, rank_in_group=0)


def _all_gather_hosts(hosts):
    def _fn(out_list, _obj, group=None):
        out_list[:] = list(hosts)

    return _fn


class TestFiA2AIntraNodeGate(CustomTestCase):
    def test_accepts_single_node_group(self):
        with patch("importlib.util.find_spec", return_value=object()), patch(
            "torch.distributed.all_gather_object",
            side_effect=_all_gather_hosts(["node-a"] * 4),
        ):
            # Must not raise: one node, FlashInfer has the SCM_RIGHTS exchanger.
            _check_fi_a2a_intra_node_fallback(_cp_group(), 4)

    def test_rejects_flashinfer_without_fd_exchange(self):
        with patch("importlib.util.find_spec", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                _check_fi_a2a_intra_node_fallback(_cp_group(), 4)
        self.assertIn("0.6.16", str(ctx.exception))

    def test_rejects_multi_node_group(self):
        with patch("importlib.util.find_spec", return_value=object()), patch(
            "torch.distributed.all_gather_object",
            side_effect=_all_gather_hosts(["node-a", "node-a", "node-b", "node-b"]),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                _check_fi_a2a_intra_node_fallback(_cp_group(), 4)
        self.assertIn("several nodes", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
