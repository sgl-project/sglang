"""fi_a2a on a system without MNNVL fabric is admitted only when every DCP rank
is on one node; a multi-node group must still be rejected.

    python -m pytest test/registered/dcp/test_fi_a2a_gate_unit.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.dcp.comm import _check_fi_a2a_single_node
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cp_group():
    return SimpleNamespace(device_group=object(), world_size=4, rank_in_group=0)


def _all_gather_hosts(hosts):
    def _fn(out_list, _obj, group=None):
        out_list[:] = list(hosts)

    return _fn


class TestFiA2AIntraNodeGate(CustomTestCase):
    def test_accepts_single_node_group(self):
        with patch(
            "torch.distributed.all_gather_object",
            side_effect=_all_gather_hosts(["node-a"] * 4),
        ):
            _check_fi_a2a_single_node(_cp_group(), 4)

    def test_rejects_multi_node_group(self):
        with patch(
            "torch.distributed.all_gather_object",
            side_effect=_all_gather_hosts(["node-a", "node-a", "node-b", "node-b"]),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                _check_fi_a2a_single_node(_cp_group(), 4)
        self.assertIn("one node", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
