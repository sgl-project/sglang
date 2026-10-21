"""PDMux retains custom collectives with grids sized to each lane's SM cap."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.distributed.device_communicators.configs.custom_all_reduce_v2 import (
    AllReduceConfig,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.multiplex import multiplexing_mixin as mixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def make_comm():
    comm = CustomAllReduceV2.__new__(CustomAllReduceV2)
    comm.disabled = False
    comm.rank = 0
    comm._capture_started = False
    comm._graph_mode_allowed = False
    comm._eager_plans = {"old": object()}
    comm.config = AllReduceConfig(
        graph=None,
        eager=None,
        num_push_blocks=148,
        num_pull_blocks=96,
        num_mc_blocks=64,
    )
    comm._init_workspace = Mock()
    return comm


class TestPDMuxCustomAllReduce(unittest.TestCase):
    def test_overlap_caps_prefill_and_preserves_full_device_decode(self):
        prefill, decode = make_comm(), make_comm()
        parallel = SimpleNamespace(tp_group=SimpleNamespace(ca_comm=decode))

        @contextmanager
        def prefill_group():
            with patch.object(parallel.tp_group, "ca_comm", prefill):
                yield

        scheduler = SimpleNamespace(sm_counts=[(148, 0), (128, 148), (0, 148)])
        with (
            patch.object(mixin, "get_parallel", return_value=parallel),
            patch.object(mixin, "pdmux_prefill_tp_group", prefill_group),
        ):
            mixin.SchedulerMultiplexMixin._limit_pdmux_collective_grids(scheduler)
        self.assertEqual(prefill.config.num_push_blocks, 128)
        self.assertEqual(prefill.config.num_pull_blocks, 96)
        prefill._init_workspace.assert_called_once_with()
        self.assertFalse(prefill._eager_plans)
        self.assertEqual(decode.config.num_push_blocks, 148)
        decode._init_workspace.assert_not_called()
        self.assertIn("old", decode._eager_plans)

    def test_small_partition_caps_all_grids_and_is_idempotent(self):
        comm = make_comm()
        comm.limit_sm_count(20)
        self.assertEqual(
            (
                comm.config.num_push_blocks,
                comm.config.num_pull_blocks,
                comm.config.num_mc_blocks,
            ),
            (20, 20, 20),
        )
        comm.limit_sm_count(128)
        comm._init_workspace.assert_called_once_with()

    def test_cannot_rebuild_graph_captured_workspace(self):
        for flag in ("_capture_started", "_graph_mode_allowed"):
            comm = make_comm()
            setattr(comm, flag, True)
            with self.assertRaisesRegex(RuntimeError, "before capture"):
                comm.limit_sm_count(128)
            comm._init_workspace.assert_not_called()
            self.assertEqual(comm.config.num_push_blocks, 148)

    def test_absent_multicast_and_disabled_communicator(self):
        comm = make_comm()
        comm.config = comm.config._replace(num_mc_blocks=None)
        comm.limit_sm_count(128)
        self.assertIsNone(comm.config.num_mc_blocks)
        comm.disabled = True
        comm.limit_sm_count(20)
        comm._init_workspace.assert_called_once_with()
        with self.assertRaises(ValueError):
            comm.limit_sm_count(0)


if __name__ == "__main__":
    unittest.main()
