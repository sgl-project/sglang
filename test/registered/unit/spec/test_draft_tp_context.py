import unittest

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _group(*, world_size, rank):
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.world_size = world_size
    group.rank_in_group = rank
    return group


class TestDraftTpContext(CustomTestCase):
    """Draft work enters the attention-TP placement only when it owns attention."""

    def setUp(self):
        topology = published_topology(tp_size=4, attn_dp_size=2, ranks=dict(dp_rank=0))
        topology.__enter__()
        self.addCleanup(topology.__exit__, None, None, None)
        self.whole_tp = _group(world_size=4, rank=0)
        self.attn_tp = _group(world_size=2, rank=0)

    def test_owning_attention_runs_on_the_attention_tp_group(self):
        with get_parallel().override(
            tp_group=self.whole_tp, attn_tp_group=self.attn_tp
        ):
            with draft_tp_context(True):
                parallel = get_parallel()
                self.assertIs(parallel.tp_group, self.attn_tp)
                self.assertEqual(parallel.tp_size, 2)
                self.assertEqual(parallel.attn_dp_size, 1)
            self.assertIs(get_parallel().tp_group, self.whole_tp)
            self.assertEqual(get_parallel().tp_size, 4)
            self.assertEqual(get_parallel().attn_dp_size, 2)

    def test_otherwise_the_target_placement_stays(self):
        with get_parallel().override(
            tp_group=self.whole_tp, attn_tp_group=self.attn_tp
        ):
            with draft_tp_context(False):
                parallel = get_parallel()
                self.assertIs(parallel.tp_group, self.whole_tp)
                self.assertEqual(parallel.tp_size, 4)
                self.assertEqual(parallel.attn_dp_size, 2)
                self.assertFalse(parallel_state._TP_STATE_PATCHED)


if __name__ == "__main__":
    unittest.main()
