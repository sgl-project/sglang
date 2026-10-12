"""An external DFlash draft uses full TP, not the target's CP placement."""

import unittest

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _group(size, rank):
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.world_size = size
    group.rank_in_group = rank
    return group


class TestDflashCpTopology(CustomTestCase):
    def test_full_tp_scope_restores_target_including_on_exception(self):
        for rank in range(4):
            for fail in (False, True):
                with published_topology(tp_size=4, attn_cp_size=4):
                    tp, attn_tp = _group(4, rank), _group(1, 0)
                    with get_parallel().override(
                        tp_group=tp,
                        attn_tp_group=attn_tp,
                        attn_cp_group=tp,
                        attn_cp_rank=rank,
                        tp_rank=rank,
                        enable_prefill_cp=True,
                        enable_cp_tp_group_sharing=True,
                    ):
                        try:
                            with draft_tp_context(True):
                                p = get_parallel()
                                self.assertIs(p.tp_group, tp)
                                self.assertIs(p.attn_tp_group, tp)
                                self.assertEqual(p.attn_tp_size, 4)
                                self.assertEqual(p.attn_tp_rank, rank)
                                self.assertEqual(p.attn_cp_size, 1)
                                self.assertFalse(p.enable_prefill_cp)
                                self.assertFalse(p.enable_cp_tp_group_sharing)
                                if fail:
                                    raise RuntimeError("draft failed")
                        except RuntimeError as exc:
                            self.assertEqual(str(exc), "draft failed")
                        p = get_parallel()
                        self.assertEqual(p.attn_cp_size, 4)
                        self.assertEqual(p.attn_cp_rank, rank)
                        self.assertIs(p.attn_tp_group, attn_tp)
                        self.assertTrue(p.enable_prefill_cp)
                        self.assertTrue(p.enable_cp_tp_group_sharing)


if __name__ == "__main__":
    unittest.main()
