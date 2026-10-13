import unittest
from types import SimpleNamespace

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
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

    def test_shared_cp_tp_draft_keeps_full_tp_without_target_cp(self):
        with get_parallel().override(
            tp_group=self.whole_tp,
            attn_tp_group=_group(world_size=1, rank=0),
            attn_tp_size=1,
            attn_dp_size=1,
            attn_cp_size=4,
            enable_cp_tp_group_sharing=True,
            enable_prefill_cp=True,
            enable_cp_decode_attn_tp=True,
        ):
            with draft_tp_context(True):
                parallel = get_parallel()
                self.assertIs(parallel.tp_group, self.whole_tp)
                self.assertEqual(parallel.attn_tp_size, 4)
                self.assertEqual(parallel.attn_cp_size, 1)
                self.assertFalse(parallel.enable_prefill_cp)
                self.assertFalse(parallel.enable_cp_decode_attn_tp)
                self.assertFalse(parallel.enable_cp_tp_group_sharing)
            self.assertEqual(get_parallel().attn_cp_size, 4)
            self.assertTrue(get_parallel().enable_prefill_cp)
            self.assertTrue(get_parallel().enable_cp_decode_attn_tp)

    def test_dflash_pool_uses_the_same_head_partition_as_draft(self):
        pools = []

        def allocate(**kwargs):
            # Eight KV heads, like DFlash2: allocate each rank's head shard.
            pools.append(torch.empty(2, 8 // get_parallel().attn_tp_size, 128))

        worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
        worker._draft_worker = SimpleNamespace(alloc_memory_pool=allocate)
        worker.draft_owns_attention = True
        worker.use_compact_draft_cache = False
        with get_parallel().override(
            tp_group=self.whole_tp,
            attn_tp_group=_group(world_size=1, rank=0),
            attn_tp_size=1,
            attn_dp_size=1,
            attn_cp_size=4,
            enable_cp_tp_group_sharing=True,
            enable_prefill_cp=True,
        ):
            worker.alloc_memory_pool()
            self.assertEqual(pools[0].shape, (2, 2, 128))
            self.assertEqual(get_parallel().attn_cp_size, 4)


if __name__ == "__main__":
    unittest.main()
