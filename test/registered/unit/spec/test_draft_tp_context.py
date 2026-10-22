import unittest
from types import SimpleNamespace

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    _resolve_dflash_draft_cell_size,
)
from sglang.srt.runtime_context import draft_attn_tp_size, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
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

    def test_cp_does_not_join_independent_dp_drafts(self):
        attn_tp = _group(world_size=1, rank=0)
        with get_parallel().override(
            tp_group=self.whole_tp,
            attn_tp_group=attn_tp,
            attn_tp_size=1,
            attn_cp_size=2,
        ):
            with draft_tp_context(True):
                self.assertIs(get_parallel().tp_group, attn_tp)
                self.assertEqual(get_parallel().attn_cp_size, 1)

    def test_cp_draft_width_matches_target_side_geometry_and_restores_on_error(self):
        with get_parallel().override(
            tp_group=self.whole_tp,
            attn_tp_group=_group(world_size=1, rank=0),
            attn_tp_size=1,
            attn_cp_size=4,
            attn_dp_size=1,
        ):
            draft_width = draft_attn_tp_size()
            self.assertEqual(draft_width, 4)
            with self.assertRaisesRegex(RuntimeError, "draft failed"):
                with draft_tp_context(True):
                    self.assertIs(get_parallel().attn_tp_group, self.whole_tp)
                    self.assertEqual(get_parallel().attn_tp_size, draft_width)
                    self.assertEqual(get_parallel().attn_cp_size, 1)
                    raise RuntimeError("draft failed")
            self.assertEqual(get_parallel().attn_tp_size, 1)
            self.assertEqual(get_parallel().attn_cp_size, 4)
            self.assertFalse(parallel_state._TP_STATE_PATCHED)

    def test_target_kv_budget_matches_draft_execution_geometry(self):
        model = SimpleNamespace(
            dtype=torch.bfloat16,
            head_dim=64,
            v_head_dim=32,
            get_num_kv_heads=lambda tp: 4 // tp,
        )
        for cp, dp, draft_width in ((1, 1, 4), (4, 1, 4), (2, 2, 1)):
            with (
                self.subTest(cp=cp, dp=dp),
                get_parallel().override(
                    tp_group=self.whole_tp,
                    attn_tp_group=_group(world_size=4 // cp // dp, rank=0),
                    attn_tp_size=4 // cp // dp,
                    attn_cp_size=cp,
                    attn_dp_size=dp,
                ),
            ):
                budget = _resolve_dflash_draft_cell_size(
                    draft_model_config=model,
                    draft_num_layers=2,
                    spec_algorithm=SpeculativeAlgorithm.DFLASH,
                )
                with draft_tp_context(True):
                    self.assertEqual(get_parallel().attn_tp_size, draft_width)
                    heads = model.get_num_kv_heads(get_parallel().attn_tp_size)
                    self.assertEqual(budget, heads * (64 + 32) * 2 * 2)


if __name__ == "__main__":
    unittest.main()
