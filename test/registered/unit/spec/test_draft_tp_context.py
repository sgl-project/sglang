import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.draft_parallel import DraftParallelSpec, DraftTPExecution
from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.mem_cache.draft_kv_budget import resolve_dflash_draft_cell_size
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.draft_parallel import (
    legacy_draft_parallel_spec,
    model_draft_parallel_spec,
    resolve_draft_parallel_layout,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _group(*, world_size, rank):
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.world_size = world_size
    group.rank_in_group = rank
    return group


class TestDraftTpContext(CustomTestCase):
    """Draft construction, execution and KV sizing share one resolved layout."""

    def setUp(self):
        topology = published_topology(tp_size=4, attn_dp_size=2, ranks=dict(dp_rank=0))
        topology.__enter__()
        self.addCleanup(topology.__exit__, None, None, None)
        self.whole_tp = _group(world_size=4, rank=0)
        self.attn_tp = _group(world_size=2, rank=0)

    def _resolve(self, *, single_group=True, collapse_cp=False):
        return resolve_draft_parallel_layout(
            DraftParallelSpec(
                tp_execution=DraftTPExecution.SINGLE_GROUP
                if single_group
                else DraftTPExecution.TARGET_LAYOUT
            ),
            collapse_cp=collapse_cp,
        )

    def test_owning_attention_runs_on_the_attention_tp_group(self):
        with get_parallel().override(
            tp_group=self.whole_tp, attn_tp_group=self.attn_tp
        ):
            with draft_tp_context(self._resolve()):
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
            with draft_tp_context(self._resolve(single_group=False)):
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
            with draft_tp_context(self._resolve()):
                self.assertIs(get_parallel().tp_group, attn_tp)
                self.assertEqual(get_parallel().attn_cp_size, 1)

    def test_cp_draft_width_matches_target_side_geometry_and_restores_on_error(self):
        with get_parallel().override(
            tp_group=self.whole_tp,
            attn_tp_group=_group(world_size=1, rank=0),
            attn_tp_size=1,
            attn_cp_size=4,
            attn_dp_size=1,
            attn_dp_enabled=False,
        ):
            layout = self._resolve(collapse_cp=True)
            draft_width = layout.attn_tp_size
            self.assertEqual(draft_width, 4)
            with self.assertRaisesRegex(RuntimeError, "draft failed"):
                with draft_tp_context(layout):
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
                    attn_dp_enabled=dp > 1,
                ),
            ):
                layout = self._resolve(collapse_cp=True)
                budget = resolve_dflash_draft_cell_size(
                    draft_model_config=model,
                    draft_num_layers=2,
                    attn_tp_size=layout.attn_tp_size,
                )
                with draft_tp_context(layout):
                    self.assertEqual(get_parallel().attn_tp_size, draft_width)
                    heads = model.get_num_kv_heads(get_parallel().attn_tp_size)
                    self.assertEqual(budget, heads * (64 + 32) * 2 * 2)

    def test_cp_and_target_gather_state_restore_after_failure(self):
        from sglang.srt.layers.cp import base as cp
        from sglang.srt.layers.dp_attention import dp_gather_width, dp_slot_in
        from sglang.srt.runtime_context import get_flags

        strategy = object()
        with (
            get_parallel().override(
                tp_group=self.whole_tp,
                attn_tp_group=_group(world_size=1, rank=0),
                attn_tp_size=1,
                attn_cp_size=2,
                attn_dp_rank=1,
                tp_rank=2,
                enable_prefill_cp=True,
                enable_cp_tp_group_sharing=True,
            ),
            get_flags().dp.override(enabled=True),
            patch.object(cp, "_STRATEGY", strategy),
        ):
            per_rank_tokens = [3, 5]
            width, slot = dp_gather_width(), dp_slot_in(per_rank_tokens)
            layout = self._resolve()
            with self.assertRaisesRegex(RuntimeError, "draft failed"):
                with draft_tp_context(layout):
                    self.assertFalse(get_parallel().enable_prefill_cp)
                    self.assertFalse(get_parallel().enable_cp_tp_group_sharing)
                    self.assertIsNone(cp._STRATEGY)
                    self.assertEqual(
                        (dp_gather_width(), dp_slot_in(per_rank_tokens)), (width, slot)
                    )
                    raise RuntimeError("draft failed")
            self.assertTrue(get_parallel().enable_prefill_cp)
            self.assertTrue(get_parallel().enable_cp_tp_group_sharing)
            self.assertIs(cp._STRATEGY, strategy)
            self.assertIsNone(get_flags().dp.scoped_gather_slot)

    def test_cp_only_is_a_worker_policy(self):
        with get_parallel().override(
            tp_group=self.whole_tp,
            attn_tp_group=_group(world_size=1, rank=0),
            attn_tp_size=1,
            attn_cp_size=4,
            attn_dp_size=1,
            attn_dp_enabled=False,
        ):
            self.assertIsNone(self._resolve().tp_group)
            self.assertIs(self._resolve(collapse_cp=True).tp_group, self.whole_tp)
            self.assertIsNone(
                self._resolve(single_group=False, collapse_cp=True).tp_group
            )

    def test_layout_does_not_reselect_groups_on_entry(self):
        with get_parallel().override(
            tp_group=self.whole_tp, attn_tp_group=self.attn_tp
        ):
            layout = self._resolve()
        with get_parallel().override(attn_tp_group=_group(world_size=2, rank=0)):
            with draft_tp_context(layout):
                self.assertIs(get_parallel().tp_group, self.attn_tp)
                self.assertEqual(get_parallel().attn_tp_size, layout.attn_tp_size)
                with self.assertRaisesRegex(AssertionError, "already patched"):
                    with draft_tp_context(layout):
                        pass
            self.assertFalse(parallel_state._TP_STATE_PATCHED)

    def test_dspark_model_and_logits_subphase_have_distinct_layouts(self):
        with get_parallel().override(
            tp_group=self.whole_tp, attn_tp_group=self.attn_tp
        ):
            model_layout = self._resolve(single_group=False)
            logits_layout = self._resolve()
            with draft_tp_context(model_layout):
                self.assertIs(get_parallel().tp_group, self.whole_tp)
            with draft_tp_context(logits_layout):
                self.assertIs(get_parallel().tp_group, self.attn_tp)
            self.assertIs(get_parallel().tp_group, self.whole_tp)

    def test_registered_model_declaration_precedes_legacy_protocol(self):
        from sglang.srt.configs.model_config import ModelImpl
        from sglang.srt.models.registry import ModelRegistry

        declared = DraftParallelSpec(tp_execution=DraftTPExecution.TARGET_LAYOUT)
        config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLMEagle3"]),
            quantization=None,
            model_impl=ModelImpl.SGLANG,
        )

        class Replacement:
            draft_parallel_spec = declared

        with patch.dict(ModelRegistry.models, {"LlamaForCausalLMEagle3": Replacement}):
            self.assertIs(
                model_draft_parallel_spec(
                    config, spec_algorithm=SpeculativeAlgorithm.EAGLE3
                ),
                declared,
            )
        # A registered implementation with no declaration keeps its old policy.
        with patch.dict(
            ModelRegistry.models, {"LlamaForCausalLMEagle3": type("External", (), {})}
        ):
            spec = model_draft_parallel_spec(
                config, spec_algorithm=SpeculativeAlgorithm.EAGLE3
            )
            self.assertIs(spec.tp_execution, DraftTPExecution.SINGLE_GROUP)

    def test_legacy_adapter_keeps_undeclared_model_policies(self):
        for algorithm, moe, expected in (
            (SpeculativeAlgorithm.EAGLE3, False, DraftTPExecution.SINGLE_GROUP),
            (SpeculativeAlgorithm.EAGLE, False, DraftTPExecution.TARGET_LAYOUT),
            (SpeculativeAlgorithm.DFLASH, False, DraftTPExecution.SINGLE_GROUP),
            (SpeculativeAlgorithm.DSPARK, False, DraftTPExecution.SINGLE_GROUP),
            (SpeculativeAlgorithm.DSPARK, True, DraftTPExecution.TARGET_LAYOUT),
        ):
            with self.subTest(algorithm=algorithm, moe=moe):
                self.assertIs(
                    legacy_draft_parallel_spec(
                        algorithm, dspark_is_moe=moe
                    ).tp_execution,
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
