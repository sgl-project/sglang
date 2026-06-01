"""Unit tests for the DS recall-oracle sink (task2), AC-1.1 force-replacement
(task5), and the per-row oracle payload builder (task1/task3 core). CPU-only.
"""

from __future__ import annotations

import os
import unittest

import torch

from sglang.srt.layers.attention.double_sparsity import oracle_artifact_sink as sink_mod
from sglang.srt.layers.attention.double_sparsity.oracle_artifact_sink import (
    OracleArtifactSink,
    oracle_enabled,
    record_oracle_sample,
    reset_sink_for_testing,
)
from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
    select_topk_sequence_order,
)
from sglang.srt.layers.attention.double_sparsity.selection_recall_oracle import (
    dense_within_window_replace,
    needle_all_tokens_in_topk,
    oracle_payload_for_row,
    selected_contains_needle,
)


class TestOracleSinkGating(unittest.TestCase):
    def setUp(self):
        self._prev = os.environ.get("SGLANG_DS_RECALL_ORACLE")
        reset_sink_for_testing(None)

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("SGLANG_DS_RECALL_ORACLE", None)
        else:
            os.environ["SGLANG_DS_RECALL_ORACLE"] = self._prev
        reset_sink_for_testing(None)

    def test_disabled_by_default_is_noop(self):
        os.environ.pop("SGLANG_DS_RECALL_ORACLE", None)
        reset_sink_for_testing(None)
        self.assertFalse(oracle_enabled())
        wrote = record_oracle_sample(
            request_id="r", trial_id=0, layer_id=3, decode_step=1, payload={"x": 1}
        )
        self.assertFalse(wrote)  # no-op when disabled
        self.assertIsNone(sink_mod.get_sink())

    def test_enabled_records_keyed_payload(self):
        os.environ["SGLANG_DS_RECALL_ORACLE"] = "1"
        reset_sink_for_testing(None)
        self.assertTrue(oracle_enabled())
        wrote = record_oracle_sample(
            request_id="req-7",
            trial_id=2,
            layer_id=5,
            decode_step=9,
            payload={"needle_worst_rank": 42},
        )
        self.assertTrue(wrote)
        rec = sink_mod.get_sink().records
        self.assertEqual(len(rec), 1)
        self.assertEqual(rec[0]["request_id"], "req-7")
        self.assertEqual(rec[0]["layer_id"], 5)
        self.assertEqual(rec[0]["decode_step"], 9)
        self.assertEqual(rec[0]["needle_worst_rank"], 42)

    def test_record_requires_key_schema(self):
        s = OracleArtifactSink()
        with self.assertRaises(ValueError):
            s.record({"request_id": "r", "trial_id": 0})  # missing layer/step


class TestDenseWithinWindowReplace(unittest.TestCase):
    def test_forces_missing_needle_evicting_lowest_score(self):
        # T=10; budget=4. top-4 by score = {2,5,4,7}; needle {0} (rank 4) is missing.
        scores = torch.tensor([[5.0, 3.0, 9.0, 1.0, 7.0, 8.0, 2.0, 6.0, 4.0, 0.0]])
        selected, _ = select_topk_sequence_order(scores, max_top_k=4)
        self.assertFalse(bool(selected_contains_needle(selected, torch.tensor([0]))))

        forced = dense_within_window_replace(selected, scores, torch.tensor([0]))
        # needle now present, count preserved (4), still ascending, lowest-score
        # non-needle (pos7 score6 is highest of the four; pos0 score5 evicts the
        # lowest selected non-needle which is pos4? no — evict lowest SCORE: among
        # {2:9,5:8,4:7,7:6} lowest is 7:6) -> {0,2,4,5}
        real = sorted(p for p in forced[0].tolist() if p >= 0)
        self.assertIn(0, real)
        self.assertEqual(len(real), 4)
        self.assertEqual(real, sorted(real))  # ascending
        self.assertTrue(bool(selected_contains_needle(forced, torch.tensor([0]))))
        # the lowest-scoring selected non-needle (pos7, score 6) was evicted
        self.assertNotIn(7, real)

    def test_noop_when_needle_already_selected(self):
        scores = torch.tensor([[5.0, 3.0, 9.0, 1.0, 7.0, 8.0, 2.0, 6.0, 4.0, 0.0]])
        selected, _ = select_topk_sequence_order(scores, max_top_k=4)
        forced = dense_within_window_replace(selected, scores, torch.tensor([2]))
        self.assertEqual(forced.tolist(), selected.tolist())  # unchanged

    def test_preserves_budget_and_count(self):
        scores = torch.tensor([[5.0, 3.0, 9.0, 1.0, 7.0, 8.0, 2.0, 6.0, 4.0, 0.0]])
        selected, _ = select_topk_sequence_order(scores, max_top_k=4)
        forced = dense_within_window_replace(selected, scores, torch.tensor([0, 9]))
        self.assertEqual(forced.shape, selected.shape)
        real = [p for p in forced[0].tolist() if p >= 0]
        self.assertEqual(len(real), 4)  # count preserved
        self.assertTrue(bool(selected_contains_needle(forced, torch.tensor([0, 9]))))

    def test_needle_larger_than_budget_raises(self):
        scores = torch.randn(1, 10)
        selected, _ = select_topk_sequence_order(scores, max_top_k=2)
        with self.assertRaises(ValueError):
            dense_within_window_replace(selected, scores, torch.tensor([0, 1, 2]))


class TestOraclePayloadRow(unittest.TestCase):
    def test_payload_fields_and_invariant(self):
        scores = torch.tensor([[5.0, 3.0, 9.0, 1.0, 7.0, 8.0, 2.0, 6.0, 4.0, 0.0]])
        budget = 4
        selected, _ = select_topk_sequence_order(scores, max_top_k=budget)
        payload = oracle_payload_for_row(
            scores[0],
            torch.tensor([4]),
            selected_indices_row=selected[0],
            stride=2,
            index_topk=budget,
        )
        self.assertEqual(payload["stride"], 2)
        self.assertEqual(payload["needle_span"], [4])
        self.assertEqual(payload["needle_worst_rank"], 2)  # pos4 is rank 2
        self.assertEqual(payload["index_topk"], budget)
        # recall@budget must equal selected_contains_needle (AC-1 invariant)
        self.assertTrue(payload["recall_at_index_topk_matches_selected"])
        self.assertEqual(
            payload["recall_at_k"][budget], payload["selected_contains_needle"]
        )

    def test_score_only_k_above_index_topk_is_tagged(self):
        scores = torch.randn(1, 6000)
        order = torch.argsort(scores[0], descending=True)
        needle_pos = int(order[3000].item())
        payload = oracle_payload_for_row(
            scores[0], torch.tensor([needle_pos]), index_topk=2048
        )
        # 4096 and 8192 exceed index_topk -> flagged score-only
        self.assertIn(4096, payload["recall_at_k_score_only_above_index_topk"])
        self.assertIn(8192, payload["recall_at_k_score_only_above_index_topk"])
        self.assertNotIn(2048, payload["recall_at_k_score_only_above_index_topk"])
        self.assertFalse(payload["recall_at_k"][2048])  # rank 3000 not in top-2048
        self.assertTrue(payload["recall_at_k"][4096])


class TestOracleOffEquivalence(unittest.TestCase):
    """task4 (CPU portion): with the oracle disabled, the selection result is
    byte-identical and the instrumentation path is a pure no-op (the graph-replay
    allocation half runs on the cluster)."""

    def test_selection_byte_identical_with_oracle_off(self):
        os.environ.pop("SGLANG_DS_RECALL_ORACLE", None)
        reset_sink_for_testing(None)
        scores = torch.randn(3, 200)
        sel_a, vl_a = select_topk_sequence_order(scores.clone(), max_top_k=64)
        # Simulate the hot-path guarded oracle call: when disabled it must not
        # write, allocate a sink, or alter the selection.
        wrote = record_oracle_sample(
            request_id="r", trial_id=0, layer_id=0, decode_step=0, payload={"a": 1}
        )
        sel_b, vl_b = select_topk_sequence_order(scores.clone(), max_top_k=64)
        self.assertFalse(wrote)
        self.assertIsNone(sink_mod.get_sink())
        self.assertTrue(torch.equal(sel_a, sel_b))
        self.assertTrue(torch.equal(vl_a, vl_b))


class TestRetrieveTopkOracleWiring(unittest.TestCase):
    """task1: the flag-gated oracle hook is wired into retrieve_topk_via_labels
    and (a) is a no-op when off, (b) records without perturbing selection."""

    def setUp(self):
        torch.manual_seed(0)
        self._prev = os.environ.get("SGLANG_DS_RECALL_ORACLE")
        sink_mod.reset_sink_for_testing(None)
        sink_mod.clear_active_trial()
        # Minimal valid physical-mode inputs (bs=1, H=2, head_dim=4, T=8, label_dim=2).
        self.kw = dict(
            queries=torch.randn(1, 2, 4),
            token_signatures=torch.randn(1, 8, 2, 2),
            written=torch.ones(1, 8, dtype=torch.bool),
            channel_selection=torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32),
            channel_weights=torch.ones(1, 2, 2, dtype=torch.float32),
            layer_id=0,
            max_top_k=4,
        )

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("SGLANG_DS_RECALL_ORACLE", None)
        else:
            os.environ["SGLANG_DS_RECALL_ORACLE"] = self._prev
        sink_mod.reset_sink_for_testing(None)
        sink_mod.clear_active_trial()

    def _run(self):
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            retrieve_topk_via_labels,
        )

        return retrieve_topk_via_labels(**self.kw)

    def _baseline_off(self):
        os.environ.pop("SGLANG_DS_RECALL_ORACLE", None)
        sink_mod.reset_sink_for_testing(None)
        sel_off, vl_off = self._run()
        self.assertIsNone(sink_mod.get_sink())
        return sel_off, vl_off

    def test_oracle_off_no_record_and_baseline(self):
        self._baseline_off()

    def test_oracle_on_records_without_perturbing_selection(self):
        sel_off, vl_off = self._baseline_off()

        os.environ["SGLANG_DS_RECALL_ORACLE"] = "1"
        sink_mod.reset_sink_for_testing(None)
        sink_mod.set_active_trial("req-1", 3, [2, 5])
        sel_on, vl_on = self._run()

        # selection is byte-identical with the oracle on (hook does not perturb)
        self.assertTrue(torch.equal(sel_off, sel_on))
        self.assertTrue(torch.equal(vl_off, vl_on))
        # and a keyed record was written for the active trial
        recs = sink_mod.get_sink().records
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r["request_id"], "req-1")
        self.assertEqual(r["trial_id"], 3)
        self.assertEqual(r["layer_id"], 0)
        self.assertEqual(r["needle_span"], [2, 5])
        self.assertIn("needle_worst_rank", r)
        self.assertIn("recall_at_k", r)
        self.assertEqual(r["index_topk"], 4)
        # invariant holds for the wired call (recall@index_topk == selected_contains_needle)
        self.assertTrue(r["recall_at_index_topk_matches_selected"])

    def test_oracle_on_without_active_trial_is_noop(self):
        os.environ["SGLANG_DS_RECALL_ORACLE"] = "1"
        sink_mod.reset_sink_for_testing(None)
        sink_mod.clear_active_trial()  # enabled, but no needle span registered
        self._run()
        sink = sink_mod.get_sink()
        # sink may be lazily created, but nothing recorded without a trial
        self.assertEqual([] if sink is None else sink.records, [])


if __name__ == "__main__":
    unittest.main()
