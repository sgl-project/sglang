"""Unit tests for the multi-position (setwise) result grouping on
``TokenizerManagerScoreMixin._process_single_item_scoring_results`` and
``_process_multi_item_extraction_results``.

Setwise scoring is expressed as SequenceClassification scoring with a
``score_extraction_token_id``: the classification head is pooled at every
occurrence of the token, so each item carries a 2-D
``[num_positions x num_labels]`` embedding. Each item's matrix is kept grouped
(``per_item_matrix=True``) so ``scores`` is nested ``[num_items][Nᵢ x num_labels]``
(one score matrix per item). These tests cover that grouping and confirm the
default pointwise behavior is unchanged.

All tests run on CPU — no GPU or tokenizer/engine state required.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.tokenizer_manager_score_mixin import (
    ScoreResult,
    TokenizerManagerScoreMixin,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Config that satisfies _validate_score_extraction (dedicated setwise endpoint).
_VALID_SETWISE_CONFIG = dict(
    disable_radix_cache=True,
    chunked_prefill_size=-1,
    allow_auto_truncate=False,
)


class _ClsHarness(TokenizerManagerScoreMixin):
    """Bare SequenceClassification harness for the pure result parser."""

    is_generation = False

    def __init__(self, architecture="Qwen3ForSequenceClassification"):
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture])
        )


class TestSingleItemScoringResults(CustomTestCase):
    @staticmethod
    def _result(embedding, phs=None, prompt_tokens=7):
        r = {
            "meta_info": {"id": "rid-1", "prompt_tokens": prompt_tokens},
            "embedding": embedding,
        }
        if phs is not None:
            r["pooled_hidden_state"] = phs
        return r

    def setUp(self):
        super().setUp()
        self.h = _ClsHarness()
        # _validate_score_extraction reads resolved config bags (get_memory /
        # get_schedule / get_serving), so publish a valid setwise config; each
        # validation test re-publishes with the one field it exercises flipped.
        self._saved_server_args = get_context()._server_args
        self.addCleanup(self._restore_server_args)
        self._publish(**_VALID_SETWISE_CONFIG)

    def _restore_server_args(self):
        if self._saved_server_args is not None:
            get_context().set_server_args(self._saved_server_args)

    def _publish(self, **fields):
        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)

    # ---- default (per-item) CLS behavior -------------------

    def test_pointwise_one_row_per_item(self):
        results = [
            self._result([0.1, 0.2]),
            self._result([0.3, 0.4]),
        ]
        res = self.h._process_single_item_scoring_results(
            results, label_token_ids=None, apply_softmax=False
        )
        self.assertIsInstance(res, ScoreResult)
        self.assertEqual(res.scores, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(res.prompt_tokens, 14)

    def test_mis_pooled_hidden_states_are_list_of_item_tensors(self):
        # Standard MIS returns one fused result with a row at each delimiter.
        # The first row is the query boundary and is discarded; the public
        # ScoreResult contract is one CPU tensor per remaining item.
        phs = torch.arange(12).reshape(3, 4)
        result = self._result([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], phs=phs)
        res = self.h._process_multi_item_scoring_results(
            [result],
            items=["item0", "item1"],
            label_token_ids=None,
            apply_softmax=False,
            return_pooled_hidden_states=True,
        )

        self.assertEqual(res.scores, [[1.0, 0.0], [0.0, 1.0]])
        self.assertIsInstance(res.pooled_hidden_states, list)
        self.assertEqual(len(res.pooled_hidden_states), 2)
        torch.testing.assert_close(res.pooled_hidden_states[0], phs[1])
        torch.testing.assert_close(res.pooled_hidden_states[1], phs[2])

    # ---- setwise: per_item_matrix=True ----------------------------------

    def test_setwise_one_matrix_per_item(self):
        # One item (single big candidate block), 3 extraction positions -> one
        # [3 x num_labels] matrix, nested under the item.
        emb = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        res = self.h._process_single_item_scoring_results(
            [self._result(emb)],
            label_token_ids=None,
            apply_softmax=False,
            per_item_matrix=True,
        )
        self.assertEqual(res.scores, [emb])  # [num_items][N x num_labels]
        self.assertEqual(len(res.scores), 1)
        self.assertEqual(len(res.scores[0]), 3)

    def test_setwise_apply_softmax_over_labels(self):
        emb = [[0.0, 0.0], [2.0, 2.0]]
        res = self.h._process_single_item_scoring_results(
            [self._result(emb)],
            label_token_ids=None,
            apply_softmax=True,
            per_item_matrix=True,
        )
        for row in res.scores[0]:
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            self.assertAlmostEqual(row[0], 0.5, places=5)

    def test_setwise_multiple_items_grouped_per_item(self):
        # Two items (two candidate sets), scored as independent sequences ->
        # one matrix per item (nested), not flattened across items.
        res = self.h._process_single_item_scoring_results(
            [self._result([[1.0, 0.0], [0.0, 1.0]]), self._result([[0.5, 0.5]])],
            label_token_ids=None,
            apply_softmax=False,
            per_item_matrix=True,
        )
        self.assertEqual(res.scores, [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5]]])

    def test_setwise_embedding_as_tensor(self):
        emb = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        res = self.h._process_single_item_scoring_results(
            [self._result(emb)],
            label_token_ids=None,
            apply_softmax=False,
            per_item_matrix=True,
        )
        self.assertEqual(len(res.scores), 1)
        self.assertEqual(len(res.scores[0]), 2)
        self.assertEqual(len(res.scores[0][0]), 2)

    def test_setwise_groups_pooled_hidden_states_per_item(self):
        emb = [[0.1, 0.2], [0.3, 0.4]]
        phs = torch.randn(2, 4)  # one pre-head vector per position
        res = self.h._process_single_item_scoring_results(
            [self._result(emb, phs=phs)],
            label_token_ids=None,
            apply_softmax=False,
            return_pooled_hidden_states=True,
            per_item_matrix=True,
        )
        self.assertIsNotNone(res.pooled_hidden_states)
        # One item -> one matrix of 2 vectors.
        self.assertEqual(len(res.pooled_hidden_states), 1)
        self.assertEqual(len(res.pooled_hidden_states[0]), 2)
        for v in res.pooled_hidden_states[0]:
            self.assertEqual(tuple(v.shape), (4,))

    def test_setwise_no_phs_when_not_requested(self):
        emb = [[0.1, 0.2]]
        phs = torch.randn(1, 4)
        res = self.h._process_single_item_scoring_results(
            [self._result(emb, phs=phs)],
            label_token_ids=None,
            apply_softmax=False,
            return_pooled_hidden_states=False,
            per_item_matrix=True,
        )
        self.assertIsNone(res.pooled_hidden_states)

    def test_setwise_rejects_non_matrix_scores(self):
        with self.assertRaisesRegex(ValueError, "expected a 2-D"):
            self.h._process_single_item_scoring_results(
                [self._result([0.1, 0.2])],
                label_token_ids=None,
                apply_softmax=False,
                per_item_matrix=True,
            )

    def test_setwise_rejects_misaligned_pooled_hidden_states(self):
        with self.assertRaisesRegex(ValueError, "one row per score position"):
            self.h._process_single_item_scoring_results(
                [self._result([[0.1], [0.2]], phs=torch.randn(1, 4))],
                label_token_ids=None,
                apply_softmax=False,
                return_pooled_hidden_states=True,
                per_item_matrix=True,
            )

    def test_validation_rejects_radix_cache(self):
        self._publish(disable_radix_cache=False, chunked_prefill_size=-1)
        with self.assertRaisesRegex(ValueError, "--disable-radix-cache"):
            self.h._validate_score_extraction(False, False)

    def test_validation_rejects_chunked_prefill(self):
        self._publish(disable_radix_cache=True, chunked_prefill_size=2048)
        with self.assertRaisesRegex(ValueError, "--chunked-prefill-size -1"):
            self.h._validate_score_extraction(False, False)

    def test_validation_rejects_cross_encoder(self):
        harness = _ClsHarness(architecture="BertForSequenceClassification")
        with self.assertRaisesRegex(ValueError, "cross-encoder"):
            harness._validate_score_extraction(False, False)

    def test_validation_rejects_reward_model(self):
        # Reward models (last-token pooling, not score_and_pool) ignore the readout
        # positions -> rejected up front, not after a wasted forward.
        harness = _ClsHarness(architecture="InternLM2ForRewardModel")
        with self.assertRaisesRegex(ValueError, "pool the head per position"):
            harness._validate_score_extraction(False, False)

    def test_validation_rejects_embedding_model(self):
        # A base embedding model likewise does not route through score_and_pool.
        harness = _ClsHarness(architecture="Qwen3Model")
        with self.assertRaisesRegex(ValueError, "pool the head per position"):
            harness._validate_score_extraction(False, False)

    def test_validation_rejects_auto_truncate(self):
        self._publish(**{**_VALID_SETWISE_CONFIG, "allow_auto_truncate": True})
        with self.assertRaisesRegex(ValueError, "--allow-auto-truncate"):
            self.h._validate_score_extraction(False, False)

    def test_validation_allows_multi_item(self):
        # Multiple items (candidate sets) are allowed with or without --enable-mis;
        # both return one score matrix per item.
        self.h._validate_score_extraction(False, False)

    def test_anchor_counts_per_item_buckets_by_delimiter(self):
        # Fused: q <d0> item0(2 anchors) <d1> item1(1 anchor) <d2>.
        # delimiter_indices point at the delimiter tokens.
        delimiter_indices = [3, 8, 12]
        anchor_positions = [5, 7, 10]  # 5,7 in item0 (3<p<8); 10 in item1 (8<p<12)
        counts = self.h._anchor_counts_per_item(delimiter_indices, anchor_positions)
        self.assertEqual(counts, [2, 1])

    def test_anchor_counts_per_item_rejects_item_without_anchor(self):
        delimiter_indices = [3, 8, 12]
        anchor_positions = [5, 7]  # nothing in item1
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.h._anchor_counts_per_item(delimiter_indices, anchor_positions)

    def test_anchor_counts_per_item_rejects_anchor_outside_items(self):
        # An anchor before the first delimiter (in the shared query prefix) is not
        # attributable to any item; reject up front so the fused forward doesn't
        # produce more rows than sum(counts) and crash post-inference.
        delimiter_indices = [3, 8, 12]
        anchor_positions = [1, 5, 7, 10]  # position 1 is in the query region
        with self.assertRaisesRegex(ValueError, "outside the candidate items"):
            self.h._anchor_counts_per_item(delimiter_indices, anchor_positions)

    def test_multi_item_extraction_groups_scores_per_item(self):
        # Fused request returns one flat [ΣNᵢ, num_labels] embedding; split back
        # into one matrix per item using the per-item anchor counts.
        res = self.h._process_multi_item_extraction_results(
            [self._result([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])],
            per_item_anchor_counts=[2, 1],
            apply_softmax=False,
        )
        self.assertEqual(res.scores, [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5]]])
        self.assertEqual(res.prompt_tokens, 7)

    def test_multi_item_extraction_softmax_over_labels(self):
        res = self.h._process_multi_item_extraction_results(
            [self._result([[0.0, 0.0], [2.0, 2.0]])],
            per_item_anchor_counts=[1, 1],
            apply_softmax=True,
        )
        for item in res.scores:
            for row in item:
                self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_multi_item_extraction_groups_pooled_hidden_states(self):
        res = self.h._process_multi_item_extraction_results(
            [self._result([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], phs=torch.randn(3, 4))],
            per_item_anchor_counts=[2, 1],
            apply_softmax=False,
            return_pooled_hidden_states=True,
        )
        self.assertIsNotNone(res.pooled_hidden_states)
        self.assertEqual(len(res.pooled_hidden_states), 2)
        self.assertEqual(len(res.pooled_hidden_states[0]), 2)
        self.assertEqual(len(res.pooled_hidden_states[1]), 1)
        self.assertEqual(tuple(res.pooled_hidden_states[0][0].shape), (4,))

    def test_multi_item_extraction_rejects_row_count_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "anchor rows"):
            self.h._process_multi_item_extraction_results(
                [self._result([[1.0, 0.0], [0.0, 1.0]])],
                per_item_anchor_counts=[2, 1],
                apply_softmax=False,
            )

    def test_text_extraction_tokenizes_combined_prompt_once(self):
        class _RecordingTokenizer:
            def __init__(self):
                self.inputs = []

            def encode(self, text):
                self.inputs.append(text)
                return [len(text)]

        self.h.tokenizer = _RecordingTokenizer()
        # anchor id 999 never appears (encode returns [len(text)]), so the
        # query-prefix guard passes; the query is scanned once and each item is
        # combined-encoded once.
        input_ids = self.h._build_score_extraction_text_inputs("query", ["a", "b"], 999)
        self.assertEqual(self.h.tokenizer.inputs, ["query", "querya", "queryb"])
        self.assertEqual(input_ids, [[6], [6]])

    def test_text_extraction_rejects_extraction_token_in_query(self):
        # The query prefix must not contain the extraction token: a query-side
        # anchor would emit an extra, misaligned score row. anchor id 5 ==
        # len("query"), so encode("query") -> [5] contains it and must reject.
        class _LenTokenizer:
            def encode(self, text):
                return [len(text)]

        self.h.tokenizer = _LenTokenizer()
        with self.assertRaisesRegex(ValueError, "query prefix"):
            self.h._build_score_extraction_text_inputs("query", ["a"], 5)

    def test_reject_query_prefix_anchor_rejects_pretokenized_query(self):
        # Pre-tokenized (token-id) query whose ids contain the extraction token
        # must be rejected too: the non-MIS path concatenates query + item and
        # scans the whole sequence, so a query-side anchor would emit an extra,
        # misaligned score row.
        with self.assertRaisesRegex(ValueError, "query prefix"):
            self.h._reject_query_prefix_anchor([1, 7, 2], score_extraction_token_id=7)

    def test_reject_query_prefix_anchor_allows_clean_query(self):
        # A query prefix without the extraction token passes.
        self.h._reject_query_prefix_anchor([1, 2, 3], score_extraction_token_id=7)

    def test_resolve_indices_returns_positions_per_sequence(self):
        # anchor id = 7; positions are per-sequence, in order.
        indices = self.h._resolve_score_extraction_indices(
            [[1, 7, 2, 7], [7, 3]], score_extraction_token_id=7
        )
        self.assertEqual(indices, [[1, 3], [0]])

    def test_resolve_indices_rejects_any_sequence_without_anchor(self):
        # Second sequence has no anchor -> must fail (not silently drop its rows),
        # even though another sequence does contain the anchor.
        with self.assertRaisesRegex(ValueError, "one or more"):
            self.h._resolve_score_extraction_indices(
                [[1, 7, 2], [3, 4]], score_extraction_token_id=7
            )

    # ---- _resolve_multi_position_pooling (extraction-token pooling) -------

    def test_resolve_multi_position_pooling_batched_returns_no_counts(self):
        # No MIS: one sequence per item; anchors resolved per sequence and no
        # per-item anchor counts (the batched path splits by result, not counts).
        token_indices, counts = self.h._resolve_multi_position_pooling(
            [[1, 7, 2, 7], [7, 3]],
            score_extraction_token_id=7,
            use_multi_item_scoring=False,
            delimiter_indices=None,
        )
        self.assertEqual(token_indices, [[1, 3], [0]])
        self.assertIsNone(counts)

    def test_resolve_multi_position_pooling_mis_buckets_anchors_by_item(self):
        # MIS: items fused into one sequence with delimiters; anchors are scanned
        # over the fused sequence and bucketed back to items via the delimiters.
        # Fused (D=0, anchor=7): q q <D> 7 9 7 <D> 9 7 <D>
        #   positions:           0 1  2  3 4 5  6  7 8  9
        # delimiters at 2,6,9; anchors at 3,5,8 -> item0 has 2, item1 has 1.
        fused = [1, 1, 0, 7, 9, 7, 0, 9, 7, 0]
        token_indices, counts = self.h._resolve_multi_position_pooling(
            [fused],
            score_extraction_token_id=7,
            use_multi_item_scoring=True,
            delimiter_indices=[2, 6, 9],
        )
        self.assertEqual(token_indices, [[3, 5, 8]])
        self.assertEqual(counts, [2, 1])

    def test_resolve_multi_position_pooling_mis_rejects_item_without_anchor(self):
        # An item with no anchor inside its fused block is rejected.
        # Fused (D=0, anchor=7): q <D> 7 <D> 9 <D> -> item1 (positions 3<p<5) empty.
        fused = [1, 0, 7, 0, 9, 0]
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.h._resolve_multi_position_pooling(
                [fused],
                score_extraction_token_id=7,
                use_multi_item_scoring=True,
                delimiter_indices=[1, 3, 5],
            )

    def test_resolve_multi_position_pooling_mis_rejects_query_side_anchor(self):
        # The extraction token (7) also appears in the query prefix (before the
        # first delimiter) -> rejected before inference (clean 400), not after the
        # forward pass.
        # Fused (D=0, anchor=7): q 7 <D> 7 <D> 7 <D>  (query anchor at position 1)
        #   positions:           0 1  2  3  4  5  6
        # delimiters at 2,4,6; in-item anchors at 3,5 but the scan also finds 1.
        fused = [1, 7, 0, 7, 0, 7, 0]
        with self.assertRaisesRegex(ValueError, "outside the candidate items"):
            self.h._resolve_multi_position_pooling(
                [fused],
                score_extraction_token_id=7,
                use_multi_item_scoring=True,
                delimiter_indices=[2, 4, 6],
            )

    # ---- _multi_position_score_rows / _multi_position_phs_matrix helpers --

    def test_multi_position_score_rows_preserves_list_values_without_softmax(self):
        # No softmax: the original list is returned as-is (no float round-trip),
        # so exact values are preserved.
        emb = [[0.1, 0.2], [0.3, 0.4]]
        rows = self.h._multi_position_score_rows(emb, apply_softmax=False)
        self.assertIs(rows, emb)

    def test_multi_position_score_rows_softmax_over_labels(self):
        rows = self.h._multi_position_score_rows(
            [[0.0, 0.0], [2.0, 2.0]], apply_softmax=True
        )
        for row in rows:
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            self.assertAlmostEqual(row[0], 0.5, places=5)

    def test_multi_position_score_rows_accepts_tensor(self):
        rows = self.h._multi_position_score_rows(
            torch.tensor([[0.1, 0.2], [0.3, 0.4]]), apply_softmax=False
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 2)

    def test_multi_position_score_rows_rejects_non_matrix(self):
        with self.assertRaisesRegex(ValueError, "expected a 2-D"):
            self.h._multi_position_score_rows([0.1, 0.2], apply_softmax=False)

    def test_multi_position_phs_matrix_returns_tensor(self):
        out = self.h._multi_position_phs_matrix(torch.randn(3, 4), expected_rows=3)
        self.assertEqual(tuple(out.shape), (3, 4))

    def test_multi_position_phs_matrix_rejects_row_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "one row per score position"):
            self.h._multi_position_phs_matrix(torch.randn(2, 4), expected_rows=3)

    def test_multi_position_phs_matrix_rejects_non_matrix(self):
        with self.assertRaisesRegex(ValueError, "one row per score position"):
            self.h._multi_position_phs_matrix(torch.randn(4), expected_rows=4)


class TestSetwisePooledHiddenStatesRoundTrip(CustomTestCase):
    """Validate per-position pooled hidden states survive scheduler serialization.

    ``_process_single_item_scoring_results`` assumes each result carries one
    ``[num_positions, hidden]`` matrix, but the tensor is produced by the pooler,
    packed by ``stream_output_embedding``, and reconstructed on the receiver side.
    These tests exercise that real transmission path (not a reimplementation):
    ``stream_output_embedding`` stacks the per-request 2-D tensors when their
    shapes match and keeps a list when they differ; the receiver recovers each
    request's matrix by index; the parser then groups it as one item's list of
    per-position vectors.
    """

    @staticmethod
    def _run_stream_output_embedding(phs_per_req):
        # Imported lazily so the heavier scheduler module is only pulled in when
        # this test actually runs.
        from sglang.srt.managers.scheduler_components.output_streamer import (
            SchedulerOutputStreamer,
        )

        captured = []
        reqs = []
        for idx, phs in enumerate(phs_per_req):
            reqs.append(
                SimpleNamespace(
                    finished=lambda: True,
                    rid=f"rid-{idx}",
                    http_worker_ipc=None,
                    finished_reason=SimpleNamespace(to_json=lambda: {"type": "stop"}),
                    embedding=[[0.0, 0.0] for _ in range(phs.shape[0])],
                    origin_input_ids=[1, 2, 3],
                    cached_tokens=0,
                    time_stats=SimpleNamespace(),
                    retraction_count=0,
                    pooled_hidden_state=phs,
                )
            )

        fake_self = SimpleNamespace(
            get_cached_tokens_details=lambda req: None,
            send_to_detokenizer=SimpleNamespace(
                send_output=lambda obj: captured.append(obj)
            ),
        )
        SchedulerOutputStreamer._stream_output_embedding(fake_self, reqs)
        out = captured[0]

        # Mirror the receiver-side reconstruction in tokenizer_manager.py: the
        # streamer stacks uniform per-request tensors into a single
        # ``[stacked(N, ...)]`` (len 1, N > 1); the receiver unwraps that back to
        # per-request indexing. Apply the same disambiguation so the returned
        # ``pooled_hidden_states`` indexes per request regardless of wire format.
        phs = out.pooled_hidden_states
        if phs is not None and len(phs) == 1 and len(reqs) > 1:
            out.pooled_hidden_states = phs[0]
        return out

    def test_uniform_anchor_counts_stack_and_reconstruct(self):
        # Two requests, each with 2 anchors -> stackable; receiver [i] must recover
        # each request's [num_positions, hidden] matrix intact.
        phs0 = torch.arange(8).float().reshape(2, 4)
        phs1 = (torch.arange(8).float() + 100).reshape(2, 4)
        out = self._run_stream_output_embedding([phs0, phs1])
        torch.testing.assert_close(torch.as_tensor(out.pooled_hidden_states[0]), phs0)
        torch.testing.assert_close(torch.as_tensor(out.pooled_hidden_states[1]), phs1)

    def test_ragged_anchor_counts_keep_list(self):
        # Different anchor counts cannot be stacked; the packer must fall back to a
        # list so per-request indexing stays correct.
        phs0 = torch.randn(3, 4)
        phs1 = torch.randn(1, 4)
        out = self._run_stream_output_embedding([phs0, phs1])
        self.assertEqual(torch.as_tensor(out.pooled_hidden_states[0]).shape, (3, 4))
        self.assertEqual(torch.as_tensor(out.pooled_hidden_states[1]).shape, (1, 4))
        torch.testing.assert_close(torch.as_tensor(out.pooled_hidden_states[0]), phs0)
        torch.testing.assert_close(torch.as_tensor(out.pooled_hidden_states[1]), phs1)

    def test_full_roundtrip_into_parser_groups_per_item(self):
        # pooler-style 2-D PHS -> stream_output_embedding -> receiver [i] -> parser:
        # the final ScoreResult must hold the item's [num_positions, hidden]
        # vectors grouped as one item (nested), in order.
        phs = torch.arange(6).float().reshape(3, 2)  # 3 positions, hidden=2
        out = self._run_stream_output_embedding([phs])
        received = out.pooled_hidden_states[0]  # receiver-side indexing

        parser = _ClsHarness()
        result = {
            "meta_info": {"id": "rid-0", "prompt_tokens": 5},
            "embedding": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "pooled_hidden_state": received,
        }
        res = parser._process_single_item_scoring_results(
            [result],
            label_token_ids=None,
            apply_softmax=False,
            return_pooled_hidden_states=True,
            per_item_matrix=True,
        )
        self.assertIsNotNone(res.pooled_hidden_states)
        # One item -> one group of 3 position vectors.
        self.assertEqual(len(res.pooled_hidden_states), 1)
        self.assertEqual(len(res.pooled_hidden_states[0]), 3)
        for i in range(3):
            torch.testing.assert_close(res.pooled_hidden_states[0][i], phs[i])


if __name__ == "__main__":
    unittest.main()
