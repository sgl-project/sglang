import json
import os
import tempfile
import unittest
import uuid

import numpy as np

from sglang.srt.speculative.cpp_ngram.external_corpus import (
    iter_external_corpus_chunks,
)
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=19, suite="stage-a-test-cpu")


def _make_corpus(match_type="BFS", **kwargs):
    external_corpus_documents = kwargs.pop("external_corpus_documents", None)
    defaults = dict(
        max_trie_depth=12,
        min_bfs_breadth=1,
        max_bfs_breadth=8,
        draft_token_num=8,
        capacity=100000,
        external_sam_budget=0,
        external_corpus_max_tokens=10000000,
    )
    defaults.update(kwargs)
    defaults["match_type"] = match_type
    corpus = NgramCorpus(**defaults)
    if external_corpus_documents is not None:
        from sglang.srt.speculative.cpp_ngram.external_corpus import SEPARATOR_TOKEN

        chunks = []
        has_prev = False
        for doc in external_corpus_documents:
            if has_prev:
                chunks.append([SEPARATOR_TOKEN] + list(doc))
            else:
                chunks.append(list(doc))
            has_prev = True
        loaded_token_count = corpus.load_external_corpus_named("test_corpus", chunks)
        corpus.commit_external_corpus_load("test_corpus", loaded_token_count)
    return corpus


def _batch_get(
    corpus: NgramCorpus,
    batch_tokens: list[list[int]],
):
    return corpus.batch_get(
        req_ids=[uuid.uuid4().hex for _ in range(len(batch_tokens))],
        batch_tokens=batch_tokens,
        total_lens=[len(tokens) for tokens in batch_tokens],
    )


def _batch_get_with_state(
    corpus: NgramCorpus,
    req_id: str,
    current_tokens: list[int],
    total_len: int,
):
    return corpus.batch_get([req_id], [current_tokens], [total_len])


class _IntTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [int(piece) for piece in text.split()]


SEED_SEQUENCES = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 2, 3, 44, 55, 66, 77, 88, 99, 100],
]

QUERY_SEQUENCES = [[1, 2, 3], [3, 44], [3, 6, 999]]

EXPECTED_BFS_IDS = [
    [3, 4, 44, 5, 55, 6, 66, 77],
    [44, 55, 66, 77, 88, 99, 100, 0],
    [999, 0, 0, 0, 0, 0, 0, 0],
]

EXPECTED_PROB_IDS = [
    [3, 44, 4, 55, 5, 66, 6, 7],
    [44, 55, 66, 77, 88, 99, 100, 0],
    [999, 0, 0, 0, 0, 0, 0, 0],
]

EXPECTED_BFS_MASKS = [
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1, 1],
    ],
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
    ],
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
    ],
]

EXPECTED_PROB_MASKS = [
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1, 1],
    ],
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
    ],
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
    ],
]


class TestNgramCorpusBFS(CustomTestCase):
    """Golden-output tests for BFS matching mode."""

    @classmethod
    def setUpClass(cls):
        cls.corpus = _make_corpus("BFS")
        cls.corpus.batch_put(SEED_SEQUENCES)
        cls.corpus.synchronize()
        ids, masks = _batch_get(cls.corpus, QUERY_SEQUENCES)
        draft = 8
        cls.ids = ids.reshape(-1, draft)
        cls.masks = masks.reshape(-1, draft, draft)

    def test_token_ids(self):
        np.testing.assert_array_equal(self.ids.tolist(), EXPECTED_BFS_IDS)

    def test_masks(self):
        np.testing.assert_array_equal(self.masks.tolist(), EXPECTED_BFS_MASKS)

    def test_output_shapes(self):
        n_queries = len(QUERY_SEQUENCES)
        draft = 8
        self.assertEqual(self.ids.shape, (n_queries, draft))
        self.assertEqual(self.masks.shape, (n_queries, draft, draft))


class TestNgramCorpusProb(CustomTestCase):
    """Golden-output tests for Prob matching mode."""

    @classmethod
    def setUpClass(cls):
        cls.corpus = _make_corpus("PROB")
        cls.corpus.batch_put(SEED_SEQUENCES)
        cls.corpus.synchronize()
        ids, masks = _batch_get(cls.corpus, QUERY_SEQUENCES)
        cls.ids = ids.reshape(-1, 8)
        cls.masks = masks.reshape(-1, 8, 8)

    def test_token_ids(self):
        np.testing.assert_array_equal(self.ids.tolist(), EXPECTED_PROB_IDS)

    def test_masks(self):
        np.testing.assert_array_equal(self.masks.tolist(), EXPECTED_PROB_MASKS)

    def test_output_shapes(self):
        n_queries = len(QUERY_SEQUENCES)
        self.assertEqual(self.ids.shape, (n_queries, 8))
        self.assertEqual(self.masks.shape, (n_queries, 8, 8))


class TestNgramCorpusReset(CustomTestCase):
    """Verify reset clears all cached state."""

    def test_reset_produces_empty_results(self):
        corpus = _make_corpus("BFS")
        corpus.batch_put(SEED_SEQUENCES)
        corpus.synchronize()

        ids_before, _ = _batch_get(corpus, [[1, 2, 3]])
        self.assertTrue(
            any(t != 0 for t in ids_before.tolist()[1:]),
            "Expected non-trivial draft tokens before reset",
        )

        corpus.reset()

        ids_after, _ = _batch_get(corpus, [[1, 2, 3]])
        self.assertEqual(
            ids_after.tolist(),
            [3, 0, 0, 0, 0, 0, 0, 0],
            "After reset, only last_token should be present (rest zero-padded)",
        )


class TestNgramCorpusNoMatch(CustomTestCase):
    """Verify behavior when query has no match in the corpus."""

    def test_unmatched_query(self):
        corpus = _make_corpus("BFS")
        corpus.batch_put([[10, 20, 30, 40, 50]])
        corpus.synchronize()

        ids, masks = _batch_get(corpus, [[999, 888, 777]])
        ids_list = ids.tolist()
        self.assertEqual(ids_list[0], 777, "First token should be last context token")
        self.assertTrue(
            all(t == 0 for t in ids_list[1:]),
            "No draft tokens expected when nothing matches",
        )

    def test_empty_corpus(self):
        corpus = _make_corpus("BFS")
        ids, masks = _batch_get(corpus, [[1, 2, 3]])
        ids_list = ids.tolist()
        self.assertEqual(ids_list[0], 3)
        self.assertTrue(all(t == 0 for t in ids_list[1:]))


class TestNgramCorpusMultipleInserts(CustomTestCase):
    """Verify that multiple inserts accumulate correctly."""

    def test_incremental_inserts(self):
        corpus = _make_corpus("BFS")
        corpus.batch_put([[1, 2, 3, 4, 5]])
        corpus.synchronize()

        corpus.batch_put([[1, 2, 3, 44, 55]])
        corpus.synchronize()

        ids, _ = _batch_get(corpus, [[1, 2, 3]])
        ids_list = ids.tolist()

        self.assertIn(4, ids_list, "Token 4 from first insert should still match")
        self.assertIn(44, ids_list, "Token 44 from second insert should also match")


class TestNgramCorpusSqueeze(CustomTestCase):
    """Verify cache eviction under memory pressure."""

    def test_small_capacity_does_not_crash(self):
        corpus = _make_corpus("BFS", capacity=200)
        long_seq = list(range(1, 101))
        corpus.batch_put([long_seq])
        corpus.synchronize()

        ids, masks = _batch_get(corpus, [[50, 51, 52]])
        self.assertEqual(len(ids), 8, "Should still produce draft_token_num outputs")

    def test_eviction_preserves_recent(self):
        corpus = _make_corpus("BFS", capacity=500, max_trie_depth=6)

        old_seq = list(range(1000, 1050))
        corpus.batch_put([old_seq])
        corpus.synchronize()

        recent_seq = list(range(2000, 2050))
        corpus.batch_put([recent_seq])
        corpus.synchronize()

        ids, _ = _batch_get(corpus, [[2000, 2001, 2002]])
        ids_list = ids.tolist()
        self.assertEqual(ids_list[0], 2002, "Last context token should be first")
        self.assertIn(2003, ids_list, "Recent sequence should still be matchable")


class TestNgramCorpusLeafPaths(CustomTestCase):
    """Verify the leaf_paths_from_mask utility."""

    def test_simple_tree(self):
        corpus = _make_corpus("BFS")
        tokens = [3, 4, 44, 5, 55]
        mask = [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
        ]
        paths = corpus.leaf_paths_from_mask(tokens, mask)

        for path in paths:
            self.assertIn(3, path, "Root token should be in every path")

        self.assertEqual(len(paths), 2, "Two leaf paths expected for a binary tree")

    def test_single_chain(self):
        corpus = _make_corpus("BFS")
        tokens = [10, 20, 30]
        mask = [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ]
        paths = corpus.leaf_paths_from_mask(tokens, mask)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], [10, 20, 30])


class TestNgramCorpusBatchConsistency(CustomTestCase):
    """Verify batch queries produce same results as individual queries."""

    def test_batch_vs_individual(self):
        corpus = _make_corpus("BFS")
        corpus.batch_put(SEED_SEQUENCES)
        corpus.synchronize()

        batch_ids, batch_masks = _batch_get(corpus, QUERY_SEQUENCES)
        draft = 8
        batch_ids = batch_ids.reshape(-1, draft)
        batch_masks = batch_masks.reshape(-1, draft, draft)

        for i, query in enumerate(QUERY_SEQUENCES):
            single_ids, single_masks = _batch_get(corpus, [query])
            single_ids = single_ids.reshape(-1, draft)
            single_masks = single_masks.reshape(-1, draft, draft)

            np.testing.assert_array_equal(
                batch_ids[i],
                single_ids[0],
                err_msg=f"Token mismatch for query {i}",
            )
            np.testing.assert_array_equal(
                batch_masks[i],
                single_masks[0],
                err_msg=f"Mask mismatch for query {i}",
            )


class TestMaskValidity(CustomTestCase):
    """Verify structural invariants of the output mask for any draft tree."""

    def _check_mask(self, masks_2d):
        n = len(masks_2d)
        for i in range(n):
            self.assertEqual(masks_2d[i][i], 1, f"Diagonal must be 1 at row {i}")
        self.assertEqual(masks_2d[0], [1] + [0] * (n - 1))

    def test_bfs_mask_invariants(self):
        corpus = _make_corpus("BFS")
        corpus.batch_put(SEED_SEQUENCES)
        corpus.synchronize()
        _, masks = _batch_get(corpus, QUERY_SEQUENCES)
        masks = masks.reshape(-1, 8, 8)
        for i in range(masks.shape[0]):
            self._check_mask(masks[i].tolist())

    def test_prob_mask_invariants(self):
        corpus = _make_corpus("PROB")
        corpus.batch_put(SEED_SEQUENCES)
        corpus.synchronize()
        _, masks = _batch_get(corpus, QUERY_SEQUENCES)
        masks = masks.reshape(-1, 8, 8)
        for i in range(masks.shape[0]):
            self._check_mask(masks[i].tolist())


class TestFrequencyBoosting(CustomTestCase):
    """Verify that repeated insertions change Prob-mode selection."""

    def test_repeated_insert_promotes_token(self):
        corpus = _make_corpus(
            "PROB",
            draft_token_num=2,
            max_bfs_breadth=1,
            min_bfs_breadth=1,
            max_trie_depth=5,
        )
        corpus.batch_put([[1, 2, 3, 10, 11]])
        corpus.synchronize()

        for _ in range(10):
            corpus.batch_put([[1, 2, 3, 20, 21]])
        corpus.synchronize()

        ids, _ = _batch_get(corpus, [[1, 2, 3]])
        ids_list = ids.tolist()

        self.assertEqual(
            ids_list[1],
            20,
            f"Token 20 should be selected over 10 after frequency boost, got {ids_list}",
        )


class TestRecencyOrdering(CustomTestCase):
    """Verify that BFS mode respects LRU recency."""

    def test_most_recent_insert_selected(self):
        corpus = _make_corpus(
            "BFS",
            draft_token_num=2,
            max_bfs_breadth=1,
            min_bfs_breadth=1,
            max_trie_depth=5,
        )
        corpus.batch_put([[1, 2, 3, 10, 11]])
        corpus.synchronize()
        corpus.batch_put([[1, 2, 3, 20, 21]])
        corpus.synchronize()

        ids, _ = _batch_get(corpus, [[1, 2, 3]])
        ids_list = ids.tolist()
        self.assertEqual(
            ids_list[1],
            20,
            f"Token 20 (recent) should be selected over 10 (old), got {ids_list}",
        )


class TestOverlappingSuffixes(CustomTestCase):
    """Verify correct matching when sequences share suffixes."""

    def test_shared_suffix_both_match(self):
        corpus = _make_corpus("BFS")
        corpus.batch_put([[100, 200, 7, 8, 9, 50, 51]])
        corpus.batch_put([[300, 400, 7, 8, 9, 60, 61]])
        corpus.synchronize()

        ids, _ = _batch_get(corpus, [[7, 8, 9]])
        ids_list = ids.tolist()
        self.assertIn(50, ids_list, "Continuation from first sequence missing")
        self.assertIn(60, ids_list, "Continuation from second sequence missing")


class TestSingleTokenContext(CustomTestCase):
    """Verify behavior with minimum-length context."""

    def test_single_token_query(self):
        corpus = _make_corpus("BFS")
        corpus.batch_put([[5, 10, 20, 30]])
        corpus.synchronize()

        ids, masks = _batch_get(corpus, [[5]])
        ids_list = ids.tolist()
        self.assertEqual(ids_list[0], 5, "First token should be last context token")
        self.assertIn(10, ids_list, "Should match continuation after single token 5")


class TestLongContext(CustomTestCase):
    """Verify behavior when query context exceeds max_trie_depth."""

    def test_context_longer_than_max_trie_depth(self):
        corpus = _make_corpus("BFS", max_trie_depth=6)
        seq = list(range(1, 20))
        corpus.batch_put([seq])
        corpus.synchronize()

        long_query = list(range(1, 16))
        ids, masks = _batch_get(corpus, [long_query])
        ids_list = ids.tolist()
        self.assertEqual(ids_list[0], 15, "First token should be last context token")
        self.assertIn(16, ids_list, "Should match via suffix despite long context")

    def test_matches_longest_stored_suffix(self):
        corpus = _make_corpus("BFS", max_trie_depth=6, draft_token_num=4)
        corpus.batch_put([[1, 2, 3, 4, 5, 6, 7]])
        corpus.batch_put([[99, 3, 4, 5, 6, 8]])
        corpus.synchronize()

        ids, _ = _batch_get(corpus, [[2, 3, 4, 5, 6]])
        ids_list = ids.tolist()
        self.assertIn(
            7, ids_list, "Longest stored suffix should contribute a continuation"
        )
        self.assertIn(
            8,
            ids_list,
            "Shorter matching suffixes should still contribute continuations",
        )


class TestDraftBudgetSaturation(CustomTestCase):
    """Verify the draft tree uses exactly draft_token_num slots."""

    def test_full_budget_used(self):
        corpus = _make_corpus("BFS", draft_token_num=8)
        seq = list(range(1, 30))
        corpus.batch_put([seq])
        corpus.synchronize()

        ids, _ = _batch_get(corpus, [[1, 2, 3]])
        ids_list = ids.tolist()
        self.assertEqual(len(ids_list), 8)
        non_zero = [t for t in ids_list[1:] if t != 0]
        self.assertGreater(
            len(non_zero),
            0,
            "Draft budget should have non-zero tokens when cache has long chains",
        )


class TestTruncate(CustomTestCase):
    """Verify truncation logic on batch_get output."""

    def test_truncate_reduces_output(self):
        corpus = _make_corpus("BFS", draft_token_num=8)
        corpus.batch_put(SEED_SEQUENCES)
        corpus.synchronize()

        ids, _ = _batch_get(corpus, [[1, 2, 3]])
        ids = ids.reshape(8)
        self.assertEqual(len(ids), 8)

        # Simulate truncate to 4
        trunc_n = 4
        trunc_ids = ids[:trunc_n]
        self.assertEqual(len(trunc_ids), trunc_n)

    def test_truncate_preserves_mask_structure(self):
        corpus = _make_corpus("BFS", draft_token_num=8)
        corpus.batch_put(SEED_SEQUENCES)
        corpus.synchronize()

        _, masks = _batch_get(corpus, [[1, 2, 3]])
        n = 8
        full_mask = masks.reshape(n, n)

        trunc_n = 4
        trunc_mask = full_mask[:trunc_n, :trunc_n]

        for i in range(trunc_n):
            for j in range(trunc_n):
                self.assertEqual(
                    trunc_mask[i, j],
                    full_mask[i, j],
                    f"Mask mismatch at ({i},{j})",
                )


class TestResetAndReinsert(CustomTestCase):
    """Verify that reset followed by new inserts works correctly."""

    def test_reset_then_reinsert(self):
        corpus = _make_corpus("BFS")
        corpus.batch_put([[1, 2, 3, 4, 5]])
        corpus.synchronize()

        corpus.reset()

        corpus.batch_put([[10, 20, 30, 40, 50]])
        corpus.synchronize()

        ids_old, _ = _batch_get(corpus, [[1, 2, 3]])
        ids_old_list = ids_old.tolist()
        self.assertTrue(
            all(t == 0 for t in ids_old_list[1:]),
            f"Old data should not match after reset+reinsert, got {ids_old_list}",
        )

        ids_new, _ = _batch_get(corpus, [[10, 20, 30]])
        ids_new_list = ids_new.tolist()
        self.assertEqual(ids_new_list[0], 30)
        self.assertIn(40, ids_new_list, "New data should match after reset+reinsert")


class TestSqueezeEvictsOld(CustomTestCase):
    """Verify that squeeze actually evicts old data, not just preserves recent."""

    def test_old_data_evicted(self):
        corpus = _make_corpus("BFS", capacity=150, max_trie_depth=6)

        old_seq = list(range(5000, 5030))
        corpus.batch_put([old_seq])
        corpus.synchronize()

        ids_before, _ = _batch_get(corpus, [[5000, 5001, 5002]])
        self.assertIn(
            5003,
            ids_before.tolist(),
            "Old data should match before eviction",
        )

        for i in range(5):
            new_seq = list(range(6000 + i * 30, 6000 + i * 30 + 30))
            corpus.batch_put([new_seq])
            corpus.synchronize()

        ids_after, _ = _batch_get(corpus, [[5000, 5001, 5002]])
        ids_after_list = ids_after.tolist()
        self.assertNotIn(
            5003,
            ids_after_list,
            f"Old data should be evicted after pressure, got {ids_after_list}",
        )


class TestNgramCorpusIncremental(CustomTestCase):
    """Verify the incremental matching path matches the stateless path."""

    def _assert_incremental_matches_stateless(self, match_type: str):
        corpus = _make_corpus(match_type, max_trie_depth=4, draft_token_num=4)
        corpus.batch_put([[1, 2, 3, 4, 5, 6], [9, 3, 4, 7, 8]])
        corpus.synchronize()

        req_id = f"req-{match_type.lower()}"

        steps = [
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5, 6],
        ]
        for full_sequence in steps:
            current_tail = full_sequence[-4:]
            inc_ids, inc_masks = _batch_get_with_state(
                corpus,
                req_id,
                current_tail,
                len(full_sequence),
            )
            full_ids, full_masks = _batch_get(corpus, [current_tail])
            np.testing.assert_array_equal(inc_ids, full_ids)
            np.testing.assert_array_equal(inc_masks, full_masks)

    def test_incremental_matches_stateless_bfs(self):
        self._assert_incremental_matches_stateless("BFS")

    def test_incremental_matches_stateless_prob(self):
        self._assert_incremental_matches_stateless("PROB")

    def test_leaf_anchor_becomes_expandable(self):
        corpus = _make_corpus("BFS", max_trie_depth=4, draft_token_num=4)
        corpus.batch_put([[1, 2, 3]])
        corpus.synchronize()

        req_id = "leaf-anchor"
        ids_before, _ = _batch_get_with_state(corpus, req_id, [2, 3], 2)
        self.assertTrue(
            all(t == 0 for t in ids_before.tolist()[1:]),
            f"Expected only the last token before extension, got {ids_before.tolist()}",
        )

        corpus.batch_put([[9, 2, 3, 4]])
        corpus.synchronize()

        inc_ids, inc_masks = _batch_get_with_state(corpus, req_id, [2, 3], 2)
        full_ids, full_masks = _batch_get(corpus, [[2, 3]])
        np.testing.assert_array_equal(inc_ids, full_ids)
        np.testing.assert_array_equal(inc_masks, full_masks)
        self.assertIn(
            4,
            inc_ids.tolist(),
            f"Expected token 4 after extension, got {inc_ids.tolist()}",
        )

    def test_stale_state_rebuilds_after_eviction(self):
        corpus = _make_corpus("BFS", capacity=150, max_trie_depth=6, draft_token_num=4)
        corpus.batch_put([list(range(5000, 5030))])
        corpus.synchronize()

        req_id = "evicted"
        _batch_get_with_state(corpus, req_id, [5000, 5001, 5002], 3)

        for i in range(5):
            new_seq = list(range(6000 + i * 30, 6000 + i * 30 + 30))
            corpus.batch_put([new_seq])
            corpus.synchronize()

        inc_ids, inc_masks = _batch_get_with_state(
            corpus, req_id, [5000, 5001, 5002], 3
        )
        full_ids, full_masks = _batch_get(corpus, [[5000, 5001, 5002]])
        np.testing.assert_array_equal(inc_ids, full_ids)
        np.testing.assert_array_equal(inc_masks, full_masks)


class TestNgramCorpusExternalSam(CustomTestCase):
    """Verify external SAM loading and fixed-budget composition."""

    def test_external_corpus_iterator_streams_documents(self):
        corpus = _make_corpus(
            "BFS",
            draft_token_num=4,
            external_sam_budget=3,
            external_corpus_max_tokens=8,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps("1 2 3 4 5"))
            f.write("\n")
            f.write(json.dumps("8 9"))
            f.write("\n")
            path = f.name
        self.addCleanup(os.remove, path)

        loaded_token_count = corpus.load_external_corpus_named(
            path,
            iter_external_corpus_chunks(path, _IntTokenizer(), max_tokens=8),
        )
        corpus.commit_external_corpus_load(path, loaded_token_count)
        # 5 doc tokens + 1 separator + 2 doc tokens = 8
        self.assertEqual(loaded_token_count, 8)

        ids, _ = _batch_get(corpus, [[1, 2, 3]])
        ids_list = ids.tolist()
        self.assertEqual(ids_list[0], 3)
        self.assertEqual(ids_list[1:3], [4, 5])

    def test_external_corpus_iterator_rejects_oversized_corpus(self):
        corpus = _make_corpus(
            "BFS",
            external_sam_budget=2,
            external_corpus_max_tokens=4,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps("1 2 3"))
            f.write("\n")
            f.write(json.dumps("4 5"))
            f.write("\n")
            path = f.name
        self.addCleanup(os.remove, path)

        with self.assertRaisesRegex(ValueError, "token limit"):
            corpus.load_external_corpus_named(
                path,
                iter_external_corpus_chunks(path, _IntTokenizer(), max_tokens=4),
            )

    def test_external_sam_only_chain(self):
        corpus = _make_corpus(
            "BFS",
            draft_token_num=4,
            external_sam_budget=3,
            external_corpus_documents=[[1, 2, 3, 4, 5]],
        )

        ids, masks = _batch_get(corpus, [[1, 2, 3]])
        ids_list = ids.tolist()
        self.assertEqual(ids_list[0], 3)
        self.assertEqual(ids_list[1:3], [4, 5])

    def test_external_sam_respects_document_boundaries(self):
        corpus = _make_corpus(
            "BFS",
            draft_token_num=4,
            external_sam_budget=3,
            external_corpus_documents=[[1, 2, 3], [4, 5, 6]],
        )

        ids, _ = _batch_get(corpus, [[2, 3]])
        ids_list = ids.tolist()
        self.assertEqual(ids_list[0], 3)
        self.assertTrue(all(token == 0 for token in ids_list[1:]), ids_list)

    def test_external_sam_adds_distinct_root_branch(self):
        corpus = _make_corpus(
            "BFS",
            draft_token_num=6,
            external_sam_budget=2,
            external_corpus_documents=[[1, 2, 3, 20, 21]],
        )
        corpus.batch_put([[1, 2, 3, 10, 11]])
        corpus.synchronize()

        ids, masks = _batch_get(corpus, [[1, 2, 3]])
        leaf_paths = corpus.leaf_paths_from_mask(
            ids.tolist(), masks.reshape(6, 6).tolist()
        )
        self.assertIn([3, 10, 11], leaf_paths)
        self.assertIn([3, 20, 21], leaf_paths)

    def test_shared_prefix_keeps_both_branches(self):
        corpus = _make_corpus(
            "BFS",
            draft_token_num=5,
            external_sam_budget=2,
            external_corpus_documents=[[1, 2, 3, 10, 99]],
        )
        corpus.batch_put([[1, 2, 3, 10, 11]])
        corpus.synchronize()

        ids, masks = _batch_get(corpus, [[1, 2, 3]])
        leaf_paths = corpus.leaf_paths_from_mask(
            ids.tolist(), masks.reshape(5, 5).tolist()
        )
        self.assertIn([3, 10, 11], leaf_paths)
        self.assertIn([3, 10, 99], leaf_paths)

    def test_shared_prefix_merge_can_underfill_budget(self):
        corpus = _make_corpus(
            "BFS",
            draft_token_num=6,
            external_sam_budget=2,
            external_corpus_documents=[[1, 2, 3, 10, 99]],
        )
        corpus.batch_put([[1, 2, 3, 10, 11]])
        corpus.synchronize()

        ids, masks = _batch_get(corpus, [[1, 2, 3]])
        ids_list = ids.tolist()
        leaf_paths = corpus.leaf_paths_from_mask(ids_list, masks.reshape(6, 6).tolist())
        self.assertIn([3, 10, 11], leaf_paths)
        self.assertIn([3, 10, 99], leaf_paths)
        self.assertEqual(ids_list.count(0), 2, ids_list)

    def test_external_sam_prob_prefers_frequent_continuation(self):
        corpus = _make_corpus(
            "PROB",
            draft_token_num=2,
            min_bfs_breadth=1,
            max_bfs_breadth=1,
            external_sam_budget=1,
            external_corpus_documents=[
                [1, 2, 3, 10],
                [1, 2, 3, 20],
                [1, 2, 3, 20],
                [1, 2, 3, 20],
            ],
        )

        ids, _ = _batch_get(corpus, [[1, 2, 3]])
        self.assertEqual(ids.tolist(), [3, 20])


class TestNgramCorpusMatchBenchmark(CustomTestCase):
    """Benchmark incremental advance vs full rebuild in match()."""

    def test_incremental_faster_than_rebuild(self):
        """Incremental advance (O(D) per token) should be faster than rebuild (O(D^2))."""
        import time

        max_trie_depth = 18
        draft_token_num = 8
        corpus = _make_corpus(
            "BFS",
            max_trie_depth=max_trie_depth,
            draft_token_num=draft_token_num,
            capacity=500000,
        )

        # Seed the trie with diverse sequences so suffix matching is non-trivial.
        seed_data = [list(range(i, i + 50)) for i in range(0, 5000, 50)]
        corpus.batch_put(seed_data)
        corpus.synchronize()

        num_steps = 500
        base_seq = list(range(1, max_trie_depth + 1))

        # --- Incremental path: same req_id, total_len grows by 1 each step ---
        req_id = "bench-incremental"
        # Warm up the state with the initial context.
        _batch_get_with_state(corpus, req_id, base_seq, len(base_seq))

        start = time.perf_counter()
        for step in range(num_steps):
            total_len = len(base_seq) + step + 1
            new_token = (step + max_trie_depth + 1) % 5000
            tail = (base_seq + [new_token])[-max_trie_depth:]
            base_seq = tail
            _batch_get_with_state(corpus, req_id, tail, total_len)
        incremental_us = (time.perf_counter() - start) / num_steps * 1e6

        # --- Rebuild path: unique req_id each call forces fresh state ---
        base_seq = list(range(1, max_trie_depth + 1))
        start = time.perf_counter()
        for step in range(num_steps):
            new_token = (step + max_trie_depth + 1) % 5000
            tail = (base_seq + [new_token])[-max_trie_depth:]
            base_seq = tail
            _batch_get(corpus, [tail])
        rebuild_us = (time.perf_counter() - start) / num_steps * 1e6

        print(
            f"\n  Incremental: {incremental_us:.1f} us/step"
            f"\n  Rebuild:     {rebuild_us:.1f} us/step"
            f"\n  Speedup:     {rebuild_us / incremental_us:.2f}x"
        )

        # The incremental path should be at least as fast; allow a small margin
        # for noise. With D=12 the theoretical speedup is ~12x (D^2/D).
        self.assertLess(
            incremental_us,
            rebuild_us * 1.1,
            f"Incremental ({incremental_us:.1f} us) should not be slower than "
            f"rebuild ({rebuild_us:.1f} us)",
        )


class TestNgramCorpusMultiSam(CustomTestCase):
    """Verify multi-SAM add/remove/list and budget splitting."""

    def test_add_and_list(self):
        corpus = _make_corpus("BFS", draft_token_num=4, external_sam_budget=3)
        loaded_token_count = corpus.load_external_corpus_named("a", [[1, 2, 3, 4, 5]])
        corpus.commit_external_corpus_load("a", loaded_token_count)
        loaded_token_count = corpus.load_external_corpus_named(
            "b", [[10, 20, 30, 40, 50]]
        )
        corpus.commit_external_corpus_load("b", loaded_token_count)
        token_counts = corpus.list_external_corpora()
        self.assertEqual(sorted(token_counts.keys()), ["a", "b"])
        self.assertEqual(token_counts["a"], 5)
        self.assertEqual(token_counts["b"], 5)

    def test_remove(self):
        corpus = _make_corpus("BFS", draft_token_num=4, external_sam_budget=3)
        loaded_token_count = corpus.load_external_corpus_named("a", [[1, 2, 3, 4, 5]])
        corpus.commit_external_corpus_load("a", loaded_token_count)
        loaded_token_count = corpus.load_external_corpus_named(
            "b", [[10, 20, 30, 40, 50]]
        )
        corpus.commit_external_corpus_load("b", loaded_token_count)
        corpus.remove_external_corpus("a")
        self.assertEqual(list(corpus.list_external_corpora().keys()), ["b"])

    def test_remove_nonexistent_is_noop(self):
        corpus = _make_corpus("BFS", draft_token_num=4, external_sam_budget=3)
        corpus.remove_external_corpus("nonexistent")
        self.assertEqual(corpus.list_external_corpora(), {})

    def test_multi_sam_candidates(self):
        corpus = _make_corpus("BFS", draft_token_num=6, external_sam_budget=4)
        loaded_token_count = corpus.load_external_corpus_named("a", [[1, 2, 3, 10, 11]])
        corpus.commit_external_corpus_load("a", loaded_token_count)
        loaded_token_count = corpus.load_external_corpus_named("b", [[1, 2, 3, 20, 21]])
        corpus.commit_external_corpus_load("b", loaded_token_count)

        ids, masks = _batch_get(corpus, [[1, 2, 3]])
        leaf_paths = corpus.leaf_paths_from_mask(
            ids.tolist(), masks.reshape(6, 6).tolist()
        )
        # Both SAMs should contribute candidates
        self.assertIn([3, 10, 11], leaf_paths)
        self.assertIn([3, 20, 21], leaf_paths)

    def test_remove_reduces_candidates(self):
        corpus = _make_corpus("BFS", draft_token_num=6, external_sam_budget=4)
        loaded_token_count = corpus.load_external_corpus_named("a", [[1, 2, 3, 10, 11]])
        corpus.commit_external_corpus_load("a", loaded_token_count)
        loaded_token_count = corpus.load_external_corpus_named("b", [[1, 2, 3, 20, 21]])
        corpus.commit_external_corpus_load("b", loaded_token_count)

        corpus.remove_external_corpus("b")

        ids, masks = _batch_get(corpus, [[1, 2, 3]])
        leaf_paths = corpus.leaf_paths_from_mask(
            ids.tolist(), masks.reshape(6, 6).tolist()
        )
        self.assertIn([3, 10, 11], leaf_paths)
        self.assertNotIn([3, 20, 21], leaf_paths)

    def test_make_corpus_with_documents(self):
        """_make_corpus helper loads documents as a named corpus."""
        corpus = _make_corpus(
            "BFS",
            draft_token_num=4,
            external_sam_budget=3,
            external_corpus_documents=[[1, 2, 3, 4, 5]],
        )
        token_counts = corpus.list_external_corpora()
        self.assertIn("test_corpus", token_counts)

    def test_remove_frees_token_budget(self):
        """Removing a corpus should free its tokens from the total budget."""
        corpus = _make_corpus(
            "BFS",
            draft_token_num=4,
            external_sam_budget=3,
            external_corpus_max_tokens=10,
        )
        loaded_token_count = corpus.load_external_corpus_named("a", [[1, 2, 3, 4, 5]])
        corpus.commit_external_corpus_load("a", loaded_token_count)
        loaded_token_count = corpus.load_external_corpus_named(
            "b", [[10, 20, 30, 40, 50]]
        )
        corpus.commit_external_corpus_load("b", loaded_token_count)
        self.assertEqual(corpus.remaining_token_budget, 0)

        corpus.remove_external_corpus("a")
        self.assertEqual(corpus.remaining_token_budget, 5)

        # Now there's room for a new corpus.
        loaded_token_count = corpus.load_external_corpus_named("c", [[100, 200, 300]])
        corpus.commit_external_corpus_load("c", loaded_token_count)
        self.assertEqual(sorted(corpus.list_external_corpora().keys()), ["b", "c"])

    def test_duplicate_corpus_id_is_rejected(self):
        """Adding a duplicate corpus_id should fail without replacing the original corpus."""
        corpus = _make_corpus(
            "BFS",
            draft_token_num=4,
            external_sam_budget=3,
            external_corpus_max_tokens=10,
        )
        loaded_token_count = corpus.load_external_corpus_named("a", [[1, 2, 3, 4, 5]])
        corpus.commit_external_corpus_load("a", loaded_token_count)
        with self.assertRaisesRegex(ValueError, "already exists"):
            corpus.load_external_corpus_named("a", [[10, 20, 30]])

        self.assertEqual(corpus.remaining_token_budget, 5)
        self.assertEqual(list(corpus.list_external_corpora().keys()), ["a"])

        # The original corpus must still be usable for matching.
        ids, masks = _batch_get(corpus, [[1, 2, 3]])
        leaf_paths = corpus.leaf_paths_from_mask(
            ids.tolist(), masks.reshape(4, 4).tolist()
        )
        self.assertTrue(
            any(4 in path or 5 in path for path in leaf_paths),
            f"Expected tokens from corpus 'a' in {leaf_paths}",
        )

    def test_error_on_load_preserves_existing_corpora(self):
        """A failed load must not wipe previously loaded corpora (staging-only cleanup)."""
        corpus = _make_corpus(
            "BFS",
            draft_token_num=4,
            external_sam_budget=3,
            external_corpus_max_tokens=10,
        )
        loaded_token_count = corpus.load_external_corpus_named("a", [[1, 2, 3, 4, 5]])
        corpus.commit_external_corpus_load("a", loaded_token_count)

        # Force an error by exceeding the budget.
        with self.assertRaises(ValueError):
            corpus.load_external_corpus_named("b", [[10, 20, 30, 40, 50, 60]])

        self.assertEqual(list(corpus.list_external_corpora().keys()), ["a"])
        self.assertEqual(corpus.remaining_token_budget, 5)

        # "a" must still be usable for matching.
        ids, masks = _batch_get(corpus, [[1, 2, 3]])
        leaf_paths = corpus.leaf_paths_from_mask(
            ids.tolist(), masks.reshape(4, 4).tolist()
        )
        # Should still find continuations from corpus "a".
        self.assertTrue(
            any(4 in path or 5 in path for path in leaf_paths),
            f"Expected tokens from corpus 'a' in {leaf_paths}",
        )


class TestMultiSamHttpMock(CustomTestCase):
    """Test HTTP endpoints for multi-SAM management with a mocked backend."""

    @classmethod
    def setUpClass(cls):
        from unittest.mock import AsyncMock, MagicMock

        try:
            from starlette.testclient import TestClient

            from sglang.srt.entrypoints.http_server import app, set_global_state
        except (ImportError, OSError):
            raise unittest.SkipTest(
                "http_server import requires CUDA libraries not available on CPU"
            )
        from sglang.srt.managers.io_struct import (
            AddExternalCorpusReqOutput,
            ListExternalCorporaReqOutput,
            RemoveExternalCorpusReqOutput,
        )

        mock_state = MagicMock()
        tm = mock_state.tokenizer_manager

        # Wire up async methods that the HTTP handlers call
        tm.add_external_corpus = AsyncMock(
            return_value=AddExternalCorpusReqOutput(
                success=True,
                corpus_id="test-id",
                message="Loaded corpus 'test-id' with 100 tokens.",
                loaded_token_count=100,
            )
        )
        tm.remove_external_corpus = AsyncMock(
            return_value=RemoveExternalCorpusReqOutput(
                success=True, message="Removed corpus 'test-id'."
            )
        )
        tm.list_external_corpora = AsyncMock(
            return_value=ListExternalCorporaReqOutput(
                success=True, corpus_token_counts={"a": 100, "b": 200}
            )
        )
        set_global_state(mock_state)
        cls.client = TestClient(app)
        cls.mock_tm = tm

    def test_add_corpus(self):
        resp = self.client.post(
            "/add_external_corpus",
            json={"corpus_id": "my-corpus", "documents": ["hello world"]},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["corpus_id"], "test-id")
        self.assertEqual(data["loaded_token_count"], 100)

    def test_add_corpus_auto_id(self):
        resp = self.client.post(
            "/add_external_corpus",
            json={"documents": ["hello world"]},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_remove_corpus(self):
        resp = self.client.post(
            "/remove_external_corpus",
            json={"corpus_id": "test-id"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_remove_corpus_missing_id(self):
        resp = self.client.post(
            "/remove_external_corpus",
            json={},
        )
        self.assertEqual(resp.status_code, 400)

    def test_list_corpora(self):
        resp = self.client.get("/list_external_corpora")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["corpus_token_counts"], {"a": 100, "b": 200})


if __name__ == "__main__":
    unittest.main(verbosity=3)
