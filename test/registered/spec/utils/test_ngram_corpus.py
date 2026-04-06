import unittest
import uuid

import numpy as np

from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, suite="stage-b-test-1-gpu-small")


def _make_corpus(match_type="BFS", **kwargs):
    defaults = dict(
        max_trie_depth=12,
        min_bfs_breadth=1,
        max_bfs_breadth=8,
        draft_token_num=8,
        capacity=100000,
    )
    defaults.update(kwargs)
    defaults["match_type"] = match_type
    return NgramCorpus(**defaults)


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


if __name__ == "__main__":
    unittest.main(verbosity=3)
