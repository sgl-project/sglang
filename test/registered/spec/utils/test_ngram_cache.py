import unittest

import numpy as np

from sglang.srt.speculative.cpp_ngram.ngram_cache import NgramCache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-small-1-gpu")


def _make_cache(match_type="BFS", cache_type="trie", **kwargs):
    defaults = dict(
        branch_length=12,
        min_match_window_size=1,
        max_match_window_size=10,
        min_bfs_breadth=1,
        max_bfs_breadth=8,
        draft_token_num=8,
        capacity=100000,
    )
    defaults.update(kwargs)
    defaults["match_type"] = match_type
    defaults["cache_type"] = cache_type
    return NgramCache(**defaults)


def _run_cache(cache_type, match_type, insert_batches, query_sequences, **kwargs):
    cache = _make_cache(match_type, cache_type=cache_type, **kwargs)
    for batch in insert_batches:
        cache.batch_put(batch)
        cache.synchronize()
    ids, masks = cache.batch_get(query_sequences)
    return cache, ids, masks


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


CACHE_TYPES = ("trie", "sam")


class NgramCacheTestBase(CustomTestCase):
    def _iter_cache_types(self):
        return CACHE_TYPES

    def _assert_golden_outputs(
        self, match_type, expected_ids, expected_masks, **kwargs
    ):
        for cache_type in self._iter_cache_types():
            with self.subTest(cache_type=cache_type, match_type=match_type):
                _, ids, masks = _run_cache(
                    cache_type, match_type, [SEED_SEQUENCES], QUERY_SEQUENCES, **kwargs
                )
                ids = ids.reshape(-1, 8)
                masks = masks.reshape(-1, 8, 8)
                np.testing.assert_array_equal(ids.tolist(), expected_ids)
                np.testing.assert_array_equal(masks.tolist(), expected_masks)
                self.assertEqual(ids.shape, (len(QUERY_SEQUENCES), 8))
                self.assertEqual(masks.shape, (len(QUERY_SEQUENCES), 8, 8))


class TestNgramCacheGoldenOutputs(NgramCacheTestBase):
    def test_bfs_golden_outputs(self):
        self._assert_golden_outputs("BFS", EXPECTED_BFS_IDS, EXPECTED_BFS_MASKS)

    def test_prob_golden_outputs(self):
        self._assert_golden_outputs("PROB", EXPECTED_PROB_IDS, EXPECTED_PROB_MASKS)


class TestNgramCacheReset(NgramCacheTestBase):
    """Verify reset clears all cached state."""

    def test_reset_produces_empty_results(self):
        for cache_type in self._iter_cache_types():
            for match_type in ("BFS", "PROB"):
                with self.subTest(cache_type=cache_type, match_type=match_type):
                    cache = _make_cache(match_type, cache_type=cache_type)
                    cache.batch_put(SEED_SEQUENCES)
                    cache.synchronize()

                    ids_before, _ = cache.batch_get([[1, 2, 3]])
                    self.assertTrue(
                        any(t != 0 for t in ids_before.tolist()[1:]),
                        "Expected non-trivial draft tokens before reset",
                    )

                    cache.reset()

                    ids_after, _ = cache.batch_get([[1, 2, 3]])
                    self.assertEqual(
                        ids_after.tolist(),
                        [3, 0, 0, 0, 0, 0, 0, 0],
                        "After reset, only last_token should be present (rest zero-padded)",
                    )


class TestNgramCacheNoMatch(NgramCacheTestBase):
    """Verify behavior when query has no match in the cache."""

    def test_unmatched_query(self):
        for cache_type in self._iter_cache_types():
            for match_type in ("BFS", "PROB"):
                with self.subTest(cache_type=cache_type, match_type=match_type):
                    cache = _make_cache(match_type, cache_type=cache_type)
                    cache.batch_put([[10, 20, 30, 40, 50]])
                    cache.synchronize()

                    ids, _ = cache.batch_get([[999, 888, 777]])
                    ids_list = ids.tolist()
                    self.assertEqual(
                        ids_list[0], 777, "First token should be last context token"
                    )
                    self.assertTrue(
                        all(t == 0 for t in ids_list[1:]),
                        "No draft tokens expected when nothing matches",
                    )

    def test_empty_cache(self):
        for cache_type in self._iter_cache_types():
            for match_type in ("BFS", "PROB"):
                with self.subTest(cache_type=cache_type, match_type=match_type):
                    cache = _make_cache(match_type, cache_type=cache_type)
                    ids, _ = cache.batch_get([[1, 2, 3]])
                    ids_list = ids.tolist()
                    self.assertEqual(ids_list[0], 3)
                    self.assertTrue(all(t == 0 for t in ids_list[1:]))


class TestNgramCacheMultipleInserts(NgramCacheTestBase):
    """Verify that multiple inserts accumulate correctly."""

    def test_incremental_inserts(self):
        for cache_type in self._iter_cache_types():
            for match_type in ("BFS", "PROB"):
                with self.subTest(cache_type=cache_type, match_type=match_type):
                    cache = _make_cache(match_type, cache_type=cache_type)
                    cache.batch_put([[1, 2, 3, 4, 5]])
                    cache.synchronize()

                    cache.batch_put([[1, 2, 3, 44, 55]])
                    cache.synchronize()

                    ids, _ = cache.batch_get([[1, 2, 3]])
                    ids_list = ids.tolist()

                    self.assertIn(4, ids_list, "Token 4 from first insert should still match")
                    self.assertIn(44, ids_list, "Token 44 from second insert should also match")


class TestNgramCacheSqueeze(NgramCacheTestBase):
    """Verify cache eviction under memory pressure."""

    def test_small_capacity_does_not_crash(self):
        for cache_type in self._iter_cache_types():
            with self.subTest(cache_type=cache_type):
                cache = _make_cache("BFS", cache_type=cache_type, capacity=200)
                long_seq = list(range(1, 101))
                cache.batch_put([long_seq])
                cache.synchronize()

                ids, _ = cache.batch_get([[50, 51, 52]])
                self.assertEqual(
                    len(ids), 8, "Should still produce draft_token_num outputs"
                )

    def test_eviction_preserves_recent(self):
        for cache_type in self._iter_cache_types():
            with self.subTest(cache_type=cache_type):
                cache = _make_cache(
                    "BFS",
                    cache_type=cache_type,
                    capacity=500,
                    branch_length=6,
                    max_match_window_size=5,
                )

                old_seq = list(range(1000, 1050))
                cache.batch_put([old_seq])
                cache.synchronize()

                recent_seq = list(range(2000, 2050))
                cache.batch_put([recent_seq])
                cache.synchronize()

                ids, _ = cache.batch_get([[2000, 2001, 2002]])
                ids_list = ids.tolist()
                self.assertEqual(
                    ids_list[0], 2002, "Last context token should be first"
                )
                self.assertIn(2003, ids_list, "Recent sequence should still be matchable")


class TestNgramCacheLeafPaths(CustomTestCase):
    """Verify the leaf_paths_from_mask utility."""

    def test_simple_tree(self):
        cache = _make_cache("BFS")
        tokens = [3, 4, 44, 5, 55]
        mask = [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
        ]
        paths = cache.leaf_paths_from_mask(tokens, mask)

        for path in paths:
            self.assertIn(3, path, "Root token should be in every path")

        self.assertEqual(len(paths), 2, "Two leaf paths expected for a binary tree")

    def test_single_chain(self):
        cache = _make_cache("BFS")
        tokens = [10, 20, 30]
        mask = [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ]
        paths = cache.leaf_paths_from_mask(tokens, mask)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], [10, 20, 30])


class TestNgramCacheBatchConsistency(NgramCacheTestBase):
    """Verify batch queries produce same results as individual queries."""

    def test_batch_vs_individual(self):
        for cache_type in self._iter_cache_types():
            for match_type in ("BFS", "PROB"):
                with self.subTest(cache_type=cache_type, match_type=match_type):
                    cache = _make_cache(match_type, cache_type=cache_type)
                    cache.batch_put(SEED_SEQUENCES)
                    cache.synchronize()

                    batch_ids, batch_masks = cache.batch_get(QUERY_SEQUENCES)
                    draft = 8
                    batch_ids = batch_ids.reshape(-1, draft)
                    batch_masks = batch_masks.reshape(-1, draft, draft)

                    for i, query in enumerate(QUERY_SEQUENCES):
                        single_ids, single_masks = cache.batch_get([query])
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


class TestMaskValidity(NgramCacheTestBase):
    """Verify structural invariants of the output mask for any draft tree."""

    def _check_mask(self, masks_2d):
        n = len(masks_2d)
        for i in range(n):
            self.assertEqual(masks_2d[i][i], 1, f"Diagonal must be 1 at row {i}")
        self.assertEqual(masks_2d[0], [1] + [0] * (n - 1))

    def test_bfs_mask_invariants(self):
        for cache_type in self._iter_cache_types():
            with self.subTest(cache_type=cache_type):
                cache = _make_cache("BFS", cache_type=cache_type)
                cache.batch_put(SEED_SEQUENCES)
                cache.synchronize()
                _, masks = cache.batch_get(QUERY_SEQUENCES)
                masks = masks.reshape(-1, 8, 8)
                for i in range(masks.shape[0]):
                    self._check_mask(masks[i].tolist())

    def test_prob_mask_invariants(self):
        for cache_type in self._iter_cache_types():
            with self.subTest(cache_type=cache_type):
                cache = _make_cache("PROB", cache_type=cache_type)
                cache.batch_put(SEED_SEQUENCES)
                cache.synchronize()
                _, masks = cache.batch_get(QUERY_SEQUENCES)
                masks = masks.reshape(-1, 8, 8)
                for i in range(masks.shape[0]):
                    self._check_mask(masks[i].tolist())


class TestFrequencyBoosting(NgramCacheTestBase):
    """Verify that repeated insertions change Prob-mode selection."""

    def test_repeated_insert_promotes_token(self):
        for cache_type in self._iter_cache_types():
            with self.subTest(cache_type=cache_type):
                cache = _make_cache(
                    "PROB",
                    cache_type=cache_type,
                    draft_token_num=2,
                    max_bfs_breadth=1,
                    min_bfs_breadth=1,
                    max_match_window_size=3,
                    branch_length=5,
                )
                cache.batch_put([[1, 2, 3, 10, 11]])
                cache.synchronize()

                for _ in range(10):
                    cache.batch_put([[1, 2, 3, 20, 21]])
                cache.synchronize()

                ids, _ = cache.batch_get([[1, 2, 3]])
                ids_list = ids.tolist()

                self.assertEqual(
                    ids_list[1],
                    20,
                    f"Token 20 should be selected over 10 after frequency boost, got {ids_list}",
                )


class TestRecencyOrdering(NgramCacheTestBase):
    """Verify that BFS mode respects LRU recency."""

    def test_most_recent_insert_selected(self):
        for cache_type in self._iter_cache_types():
            with self.subTest(cache_type=cache_type):
                cache = _make_cache(
                    "BFS",
                    cache_type=cache_type,
                    draft_token_num=2,
                    max_bfs_breadth=1,
                    min_bfs_breadth=1,
                    max_match_window_size=3,
                    branch_length=5,
                )
                cache.batch_put([[1, 2, 3, 10, 11]])
                cache.synchronize()
                cache.batch_put([[1, 2, 3, 20, 21]])
                cache.synchronize()

                ids, _ = cache.batch_get([[1, 2, 3]])
                ids_list = ids.tolist()
                self.assertEqual(
                    ids_list[1],
                    20,
                    f"Token 20 (recent) should be selected over 10 (old), got {ids_list}",
                )


class TestOverlappingSuffixes(NgramCacheTestBase):
    """Verify correct matching when sequences share suffixes."""

    def test_shared_suffix_both_match(self):
        for cache_type in self._iter_cache_types():
            for match_type in ("BFS", "PROB"):
                with self.subTest(cache_type=cache_type, match_type=match_type):
                    cache = _make_cache(match_type, cache_type=cache_type)
                    cache.batch_put([[100, 200, 7, 8, 9, 50, 51]])
                    cache.batch_put([[300, 400, 7, 8, 9, 60, 61]])
                    cache.synchronize()

                    ids, _ = cache.batch_get([[7, 8, 9]])
                    ids_list = ids.tolist()
                    self.assertIn(50, ids_list, "Continuation from first sequence missing")
                    self.assertIn(60, ids_list, "Continuation from second sequence missing")


class TestSingleTokenContext(NgramCacheTestBase):
    """Verify behavior with minimum-length context."""

    def test_single_token_query(self):
        for cache_type in self._iter_cache_types():
            for match_type in ("BFS", "PROB"):
                with self.subTest(cache_type=cache_type, match_type=match_type):
                    cache = _make_cache(
                        match_type, cache_type=cache_type, min_match_window_size=1
                    )
                    cache.batch_put([[5, 10, 20, 30]])
                    cache.synchronize()

                    ids, _ = cache.batch_get([[5]])
                    ids_list = ids.tolist()
                    self.assertEqual(
                        ids_list[0], 5, "First token should be last context token"
                    )
                    self.assertIn(
                        10, ids_list, "Should match continuation after single token 5"
                    )


class TestLongContext(NgramCacheTestBase):
    """Verify behavior when query context exceeds branch_length."""

    def test_context_longer_than_branch_length(self):
        for cache_type in self._iter_cache_types():
            for match_type in ("BFS", "PROB"):
                with self.subTest(cache_type=cache_type, match_type=match_type):
                    cache = _make_cache(
                        match_type,
                        cache_type=cache_type,
                        branch_length=6,
                        max_match_window_size=5,
                    )
                    seq = list(range(1, 20))
                    cache.batch_put([seq])
                    cache.synchronize()

                    long_query = list(range(1, 16))
                    ids, _ = cache.batch_get([long_query])
                    ids_list = ids.tolist()
                    self.assertEqual(
                        ids_list[0], 15, "First token should be last context token"
                    )
                    self.assertIn(
                        16, ids_list, "Should match via suffix despite long context"
                    )


class TestDraftBudgetSaturation(NgramCacheTestBase):
    """Verify the draft tree uses exactly draft_token_num slots."""

    def test_full_budget_used(self):
        for cache_type in self._iter_cache_types():
            for match_type in ("BFS", "PROB"):
                with self.subTest(cache_type=cache_type, match_type=match_type):
                    cache = _make_cache(
                        match_type, cache_type=cache_type, draft_token_num=8
                    )
                    seq = list(range(1, 30))
                    cache.batch_put([seq])
                    cache.synchronize()

                    ids, _ = cache.batch_get([[1, 2, 3]])
                    ids_list = ids.tolist()
                    self.assertEqual(len(ids_list), 8)
                    non_zero = [t for t in ids_list[1:] if t != 0]
                    self.assertGreater(
                        len(non_zero),
                        0,
                        "Draft budget should have non-zero tokens when cache has long chains",
                    )


class TestTruncate(NgramCacheTestBase):
    """Verify the Result.truncate method via the Python binding."""

    def test_truncate_reduces_output(self):
        for cache_type in self._iter_cache_types():
            with self.subTest(cache_type=cache_type):
                cache = _make_cache("BFS", cache_type=cache_type, draft_token_num=8)
                cache.batch_put(SEED_SEQUENCES)
                cache.synchronize()

                result = cache.cache.batchMatch([[1, 2, 3]])
                original_len = len(result.token)
                self.assertEqual(original_len, 8)

                result.truncate(4)
                self.assertEqual(len(result.token), 4)
                self.assertEqual(len(result.mask), 4 * 4)

    def test_truncate_preserves_mask_structure(self):
        for cache_type in self._iter_cache_types():
            with self.subTest(cache_type=cache_type):
                cache = _make_cache("BFS", cache_type=cache_type, draft_token_num=8)
                cache.batch_put(SEED_SEQUENCES)
                cache.synchronize()

                result = cache.cache.batchMatch([[1, 2, 3]])
                full_ids = list(result.token)
                full_mask = list(result.mask)
                n = len(full_ids)

                result_copy = cache.cache.batchMatch([[1, 2, 3]])
                trunc_n = 4
                result_copy.truncate(trunc_n)
                trunc_mask = list(result_copy.mask)

                for i in range(trunc_n):
                    for j in range(trunc_n):
                        self.assertEqual(
                            trunc_mask[i * trunc_n + j],
                            full_mask[i * n + j],
                            f"Mask mismatch at ({i},{j})",
                        )


class TestResetAndReinsert(NgramCacheTestBase):
    """Verify that reset followed by new inserts works correctly."""

    def test_reset_then_reinsert(self):
        for cache_type in self._iter_cache_types():
            for match_type in ("BFS", "PROB"):
                with self.subTest(cache_type=cache_type, match_type=match_type):
                    cache = _make_cache(match_type, cache_type=cache_type)
                    cache.batch_put([[1, 2, 3, 4, 5]])
                    cache.synchronize()

                    cache.reset()

                    cache.batch_put([[10, 20, 30, 40, 50]])
                    cache.synchronize()

                    ids_old, _ = cache.batch_get([[1, 2, 3]])
                    ids_old_list = ids_old.tolist()
                    self.assertTrue(
                        all(t == 0 for t in ids_old_list[1:]),
                        f"Old data should not match after reset+reinsert, got {ids_old_list}",
                    )

                    ids_new, _ = cache.batch_get([[10, 20, 30]])
                    ids_new_list = ids_new.tolist()
                    self.assertEqual(ids_new_list[0], 30)
                    self.assertIn(
                        40, ids_new_list, "New data should match after reset+reinsert"
                    )


class TestSqueezeEvictsOld(NgramCacheTestBase):
    """Verify that squeeze actually evicts old data, not just preserves recent."""

    def test_old_data_evicted(self):
        for cache_type in self._iter_cache_types():
            with self.subTest(cache_type=cache_type):
                cache = _make_cache(
                    "BFS",
                    cache_type=cache_type,
                    capacity=300,
                    branch_length=6,
                    max_match_window_size=5,
                )

                old_seq = list(range(5000, 5030))
                cache.batch_put([old_seq])
                cache.synchronize()

                ids_before, _ = cache.batch_get([[5000, 5001, 5002]])
                self.assertIn(
                    5003,
                    ids_before.tolist(),
                    "Old data should match before eviction",
                )

                for i in range(5):
                    new_seq = list(range(6000 + i * 30, 6000 + i * 30 + 30))
                    cache.batch_put([new_seq])
                    cache.synchronize()

                ids_after, _ = cache.batch_get([[5000, 5001, 5002]])
                ids_after_list = ids_after.tolist()
                self.assertNotIn(
                    5003,
                    ids_after_list,
                    f"Old data should be evicted after pressure, got {ids_after_list}",
                )


class TestNgramCacheSAMSpecific(CustomTestCase):
    """Cover SAM-only cases that exercise interval matching behavior."""

    def test_sam_shorter_window_fail_link_match(self):
        cache = _make_cache(
            "BFS",
            cache_type="sam",
            max_match_window_size=3,
            branch_length=5,
        )
        cache.batch_put([[5, 2, 3, 40, 41]])
        cache.batch_put([[6, 2, 3, 50, 51]])
        cache.synchronize()

        ids, _ = cache.batch_get([[99, 2, 3]])
        ids_list = ids.tolist()
        self.assertIn(40, ids_list)
        self.assertIn(50, ids_list)

    def test_sam_clone_heavy_overlap_stays_matchable(self):
        cache = _make_cache(
            "PROB",
            cache_type="sam",
            max_match_window_size=3,
            branch_length=6,
            draft_token_num=4,
        )
        cache.batch_put([[1, 2, 3, 1, 2, 4]])
        cache.batch_put([[1, 2, 3, 1, 2, 5]])
        cache.synchronize()

        ids, _ = cache.batch_get([[1, 2, 3]])
        ids_list = ids.tolist()
        self.assertEqual(ids_list[0], 3)
        self.assertTrue(any(token in ids_list for token in (1, 4, 5)))


if __name__ == "__main__":
    unittest.main(verbosity=3)
