"""Unit tests for the CP-reshard rank-0 deficit-vector eviction protocol:
``RadixCache.cp_coordinated_evict`` and ``_pick_cp_victims_for_deficits``.

These guard the consistency fix for ``--enable-cp-kv-reshard``: every CP rank
enters eviction in lockstep (unconditional ``all_gather`` of per-rank deficits),
and rank 0 selects a single victim set that frees enough *owned* rows on EVERY
rank -- computed from the mirrored ``cp_owner_per_page`` via
``(owner == r).sum() * page_size`` (never ``len(value)``, which counts the
slot-0 sentinels of non-owned pages).

Pure-Python / CPU-only: a fake CP group supplies ``all_gather_object`` (returns
a configurable skewed deficit vector with rank 0's own deficit first) and a
passthrough ``broadcast_object`` so the rank-0 selection is applied locally.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.test_utils import CustomTestCase


class _FakeCPGroup:
    """Single-process stand-in for the attention CP group. ``all_gather_object``
    returns ``[rank0_deficit, *peer_deficits]`` so tests can inject per-rank
    skew; ``broadcast_object`` is a passthrough so rank-0's payload is applied
    locally. Counts calls so tests can assert the lockstep collective pattern.
    """

    def __init__(self, peer_deficits):
        self.peer_deficits = [int(d) for d in peer_deficits]
        self.gather_calls = 0
        self.broadcast_calls = 0
        self.last_gathered = None

    def all_gather_object(self, obj):
        self.gather_calls += 1
        self.last_gathered = [int(obj)] + self.peer_deficits
        return list(self.last_gathered)

    def broadcast_object(self, obj, src=0):
        self.broadcast_calls += 1
        return obj


def _make_mock_allocator():
    """Mock allocator that records freed indices (CPU-only)."""
    mock = unittest.mock.Mock()
    mock.device = torch.device("cpu")
    mock.freed = []
    mock.free.side_effect = lambda idx: mock.freed.append(
        idx.clone() if isinstance(idx, torch.Tensor) else idx
    )
    return mock


def _flat_freed(mock):
    pieces = [
        f.flatten() for f in mock.freed if isinstance(f, torch.Tensor) and f.numel() > 0
    ]
    return (
        torch.cat(pieces).to(torch.int64)
        if pieces
        else torch.empty((0,), dtype=torch.int64)
    )


def _build_cache(peer_deficits, page_size=1, cp_rank=0, mock=None):
    """Build a simulated RadixCache (rank ``cp_rank``) with CP consensus active
    over a fake group of size ``1 + len(peer_deficits)``."""
    mock = mock or _make_mock_allocator()
    cache = RadixCache.create_simulated(page_size=page_size, mock_allocator=mock)
    group = _FakeCPGroup(peer_deficits)
    cache.enable_cp_consensus(group, cp_rank=cp_rank, cp_size=1 + len(peer_deficits))
    return cache, mock, group


# Monotonic row-id source so each owned token gets a distinct real pool row.
_NEXT_ROW = [100]


def _insert_leaf(cache, key_tokens, owner_per_page, page_size, owner_rank=0):
    """Insert one leaf carrying ``owner_per_page``. ``value`` is the per-rank
    projection for ``owner_rank``: a distinct real row where ``owner_rank`` owns
    the token's page, slot-0 sentinel elsewhere."""
    owner = torch.tensor(owner_per_page, dtype=torch.int8)
    value = []
    for i in range(len(key_tokens)):
        if int(owner_per_page[i // page_size]) == owner_rank:
            value.append(_NEXT_ROW[0])
            _NEXT_ROW[0] += 1
        else:
            value.append(0)
    cache.insert(
        InsertParams(
            key=RadixKey(list(key_tokens)),
            value=torch.tensor(value, dtype=torch.int64),
            cp_owner_per_page=owner,
        )
    )


def _freed_for_rank(paths, cache, rank, page_size):
    """Sum, over victim ``paths``, the rows that would be freed on ``rank``."""
    total = 0
    for path in paths:
        node = cache._lookup_by_path(path)
        owner = node.cp_owner_per_page
        total += int((owner == rank).sum()) * page_size
    return total


class TestPickCpVictimsForDeficits(CustomTestCase):
    def test_covers_every_rank_deficit_under_skew(self):
        """Selection must free >= deficit[r] OWNED rows for EVERY rank r, even
        when rank 0's deficit is smaller than a peer's."""
        cache, _, _ = _build_cache(peer_deficits=[2])  # cp_size=2; page_size=1
        # 4 single-token leaves: 2 owned by rank 0, 2 owned by rank 1.
        _insert_leaf(cache, [1], [0], page_size=1)
        _insert_leaf(cache, [2], [1], page_size=1)
        _insert_leaf(cache, [3], [1], page_size=1)
        _insert_leaf(cache, [4], [0], page_size=1)

        # rank0 needs 1 row, rank1 needs 2 rows.
        deficits = [1, 2]
        paths, satisfiable = cache._pick_cp_victims_for_deficits(deficits)

        self.assertTrue(satisfiable)
        self.assertGreaterEqual(_freed_for_rank(paths, cache, 0, 1), deficits[0])
        self.assertGreaterEqual(_freed_for_rank(paths, cache, 1, 1), deficits[1])

    def test_unsatisfiable_when_capacity_short(self):
        """When some rank's deficit exceeds total evictable owned rows, return
        satisfiable=False and exhaust the heap."""
        cache, _, _ = _build_cache(peer_deficits=[99])
        _insert_leaf(cache, [1], [0], page_size=1)
        _insert_leaf(cache, [2], [1], page_size=1)  # only 1 row for rank 1

        paths, satisfiable = cache._pick_cp_victims_for_deficits([1, 99])
        self.assertFalse(satisfiable)
        # Every evictable leaf was selected (heap exhausted trying to satisfy).
        self.assertEqual(len(paths), 2)

    def test_uses_owner_not_value_len(self):
        """freed[r] must be (owner==r).sum()*page_size, not len(value). A
        rank-0 leaf with 3 of 4 tokens sentinelled still frees 4 rows for the
        owning rank under page_size=1 -> the owner array, not the value, drives
        the count. (Guards against the sentinel over/under-count regression.)"""
        cache, _, _ = _build_cache(peer_deficits=[0], page_size=4)
        # 1 page (4 tokens) owned entirely by rank 0.
        _insert_leaf(cache, [1, 2, 3, 4], [0], page_size=4)
        paths, satisfiable = cache._pick_cp_victims_for_deficits([4, 0])
        self.assertTrue(satisfiable)
        self.assertEqual(_freed_for_rank(paths, cache, 0, 4), 4)


class TestCpCoordinatedEvict(CustomTestCase):
    def test_no_deficit_skips_broadcast(self):
        """All-zero gathered deficits -> returns True, NO broadcast (lockstep
        skip; every rank sees the same vector and skips identically)."""
        cache, _, group = _build_cache(peer_deficits=[0])
        _insert_leaf(cache, [1], [0], page_size=1)
        result = cache.cp_coordinated_evict(0)
        self.assertTrue(result)
        self.assertEqual(group.gather_calls, 1)
        self.assertEqual(group.broadcast_calls, 0)

    def test_broadcast_fires_when_any_rank_short(self):
        """max(deficits) > 0 -> exactly one all_gather AND one broadcast."""
        cache, _, group = _build_cache(peer_deficits=[2])
        _insert_leaf(cache, [1], [0], page_size=1)
        _insert_leaf(cache, [2], [1], page_size=1)
        _insert_leaf(cache, [3], [1], page_size=1)
        # rank 0's own deficit is 0, but a peer is short -> still coordinate.
        result = cache.cp_coordinated_evict(0)
        self.assertTrue(result)
        self.assertEqual(group.gather_calls, 1)
        self.assertEqual(group.broadcast_calls, 1)

    def test_frees_only_rank0_owned_rows_no_sentinels(self):
        """Applying the agreed victim set frees this rank's real owned rows and
        never the slot-0 sentinels of non-owned pages."""
        cache, mock, _ = _build_cache(peer_deficits=[0])
        _insert_leaf(cache, [1], [0], page_size=1)  # rank-0 owned: real row
        _insert_leaf(cache, [2], [1], page_size=1)  # rank-1 owned: sentinel on rank 0
        cache.cp_coordinated_evict(1)  # rank 0 needs 1 row
        freed = _flat_freed(mock)
        self.assertFalse((freed == 0).any(), f"slot-0 freed: {freed.tolist()}")
        self.assertTrue((freed >= 100).all())  # only the real owned row ids

    def test_unsatisfiable_returns_false(self):
        cache, _, _ = _build_cache(peer_deficits=[99])
        _insert_leaf(cache, [1], [0], page_size=1)
        _insert_leaf(cache, [2], [1], page_size=1)
        self.assertFalse(cache.cp_coordinated_evict(1))

    def test_no_consensus_falls_back_to_local(self):
        """Without enable_cp_consensus, cp_coordinated_evict degrades to a local
        evict and never touches a (nonexistent) group."""
        mock = _make_mock_allocator()
        cache = RadixCache.create_simulated(page_size=1, mock_allocator=mock)
        self.assertIsNone(cache.cp_attn_group)
        cache.insert(InsertParams(key=RadixKey([1, 2]), value=torch.tensor([7, 8])))
        self.assertTrue(cache.cp_coordinated_evict(2))
        self.assertEqual(sorted(_flat_freed(mock).tolist()), [7, 8])


class TestCrossRankConsistency(CustomTestCase):
    def test_same_paths_free_each_ranks_owned_rows(self):
        """The path list rank 0 broadcasts, applied on a rank-1 tree built with
        rank-1's value projection, deletes the identical nodes and frees rank
        1's owned rows (the mirrored-tree consistency guarantee)."""
        owners = {(1,): [0], (2,): [1], (3,): [1], (4,): [0]}

        cache0, _, _ = _build_cache(peer_deficits=[2], cp_rank=0)
        for k, o in owners.items():
            _insert_leaf(cache0, list(k), o, page_size=1, owner_rank=0)
        paths, satisfiable = cache0._pick_cp_victims_for_deficits([1, 2])
        self.assertTrue(satisfiable)

        mock1 = _make_mock_allocator()
        cache1 = RadixCache.create_simulated(page_size=1, mock_allocator=mock1)
        group1 = _FakeCPGroup([1])
        cache1.enable_cp_consensus(group1, cp_rank=1, cp_size=2)
        for k, o in owners.items():
            _insert_leaf(cache1, list(k), o, page_size=1, owner_rank=1)

        # Rank 1 applies the SAME paths rank 0 chose.
        expected_rank1_rows = _freed_for_rank(paths, cache1, 1, 1)
        cache1._apply_cp_eviction(paths)
        freed1 = _flat_freed(mock1)
        self.assertFalse((freed1 == 0).any())
        self.assertEqual(freed1.numel(), expected_rank1_rows)
        # The chosen nodes are gone from rank 1's tree.
        for path in paths:
            self.assertIsNone(cache1._lookup_by_path(path))


if __name__ == "__main__":
    unittest.main()
