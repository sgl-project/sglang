"""CPU contracts for the page-striped logical host pool."""

import random
import unittest
from collections import deque

import torch

from sglang.srt.mem_cache.pool_host.striped_host import (
    StripedHostInvalidFree,
    StripedHostInvalidRollback,
    StripedHostInvariantError,
    StripedHostPlacement,
    StripedLogicalHostPool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 4
NUM_RANKS = 8
PER_RANK_PAGES = 37


def _make_pools(num_ranks=NUM_RANKS, per_rank_pages=PER_RANK_PAGES, ps=PAGE_SIZE):
    return [
        StripedLogicalHostPool(
            per_rank_size=per_rank_pages * ps,
            page_size=ps,
            placement=StripedHostPlacement(num_ranks=num_ranks, rank=r, page_size=ps),
        )
        for r in range(num_ranks)
    ]


def _check_pages_whole(test, indices, ps):
    rows = indices.view(-1, ps)
    starts = rows[:, 0]
    test.assertTrue(bool((starts % ps == 0).all()))
    test.assertTrue(
        torch.equal(rows, starts.unsqueeze(1) + torch.arange(ps, dtype=torch.int64))
    )


class _RefStripedAllocator:
    """Per-page reference: one visit per page from a rotating cursor."""

    def __init__(self, num_ranks, per_rank_pages):
        self.num_ranks = num_ranks
        self.free = [
            deque(range(c, num_ranks * per_rank_pages, num_ranks))
            for c in range(num_ranks)
        ]
        self.cursor = 0

    def alloc(self, num_pages):
        if num_pages > sum(len(f) for f in self.free):
            return None
        cls, pages = self.cursor, []
        for _ in range(num_pages):
            while not self.free[cls]:
                cls = (cls + 1) % self.num_ranks
            pages.append(self.free[cls].popleft())
            cls = (cls + 1) % self.num_ranks
        if num_pages:
            self.cursor = cls
        return pages

    def free_pages(self, pages):
        for page in pages:
            self.free[page % self.num_ranks].append(page)


class TestStripedHostPlacement(CustomTestCase):
    def test_owner_local_is_a_bijection_per_rank(self):
        ps, n, per_rank_pages = 4, 8, 5
        logical = torch.arange(n * per_rank_pages * ps, dtype=torch.int64)
        seen_owner = torch.zeros_like(logical)
        for rank in range(n):
            placement = StripedHostPlacement(num_ranks=n, rank=rank, page_size=ps)
            owned = logical[placement.owned_mask(logical)]
            seen_owner[owned] += 1
            local = placement.local(owned)
            self.assertTrue(
                torch.equal(
                    local.sort().values,
                    torch.arange(per_rank_pages * ps, dtype=torch.int64),
                )
            )
            # Whole pages stay whole and keep in-page offsets.
            self.assertTrue(torch.equal(local % ps, owned % ps))
        self.assertTrue(bool((seen_owner == 1).all()))


class TestStripedLogicalHostPool(CustomTestCase):
    def test_capacity_is_n_times_per_rank(self):
        pool = _make_pools()[3]
        total = NUM_RANKS * PER_RANK_PAGES * PAGE_SIZE
        self.assertEqual(pool.size, total)
        self.assertEqual(pool.logical_size, total)
        self.assertEqual(pool.available_size(), total)

        indices = pool.alloc(total)
        self.assertIsNotNone(indices)
        _check_pages_whole(self, indices, PAGE_SIZE)
        self.assertEqual(torch.unique(indices).numel(), total)
        self.assertEqual(pool.used_pages_per_class(), [PER_RANK_PAGES] * NUM_RANKS)
        self.assertEqual(pool.available_size(), 0)
        self.assertIsNone(pool.alloc(PAGE_SIZE))

        pool.free(indices)
        self.assertEqual(pool.available_size(), total)

    def test_free_rejects_partial_duplicate_and_double_free(self):
        pool = _make_pools(num_ranks=2, per_rank_pages=4, ps=4)[0]
        indices = pool.alloc(8)
        digest, seq = pool.digest(), pool.op_seq
        bad_frees = {
            "partial": indices[:3],
            "unaligned": indices[1:5],
            "duplicate": torch.cat([indices[:4], indices[:4]]),
            "out_of_range": torch.arange(pool.size, pool.size + 4),
            "negative": torch.arange(-4, 0),
        }
        for name, bad in bad_frees.items():
            with self.subTest(name=name):
                with self.assertRaises(StripedHostInvalidFree):
                    pool.free(bad)
        # A rejected free leaves the state untouched.
        self.assertEqual((pool.digest(), pool.op_seq), (digest, seq))
        with self.assertRaises(ValueError):
            pool.alloc(3)
        pool.free(indices)
        with self.assertRaises(StripedHostInvalidFree):
            pool.free(indices)
        # 2-D and strided index tensors free whole pages too.
        self.assertEqual(pool.free(pool.alloc(8).view(2, 4)), 8)
        strided = torch.stack([pool.alloc(8)] * 2, dim=1)[:, 0]
        self.assertFalse(strided.is_contiguous())
        self.assertEqual(pool.free(strided), 8)
        self.assertEqual(pool.available_size(), pool.size)

    def test_empty_alloc_and_free_are_not_recorded(self):
        pool = _make_pools(num_ranks=2, per_rank_pages=4, ps=2)[0]
        digest, seq = pool.digest(), pool.op_seq
        self.assertEqual(pool.alloc(0).numel(), 0)
        self.assertEqual(pool.free(torch.empty(0, dtype=torch.int64)), 0)
        self.assertEqual((pool.digest(), pool.op_seq), (digest, seq))

    def test_matches_per_page_reference(self):
        num_ranks, per_rank_pages, ps = 8, 23, 2
        pool = _make_pools(num_ranks, per_rank_pages, ps)[0]
        ref = _RefStripedAllocator(num_ranks, per_rank_pages)
        rng = random.Random(7)
        live = []
        for step in range(3000):
            if live and rng.random() < 0.45:
                victim = live.pop(rng.randrange(len(live)))
                order = list(range(len(victim)))
                rng.shuffle(order)
                victim = [victim[i] for i in order]
                pool.free(
                    torch.cat([torch.arange(p * ps, p * ps + ps) for p in victim])
                )
                ref.free_pages(victim)
            else:
                num_pages = rng.randint(0, num_ranks * per_rank_pages // 5)
                got = pool.alloc(num_pages * ps)
                want = ref.alloc(num_pages)
                if want is None:
                    self.assertIsNone(got, f"step {step}")
                    continue
                self.assertEqual((got[::ps] // ps).tolist(), want, f"step {step}")
                if want:
                    live.append(want)
            self.assertEqual(pool._cursor, ref.cursor, f"step {step}")

    def test_digest_depends_on_page_order_and_config(self):
        a, b = _make_pools(num_ranks=2, per_rank_pages=4, ps=2)[:2]
        self.assertEqual(a.digest(), b.digest())
        ia = a.alloc(8)
        ib = b.alloc(8)
        self.assertTrue(torch.equal(ia, ib))
        a.free(ia)
        b.free(torch.cat([ib[4:], ib[:4]]))
        self.assertEqual(a.available_size(), b.available_size())
        self.assertNotEqual(a.digest(), b.digest())

        other = _make_pools(num_ranks=4, per_rank_pages=2, ps=2)[0]
        self.assertEqual(
            other.size, _make_pools(num_ranks=2, per_rank_pages=4, ps=2)[0].size
        )
        self.assertNotEqual(
            other.digest(), _make_pools(num_ranks=2, per_rank_pages=4, ps=2)[0].digest()
        )

        # clear is a recorded op: it rolls the digest and advances op_seq.
        before, seq = a.digest(), a.op_seq
        a.clear()
        self.assertEqual(a.available_size(), a.size)
        self.assertNotEqual(a.digest(), before)
        self.assertEqual(a.op_seq, seq + 1)

    def test_rollback_alloc_matches_a_pool_that_never_allocated(self):
        num_ranks, per_rank_pages, ps = 4, 20, 2
        pool = _make_pools(num_ranks, per_rank_pages, ps)[0]
        ref = _make_pools(num_ranks, per_rank_pages, ps)[0]
        rng = random.Random(11)
        live = []
        rolled_back = 0
        # > _OP_TAIL_LEN ops so rollbacks also restore an evicted tail record.
        for step in range(400):
            if live and rng.random() < 0.4:
                victim = live.pop(rng.randrange(len(live)))
                self.assertEqual(pool.free(victim), ref.free(victim.clone()))
            else:
                need = rng.randint(1, num_ranks * per_rank_pages // 4) * ps
                if rng.random() < 0.3:
                    vetoed = pool.alloc(need)
                    if vetoed is not None:
                        pool.rollback_alloc(vetoed)
                        rolled_back += 1
                        with self.assertRaises(StripedHostInvariantError):
                            pool.rollback_alloc(vetoed)
                else:
                    got = pool.alloc(need)
                    want = ref.alloc(need)
                    self.assertEqual(
                        None if got is None else got.tolist(),
                        None if want is None else want.tolist(),
                        f"step {step}",
                    )
                    if got is not None:
                        live.append(got)
            self.assertEqual(
                (pool.digest(), pool.op_seq, pool._cursor, list(pool._op_tail)),
                (ref.digest(), ref.op_seq, ref._cursor, list(ref._op_tail)),
                f"step {step}",
            )
            self.assertEqual(pool.free_pages_per_class(), ref.free_pages_per_class())
        self.assertGreater(rolled_back, 20)

    def test_rollback_alloc_rejects_stale_or_foreign_indices(self):
        pool = _make_pools(num_ranks=2, per_rank_pages=4, ps=2)[0]
        with self.assertRaises(StripedHostInvalidRollback):
            pool.rollback_alloc(torch.empty(0, dtype=torch.int64))
        first = pool.alloc(4)
        second = pool.alloc(4)
        with self.assertRaises(StripedHostInvalidRollback):
            pool.rollback_alloc(first)
        with self.assertRaises(StripedHostInvalidRollback):
            pool.rollback_alloc(second[:2])
        pool.rollback_alloc(second)
        pool.alloc(0)
        # An empty alloc or a free closes the window too.
        third = pool.alloc(4)
        pool.free(first)
        with self.assertRaises(StripedHostInvalidRollback):
            pool.rollback_alloc(third)


if __name__ == "__main__":
    unittest.main()
