"""CPU unit test for the DCP extend KV write's owner filter.

Pins ``plan_dcp_owner_write`` (``layers/dcp/layout.py``), which is what the NPU
MLA pool uses at extend instead of the capturable row-0 trick it still uses at
decode (``_resolve_dcp_write``).

Under DCP a rank owns the positions where ``loc % dcp_size == rank`` and keeps
them at ``loc // dcp_size``. The decode path writes EVERY row on EVERY rank and
aims the ones it does not own at physical row 0, because a boolean filter has a
data-dependent shape and cannot run inside a captured stream. Extend is not
captured, so it can drop those rows -- and must, because a 13,855-token tail was
writing 13,855 rows per rank per layer instead of ~866, with 12,989 of them
colliding on one row.

The risk in swapping one for the other is silent: a wrong filter writes real KV
to the wrong row, or drops a row nobody else writes, and attention then reads
plausible-looking garbage rather than failing. So the central test here does not
check the indices -- it runs BOTH write paths against the same KV and requires
the resulting pools to be identical everywhere except physical row 0, which is
the padding row no real token is ever issued.

Usage:
    python -m pytest test_dcp_owner_write.py -v
    python test_dcp_owner_write.py
"""

import unittest

import torch

from sglang.srt.layers.dcp.layout import plan_dcp_owner_write
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

DCP_SIZES = [2, 3, 4, 16]
DIM = 3
# The allocator seeds free_pages from 1, so no real token is ever issued a
# virtual loc below one widened page: page_size * dcp_size. Physical page 0 is
# therefore never allocated on any rank, which is what makes it safe to aim the
# decode path's non-owned rows there.
PAGE_SIZE = 4


def _widened_pool() -> torch.Tensor:
    """A per-rank pool, NaN-filled so an unwritten row cannot pass as a match."""
    return torch.full((64, DIM), float("nan"))


def _kv(n: int) -> torch.Tensor:
    """Distinct rows, so a row landing in the wrong place is visible."""
    return torch.arange(1, n * DIM + 1, dtype=torch.float32).view(n, DIM)


def _write_row0(pool, loc, kv, dcp_size, rank):
    """The decode path: every row written, non-owned aimed at physical row 0."""
    owned = (loc % dcp_size) == rank
    dest = torch.where(owned, loc // dcp_size, loc.new_zeros(()))
    pool[dest] = kv


def _write_filtered(pool, loc, kv, dcp_size, rank):
    """The extend path: only this rank's rows, written where they belong."""
    owned_idx, dest = plan_dcp_owner_write(loc, dcp_size, rank)
    pool[dest] = kv.index_select(0, owned_idx)


class TestDcpOwnerWrite(CustomTestCase):
    def _locs(self, dcp_size):
        """Write locations, in the widened virtual space.

        The first is what a prefix-cache-hit tail actually produces: one
        contiguous run starting on a widened page boundary. The rest are the
        ragged cases the filter still has to get right -- a run that starts
        mid-page, several disjoint runs, and a single row.
        """
        base = PAGE_SIZE * dcp_size
        return [
            torch.arange(base, base + 8 * dcp_size, dtype=torch.int64),
            torch.arange(base + 3, base + 3 + 5 * dcp_size, dtype=torch.int64),
            torch.cat(
                [
                    torch.arange(base, base + dcp_size + 1, dtype=torch.int64),
                    torch.arange(base + 4 * dcp_size, base + 5 * dcp_size + 2),
                ]
            ),
            torch.tensor([base + 1], dtype=torch.int64),
        ]

    def test_the_filtered_write_matches_the_row0_write_where_it_matters(self):
        # THE test. Two paths, same KV, same pool -- identical everywhere a real
        # token can live. Row 0 is excluded because that is precisely where the
        # decode path dumps what it does not own.
        for dcp_size in DCP_SIZES:
            for loc in self._locs(dcp_size):
                kv = _kv(loc.numel())
                for rank in range(dcp_size):
                    a, b = _widened_pool(), _widened_pool()
                    _write_row0(a, loc, kv, dcp_size, rank)
                    _write_filtered(b, loc, kv, dcp_size, rank)
                    self.assertTrue(
                        torch.equal(
                            torch.nan_to_num(a[1:], nan=-1.0),
                            torch.nan_to_num(b[1:], nan=-1.0),
                        ),
                        f"dcp_size={dcp_size} rank={rank} n={loc.numel()}",
                    )

    def test_the_ranks_partition_every_row_exactly_once(self):
        # A row dropped by every rank is KV that no rank holds; a row claimed by
        # two is a write the group disagrees about.
        for dcp_size in DCP_SIZES:
            for loc in self._locs(dcp_size):
                claimed = torch.cat(
                    [plan_dcp_owner_write(loc, dcp_size, r)[0] for r in range(dcp_size)]
                )
                self.assertTrue(
                    torch.equal(
                        claimed.sort().values,
                        torch.arange(loc.numel(), dtype=claimed.dtype),
                    ),
                    f"dcp_size={dcp_size} n={loc.numel()}",
                )

    def test_no_filtered_write_ever_touches_physical_row_zero(self):
        # What lets the decode path use row 0 as a bin. If a real write could
        # land there, the two paths would corrupt each other.
        for dcp_size in DCP_SIZES:
            for loc in self._locs(dcp_size):
                for rank in range(dcp_size):
                    _, dest = plan_dcp_owner_write(loc, dcp_size, rank)
                    if dest.numel():
                        self.assertGreaterEqual(int(dest.min()), PAGE_SIZE)

    def test_a_rank_that_owns_nothing_returns_empty_not_garbage(self):
        # Fewer rows than ranks: some ranks legitimately own no row of this
        # batch, and must write none rather than write row 0.
        loc = torch.tensor([64, 65], dtype=torch.int64)
        owned_idx, dest = plan_dcp_owner_write(loc, 16, 7)
        self.assertEqual(owned_idx.numel(), 0)
        self.assertEqual(dest.numel(), 0)


if __name__ == "__main__":
    unittest.main()
