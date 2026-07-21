"""CPU unit test for the decode-context-parallel (DCP) per-rank KV-length math.

Pins ``get_dcp_lens`` (the single, superset implementation in
``layers/dcp/layout.py``) to a brute-force owner-count reference, and proves
it is bit-identical to the legacy in-place formula that
``update_local_kv_lens_for_dcp`` used before it was collapsed into a wrapper:

    floor((len - rank - 1) / N) + 1   ==   len // N + (rank < len % N)   (len >= 0)

Usage:
    python -m pytest test_dcp_layout_unit.py -v
    python test_dcp_layout_unit.py
"""

import unittest

import torch

from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 3, 4, 8]
LENS = list(range(0, 41))
STARTS = [0, 1, 2, 5, 7, 13, 31]


def _owner_count(length: int, n: int, rank: int, start: int) -> int:
    """Ground truth: # of absolute positions p in [start, start+length) with p % n == rank."""
    return sum(1 for p in range(start, start + length) if p % n == rank)


def _legacy_inplace_formula(length: int, n: int, rank: int) -> int:
    """The pre-refactor update_local_kv_lens_for_dcp body (start == 0 case)."""
    return (length - rank - 1) // n + 1


class TestGetDcpLens(unittest.TestCase):
    def test_start_none_matches_owner_count(self):
        for n in DCP_SIZES:
            for rank in range(n):
                lens = torch.tensor(LENS, dtype=torch.int32)
                got = get_dcp_lens(lens, n, rank)
                expected = torch.tensor(
                    [_owner_count(L, n, rank, 0) for L in LENS], dtype=torch.int32
                )
                self.assertTrue(
                    torch.equal(got.to(torch.int32), expected),
                    f"start=None mismatch at n={n}, rank={rank}: {got.tolist()} != {expected.tolist()}",
                )

    def test_start_none_matches_legacy_inplace_formula(self):
        # The collapse claim: get_dcp_lens (start=None) == legacy floor((L-rank-1)/N)+1.
        for n in DCP_SIZES:
            for rank in range(n):
                lens = torch.tensor(LENS, dtype=torch.int64)
                got = get_dcp_lens(lens, n, rank)
                legacy = torch.tensor(
                    [_legacy_inplace_formula(L, n, rank) for L in LENS],
                    dtype=torch.int64,
                )
                self.assertTrue(
                    torch.equal(got.to(torch.int64), legacy),
                    f"legacy-formula mismatch at n={n}, rank={rank}",
                )

    def test_start_tensor_matches_owner_count(self):
        for n in DCP_SIZES:
            for rank in range(n):
                for start in STARTS:
                    lens = torch.tensor(LENS, dtype=torch.int64)
                    start_t = torch.full_like(lens, start)
                    got = get_dcp_lens(lens, n, rank, start=start_t)
                    expected = torch.tensor(
                        [_owner_count(L, n, rank, start) for L in LENS],
                        dtype=torch.int64,
                    )
                    self.assertTrue(
                        torch.equal(got.to(torch.int64), expected),
                        f"start={start} mismatch at n={n}, rank={rank}: "
                        f"{got.tolist()} != {expected.tolist()}",
                    )

    def test_dcp_size_one_is_identity(self):
        lens = torch.tensor(LENS, dtype=torch.int32)
        self.assertTrue(torch.equal(get_dcp_lens(lens, 1, 0), lens))


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class TestGetDcpLensPageBoundary(unittest.TestCase):
    """Bench-scale boundary sweep pinning the verify-DCP pass-1 table contract.

    The pass-1 prefix block table is built with cdiv(global_len, page*world)
    GLOBAL page entries per request, while the decode kernel on rank r reads
    cdiv(local_len_r, page) LOCAL pages ("local page N == global page N").
    The contract that makes the table wide enough for every rank is

        cdiv(get_dcp_lens(L, N, r), page) <= cdiv(L, page * N)

    together with conservation, sum_r get_dcp_lens(L, N, r) == L. A violation
    is an out-of-bounds-read class in ``_forward_verify_dcp`` pass-1 (the
    consumer would dereference the -1 fill / stale tail of the table row).
    Swept at the exact shapes the cc16 crash regime uses (50K prefixes,
    page 32/64, dcp 8) plus +-1 neighborhoods of effective-page multiples.
    """

    PAGE_SIZES = [32, 64]
    DCP_SIZES = [2, 4, 8]

    @staticmethod
    def _boundary_lens(eff_page: int) -> list[int]:
        lens = set()
        for k in (1, 2, 3, 7, 97, 100, 781, 782, 1024):
            for delta in (-1, 0, 1):
                v = k * eff_page + delta
                if v >= 0:
                    lens.add(v)
        # cc16 bench regime: 50K shared prefix + a few thousand decoded tokens.
        lens.update([0, 1, 49999, 50000, 50007, 53248, 65535, 65536, 131072])
        return sorted(lens)

    def test_local_pages_fit_global_page_table(self):
        for page in self.PAGE_SIZES:
            for n in self.DCP_SIZES:
                eff_page = page * n
                lens_list = self._boundary_lens(eff_page)
                lens = torch.tensor(lens_list, dtype=torch.int64)
                table_width = [_cdiv(L, eff_page) for L in lens_list]
                for rank in range(n):
                    local = get_dcp_lens(lens, n, rank)
                    for i, L in enumerate(lens_list):
                        pages_read = _cdiv(int(local[i]), page)
                        self.assertLessEqual(
                            pages_read,
                            table_width[i],
                            f"rank {rank} reads {pages_read} local pages but the "
                            f"pass-1 table only has {table_width[i]} valid entries "
                            f"(L={L}, page={page}, dcp={n})",
                        )

    def test_local_lens_conserve_global(self):
        for page in self.PAGE_SIZES:
            for n in self.DCP_SIZES:
                eff_page = page * n
                lens = torch.tensor(self._boundary_lens(eff_page), dtype=torch.int64)
                total = torch.zeros_like(lens)
                for rank in range(n):
                    total += get_dcp_lens(lens, n, rank)
                self.assertTrue(
                    torch.equal(total, lens),
                    f"sum over ranks != global lens at page={page}, dcp={n}",
                )


if __name__ == "__main__":
    unittest.main()
