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

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

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


if __name__ == "__main__":
    unittest.main()
