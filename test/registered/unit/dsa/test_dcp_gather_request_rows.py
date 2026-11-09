"""The DCP prefill gather must rebuild every request's rows in position order from the ranks' interleaved shares."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa import dsa_dcp_sharded as sharded
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _shares(rows, lens, size):
    """Each rank's rows, request by request, positions p with p % size == rank."""
    starts = torch.cumsum(lens, 0) - lens
    return [
        torch.cat([rows[s + r : s + n : size] for s, n in zip(starts.tolist(), lens.tolist())])
        for r in range(size)
    ]


class _Group:
    def __init__(self, sends, rank):
        self.sends, self.rank = sends, rank

    def all_gather_into_tensor(self, out, inp):
        sends = list(self.sends)
        sends[self.rank] = inp
        out.copy_(torch.cat(sends))


class TestDCPGatherRequestRows(CustomTestCase):
    def setUp(self):
        sharded._REQUEST_ROWS_PLAN.clear()

    def _check(self, lens, size, width, page_size=1):
        lens = torch.tensor(lens)
        rows = torch.randint(0, 256, (int(lens.sum()), width), dtype=torch.uint8)
        shares = _shares(rows, lens, size)
        plans = [sharded.request_rows_plan(lens, size, r, torch.device("cpu")) for r in range(size)]
        for r, (send, _, num_slots) in enumerate(plans):
            self.assertEqual(shares[r].shape[0], int(get_dcp_lens(lens, size, r).sum()))
            self.assertEqual(send.shape[0], num_slots)
        sends = [
            s[p[0]] if s.shape[0] else s.new_empty((p[2], width)) for s, p in zip(shares, plans)
        ]
        expected = torch.cat([rows, rows.new_zeros((-rows.shape[0] % page_size, width))])
        for r in range(size):
            sharded._REQUEST_ROWS_PLAN.clear()
            parallel = SimpleNamespace(dcp_size=size, dcp_rank=r, dcp_group=_Group(sends, r))
            with mock.patch.object(sharded, "get_parallel", return_value=parallel):
                out = sharded.dcp_gather_request_rows(shares[r], lens, page_size=page_size)
                # the cache-rows form: this rank's rows scattered through a larger table
                table = torch.randint(0, 256, (shares[r].shape[0] * 2 + 3, width), dtype=torch.uint8)
                ids = torch.randperm(table.shape[0])[: shares[r].shape[0]]
                table[ids] = shares[r]
                by_ids = sharded.dcp_gather_request_rows(table, lens, row_ids=ids, page_size=page_size)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            torch.testing.assert_close(by_ids, expected, rtol=0, atol=0)

    def test_short_and_ragged_requests(self):
        # requests shorter than the group leave some ranks with no rows at all
        self._check([1, 3, 7, 8, 9, 16, 37], size=8, width=656, page_size=64)

    def test_word_and_byte_widths(self):
        self._check([5, 64, 130], size=8, width=68)
        self._check([5, 64, 130], size=2, width=5)

    def test_single_short_request(self):
        self._check([3], size=8, width=656)


if __name__ == "__main__":
    unittest.main()
