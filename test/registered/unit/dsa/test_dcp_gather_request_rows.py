"""DCP prefill gathers rebuild request rows from the ranks' interleaved shares."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa import dsa_dcp_sharded as cuda_sharded
from sglang.srt.layers.attention.dsa import dsa_dcp_sharded_hip as sharded
from sglang.srt.layers.dcp import comm
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


def _shares(rows, lens, size):
    """Each rank's rows, request by request, positions p with p % size == rank."""
    starts = torch.cumsum(lens, 0) - lens
    return [
        torch.cat(
            [rows[s + r : s + n : size] for s, n in zip(starts.tolist(), lens.tolist())]
        )
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
        plans = [
            sharded.request_rows_plan(lens, size, r, torch.device("cpu"))
            for r in range(size)
        ]
        for r, (send, _, num_slots) in enumerate(plans):
            self.assertEqual(shares[r].shape[0], int(get_dcp_lens(lens, size, r).sum()))
            self.assertEqual(send.shape[0], num_slots)
        sends = [
            s[p[0]] if s.shape[0] else s.new_empty((p[2], width))
            for s, p in zip(shares, plans)
        ]
        expected = torch.cat(
            [rows, rows.new_zeros((-rows.shape[0] % page_size, width))]
        )
        for r in range(size):
            sharded._REQUEST_ROWS_PLAN.clear()
            parallel = SimpleNamespace(
                dcp_size=size, dcp_rank=r, dcp_group=_Group(sends, r)
            )
            with mock.patch.object(sharded, "get_parallel", return_value=parallel):
                out = sharded.dcp_gather_request_rows(
                    shares[r], lens, page_size=page_size
                )
                # the cache-rows form: this rank's rows scattered through a larger table
                table = torch.randint(
                    0, 256, (shares[r].shape[0] * 2 + 3, width), dtype=torch.uint8
                )
                ids = torch.randperm(table.shape[0])[: shares[r].shape[0]]
                table[ids] = shares[r]
                by_ids = sharded.dcp_gather_request_rows(
                    table, lens, row_ids=ids, page_size=page_size
                )
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            torch.testing.assert_close(by_ids, expected, rtol=0, atol=0)

    def test_short_and_ragged_requests(self):
        # DCP8 KV rows (656 bytes, page 64); requests shorter than the group leave
        # some ranks with no rows at all
        self._check([1, 3, 7, 8, 9, 16, 37], size=8, width=656, page_size=64)

    def _check_cuda(self, lengths, size, width, chunk_slots, *, page_size=1):
        lens = torch.tensor(lengths)
        rows = torch.randint(
            0,
            256,
            (sum(lengths), width),
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(42),
        )
        shares = _shares(rows, lens, size)
        padded = torch.cat(
            [
                torch.cat((part, part.new_zeros((-part.shape[0] % size, width))))
                for part in rows.split(lengths)
            ]
        )
        sends = padded.view(-1, size, width).transpose(0, 1)
        expected_calls = None
        expected = torch.cat((rows, rows.new_zeros((-len(rows) % page_size, width))))
        platform = SimpleNamespace(is_cuda=True, is_gfx95=False, is_sm100=False)
        for rank, local in enumerate(shares):
            offset = 0
            calls = []

            def all_gather_into_tensor(out, inp):
                nonlocal offset
                self.assertLessEqual(inp.shape[0], chunk_slots)
                self.assertEqual(inp.dtype, torch.uint8)
                end = offset + inp.shape[0]
                recv = out.view(size, *inp.shape)
                recv.copy_(sends[:, offset:end])
                recv[rank].copy_(inp)
                calls.append(inp.shape[0])
                offset = end

            parallel = SimpleNamespace(
                dcp_enabled=True,
                dcp_size=size,
                dcp_rank=rank,
                dcp_group=SimpleNamespace(
                    all_gather_into_tensor=all_gather_into_tensor
                ),
            )
            storage = local.new_empty((len(local), width + 1))
            local = storage[:, 1:].copy_(local)
            kwargs = {"page_size": page_size} if page_size != 1 else {}
            with (
                mock.patch.object(cuda_sharded, "get_parallel", return_value=parallel),
                mock.patch.object(comm, "get_parallel", return_value=parallel),
                mock.patch.object(cuda_sharded, "get_platform", return_value=platform),
                mock.patch.object(comm, "get_platform", return_value=platform),
                mock.patch.object(
                    cuda_sharded,
                    "_DCP_GATHER_CHUNK_BYTES",
                    chunk_slots * size * width,
                    create=True,
                ),
            ):
                actual = cuda_sharded.dcp_gather_request_rows(local, lens, **kwargs)
            self.assertEqual(offset, sends.shape[1])
            if offset <= chunk_slots:
                self.assertEqual(calls, [offset])
            else:
                self.assertLessEqual(2 * max(calls), offset)
            if expected_calls is None:
                expected_calls = calls
            self.assertEqual(calls, expected_calls)
            self.assertTrue(actual.is_contiguous())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cuda_bounds_collective_size(self):
        # Split inside a request, mix requests within a chunk, and trim ragged
        # tails without dropping the positions owned by a later rank.
        for size, width in ((2, 133), (4, 132), (8, 656)):
            with self.subTest(size=size, width=width):
                self._check_cuda([1, 0, 3, 49, 2, 8, 16], size, width, chunk_slots=3)

    def test_cuda_one_collective_and_page_padding(self):
        self._check_cuda([1, 0, 7, 9], 8, 656, chunk_slots=4)
        self._check_cuda([1, 0, 7, 9], 8, 656, chunk_slots=3, page_size=64)

    def test_cuda_empty_batch_skips_collective(self):
        group = mock.Mock()
        parallel = SimpleNamespace(
            dcp_enabled=True, dcp_size=8, dcp_rank=7, dcp_group=group
        )
        platform = SimpleNamespace(is_cuda=True, is_gfx95=False, is_sm100=False)
        with (
            mock.patch.object(cuda_sharded, "get_platform", return_value=platform),
            mock.patch.object(cuda_sharded, "get_parallel", return_value=parallel),
            mock.patch.object(comm, "get_parallel", return_value=parallel),
            mock.patch.object(comm, "get_platform", return_value=platform),
        ):
            for lengths in ([], [0, 0]):
                lens = torch.tensor(lengths, dtype=torch.int64)
                rows = torch.empty((0, 656), dtype=torch.uint8)
                self.assertEqual(
                    cuda_sharded.dcp_gather_request_rows(rows, lens).shape, rows.shape
                )
        group.all_gather_into_tensor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
