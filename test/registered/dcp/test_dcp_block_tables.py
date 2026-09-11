"""CPU unit test for the two DCP page tables over one allocation.

[Test Category] Correctness
[Test Target] layers/dcp/layout.py::dcp_local_kv_block_table

Under DCP the NPU attention backend has to hand two different page tables to two
different consumers reading the same allocation:

    indexer    replicated buffer, pages of page_size over the full virtual span
    latent KV  sharded pool,      pages of page_size over this rank's own rows

Both are derived from the same ``req_to_token`` rows. The interesting claim,
which this file pins, is that **the backend's existing expression already
produces the indexer's table and needs no change** -- only the second table is
new. Getting that backwards is the expensive mistake here: feeding the indexer a
rank-local table selects top-k from 1/c of the candidates and the model is
quietly wrong, with no crash and no shape error to catch it.

The fixture deliberately allocates pages out of order. With an identity free
list every wrong formula that is off by a factor of c still lines up on page 0,
and several of them agree with the right answer for the first request. Shuffled
physical ids are what separate them.

Usage:
    python -m pytest test_dcp_block_tables.py -v
    python test_dcp_block_tables.py
"""

import unittest

import torch

from sglang.srt.layers.dcp.layout import dcp_local_kv_block_table
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE_SIZE = 8


def _req_to_token(physical_pages, page_size, dcp_size):
    """One request's ``req_to_token`` row, as the allocator would fill it.

    The allocator pages in virtual space at ``page_size * dcp_size`` and writes
    that many consecutive virtual locations per page, so page with physical id
    ``q`` contributes ``q * P * c + arange(P * c)``.
    """
    stride = page_size * dcp_size
    return torch.cat(
        [q * stride + torch.arange(stride, dtype=torch.int64) for q in physical_pages]
    ).unsqueeze(0)


def _index_block_table(loc_rows, page_size):
    """The backend's existing expression, verbatim (ascend_backend.py:465-470)."""
    return loc_rows[:, ::page_size] // page_size


def _local(loc_rows, page_size, dcp_size, rank=0):
    with get_parallel().override(
        dcp_enabled=dcp_size > 1,
        attn_dcp_size=dcp_size,
        attn_dcp_rank=rank,
    ):
        return dcp_local_kv_block_table(loc_rows, page_size)


class TestDcpBlockTables(CustomTestCase):
    def test_without_dcp_the_two_tables_are_the_same(self):
        """At c == 1 there is one pool and one coordinate system, so the new
        expression must reduce to the existing one exactly -- that equality is
        what makes it safe to route the non-DCP path through it too."""
        pages = [5, 2, 9, 0, 7]
        rows = _req_to_token(pages, PAGE_SIZE, 1)
        self.assertTrue(
            torch.equal(_local(rows, PAGE_SIZE, 1), _index_block_table(rows, PAGE_SIZE))
        )

    def test_the_local_table_names_the_allocator_pages(self):
        pages = [5, 2, 9, 0, 7]
        for dcp_size in (2, 4, 8, 16):
            with self.subTest(dcp_size=dcp_size):
                rows = _req_to_token(pages, PAGE_SIZE, dcp_size)
                got = _local(rows, PAGE_SIZE, dcp_size)
                self.assertTrue(
                    torch.equal(got[0], torch.tensor(pages, dtype=torch.int64))
                )

    def test_the_existing_expression_still_yields_the_indexer_table(self):
        """The claim that the indexer call site needs no change.

        Indexer page ids run over the *virtual* span at page_size granularity,
        so allocator page q contributes c consecutive ids ``q*c .. q*c+c-1``.
        """
        pages = [5, 2, 9]
        for dcp_size in (2, 4, 8):
            with self.subTest(dcp_size=dcp_size):
                rows = _req_to_token(pages, PAGE_SIZE, dcp_size)
                got = _index_block_table(rows, PAGE_SIZE)[0]
                expected = torch.tensor(
                    [q * dcp_size + j for q in pages for j in range(dcp_size)],
                    dtype=torch.int64,
                )
                self.assertTrue(torch.equal(got, expected))

    def test_the_two_tables_are_not_accidentally_equal(self):
        # Guards against a change that quietly makes one alias the other. They
        # differ in both width (by c) and value, and a test suite that only
        # checked shapes would not notice.
        rows = _req_to_token([5, 2, 9], PAGE_SIZE, 4)
        local = _local(rows, PAGE_SIZE, 4)
        index = _index_block_table(rows, PAGE_SIZE)
        self.assertEqual(index.shape[1], local.shape[1] * 4)
        self.assertFalse(torch.equal(index[:, : local.shape[1]], local))

    def test_the_local_table_addresses_the_row_the_pool_actually_wrote(self):
        """The end-to-end check, against P2's write rule rather than against
        another formula.

        For every global position this rank owns, walking the local page table
        the way the attention kernel does -- ``table[page] * page_size +
        offset`` -- must land on ``loc // dcp_size``, which is where
        ``_resolve_dcp_write`` put that token. This is the assertion that ties
        P3a's page table to P2's storage geometry; everything else in this file
        compares one index expression to another.
        """
        pages = [5, 2, 9, 0]
        for dcp_size in (2, 4, 8):
            rows = _req_to_token(pages, PAGE_SIZE, dcp_size)
            for rank in range(dcp_size):
                with self.subTest(dcp_size=dcp_size, rank=rank):
                    table = _local(rows, PAGE_SIZE, dcp_size, rank)[0]
                    for pos, loc in enumerate(rows[0].tolist()):
                        if loc % dcp_size != rank:
                            continue
                        local_pos = pos // dcp_size
                        page, offset = divmod(local_pos, PAGE_SIZE)
                        self.assertEqual(
                            int(table[page]) * PAGE_SIZE + offset,
                            loc // dcp_size,
                            f"rank {rank} would read the wrong row for global "
                            f"position {pos} (loc {loc})",
                        )

    def test_a_partial_final_page_is_still_addressable(self):
        # Requests are not page-aligned. The table is built from a
        # `:seq_lens_max` slice, so the last column must still be present and
        # correct when the slice stops mid-page.
        dcp_size = 4
        rows = _req_to_token([5, 2, 9], PAGE_SIZE, dcp_size)
        stride = PAGE_SIZE * dcp_size
        truncated = rows[:, : 2 * stride + 3]
        table = _local(truncated, PAGE_SIZE, dcp_size)[0]
        self.assertEqual(table.tolist(), [5, 2, 9])


if __name__ == "__main__":
    unittest.main()
