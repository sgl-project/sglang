"""CPU unit test for the DCP extend planner's prefix selection.

``prepare_decode_context_parallel_metadata`` (``layers/dcp/planner.py``) builds
this rank's prefix KV slot list. ``all_gather_kv_cache_for_dcp``
(``layers/dcp/comm.py``) then cuts that buffer row by row, with each row's own
length and a start of zero. The producer must select the slots with the rule the
consumer cuts with.

The test drives the real planner and replaces only its environment, so it pins
the selection in the shipped code instead of a copy of it.

Usage:
    python -m pytest test_planner.py -v
    python test_planner.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt import runtime_context as rc
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeKernel:
    """Stand-in for a triton kernel launch: `kernel[grid](*args)`."""

    def __init__(self, fill=None):
        self.fill = fill

    def __getitem__(self, _grid):
        def launch(*args):
            if self.fill is not None:
                self.fill(*args)

        return launch


class TestPlannerPrefixSelection(CustomTestCase):
    """The planner must give each row the prefix slots the gather expects.

    A mixed batch holds decode rows whose prefix length is not a multiple of
    ``dcp_size``. That is where one stride over the concatenated prefixes moves
    the later rows to the wrong phase.
    """

    def _row_slots(self, prefix_lens, dcp_size):
        # Keep each row's base a multiple of dcp_size, so a slot id has the same
        # phase as its position in the row. The widened allocator page gives the
        # same property in production.
        return [
            torch.arange(i * 4096 * dcp_size, i * 4096 * dcp_size + length)
            for i, length in enumerate(prefix_lens)
        ]

    def _plan(self, prefix_lens, dcp_size, dcp_rank):
        """Run the real planner on CPU and return its prefix selection."""
        from sglang.srt.layers.dcp import planner

        rows = self._row_slots(prefix_lens, dcp_size)
        slots = (torch.cat(rows) if rows else torch.empty(0, dtype=torch.int64)).to(
            torch.int32
        )
        seq_lens = torch.tensor([length + 1 for length in prefix_lens])

        def fill_prefix_indices(*args):
            # args[5] is dcp_prefix_kv_indices. A size change makes copy_ raise.
            args[5].copy_(slots)

        translator = SimpleNamespace(translate_dcp_read_ids=lambda ids: ids)
        with (
            rc.get_parallel().override(
                dcp_enabled=True, dcp_size=dcp_size, dcp_rank=dcp_rank
            ),
            patch.object(planner, "get_device", lambda: SimpleNamespace(device="cpu")),
            patch.object(planner, "create_dcp_kv_indices", _FakeKernel()),
            patch.object(
                planner,
                "get_attn_backend",
                lambda: SimpleNamespace(kv_index_translator=translator),
            ),
        ):
            metadata = planner.prepare_decode_context_parallel_metadata(
                seq_lens=seq_lens,
                extend_prefix_lens=torch.tensor(prefix_lens),
                extend_prefix_lens_cpu=list(prefix_lens),
                extend_seq_lens=torch.ones(len(prefix_lens), dtype=torch.int32),
                req_pool_indices=torch.arange(len(prefix_lens), dtype=torch.int32),
                req_to_token=torch.zeros((len(prefix_lens), 16), dtype=torch.int32),
                seq_lens_sum=int(seq_lens.sum()),
                kv_buffer_shape=torch.Size([1, 1, 8]),
                kv_cache_dtype=torch.bfloat16,
                kv_cache_device="cpu",
                create_chunked_prefix_cache_kv_indices_fn=_FakeKernel(
                    fill_prefix_indices
                ),
            )
        return rows, metadata.dcp_local_prefix_kv_indices

    def test_each_row_is_cut_from_its_own_start(self):
        cases = [
            # Aligned lengths must keep working: these pass before the fix too.
            [512, 1024],
            [8, 8, 8],
            # Lengths that are not multiples of dcp_size.
            [1004, 484],
            [1004, 484, 777],
            # A zero-length prefix, then an aligned row after unaligned ones.
            [3, 0, 5, 64],
        ]
        for dcp_size in [2, 4, 8]:
            for dcp_rank in range(dcp_size):
                for prefix_lens in cases:
                    rows, got = self._plan(prefix_lens, dcp_size, dcp_rank)
                    expected = torch.cat([row[dcp_rank::dcp_size] for row in rows]).to(
                        got.dtype
                    )
                    torch.testing.assert_close(
                        got,
                        expected,
                        msg=f"size={dcp_size} rank={dcp_rank} lens={prefix_lens}",
                    )

    def test_row_counts_match_the_gather(self):
        """Row counts must equal what `all_gather_kv_cache_for_dcp` cuts."""
        prefix_lens = [1004, 484, 777]
        for dcp_size in [2, 4, 8]:
            for dcp_rank in range(dcp_size):
                _, got = self._plan(prefix_lens, dcp_size, dcp_rank)
                expected = get_dcp_lens(torch.tensor(prefix_lens), dcp_size, dcp_rank)
                self.assertEqual(
                    got.numel(),
                    int(expected.sum()),
                    f"size={dcp_size} rank={dcp_rank}",
                )


if __name__ == "__main__":
    unittest.main()
