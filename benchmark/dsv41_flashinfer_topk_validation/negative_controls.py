"""Negative controls: each new test must fail when the behavior it guards is broken.

Run on the devbox with the branch tree on PYTHONPATH from the sglang tree root:
    python3 /hai-workspace/dsv41-fi-topk/tools/negative_controls.py
Each control patches one guarded behavior, runs the guarding test, and expects a
failure; the script exits non-zero if any control passes.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, "test/registered/kernels/ops/attention")
sys.path.insert(0, "test/registered/unit/layers")

from sglang.srt.layers.attention.dsv4.v41_indexer import selector as selector_mod  # noqa: E402


def run(module_name: str, test_id: str) -> unittest.TestResult:
    module = importlib.import_module(module_name)
    suite = unittest.defaultTestLoader.loadTestsFromName(test_id, module)
    result = unittest.TestResult()
    suite.run(result)
    return result


def _identity_order(self, blocks):
    return blocks


_original_init = selector_mod.TopKSelector.__init__


def _overlapping_init(self, backend):
    _original_init(self, backend)
    self.can_overlap_streams = True


def _no_tie_break_modes():
    modes = dict(_original_modes())
    modes["tie_break"] = 0
    return modes


_original_modes = selector_mod._flashinfer_modes


def _sorting_dense(self, scores, k, *, sorted=False):
    import flashinfer

    return flashinfer.top_k(scores.contiguous(), k, sorted=sorted, **_original_modes())


def _unsorted_dense(self, scores, k, *, sorted=False):
    import flashinfer

    return flashinfer.top_k(scores.contiguous(), k, sorted=False, **_original_modes())


def _values_only_sorted_dense(self, scores, k, *, sorted=False):
    import flashinfer

    values, indices = flashinfer.top_k(
        scores.contiguous(), k, sorted=False, **_original_modes()
    )
    if sorted:
        values = values.sort(dim=-1, descending=True).values
    return values, indices


def _chunked_paged(*, slice_table=True, chunk=True):
    """The unfused branch of TopKSelector._paged with one chunking defect."""

    def _paged(self, scores, lens, page_table, out, page_size, *, out_raw_indices=None):
        assert not self.flashinfer_fused
        lens = lens.reshape(-1).to(torch.int32)
        row_bytes = max(1, scores.shape[1] * scores.element_size())
        rows_per_chunk = (
            max(1, selector_mod._UNFUSED_CHUNK_BYTES // row_bytes)
            if chunk
            else scores.shape[0]
        )
        for start in range(0, scores.shape[0], rows_per_chunk):
            rows = slice(start, start + rows_per_chunk)
            columns = self.backend.topk_func(scores[rows], lens[rows], out.shape[1])
            table = page_table[rows] if slice_table else page_table
            out[rows].copy_(selector_mod._through_table(columns, table, page_size))
            if out_raw_indices is not None:
                out_raw_indices[rows].copy_(columns)

    return _paged


def _v4_cached_paged(self, scores, lens, page_table, out, page_size, *, out_raw_indices=None):
    """The earlier unfused branch: V4's transforms, which cache aranges per width."""
    from sglang.srt.layers.attention.dsv4.indexer import (
        topk_transform_flashinfer_unfused,
        topk_transform_pytorch_vectorized,
    )

    transform = (
        topk_transform_pytorch_vectorized
        if self.backend.is_torch()
        else topk_transform_flashinfer_unfused
    )
    transform(
        scores,
        lens.reshape(-1).to(torch.int32),
        page_table,
        out,
        page_size,
        out_raw_indices=out_raw_indices,
    )


def _translation_keeps_padding(columns, table, page_size):
    pages = table.gather(1, (columns.clamp_min(0) // page_size).long())
    return (pages * page_size + columns % page_size).to(torch.int32)


CONTROLS = [
    (
        "dense-block publishes skip order_candidates",
        mock.patch.object(selector_mod.TopKSelector, "order_candidates", _identity_order),
        "test_dsv4_nonpaged_indexer",
        "TestDSV41FlashInferTopK.test_flashinfer_dense_blocks_publish_ascending_ids",
    ),
    (
        "FlashInfer top_k is asked to sort (no graph-safe path on SM120)",
        mock.patch.object(selector_mod.TopKSelector, "dense", _sorting_dense),
        "test_dsv4_nonpaged_indexer",
        "TestDSV41FlashInferTopK.test_flashinfer_modes_reach_every_selection",
    ),
    (
        "order_candidates is the identity",
        mock.patch.object(selector_mod.TopKSelector, "order_candidates", _identity_order),
        "test_dsv41_topk_selector",
        "TestDSV41TopKSelector.test_flashinfer_ties_follow_logical_positions",
    ),
    (
        "FlashInfer drops the tie-break mode",
        mock.patch.object(selector_mod, "_flashinfer_modes", _no_tie_break_modes),
        "test_dsv41_topk_selector",
        "TestDSV41TopKSelector.test_flashinfer_ties_follow_logical_positions",
    ),
    (
        "FlashInfer overlaps the block selection on the side stream",
        mock.patch.object(selector_mod.TopKSelector, "__init__", _overlapping_init),
        "test_dsv4_nonpaged_indexer",
        "TestDSV41FlashInferTopK.test_flashinfer_block_selection_stays_on_the_main_stream",
    ),
    (
        "dense(sorted=True) returns FlashInfer's unsorted order",
        mock.patch.object(selector_mod.TopKSelector, "dense", _unsorted_dense),
        "test_dsv4_nonpaged_indexer",
        "TestDSV41FlashInferTopK.test_flashinfer_modes_reach_every_selection",
    ),
    (
        "dense(sorted=True) sorts values without regathering indices",
        mock.patch.object(selector_mod.TopKSelector, "dense", _values_only_sorted_dense),
        "test_dsv4_nonpaged_indexer",
        "TestDSV41FlashInferTopK.test_flashinfer_modes_reach_every_selection",
    ),
    (
        "unfused selections go back through V4's per-width arange cache",
        mock.patch.object(selector_mod.TopKSelector, "_paged", _v4_cached_paged),
        "test_dsv4_nonpaged_indexer",
        "TestDSV41FlashInferTopK.test_unfused_selections_hold_no_per_width_state",
    ),
    (
        "unfused chunks translate through the unsliced page table",
        mock.patch.object(
            selector_mod.TopKSelector, "_paged", _chunked_paged(slice_table=False)
        ),
        "test_dsv41_topk_selector",
        "TestDSV41TopKSelector.test_unfused_selections_split_rows_into_chunks",
    ),
    (
        "unfused selections ignore the scratch cap (one chunk)",
        mock.patch.object(selector_mod.TopKSelector, "_paged", _chunked_paged(chunk=False)),
        "test_dsv41_topk_selector",
        "TestDSV41TopKSelector.test_unfused_selections_split_rows_into_chunks",
    ),
    (
        "unfused translation turns -1 padding into slots",
        mock.patch.object(selector_mod, "_through_table", _translation_keeps_padding),
        "test_dsv41_topk_selector",
        "TestDSV41TopKSelector.test_unfused_selections_split_rows_into_chunks",
    ),
]


def main() -> int:
    survived = []
    for name, patch, module_name, test_id in CONTROLS:
        with patch:
            result = run(module_name, test_id)
        caught = bool(result.failures or result.errors)
        print(f"{'CAUGHT ' if caught else 'MISSED '} {name}: {test_id} "
              f"(failures={len(result.failures)}, errors={len(result.errors)}, run={result.testsRun})")
        for _, trace in (result.failures + result.errors)[:1]:
            print("    " + trace.strip().splitlines()[-1])
        if not caught:
            survived.append(name)
    print("all controls caught" if not survived else f"controls missed: {survived}")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
