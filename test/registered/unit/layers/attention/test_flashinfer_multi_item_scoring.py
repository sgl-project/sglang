"""Regression coverage for FlashInfer multi-item scoring positions."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.flashinfer_backend import (  # noqa: E402
    FlashInferAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_backend(*, enable_mis: bool = True) -> FlashInferAttnBackend:
    backend = object.__new__(FlashInferAttnBackend)
    backend.enable_mis = enable_mis
    return backend


def _make_forward_batch(
    *,
    seq_lens: list[int],
    delimiter_indices: list[list[int]] | None,
    positions: torch.Tensor,
    mrope_positions: torch.Tensor | None = None,
    mm_inputs: list[object | None] | None = None,
    prefix_lens: list[int] | None = None,
    mode: ForwardMode = ForwardMode.EXTEND,
) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=mode,
        multi_item_delimiter_indices=(
            [torch.tensor(indices, dtype=torch.int64) for indices in delimiter_indices]
            if delimiter_indices is not None
            else None
        ),
        extend_prefix_lens_cpu=(
            prefix_lens if prefix_lens is not None else [0] * len(seq_lens)
        ),
        extend_seq_lens_cpu=seq_lens,
        input_ids=torch.zeros(sum(seq_lens), dtype=torch.int64),
        positions=positions,
        mrope_positions=mrope_positions,
        mm_inputs=mm_inputs,
    )


class TestFlashInferMultiItemScoringPositions(CustomTestCase):
    def test_text_mrope_positions_follow_query_relative_item_positions(self):
        """MRoPE text items must follow the query, not preceding packed items."""
        batch = _make_forward_batch(
            seq_lens=[12],
            delimiter_indices=[[2, 5, 9, 11]],
            positions=torch.arange(12, dtype=torch.int64),
            mrope_positions=torch.arange(12, dtype=torch.int64).repeat(3, 1),
        )

        params = _make_backend()._process_multi_item_scoring(batch)

        self.assertTrue(params.is_enabled())
        expected = torch.tensor([0, 1, 1, 2, 3, 1, 2, 3, 4, 1, 2, 1])
        torch.testing.assert_close(batch.positions, expected)
        torch.testing.assert_close(
            batch.mrope_positions, expected.unsqueeze(0).expand(3, -1)
        )

    def test_empty_query_preserves_logical_positions_and_safe_rotary_indices(self):
        """Only MRoPE cache indices are clamped; logical delimiters stay masked."""
        for has_mrope in (False, True):
            with self.subTest(has_mrope=has_mrope):
                batch = _make_forward_batch(
                    seq_lens=[6],
                    delimiter_indices=[[0, 3, 5]],
                    positions=torch.arange(6, dtype=torch.int64),
                    mrope_positions=(
                        torch.arange(6, dtype=torch.int64).repeat(3, 1)
                        if has_mrope
                        else None
                    ),
                )

                params = _make_backend()._process_multi_item_scoring(batch)

                self.assertEqual(batch.positions.tolist(), [-1, 0, 1, -1, 0, -1])
                if has_mrope:
                    self.assertEqual(
                        batch.mrope_positions.tolist(), [[0, 0, 1, 0, 0, 0]] * 3
                    )
                else:
                    self.assertIsNone(batch.mrope_positions)
                self.assertEqual(params.prefix_len_ptr.tolist(), [0])
                self.assertEqual(
                    params.token_pos_in_items_ptr.tolist(), [0, 1, 2, 0, 1, 0]
                )

                # FlashInfer's causal MIS predicate must exclude delimiter keys.
                queries = torch.tensor([1, 2, 4])[:, None]
                keys = torch.arange(6)[None, :]
                within_item = params.token_pos_in_items_ptr.to(torch.int64)[queries]
                visible = (keys <= queries) & (
                    (keys < params.prefix_len_ptr.item())
                    | (queries < keys + within_item)
                )
                self.assertEqual(
                    visible.tolist(),
                    [
                        [False, True, False, False, False, False],
                        [False, True, True, False, False, False],
                        [False, False, False, False, True, False],
                    ],
                )

    def test_mixed_batch_preserves_multimodal_mrope_axes(self):
        """Preserve spatial axes and cached-prefix offsets across mixed requests."""
        for mm_delimiters in ([], [1, 3]):
            with self.subTest(mm_delimiters=mm_delimiters):
                positions = torch.tensor(
                    [5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8]
                )
                mrope_positions = positions.repeat(3, 1)
                mrope_positions[:, 6:10] = torch.tensor(
                    [[0, 1, 1, 1], [0, 2, 3, 4], [0, 4, 3, 2]]
                )
                multimodal_positions = mrope_positions[:, 6:10].clone()
                batch = _make_forward_batch(
                    seq_lens=[6, 4, 7],
                    delimiter_indices=[[2, 5], mm_delimiters, [1, 3, 6]],
                    positions=positions,
                    mrope_positions=mrope_positions,
                    mm_inputs=[None, object(), None],
                    prefix_lens=[5, 0, 2],
                )

                _make_backend()._process_multi_item_scoring(batch)

                self.assertEqual(
                    batch.mrope_positions[:, :6].tolist(), [[5, 6, 6, 7, 8, 6]] * 3
                )
                torch.testing.assert_close(
                    batch.mrope_positions[:, 6:10], multimodal_positions
                )
                self.assertEqual(
                    batch.mrope_positions[:, 10:].tolist(), [[2, 2, 3, 2, 3, 4, 2]] * 3
                )

    def test_non_mis_paths_leave_positions_unchanged(self):
        """Disabled MIS, ordinary prefill, and decode must not rewrite positions."""
        for enabled, mode, delimiters in (
            (False, ForwardMode.EXTEND, [[2, 5]]),
            (True, ForwardMode.DECODE, [[2, 5]]),
            (True, ForwardMode.EXTEND, None),
            (True, ForwardMode.EXTEND, [[]]),
        ):
            with self.subTest(enabled=enabled, mode=mode, delimiters=delimiters):
                positions = torch.arange(6, dtype=torch.int64)
                mrope_positions = torch.tensor(
                    [[0, 1, 2, 2, 2, 3], [0, 1, 2, 3, 4, 5], [0, 1, 4, 3, 2, 5]]
                )
                batch = _make_forward_batch(
                    seq_lens=[6],
                    delimiter_indices=delimiters,
                    positions=positions.clone(),
                    mrope_positions=mrope_positions.clone(),
                    mm_inputs=[object()],
                    mode=mode,
                )

                params = _make_backend(enable_mis=enabled)._process_multi_item_scoring(
                    batch
                )

                self.assertFalse(params.is_enabled())
                torch.testing.assert_close(batch.positions, positions)
                torch.testing.assert_close(batch.mrope_positions, mrope_positions)


if __name__ == "__main__":
    unittest.main()
