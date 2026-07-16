from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.speculative.topk1 import draft_extend_topk1_postprocess
from sglang.test.test_utils import CustomTestCase


def _make_logits_with_unique_argmax(
    num_rows: int,
    vocab_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(
        (num_rows, vocab_size), dtype=dtype, device=device, generator=generator
    )
    max_indices = (
        torch.arange(num_rows, dtype=torch.long, device=device) * 9973 + 17
    ) % vocab_size
    logits.scatter_(1, max_indices[:, None], 1000.0)
    return logits


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDraftExtendTopk1Triton(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def _assert_matches_reference(
        self,
        logits: torch.Tensor,
        row_indices: torch.Tensor,
        hidden_states: torch.Tensor | None,
        dsa_topk_indices: torch.Tensor | None,
    ):
        source_snapshots = [
            tensor.clone() if tensor is not None else None
            for tensor in (logits, hidden_states, dsa_topk_indices)
        ]
        reference_indices = row_indices.to(torch.long)

        topk_p, topk_index, selected_hidden, selected_dsa = (
            draft_extend_topk1_postprocess(
                logits,
                row_indices,
                hidden_states,
                dsa_topk_indices,
            )
        )

        expected_topk_index = torch.argmax(
            logits[reference_indices], dim=-1, keepdim=True
        )
        torch.testing.assert_close(topk_index, expected_topk_index, rtol=0, atol=0)
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)

        for selected, source in (
            (selected_hidden, hidden_states),
            (selected_dsa, dsa_topk_indices),
        ):
            if source is None:
                self.assertIsNone(selected)
            else:
                self.assertIsNotNone(selected)
                self.assertEqual(selected.dtype, source.dtype)
                self.assertEqual(
                    selected.shape, (row_indices.shape[0], source.shape[1])
                )
                torch.testing.assert_close(
                    selected, source[reference_indices], rtol=0, atol=0
                )
                self.assertIsNot(selected, source)
                if selected.numel() > 0:
                    self.assertNotEqual(selected.data_ptr(), source.data_ptr())

        for source, snapshot in zip(
            (logits, hidden_states, dsa_topk_indices), source_snapshots
        ):
            if source is not None:
                torch.testing.assert_close(source, snapshot, rtol=0, atol=0)

    def test_matches_reference_across_shapes_and_dtypes(self):
        configs = [
            (9, 127, 33, 7, torch.float32),
            (11, 8193, 1025, 257, torch.float16),
            (7, 50000, 4097, 513, torch.bfloat16),
            (8, 154880, 6144, 2048, torch.float32),
        ]
        for num_rows, vocab_size, hidden_size, dsa_topk, dtype in configs:
            with self.subTest(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                dsa_topk=dsa_topk,
                dtype=dtype,
            ):
                logits = _make_logits_with_unique_argmax(
                    num_rows,
                    vocab_size,
                    dtype=dtype,
                    device=self.device,
                    seed=vocab_size,
                )
                hidden_states = torch.randn(
                    (num_rows, hidden_size),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                # The graph-owned DSA buffer can have a larger leading dimension
                # than the logits and hidden-state views returned by replay.
                dsa_topk_indices = torch.randint(
                    0,
                    32000,
                    (num_rows * 3, dsa_topk),
                    dtype=torch.int32,
                    device=self.device,
                )
                row_indices = torch.tensor(
                    [num_rows - 1, 1, num_rows - 1, 0],
                    dtype=torch.int64,
                    device=self.device,
                )
                self._assert_matches_reference(
                    logits, row_indices, hidden_states, dsa_topk_indices
                )

    def test_noncontiguous_int32_indices_and_row_strides(self):
        num_rows = 7
        vocab_size = 8193
        hidden_size = 2051
        dsa_topk = 1031

        dense_logits = _make_logits_with_unique_argmax(
            num_rows,
            vocab_size,
            dtype=torch.float32,
            device=self.device,
            seed=0,
        )
        logits_backing = torch.full(
            (num_rows, vocab_size + 31),
            2000.0,
            dtype=torch.float32,
            device=self.device,
        )
        logits_backing[:, :vocab_size] = dense_logits
        logits = logits_backing[:, :vocab_size]

        hidden_backing = torch.randn(
            (num_rows, hidden_size + 17),
            dtype=torch.float16,
            device=self.device,
        )
        hidden_states = hidden_backing[:, :hidden_size]
        dsa_backing = torch.randint(
            0,
            32000,
            (num_rows, dsa_topk + 13),
            dtype=torch.int32,
            device=self.device,
        )
        dsa_topk_indices = dsa_backing[:, :dsa_topk]
        index_pairs = torch.tensor(
            [[6, -1], [2, -1], [6, -1], [0, -1]],
            dtype=torch.int32,
            device=self.device,
        )
        row_indices = index_pairs[:, 0]

        self.assertFalse(logits.is_contiguous())
        self.assertFalse(hidden_states.is_contiguous())
        self.assertFalse(dsa_topk_indices.is_contiguous())
        self.assertFalse(row_indices.is_contiguous())
        self._assert_matches_reference(
            logits, row_indices, hidden_states, dsa_topk_indices
        )

    def test_trace_shapes_select_first_and_last_tree_rows(self):
        logits = _make_logits_with_unique_argmax(
            6,
            154880,
            dtype=torch.float32,
            device=self.device,
            seed=2,
        )
        hidden_states = torch.randn((6, 6144), dtype=torch.bfloat16, device=self.device)
        dsa_topk_indices = torch.randint(
            0, 32000, (720, 2048), dtype=torch.int32, device=self.device
        )

        for selected_row in (0, 5):
            with self.subTest(selected_row=selected_row):
                row_indices = torch.tensor(
                    [selected_row], dtype=torch.long, device=self.device
                )
                self._assert_matches_reference(
                    logits, row_indices, hidden_states, dsa_topk_indices
                )

    def test_optional_gather_outputs(self):
        logits = _make_logits_with_unique_argmax(
            6,
            50000,
            dtype=torch.float32,
            device=self.device,
            seed=1,
        )
        hidden_states = torch.randn((6, 6144), dtype=torch.bfloat16, device=self.device)
        dsa_topk_indices = torch.randint(
            0, 32000, (12, 2048), dtype=torch.int32, device=self.device
        )
        row_indices = torch.tensor([5, 2, 0], device=self.device)

        for hidden, dsa in (
            (hidden_states, None),
            (None, dsa_topk_indices),
            (None, None),
            (
                torch.empty((6, 0), dtype=torch.bfloat16, device=self.device),
                torch.empty((12, 0), dtype=torch.int32, device=self.device),
            ),
        ):
            with self.subTest(hidden=hidden is not None, dsa=dsa is not None):
                self._assert_matches_reference(logits, row_indices, hidden, dsa)

    def test_argmax_tie_breaks_to_lowest_vocab_index(self):
        vocab_size = 20000
        logits = torch.full(
            (3, vocab_size), -1.0, dtype=torch.float32, device=self.device
        )
        # Cover ties within one reduction split and across the 8192 boundary.
        logits[:, 11] = 10.0
        logits[:, 17] = 10.0
        logits[:, 8194] = 10.0
        row_indices = torch.tensor([2, 0, 2], device=self.device)

        topk_p, topk_index, selected_hidden, selected_dsa = (
            draft_extend_topk1_postprocess(logits, row_indices, None)
        )

        torch.testing.assert_close(
            topk_index,
            torch.full((3, 1), 11, dtype=torch.long, device=self.device),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)
        self.assertIsNone(selected_hidden)
        self.assertIsNone(selected_dsa)

    def test_empty_selection(self):
        logits = torch.empty((6, 1024), dtype=torch.float32, device=self.device)
        hidden_states = torch.empty((6, 64), dtype=torch.float16, device=self.device)
        dsa_topk_indices = torch.empty((12, 32), dtype=torch.int32, device=self.device)
        row_indices = torch.empty((0,), dtype=torch.long, device=self.device)

        topk_p, topk_index, selected_hidden, selected_dsa = (
            draft_extend_topk1_postprocess(
                logits, row_indices, hidden_states, dsa_topk_indices
            )
        )

        self.assertEqual(topk_p.shape, (0, 1))
        self.assertEqual(topk_index.shape, (0, 1))
        self.assertEqual(selected_hidden.shape, (0, 64))
        self.assertEqual(selected_dsa.shape, (0, 32))


if __name__ == "__main__":
    unittest.main()
