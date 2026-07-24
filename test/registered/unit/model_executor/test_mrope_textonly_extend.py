"""Unit tests for the vectorized text-only extend branch of
``ForwardBatch._compute_mrope_positions``.

The text-only extend branch used to build mrope positions with a Python list
comprehension over every extend token (``torch.tensor([[pos for pos in
range(...)]] * 3)``), which costs ~1.4 ms per 4k-token chunk on the scheduler
hot path. It is now a ``torch.arange(...).unsqueeze(0).expand(3, -1)``. These
tests pin the contract: identical values/shape, int64 dtype, and unchanged
behavior when text-only and multimodal requests share a batch.

Usage:
    python -m pytest test/registered/unit/model_executor/test_mrope_textonly_extend.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def _make_mm_input(prompt_len: int) -> SimpleNamespace:
    """Minimal stand-in for MultimodalInputs as read by mrope computation."""
    return SimpleNamespace(
        mrope_positions=torch.arange(prompt_len, dtype=torch.int64)
        .unsqueeze(0)
        .repeat(3, 1),
        mrope_position_delta=torch.tensor([[0]], dtype=torch.int64),
        mrope_position_delta_repeated_cache=None,
    )


def _fb(forward_mode: ForwardMode, seq_lens_cpu: torch.Tensor) -> ForwardBatch:
    fb = ForwardBatch.__new__(ForwardBatch)
    fb.forward_mode = forward_mode
    fb.seq_lens_cpu = seq_lens_cpu
    return fb


class TestTextOnlyExtendMropePositions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        cls.model_runner = SimpleNamespace(device="cpu")

    def test_text_only_extend_matches_enumeration(self):
        """One column per extend token: prefix, prefix+1, ..., on all 3 axes."""
        extend_len, prefix_len = 4096, 128
        batch = SimpleNamespace(
            multimodal_inputs=[None],
            extend_lens=[extend_len],
            prefix_lens=[prefix_len],
        )
        fb = _fb(ForwardMode.EXTEND, torch.tensor([prefix_len + extend_len]))

        fb._compute_mrope_positions(self.model_runner, batch)

        self.assertEqual(fb.mrope_positions.shape, (3, extend_len))
        self.assertEqual(fb.mrope_positions.dtype, torch.int64)
        expected = torch.arange(prefix_len, prefix_len + extend_len)
        for axis in range(3):
            self.assertTrue(torch.equal(fb.mrope_positions[axis], expected))

    def test_mixed_text_and_mm_batch_concatenates(self):
        """Text-only and multimodal requests in one extend batch: widths add up
        and each request keeps its own positions."""
        text_extend, mm_prompt = 7, 5
        batch = SimpleNamespace(
            multimodal_inputs=[None, _make_mm_input(mm_prompt)],
            extend_lens=[text_extend, mm_prompt],
            prefix_lens=[0, 0],
        )
        fb = _fb(ForwardMode.EXTEND, torch.tensor([text_extend, mm_prompt]))

        fb._compute_mrope_positions(self.model_runner, batch)

        self.assertEqual(fb.mrope_positions.shape, (3, text_extend + mm_prompt))
        self.assertTrue(
            torch.equal(fb.mrope_positions[0, :text_extend], torch.arange(text_extend))
        )
        self.assertTrue(
            torch.equal(
                fb.mrope_positions[:, text_extend:],
                _make_mm_input(mm_prompt).mrope_positions,
            )
        )

    def test_text_only_decode_single_column(self):
        """Decode path is untouched: one column with seq_len - 1."""
        seq_len = 25
        batch = SimpleNamespace(
            multimodal_inputs=[None],
            extend_lens=None,
            prefix_lens=None,
        )
        fb = _fb(ForwardMode.DECODE, torch.tensor([seq_len]))

        fb._compute_mrope_positions(self.model_runner, batch)

        self.assertEqual(fb.mrope_positions.shape, (3, 1))
        self.assertEqual(fb.mrope_positions[0, 0].item(), seq_len - 1)


if __name__ == "__main__":
    unittest.main()
