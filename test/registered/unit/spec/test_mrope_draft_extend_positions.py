"""Draft-extend gives a multimodal request one mRoPE position per extended token.

EAGLE's draft-extend re-enters the draft model with ``num_draft_tokens`` tokens
per request. A multimodal request's precomputed ``mrope_positions`` only span
the original prompt, so once decoding has moved past it the prompt slice is
empty and the code falls back. The fallback has to widen to ``extend_seq_len``
positions: emitting a single decode-shaped position leaves ``mrope_positions``
with ``batch_size`` entries where the forward pass has ``sum(extend_lens)``
tokens, and the rotary embedding then reads out of bounds.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PROMPT_LEN = 16
MROPE_DELTA = 5


def _mm_input():
    """A multimodal request whose mRoPE positions only span its prompt."""
    return SimpleNamespace(
        mrope_positions=torch.arange(PROMPT_LEN).unsqueeze(0).repeat(3, 1),
        mrope_position_delta=torch.tensor([[MROPE_DELTA]]),
        mrope_position_delta_repeated_cache=None,
    )


class TestMropeDraftExtendPositions(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(model_path="dummy")
        override.install()
        self.addCleanup(override.restore)

    def _run(self, batch_size, num_draft_tokens, seq_len):
        forward_batch = ForwardBatch.__new__(ForwardBatch)
        forward_batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        forward_batch.seq_lens_cpu = torch.full(
            (batch_size,), seq_len, dtype=torch.int64
        )
        batch = SimpleNamespace(
            multimodal_inputs=[_mm_input() for _ in range(batch_size)],
            extend_lens=[num_draft_tokens] * batch_size,
            prefix_lens=[seq_len] * batch_size,
        )
        forward_batch._compute_mrope_positions(SimpleNamespace(device="cpu"), batch)
        return forward_batch.mrope_positions

    def test_one_position_per_extended_token(self):
        batch_size, num_draft_tokens = 3, 4
        positions = self._run(batch_size, num_draft_tokens, seq_len=PROMPT_LEN + 20)
        self.assertEqual(positions.shape[0], 3)
        self.assertEqual(positions.shape[1], batch_size * num_draft_tokens)

    def test_positions_are_the_text_only_ones_shifted_by_the_delta(self):
        num_draft_tokens, seq_len = 4, PROMPT_LEN + 20
        positions = self._run(1, num_draft_tokens, seq_len)

        # Text-only extend covers [prefix_len, prefix_len + extend_seq_len);
        # a multimodal request sits mrope_position_delta above that.
        expected = torch.arange(seq_len, seq_len + num_draft_tokens) + MROPE_DELTA
        for section in range(3):
            self.assertTrue(torch.equal(positions[section], expected))

    def test_single_token_extend_matches_the_decode_position(self):
        seq_len = PROMPT_LEN + 20
        extend = self._run(1, 1, seq_len)

        forward_batch = ForwardBatch.__new__(ForwardBatch)
        decode = forward_batch._expand_mrope_from_input(_mm_input(), seq_len + 1)

        self.assertTrue(torch.equal(extend, decode))

    def test_prompt_slice_still_wins_while_it_covers_the_extend(self):
        # Prefill-shaped extend: the precomputed prompt positions are in range
        # and must be used as-is rather than going through the fallback.
        batch_size, extend_len, prefix_len = 2, 4, 8
        forward_batch = ForwardBatch.__new__(ForwardBatch)
        forward_batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        forward_batch.seq_lens_cpu = torch.full(
            (batch_size,), prefix_len + extend_len, dtype=torch.int64
        )
        batch = SimpleNamespace(
            multimodal_inputs=[_mm_input() for _ in range(batch_size)],
            extend_lens=[extend_len] * batch_size,
            prefix_lens=[prefix_len] * batch_size,
        )
        forward_batch._compute_mrope_positions(SimpleNamespace(device="cpu"), batch)

        expected = torch.arange(prefix_len, prefix_len + extend_len)
        self.assertEqual(
            forward_batch.mrope_positions.shape[1], batch_size * extend_len
        )
        self.assertTrue(
            torch.equal(forward_batch.mrope_positions[0][:extend_len], expected)
        )


if __name__ == "__main__":
    unittest.main()
