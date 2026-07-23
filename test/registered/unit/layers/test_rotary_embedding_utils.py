"""Numerical tests for the RoPE apply/reverse rotation helpers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.rotary_embedding.utils import (
    apply_rotary_emb,
    reverse_rotary_emb,
)
from sglang.test.test_utils import CustomTestCase


def _cos_sin(positions: torch.Tensor, head_size: int, base: float = 10000.0):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)
    )
    freqs = torch.einsum("i,j->ij", positions.to(torch.float32), inv_freq)
    return freqs.cos(), freqs.sin()


class TestReverseRotaryEmb(CustomTestCase):
    HEAD_SIZE = 64
    NUM_HEADS = 4

    def _rand_x(self, num_tokens: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(1234)
        return torch.randn(
            num_tokens,
            self.NUM_HEADS,
            self.HEAD_SIZE,
            generator=generator,
            dtype=torch.float32,
        )

    def test_apply_then_reverse_roundtrips(self):
        """reverse_rotary_emb must be the exact inverse of apply_rotary_emb
        (rotations are orthonormal: cos^2 + sin^2 = 1). A sign or
        interleave-layout error in either helper silently corrupts every
        position-corrected KV entry; this pins the algebra for both the
        Neox (chunked-half) and GPT-J (interleaved) layouts."""
        positions = torch.tensor([0, 1, 17, 255, 4095])
        cos, sin = _cos_sin(positions, self.HEAD_SIZE)
        x = self._rand_x(len(positions))
        for is_neox_style in (True, False):
            with self.subTest(is_neox_style=is_neox_style):
                rotated = apply_rotary_emb(x, cos, sin, is_neox_style)
                # Non-degenerate check: rotation must change the nonzero
                # positions, otherwise the roundtrip below would also pass
                # for a pair of no-op helpers.
                self.assertFalse(torch.allclose(rotated[1:], x[1:], atol=1e-3))
                restored = reverse_rotary_emb(rotated, cos, sin, is_neox_style)
                torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)

    def test_reverse_then_reapply_relocates_positions(self):
        """The KV realization contract: K rotated at donor position p, then
        reverse-rotated at p and re-rotated at target position q, must equal
        K rotated at q directly. If reverse and apply disagree on layout or
        angle sign, relocated donor KV carries the wrong positional phase."""
        donor_pos = torch.tensor([8, 9, 10, 11])
        target_pos = torch.tensor([789, 790, 791, 792])
        cos_p, sin_p = _cos_sin(donor_pos, self.HEAD_SIZE)
        cos_q, sin_q = _cos_sin(target_pos, self.HEAD_SIZE)
        x = self._rand_x(len(donor_pos))
        for is_neox_style in (True, False):
            with self.subTest(is_neox_style=is_neox_style):
                at_p = apply_rotary_emb(x, cos_p, sin_p, is_neox_style)
                unrotated = reverse_rotary_emb(at_p, cos_p, sin_p, is_neox_style)
                at_q = apply_rotary_emb(unrotated, cos_q, sin_q, is_neox_style)
                expected = apply_rotary_emb(x, cos_q, sin_q, is_neox_style)
                torch.testing.assert_close(at_q, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
