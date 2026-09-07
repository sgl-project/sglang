import unittest

import torch
from dsv41_rope import (
    apply_rotary_emb,
    apply_rotary_emb_tail,
    precompute_freqs_cis,
)
from ref_loader import RefTestCase, assert_equal, requires_ref


@requires_ref
class TestRope(RefTestCase):
    def _freqs(self, original_seq_len):
        args = (64, 128, original_seq_len, 160000.0, 16, 32, 1)
        return precompute_freqs_cis(*args), self.model.precompute_freqs_cis(*args)

    def test_freqs_cis(self):
        for original_seq_len in (0, 65536):
            ours, ref = self._freqs(original_seq_len)
            assert_equal(torch.view_as_real(ours), torch.view_as_real(ref))

    def test_apply(self):
        freqs, _ = self._freqs(65536)
        for shape in ((2, 7, 64), (2, 7, 4, 64)):
            x = torch.randn(shape)
            for inverse in (False, True):
                expected = x.clone()
                self.model.apply_rotary_emb(expected, freqs[3:10], inverse)
                assert_equal(apply_rotary_emb(x, freqs[3:10], inverse), expected)

    def test_apply_tail(self):
        freqs, _ = self._freqs(0)
        x = torch.randn(2, 7, 4, 128)
        expected = x.clone()
        self.model.apply_rotary_emb(expected[..., -64:], freqs[:7])
        assert_equal(apply_rotary_emb_tail(x, 64, freqs[:7]), expected)


if __name__ == "__main__":
    unittest.main()
