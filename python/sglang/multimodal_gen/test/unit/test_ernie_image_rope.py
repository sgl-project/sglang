import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.ernie_image import (
    _apply_rotary_bshd,
    _rotary_cos_sin_bshd,
)


def _apply_rotary_ref(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    freqs = freqs.permute(1, 0, 2, 3)
    rot_dim = freqs.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]

    cos_ = torch.cos(freqs).to(x.dtype)
    sin_ = torch.sin(freqs).to(x.dtype)

    x1, x2 = x_rot.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat((x_rot * cos_ + x_rotated * sin_, x_pass), dim=-1)


class TestErnieImageRope(unittest.TestCase):
    def test_precomputed_cos_sin_matches_inline_trig(self):
        torch.manual_seed(0)
        x = torch.randn(2, 7, 4, 80, dtype=torch.float32)
        freqs = torch.randn(7, 2, 1, 64, dtype=torch.float32)

        cos_, sin_ = _rotary_cos_sin_bshd(freqs, x.dtype)
        actual = _apply_rotary_bshd(x, cos_, sin_)
        expected = _apply_rotary_ref(x, freqs)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
