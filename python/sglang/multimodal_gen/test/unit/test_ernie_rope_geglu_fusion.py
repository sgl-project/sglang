import unittest

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.dits.ernie_image import (
    _ernie_geglu,
    _ernie_rope,
    _precompute_rope_cos_sin,
)


def _reference_rotary_bshd(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """The pre-fusion eager chain (per-layer cos/sin) verbatim."""
    freqs = freqs.permute(1, 0, 2, 3)
    rot_dim = freqs.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]

    cos_ = torch.cos(freqs).to(x.dtype)
    sin_ = torch.sin(freqs).to(x.dtype)

    x1, x2 = x_rot.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)

    x_rot = x_rot * cos_ + x_rotated * sin_
    return torch.cat((x_rot, x_pass), dim=-1)


def _make_freqs(batch: int, seq: int, rot: int, device) -> torch.Tensor:
    # EmbedND3 layout: (S, B, 1, rot), interleave-duplicated frequencies.
    uniq = torch.randn(seq, batch, 1, rot // 2, device=device) * 3.0
    return torch.stack([uniq, uniq], dim=-1).reshape(seq, batch, 1, rot)


class TestErnieRopeFusion(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_rope_matches_prefusion_chain(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        for batch, seq, heads, head_dim, rot in [
            (2, 257, 16, 128, 64),
            (1, 64, 3, 128, 128),
            (2, 33, 8, 64, 56),
        ]:
            x = torch.randn(
                batch, seq, heads, head_dim, device=device, dtype=torch.bfloat16
            )
            freqs = _make_freqs(batch, seq, rot, device)
            reference = _reference_rotary_bshd(x, freqs)

            cos_, sin_ = _precompute_rope_cos_sin(freqs, torch.bfloat16)
            fused = _ernie_rope(x, cos_, sin_)
            self.assertTrue(
                torch.equal(reference, fused),
                f"rope mismatch at {(batch, seq, heads, head_dim, rot)}",
            )

    def test_eager_fallback_matches_prefusion_chain_cpu(self):
        torch.manual_seed(1)
        x = torch.randn(2, 17, 4, 32, dtype=torch.float32)
        freqs = _make_freqs(2, 17, 16, x.device)
        reference = _reference_rotary_bshd(x, freqs)
        cos_, sin_ = _precompute_rope_cos_sin(freqs, torch.float32)
        self.assertTrue(torch.equal(reference, _ernie_rope(x, cos_, sin_)))


class TestErnieGegluFusion(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_geglu_matches_eager(self):
        torch.manual_seed(0)
        gate_up = torch.randn(2, 129, 2 * 3584, device="cuda", dtype=torch.bfloat16)
        gate, up = gate_up.chunk(2, dim=-1)
        reference = up * F.gelu(gate)
        self.assertTrue(torch.equal(reference, _ernie_geglu(gate_up)))

    def test_eager_fallback_cpu(self):
        torch.manual_seed(1)
        gate_up = torch.randn(3, 8, dtype=torch.float32)
        gate, up = gate_up.chunk(2, dim=-1)
        self.assertTrue(torch.equal(up * F.gelu(gate), _ernie_geglu(gate_up)))


if __name__ == "__main__":
    unittest.main()
