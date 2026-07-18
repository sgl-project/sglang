# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVarN Hadamard utilities (CPU-only)."""

import math

import pytest
import torch

from sglang.srt.layers.quantization.kvarn.hadamard import (
    build_hadamard,
    hadamard_cached,
)


class TestHadamard:
    @pytest.mark.parametrize("d", [16, 32, 64, 128, 256])
    def test_orthonormal(self, d):
        """H * H^T should be identity (orthonormal)."""
        H = build_hadamard(d, torch.device("cpu"))
        I = torch.mm(H, H.t())
        torch.testing.assert_close(I, torch.eye(d), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("d", [16, 32, 64, 128])
    def test_symmetric(self, d):
        """Sylvester Hadamard matrices are symmetric."""
        H = build_hadamard(d, torch.device("cpu"))
        torch.testing.assert_close(H, H.t(), rtol=1e-6, atol=1e-6)

    def test_cached(self):
        """hadamard_cached returns the same tensor for the same (d, device)."""
        H1 = hadamard_cached(128, "cpu")
        H2 = hadamard_cached(128, "cpu")
        assert H1 is H2

    def test_values(self):
        """Entries should be ±1/sqrt(d)."""
        d = 128
        H = build_hadamard(d, torch.device("cpu"))
        expected = 1.0 / math.sqrt(d)
        assert H.abs().min().item() == pytest.approx(expected, rel=1e-6)
        assert H.abs().max().item() == pytest.approx(expected, rel=1e-6)

    def test_rotation_preserves_norm(self):
        """Rotating a vector by H should preserve its L2 norm."""
        torch.manual_seed(42)
        d = 128
        H = build_hadamard(d, torch.device("cpu"))
        x = torch.randn(d)
        x_rot = torch.mv(H, x)
        assert x_rot.norm().item() == pytest.approx(x.norm().item(), rel=1e-4)

    def test_rotation_preserves_dot_product(self):
        """QK^T should be invariant under Hadamard rotation: (QH)(KH)^T = Q(HH^T)K^T = QK^T."""
        torch.manual_seed(42)
        d = 128
        H = build_hadamard(d, torch.device("cpu"))
        q = torch.randn(1, d)
        k = torch.randn(4, d)
        score_orig = torch.mm(q, k.t())
        q_rot = torch.mm(q, H)
        k_rot = torch.mm(k, H)
        score_rot = torch.mm(q_rot, k_rot.t())
        torch.testing.assert_close(score_rot, score_orig, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
