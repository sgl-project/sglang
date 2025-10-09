import pytest
import torch

from sglang.srt.layers.attention.nsa.fused_logits_head_gate import (
    fused_logits_head_gate,
)


def test_fused_logits_head_gate():
    n_heads = 64
    head_dim = 128
    softmax_scale = head_dim**-0.5

    weights = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    q_scale = torch.randn(128, 64, 1, dtype=torch.float32, device="cuda")

    # Reference
    weights_ref = weights.clone()
    weights_ref = weights_ref * n_heads**-0.5
    weights_ref = weights_ref.unsqueeze(-1) * q_scale * softmax_scale

    # Fused kernel
    weights_fused = fused_logits_head_gate(weights, q_scale, n_heads, softmax_scale)

    torch.testing.assert_close(weights_ref, weights_fused, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_fused_logits_head_gate()
