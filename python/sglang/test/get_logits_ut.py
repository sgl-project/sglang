import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self, d_in=2048, n_heads=128, softmax_scale=0.5):
        super().__init__()
        self.weights_proj = nn.Linear(d_in, 1024)
        self.n_heads = n_heads
        self.softmax_scale = softmax_scale

    def _get_logits_head_gate_orig(self, x: torch.Tensor, q_scale: torch.Tensor):
        weights = self.weights_proj(x)
        weights = weights * self.n_heads**-0.5
        q_scale = q_scale.unsqueeze(1)  # (B,1,1)
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    def _get_logits_head_gate_opt(self, x: torch.Tensor, q_scale: torch.Tensor):
        weights = self.weights_proj(x)
        q_scale = q_scale.unsqueeze(1)  # (B,1,1)
        scale_const = self.n_heads**-0.5 * q_scale * self.softmax_scale  # (B,1,1)
        weights = weights.unsqueeze(-1) * scale_const  # (B,1024,1)
        return weights


def main():
    torch.manual_seed(0)
    model = DummyModel(d_in=2048, n_heads=128, softmax_scale=0.5)
    x = torch.randn(128, 2048)  # batch=128, d_in=2048
    q_scale = torch.randn(128, 1)

    import time

    start = time.time()
    for _ in range(1000):
        out_orig = model._get_logits_head_gate_orig(x, q_scale)
    print("Original version time:", time.time() - start)

    start = time.time()
    for _ in range(1000):
        out_opt = model._get_logits_head_gate_opt(x, q_scale)
    print("Optimized version time:", time.time() - start)

    print("Difference:", (out_orig - out_opt).abs().max().item())
    assert torch.allclose(out_orig, out_opt), "Mismatch between original and optimized"


if __name__ == "__main__":
    main()


"""
Original version time: 0.49235057830810547
Optimized version time: 0.4087331295013428
Difference: 1.4901161193847656e-08
"""
