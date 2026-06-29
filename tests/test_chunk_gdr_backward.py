"""Test backward pass for SGLang's chunk_gated_delta_rule."""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/sgl-workspace/sglang/python")


def test_backward_basic():
    """Basic gradient flow test: verify gradients are non-zero and finite."""
    torch.manual_seed(42)
    B, T, H, K, V = 2, 128, 4, 64, 64
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=dtype), p=2, dim=-1).requires_grad_(True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype)).requires_grad_(True)
    beta = torch.randn(B, T, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
    h0 = torch.randn(B, H, V, K, device=device, dtype=dtype, requires_grad=True)

    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    initial_state_indices = torch.arange(B, device=device, dtype=torch.long)

    o, _, h = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=h0,
        initial_state_indices=initial_state_indices,
    )

    loss = o.sum()
    loss.backward()

    for name, param in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("h0", h0)]:
        assert param.grad is not None, f"{name}.grad is None"
        assert param.grad.isfinite().all(), f"{name}.grad has non-finite values"
        assert param.grad.abs().max() > 0, f"{name}.grad is all zeros"
        print(f"  {name}.grad: max={param.grad.abs().max().item():.6f}, "
              f"mean={param.grad.abs().mean().item():.6f}")

    print("PASSED: basic gradient flow")


def test_backward_vs_torch_reference():
    """Compare gradients against the pure-torch reference implementation from FLA."""
    torch.manual_seed(123)
    B, T, H, K, V = 1, 128, 2, 64, 32
    device = "cuda"
    dtype = torch.float32

    # torch_chunk_gated_delta_rule expects initial_state as [B, H, K, V]
    # SGLang's chunk_gated_delta_rule expects initial_state as [N, H, V, K]
    # Use V != K to catch layout bugs
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=dtype), p=2, dim=-1).detach().requires_grad_(True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype)).detach().requires_grad_(True)
    beta = torch.randn(B, T, H, device=device, dtype=dtype).sigmoid().detach().requires_grad_(True)
    # Shared state data in [B, H, K, V] layout (torch reference convention)
    h0_data = torch.randn(B, H, K, V, device=device, dtype=dtype)

    # Run torch reference (state layout: [B, H, K, V])
    sys.path.insert(0, "/workspace/Megatron-LM")
    from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    g_ref = g.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    h0_ref = h0_data.detach().clone().requires_grad_(True)

    o_ref, _ = torch_chunk_gated_delta_rule(
        query=q_ref, key=k_ref, value=v_ref, g=g_ref, beta=beta_ref,
        initial_state=h0_ref, chunk_size=64,
    )
    loss_ref = o_ref.sum()
    loss_ref.backward()

    # Run SGLang triton (state layout: [N, H, V, K])
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule

    q_tri = q.detach().clone().to(torch.bfloat16).requires_grad_(True)
    k_tri = k.detach().clone().to(torch.bfloat16).requires_grad_(True)
    v_tri = v.detach().clone().to(torch.bfloat16).requires_grad_(True)
    g_tri = g.detach().clone().to(torch.bfloat16).requires_grad_(True)
    beta_tri = beta.detach().clone().to(torch.bfloat16).requires_grad_(True)
    # Transpose to [N, H, V, K] for SGLang
    h0_tri = h0_data.detach().clone().transpose(-1, -2).to(torch.bfloat16).requires_grad_(True)

    initial_state_indices = torch.arange(B, device=device, dtype=torch.long)
    o_tri, _, h_tri = chunk_gated_delta_rule(
        q=q_tri, k=k_tri, v=v_tri, g=g_tri, beta=beta_tri,
        initial_state=h0_tri,
        initial_state_indices=initial_state_indices,
    )
    loss_tri = o_tri.float().sum()
    loss_tri.backward()

    print(f"\n  Forward diff: {(o_ref - o_tri.float()).abs().max().item():.6e}")

    for name, ref_g, tri_g in [
        ("dq", q_ref.grad, q_tri.grad),
        ("dk", k_ref.grad, k_tri.grad),
        ("dv", v_ref.grad, v_tri.grad),
        ("dg", g_ref.grad, g_tri.grad),
        ("dbeta", beta_ref.grad, beta_tri.grad),
        ("dh0", h0_ref.grad, h0_tri.grad.transpose(-1, -2) if h0_tri.grad is not None else None),
    ]:
        if ref_g is None or tri_g is None:
            print(f"  {name}: ref={ref_g is not None}, tri={tri_g is not None}")
            continue
        diff = (ref_g.float() - tri_g.float()).abs()
        rel = diff / (ref_g.float().abs() + 1e-8)
        print(f"  {name}: max_abs_diff={diff.max().item():.6e}, "
              f"max_rel_diff={rel.max().item():.6e}, "
              f"mean_abs_diff={diff.mean().item():.6e}")

    print("PASSED: backward comparison")


def test_backward_short_seq():
    """Short sequence (single chunk) test."""
    torch.manual_seed(7)
    B, T, H, K, V = 1, 32, 2, 64, 64
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=dtype), p=2, dim=-1).requires_grad_(True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype)).requires_grad_(True)
    beta = torch.randn(B, T, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)

    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    initial_state_indices = torch.arange(B, device=device, dtype=torch.long)

    o, _, h = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state_indices=initial_state_indices,
    )

    loss = o.sum()
    loss.backward()

    for name, param in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]:
        assert param.grad is not None, f"{name}.grad is None"
        assert param.grad.isfinite().all(), f"{name}.grad has non-finite values"
    print("PASSED: short sequence backward")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Basic gradient flow")
    print("=" * 60)
    test_backward_basic()

    print("\n" + "=" * 60)
    print("Test 2: Short sequence (single chunk)")
    print("=" * 60)
    test_backward_short_seq()

    print("\n" + "=" * 60)
    print("Test 3: Backward vs torch reference")
    print("=" * 60)
    test_backward_vs_torch_reference()

    print("\nAll tests passed!")
