import torch
import torch.nn.functional as F


def original_calc_linear(q, k, v):
    kvsum = k.transpose(-1, -2) @ v
    ksum = torch.sum(k, dim=-2, keepdim=True)
    return (q @ kvsum) / (1e-5 + (q * ksum).sum(dim=-1, keepdim=True))


def torch_calc_linear(q, k, v):
    kv = torch.matmul(k.transpose(-1, -2), v)
    k_sum = torch.sum(k, dim=-2, keepdim=True)

    q_kv = torch.matmul(q, kv)
    q_k_sum = torch.matmul(q, k_sum.transpose(-1, -2))

    denominator = q_k_sum + 1e-5

    return q_kv / denominator


for seed in [0, 42, 128, 1024]:
    # set seed
    torch.manual_seed(seed)

    # test case
    # B, H, L, D = 1, 12, 32760, 128
    B, H, L, D = 1, 40, 32760, 128
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    q = F.softmax(q, dim=-1)
    k = F.softmax(k, dim=-1)

    # result
    original = original_calc_linear(q, k, v)
    optimized = torch_calc_linear(q, k, v)

    # diff
    abs_error = torch.abs(original - optimized)
    rel_error = abs_error / (torch.abs(original) + 1e-8)

    print(f"=== seed {seed} result ===")
    print(f"Maximum Absolute Error: {abs_error.max().item():.2e}")
    print(f"Mean Absolute Error: {abs_error.mean().item():.2e}")
    print(f"Maximum Relative Error: {rel_error.max().item():.2e}")
    print(f"Mean Relative Error: {rel_error.mean().item():.2e}")

    # Cosine Similarity
    original_flat = original.flatten()
    optimized_flat = optimized.flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(
        original_flat, optimized_flat, dim=0
    )
    print(f"Cosine Similarity: {cosine_sim.item():.6f}")

    # range
    print(f"original range: [{original.min().item():.4f}, {original.max().item():.4f}]")
    print(
        f"optimized range: [{optimized.min().item():.4f}, {optimized.max().item():.4f}]"
    )
    print(f"=========================")
