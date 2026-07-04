import torch


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """PyTorch CPU implementation of DSA FP8 index score.

    Semantics aligned with CUDA/HIP kernel path:
    1) fp8 q @ fp8 k -> fp32 logits
    2) relu(logits) * q_s
    3) sum over heads
    4) multiply by k_s

    Args:
        q   : [B, M, H, D]  float8_e4m3fn
        q_s : [B, M, H]     float32
        k   : [B, N, D]     float8_e4m3fn
        k_s : [B, N]        float32
    Returns:
        index_score : [B, M, N] float32
    """
    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)

    # (b, m, h, d) x (b, n, d) -> (b, m, n, h)
    logits = torch.einsum("bmhd,bnd->bmnh", q_f32, k_f32)
    logits = torch.relu(logits)
    logits = logits * q_s.to(torch.float32).unsqueeze(2)
    index_score = logits.sum(dim=-1)
    index_score = index_score * k_s.to(torch.float32).unsqueeze(1)
    return index_score
