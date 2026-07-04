import torch


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """CPU DSA FP8 index score, backed by torch.ops.sgl_kernel.fp8_index_cpu.

    Shapes (all contiguous):
        q   : [B, M, H, D]  float8_e4m3fn
        q_s : [B, M, H]     float32
        k   : [B, N, D]     float8_e4m3fn
        k_s : [B, N]        float32
    Returns:
        index_score : [B, M, N] float32
    """
    return torch.ops.sgl_kernel.fp8_index_cpu(
        q, q_s.to(torch.float32), k, k_s.to(torch.float32)
    )
