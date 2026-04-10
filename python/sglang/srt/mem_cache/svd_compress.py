import logging

import torch

logger = logging.getLogger(__name__)


def apply_svd_lowrank_to_temporal_state(
    temporal: torch.Tensor,
    pool_idx: torch.Tensor,
    rank: int = 16,
    oversample: int = 4,
    niter: int = 1,
) -> None:
    """Apply low-rank SVD approximation to a request's Mamba temporal state in-place.

    Operates on the last two dimensions (head_dim, state_size) of the temporal
    cache tensor, compressing them to the given rank via randomized SVD on CPU
    in float32 (which is faster than GPU for this operation).

    Args:
        temporal: Full temporal cache, shape [num_layers, pool_size+1, heads, head_dim, state_size].
        pool_idx: Scalar tensor — index into dim 1 for the target request slot.
        rank: Target rank for the low-rank approximation.
        oversample: Extra dimensions sampled for numerical stability.
        niter: Number of power iterations for the randomized SVD.
    """
    idx = pool_idx.item()
    state = temporal[:, idx]

    orig_device = state.device
    orig_dtype = state.dtype

    cpu_state = state.detach().to(device="cpu", dtype=torch.float32)

    # Total energy = squared Frobenius norm of original state
    total_energy = torch.sum(cpu_state**2).item()

    q = min(rank + oversample, min(cpu_state.shape[-2:]))

    try:
        u, s, v = torch.svd_lowrank(cpu_state, q=q, niter=niter)
    except RuntimeError as e:
        logger.warning(
            "SVD failed for mamba temporal state (pool_idx=%d), skipping: %s", idx, e
        )
        return

    u, s, v = u[..., :rank], s[..., :rank], v[..., :rank]
    approx = (u * s.unsqueeze(-2)) @ v.transpose(-2, -1)

    # Energy retained by the rank-k approximation = sum of squared singular values kept.
    # Energy lost = 1 - retained/total.  Relative Frobenius error = ||orig - approx||/||orig||.
    retained_energy = torch.sum(s**2).item()
    error_norm = torch.norm(cpu_state - approx).item()
    orig_norm = total_energy**0.5

    if total_energy > 0:
        energy_ratio = retained_energy / total_energy
        relative_error = error_norm / orig_norm
    else:
        energy_ratio = 1.0
        relative_error = 0.0

    logger.info(
        "SVD compression (pool_idx=%d, rank=%d): "
        "energy_retained=%.6f, energy_lost=%.6f, "
        "relative_frobenius_error=%.6f, "
        "orig_norm=%.4f, error_norm=%.4f, "
        "state_shape=%s, "
        "u_shape=%s, s_shape=%s, v_shape=%s",
        idx,
        rank,
        energy_ratio,
        1.0 - energy_ratio,
        relative_error,
        orig_norm,
        error_norm,
        tuple(cpu_state.shape),
        tuple(u.shape),
        tuple(s.shape),
        tuple(v.shape),
    )

    temporal[:, idx] = approx.to(device=orig_device, dtype=orig_dtype)
