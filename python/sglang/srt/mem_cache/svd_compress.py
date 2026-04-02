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

    temporal[:, idx] = approx.to(device=orig_device, dtype=orig_dtype)
