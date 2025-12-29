from typing import Optional, Tuple

import torch


def apply_flashinfer_rope_qk_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    head_size: Optional[int] = None,
    is_neox: bool = False,
    positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(
            f"Expected q/k to be 4D [bsz, seqlen, nheads, head_size], "
            f"got q:{tuple(q.shape)} k:{tuple(k.shape)}"
        )
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} vs {k.shape}")

    if not (isinstance(cos_sin_cache, torch.Tensor) and cos_sin_cache.dim() == 2):
        raise ValueError("cos_sin_cache must be a 2D torch.Tensor")

    bsz, seqlen, nheads, d = q.shape
    if head_size is None:
        head_size = d
    if head_size != d:
        raise ValueError(f"head_size mismatch: inferred {d}, but head_size={head_size}")

    try:
        from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
    except Exception as e:
        raise RuntimeError(
            "flashinfer is required for apply_flashinfer_rope_qk_inplace. "
            "Please install flashinfer or disable this optimization."
        ) from e

    if positions is None:
        pos_1d = torch.arange(seqlen, device="cpu", dtype=torch.long)
        positions = pos_1d if bsz == 1 else pos_1d.repeat(bsz)
    else:
        if not (isinstance(positions, torch.Tensor) and positions.dtype == torch.long and positions.dim() == 1):
            raise ValueError("positions must be a 1D torch.long Tensor")
        if positions.numel() != bsz * seqlen:
            raise ValueError(
                f"positions length must be bsz*seqlen={bsz*seqlen}, got {positions.numel()}"
            )

    positions = positions.to(q.device, non_blocking=True)

    q_flat = q.reshape(bsz * seqlen, nheads * d).contiguous()
    k_flat = k.reshape(bsz * seqlen, nheads * d).contiguous()
    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=q_flat,
        key=k_flat,
        head_size=d,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )
    return q_flat.view(bsz, seqlen, nheads, d), k_flat.view(bsz, seqlen, nheads, d)

