import math

import torch


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    """Complex rotary factors [seqlen, dim // 2]. original_seq_len > 0 enables YaRN:
    frequencies whose wavelength fits the training context are kept, those far
    beyond it are divided by factor, with a linear ramp between beta_fast and beta_slow."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return (
                dim
                * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Rotate adjacent element pairs of x ([b, s, d] or [b, s, h, d]) by freqs_cis [s, d // 2].
    inverse conjugates the rotation. Returns a new tensor in x's dtype."""
    dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1) // 2)
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1) // 2)
    return torch.view_as_real(xc * freqs_cis).flatten(-2).to(dtype)


def apply_rotary_emb_tail(
    x: torch.Tensor, rope_dim: int, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """apply_rotary_emb on the last rope_dim features only; the rest pass through."""
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    return torch.cat([head, apply_rotary_emb(tail, freqs_cis, inverse)], dim=-1)
