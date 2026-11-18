"""Independent PyTorch references and rounding bounds for the Muse Spark Quark PTPC-FP8 tests.

Quark PTPC: weights are FP8 e4m3 codes times one fp32 scale per output channel;
activations are quantized per token at run time with scale = absmax(row) / 448.

Bounds follow the standard floating-point error model (Higham, "Accuracy and
Stability of Numerical Algorithms", ch. 3) with BF16 unit roundoff u = 2^-8 and
FP32 unit roundoff u32 = 2^-24:

* a GEMM row accumulated in FP32 over K products, scaled twice in FP32 and
  rounded once to BF16 deviates from the exact row r by at most
  ``u*|r| + (1+u)*(gamma_K*|abs_r| + 2*u32*|r|)`` (``abs_r`` = |A| @ |B|^T,
  gamma_K = K*u32 / (1 - K*u32)); this holds for any summation order.
* the statistical variant replaces gamma_K by ``lam*sqrt(K)*u32``: it assumes
  the accumulation rounding errors are independent and zero-mean, so the error
  of a K-term sum grows like sqrt(K); with lam = 6 a row exceeds it with
  probability of order 1e-9. It is an engineering check, not a hardware bound.
"""

import math

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
U_BF16 = 2.0**-8
U_FP32 = 2.0**-24
STAT_LAMBDA = 6.0


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row FP8 codes and fp32 scales of ``weight`` [..., N, K]."""
    w = weight.float()
    scale = w.abs().amax(dim=-1).clamp_min(1e-12) / FP8_MAX
    codes = (w / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return codes, scale


def per_token_dequant(x: torch.Tensor) -> torch.Tensor:
    """What a per-token dynamic FP8 quantization of ``x`` [M, K] represents, fp32 (x / s, round to nearest even)."""
    xf = x.float()
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    return (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8).float() * scale


def fp8_grid(device) -> torch.Tensor:
    """Every finite FP8 e4m3 value, ascending (float64)."""
    values = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(FP8).double()
    return values[torch.isfinite(values)].unique().sort().values.to(device)


def quantizer_report(x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor) -> dict:
    """How production per-token codes/scales relate to the exact PTPC definition.

    The scale must be absmax/448 within one FP32 ulp. With q = x / scale (exact, float64) the code must be a
    neighbour of q; it may differ from round-to-nearest-even of q only where q lies within 2*u32*|q| of a
    rounding midpoint, because production forms q in FP32 (x * (1/s) or x / s, at most two roundings)."""
    x64 = x.double()
    s64 = scale.double().reshape(-1, 1)
    s32 = scale.float().reshape(-1, 1)
    ulp = (torch.nextafter(s32, torch.full_like(s32, math.inf)) - s32).double()
    exact = x64.abs().amax(dim=-1, keepdim=True) / FP8_MAX
    q = (x64 / s64).clamp(-FP8_MAX, FP8_MAX)
    grid = fp8_grid(x.device)
    hi_idx = torch.searchsorted(grid, q).clamp(max=grid.numel() - 1)
    lo_idx = (hi_idx - 1).clamp(min=0)
    lo, hi = grid[lo_idx], grid[hi_idx]
    lo = torch.where(grid[hi_idx] == q, q, lo)
    mid = (lo + hi) / 2
    hi_even = (hi.float().to(FP8).view(torch.uint8) & 1) == 0
    rne = torch.where(
        q < mid, lo, torch.where(q > mid, hi, torch.where(hi_even, hi, lo))
    )
    got = codes.double()
    apart = got != rne
    near_mid = (q - mid).abs() <= 2 * U_FP32 * q.abs()
    return {
        "scale_err_ulps_max": float(((s64 - exact).abs() / ulp).max()),
        "not_a_neighbour": int(((got != lo) & (got != hi)).sum()),
        "differs_from_rne": int(apart.sum()),
        "differs_from_rne_away_from_midpoint": int((apart & ~near_mid).sum()),
        "elements": x.numel(),
    }


def gemm_row_bounds(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> dict:
    """Per-row error of ``out`` against the exact ``a @ b.T`` (a, b already dequantized, float64) and both bounds."""
    ref = a @ b.t()
    abs_ref = a.abs() @ b.abs().t()
    k = a.shape[1]
    gamma = k * U_FP32 / (1 - k * U_FP32)
    ref_norm, abs_norm = ref.norm(dim=-1), abs_ref.norm(dim=-1)
    err = (out.double() - ref).norm(dim=-1)
    tail = 2 * U_FP32 * ref_norm
    return {
        "err": err,
        "ref_norm": ref_norm,
        "cancellation": abs_norm / ref_norm.clamp_min(1e-300),
        "k": k,
        "deterministic": U_BF16 * ref_norm + (1 + U_BF16) * (gamma * abs_norm + tail),
        "statistical": U_BF16 * ref_norm
        + (1 + U_BF16) * (STAT_LAMBDA * math.sqrt(k) * U_FP32 * abs_norm + tail),
    }


def finalize_reference(
    y: torch.Tensor,
    scores: torch.Tensor,
    slots: torch.Tensor,
    m: int,
    norm_weight: torch.Tensor | None,
    gain: float,
    eps: float,
    has_norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Muse finalize of per-slot outputs ``y`` [slots, H] (float64): per slot z = BF16(y * BF16(w + gain)),
    n = z / sqrt(mean(z^2) + eps); per token the score-weighted sum. Returns (out [m, H], C_row [m]) with
    C_row = sum_s w_s |n_s| / |sum_s w_s n_s|, the cancellation of the weighted sum."""
    z = y.double()
    if has_norm:
        if norm_weight is not None:
            scale = (norm_weight.float() + gain).to(norm_weight.dtype)
            z = (y.float() * scale.float()).to(torch.bfloat16).double()
        z = z / torch.sqrt(z.pow(2).mean(dim=-1, keepdim=True) + eps)
    weights = scores.double()
    tokens = slots // scores.shape[-1] if scores.dim() == 2 else slots
    flat = weights.reshape(-1)[slots]
    out = torch.zeros(m, y.shape[-1], dtype=torch.float64, device=y.device)
    out.index_add_(0, tokens, flat.unsqueeze(-1) * z)
    mass = torch.zeros(m, dtype=torch.float64, device=y.device)
    mass.index_add_(0, tokens, flat.abs() * z.norm(dim=-1))
    return out, mass / out.norm(dim=-1).clamp_min(1e-300)


def finalize_bound(
    ref: torch.Tensor, cancellation: torch.Tensor, h: int
) -> torch.Tensor:
    """Row error bound of the finalize kernel: C_row * (2u + gamma_H + u32) + u, relative to |ref_row|.

    2u covers the BF16 per-slot rounding the kernel reads plus the final BF16 store, gamma_H the FP32 mean of
    squares over H, u32 the FP32 rsqrt / multiply-add of each slot."""
    gamma = h * U_FP32 / (1 - h * U_FP32)
    return (cancellation * (2 * U_BF16 + gamma + U_FP32) + U_BF16) * ref.norm(dim=-1)


def muse_moe_fp8_reference(
    hidden_states: torch.Tensor,
    topk_scores: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    local_expert_start: int,
    post_norm_weight: torch.Tensor | None,
    eps: float,
    post_norm_gain: float = 0.0,
) -> torch.Tensor:
    """End-to-end oracle with its own quantization and one BF16 rounding per stage, [M, H] bf16.

    It re-derives every intermediate instead of reading production's, so it differs from the production path by
    requantized BF16 ulps (production SiLU-and-mul rounds twice); use it to characterize gross errors."""
    M = hidden_states.shape[0]
    E, H = w13.shape[0], w2.shape[1]
    out = torch.zeros(M, H, dtype=torch.float32, device=hidden_states.device)
    x = per_token_dequant(hidden_states)
    if post_norm_weight is not None:
        norm_scale = (post_norm_weight.float() + post_norm_gain).to(
            post_norm_weight.dtype
        )
    for local in range(E):
        token, slot = torch.nonzero(
            topk_ids == local_expert_start + local, as_tuple=True
        )
        if token.numel() == 0:
            continue
        w1 = w13[local].float() * w13_scale[local].unsqueeze(-1)
        gate_up = (x[token] @ w1.t()).to(torch.bfloat16).float()
        gate, up = gate_up.chunk(2, dim=-1)
        inter = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        wd = w2[local].float() * w2_scale[local].unsqueeze(-1)
        y = (per_token_dequant(inter) @ wd.t()).to(torch.bfloat16)
        if post_norm_weight is not None:
            y = (y.float() * norm_scale.float()).to(torch.bfloat16)
        y = y.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
        out.index_add_(0, token, topk_scores[token, slot].float().unsqueeze(-1) * y)
    return out.to(hidden_states.dtype)
