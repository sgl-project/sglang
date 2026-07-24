"""Fused Walsh-Hadamard transform + block fp8 quant for the DSA indexer query.

Replaces the two-pass `act_quant(rotate_activation(q))` (hadamard writes a bf16
tensor, then act_quant reads it back and quantizes) with a single kernel: the
post-hadamard row stays in registers and is quantized in place, removing one
bf16 round-trip of the query per layer. Decode is bandwidth-bound, so dropping a
memory pass is the relevant win.

The hadamard is done as a matmul against the +/-1 Sylvester matrix (accumulated
in fp32) then scaled by 1/sqrt(N) -- numerically equivalent to
`fast_hadamard_transform(x, scale=N**-0.5)`. The quant matches tilelang
`act_quant`: block-wise (group=128) absmax, s = max(absmax, 1e-4)/fp8_max
(round_scale when scale_fmt is set), e4m3fn output + fp32 scale.
"""

from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

_IS_FNUZ = is_fp8_fnuz()
_FP8_DTYPE = torch.float8_e4m3fnuz if _IS_FNUZ else torch.float8_e4m3fn
_FP8_MAX = 224.0 if _IS_FNUZ else 448.0


@lru_cache(maxsize=4)
def _hadamard_pm1(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """+/-1 Sylvester (natural-order) Hadamard matrix [n, n]."""
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    assert h.shape[0] == n, "n must be a power of 2"
    return h.to(device=device, dtype=dtype)


@triton.jit
def _hadamard_quant_kernel(
    x_ptr,
    h_ptr,
    y_ptr,
    s_ptr,
    w_ptr,
    wo_ptr,
    M,
    inv_sqrt_n,
    fp8_max,
    fp8_min,
    softmax_scale,
    ROUND_SCALE: tl.constexpr,
    FUSE_WEIGHTS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    n = tl.arange(0, N)

    x = tl.load(
        x_ptr + rows[:, None] * N + n[None, :], mask=rmask[:, None], other=0.0
    ).to(
        h_ptr.dtype.element_ty
    )  # [BLOCK_M, N]
    h = tl.load(h_ptr + n[:, None] * N + n[None, :])  # [N, N] +/-1

    # hadamard: (x @ H_pm1) accumulated in fp32, then * 1/sqrt(N)
    had = tl.dot(x, h).to(tl.float32) * inv_sqrt_n  # [BLOCK_M, N]

    amax = tl.maximum(tl.max(tl.abs(had), axis=1), 1e-4)  # [BLOCK_M]
    if ROUND_SCALE:
        # match fast_round_scale: 2 ** ceil(log2(amax / fp8_max))
        s = tl.exp2(tl.ceil(tl.log2(amax / fp8_max)))
    else:
        s = amax / fp8_max
    y = tl.clamp(had / s[:, None], fp8_min, fp8_max).to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + rows[:, None] * N + n[None, :], y, mask=rmask[:, None])
    tl.store(s_ptr + rows, s, mask=rmask)

    # Optional fold of `_apply_q_scale_and_softmax_scale`: the indexer rescales the
    # per-(token,head) head-gate `weights` by this row's quant scale `s` (==q_scale)
    # and the softmax scale. `w`/`s` are row-aligned, so emit it here instead of a
    # separate elementwise launch.
    if FUSE_WEIGHTS:
        w = tl.load(w_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(wo_ptr + rows, w * s * softmax_scale, mask=rmask)


def fused_hadamard_act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    weights: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
):
    """Fused hadamard(scale=N**-0.5) + block fp8 quant.

    Drop-in for `act_quant(rotate_activation(x), block_size, scale_fmt)` when the
    last dim N is a power of 2 and == block_size (one quant group). Returns
    (fp8 tensor same shape as x, fp32 scale of shape x.shape[:-1] + (1,)).

    If `weights` (the indexer head-gate, shape x.shape[:-1]) is given, also folds
    `_apply_q_scale_and_softmax_scale` (weights * q_scale * softmax_scale) into the
    same kernel and returns it as a third value `(y, s, weights_scaled)` of shape
    x.shape[:-1] + (1,) -- eliminating the separate elementwise launch.
    """
    assert x.is_contiguous(), "input must be contiguous"
    N = x.size(-1)
    assert N == block_size, "fused path requires head_dim == block_size (one group)"
    assert (N & (N - 1)) == 0, "N must be a power of 2 for the hadamard"

    y = torch.empty_like(x, dtype=_FP8_DTYPE)
    s = x.new_empty(*x.size()[:-1], 1, dtype=torch.float32)
    x2 = x.view(-1, N)
    M = x2.shape[0]
    h = _hadamard_pm1(N, x.device, torch.bfloat16)

    fuse_weights = weights is not None
    if fuse_weights:
        assert (
            weights.shape == x.shape[:-1]
        ), f"weights {tuple(weights.shape)} must match x.shape[:-1] {tuple(x.shape[:-1])}"
        w_in = weights.contiguous().view(-1)
        w_out = torch.empty(M, dtype=torch.float32, device=x.device)
    else:
        w_in = x2  # unused dummy ptr
        w_out = s.view(-1)  # unused dummy ptr

    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M),)
    _hadamard_quant_kernel[grid](
        x2,
        h,
        y.view(-1, N),
        s.view(-1),
        w_in,
        w_out,
        M,
        float(N) ** -0.5,
        _FP8_MAX,
        -_FP8_MAX,
        float(softmax_scale),
        ROUND_SCALE=scale_fmt is not None,
        FUSE_WEIGHTS=fuse_weights,
        N=N,
        BLOCK_M=BLOCK_M,
    )
    if fuse_weights:
        return y, s, w_out.view(*weights.shape, 1)
    return y, s


# Standalone correctness check vs act_quant(rotate_activation(x)).
# Run on the box: PYTHONPATH=python python3 -m sglang.srt.layers.attention.dsa.triton_hadamard_quant
if __name__ == "__main__":
    from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation
    from sglang.srt.layers.attention.dsa.tilelang_kernel import act_quant

    torch.manual_seed(0)
    softmax_scale = 0.1337
    for shape in [(8, 32, 128), (4096, 32, 128), (1, 8, 128)]:
        x = (
            torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.3
        ).contiguous()
        y_ref, s_ref = act_quant(rotate_activation(x.clone()), 128, None)
        y_f, s_f = fused_hadamard_act_quant(x.clone(), 128, None)
        # compare fp8 bytes and scales
        same_fp8 = (
            (y_ref.view(torch.uint8) == y_f.view(torch.uint8)).float().mean().item()
        )
        s_maxrel = ((s_ref - s_f).abs() / (s_ref.abs() + 1e-9)).max().item()
        # dequantized cosine
        dq_ref = y_ref.float() * s_ref.float()
        dq_f = y_f.float() * s_f.float()
        cos = torch.nn.functional.cosine_similarity(
            dq_ref.flatten(), dq_f.flatten(), dim=0
        ).item()
        # weights fold vs _apply_q_scale_and_softmax_scale
        w = (torch.randn(*shape[:-1], device="cuda", dtype=torch.bfloat16)).contiguous()
        w_ref = w.unsqueeze(-1) * s_f * softmax_scale
        _, _, w_fused = fused_hadamard_act_quant(
            x.clone(), 128, None, weights=w, softmax_scale=softmax_scale
        )
        w_maxabs = (w_ref.float() - w_fused.float()).abs().max().item()
        print(
            f"shape={shape}: fp8_exact_frac={same_fp8:.5f} "
            f"scale_maxrel={s_maxrel:.3e} dequant_cos={cos:.6f} "
            f"weights_fold_maxabs={w_maxabs:.3e}"
        )
