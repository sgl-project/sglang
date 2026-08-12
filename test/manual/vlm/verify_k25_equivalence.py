"""Prove the two GPU-only rewrites in the K2.5 port are equivalent to main.

1. normalize_and_patchify(scale/bias) == pad -> /255 -> (x-mean)*inv_std -> patchify
2. apply_fused_qk_complex_rope_inplace == the torch complex reference
"""

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.vision_rope import (
    apply_fused_qk_complex_rope_inplace,
    prepare_fused_qk_complex_rope_inplace,
)
from sglang.kernels.ops.mm.process import normalize_and_patchify

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
ASYM_MEAN = [0.481, 0.457, 0.408]
ASYM_STD = [0.268, 0.261, 0.275]


def reference_preprocess(batch_u8, mean, std, patch_size, padded_h, padded_w):
    """Exactly what main does, in main's order."""
    image_mean = torch.tensor(mean, device="cuda", dtype=torch.float32).view(1, 3, 1, 1)
    image_std_inv = (1.0 / torch.tensor(std, device="cuda", dtype=torch.float32)).view(
        1, 3, 1, 1
    )
    x = batch_u8.float()
    pad_h = padded_h - x.shape[-2]
    pad_w = padded_w - x.shape[-1]
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
    x = x / 255.0
    x = (x - image_mean) * image_std_inv
    B, C, H, W = x.shape
    gh, gw = H // patch_size, W // patch_size
    x = x.view(B, C, gh, patch_size, gw, patch_size)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C, patch_size, patch_size)


def check_patchify():
    print("== normalize_and_patchify vs main's pad/normalize/patchify ==")
    torch.manual_seed(0)
    cases = [
        # (H, W, padded_h, padded_w, patch, mean, std, label)
        (32, 24, 32, 24, 8, MEAN, STD, "no padding, symmetric norm"),
        (30, 22, 32, 24, 8, MEAN, STD, "padded, symmetric norm"),
        (30, 22, 32, 24, 8, ASYM_MEAN, ASYM_STD, "padded, per-channel norm"),
        (64, 64, 64, 64, 16, ASYM_MEAN, ASYM_STD, "large patch"),
    ]
    ok = True
    for h, w, ph, pw, patch, mean, std, label in cases:
        raw = torch.randint(0, 256, (3, 3, h, w), dtype=torch.uint8, device="cuda")
        ref = reference_preprocess(raw, mean, std, patch, ph, pw)

        scale = torch.tensor(
            [1.0 / (255.0 * s) for s in std], device="cuda", dtype=torch.float32
        ).view(1, 3, 1, 1)
        bias = torch.tensor(
            [-m / s for m, s in zip(mean, std)], device="cuda", dtype=torch.float32
        ).view(1, 3, 1, 1)
        got = normalize_and_patchify(raw.float(), scale, bias, patch, ph, pw)

        max_abs = (got - ref).abs().max().item()
        # The padded rows must carry -mean/std, not zero.
        pad_ok = True
        if ph > h or pw > w:
            pad_ok = torch.allclose(
                got.flatten()[(got - ref).abs().argmax()],
                ref.flatten()[(got - ref).abs().argmax()],
                atol=1e-5,
            )
        good = max_abs < 1e-5 and pad_ok
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label:32s} max|d|={max_abs:.3e}")
    return ok


def check_padded_value_is_not_zero():
    """The old pipeline padded in raw space, so pad cells become -mean/std."""
    print("== padded cells carry -mean/std, not 0 ==")
    raw = torch.full((1, 3, 8, 8), 128, dtype=torch.uint8, device="cuda")
    scale = torch.tensor(
        [1.0 / (255.0 * s) for s in ASYM_STD], device="cuda", dtype=torch.float32
    ).view(1, 3, 1, 1)
    bias = torch.tensor(
        [-m / s for m, s in zip(ASYM_MEAN, ASYM_STD)],
        device="cuda",
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    got = normalize_and_patchify(raw.float(), scale, bias, 8, 16, 16)
    # patch index 1 is the (row 0, col 1) patch -- entirely padding.
    pad_patch = got[0, 1]
    expected = bias.view(3, 1, 1).expand(3, 8, 8)
    good = torch.allclose(pad_patch, expected, atol=1e-6)
    print(
        f"  {'PASS' if good else 'FAIL'}  pad cell = {pad_patch[0, 0, 0].item():.6f}, "
        f"expected -mean/std = {expected[0, 0, 0].item():.6f}"
    )
    return good


def reference_rope(xq, xk, freqs_cis):
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def check_rope():
    print("== fused vision RoPE vs the torch complex reference ==")
    torch.manual_seed(0)
    ok = True
    for dtype, tol in ((torch.bfloat16, 8e-3), (torch.float16, 2e-3)):
        for tokens, heads, head_dim in ((1024, 16, 72), (4096, 8, 128), (37, 4, 64)):
            xq = torch.randn(tokens, heads, head_dim, device="cuda", dtype=dtype)
            xk = torch.randn(tokens, heads, head_dim, device="cuda", dtype=dtype)
            angle = torch.randn(tokens, head_dim // 2, device="cuda")
            freqs_cis = torch.polar(torch.ones_like(angle), angle)

            ref_q, ref_k = reference_rope(xq, xk, freqs_cis)
            prepared = prepare_fused_qk_complex_rope_inplace(freqs_cis)
            got_q, got_k = apply_fused_qk_complex_rope_inplace(
                xq.clone(), xk.clone(), prepared
            )

            dq = (got_q.float() - ref_q.float()).abs().max().item()
            dk = (got_k.float() - ref_k.float()).abs().max().item()
            good = dq < tol and dk < tol
            ok &= good
            print(
                f"  {'PASS' if good else 'FAIL'}  {str(dtype):16s} "
                f"t={tokens:5d} h={heads:2d} d={head_dim:3d}  "
                f"max|dq|={dq:.2e} max|dk|={dk:.2e}"
            )
    return ok


if __name__ == "__main__":
    results = [check_patchify(), check_padded_value_is_not_zero(), check_rope()]
    print()
    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    raise SystemExit(0 if all(results) else 1)
