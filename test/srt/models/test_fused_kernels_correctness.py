"""Comprehensive correctness check for both fused kernels.

References match production:
  hc_head      : eager-torch impl (the original DeepseekV4Model.hc_head)
                 — there is no sgl_kernel optimized hc_head; eager IS production.
  mhc_pre+norm : TileLang mhc_pre()  →  sgl_kernel RMSNorm
                 — the actual production path.

Sweep: hidden_size ∈ {4096, 7168}, T ∈ {1, 8, 64, 256, 1024, 8192, 32768},
seeds ∈ {0, 17, 42, 12345}. Reports max_abs / max_rel per (shape, seed) and
asserts a bf16 tolerance (abs < 6e-2 ≈ 8 ULPs, OR rel < 1.5e-1).
"""

from __future__ import annotations
import sys
import torch
import torch.nn.functional as F
from sglang.srt.layers.layernorm import RMSNorm


HC_MULT = 4
HIDDENS = (4096, 7168)
TS = (1, 8, 64, 256, 1024, 8192, 32768)
SEEDS = (0, 17, 42, 12345)
ABS_TOL = 6e-2   # ~8 ULPs of bf16 near magnitude 1
REL_TOL = 1.5e-1


def bf16_check(name, T, hidden, seed, ref, fused):
    abs_err = (fused.float() - ref.float()).abs()
    rel_err = abs_err / (ref.float().abs() + 1e-6)
    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()
    ok = (max_abs < ABS_TOL) or (max_rel < REL_TOL)
    flag = "OK " if ok else "FAIL"
    print(
        f"  [{flag}] {name:<14} T={T:>5d} hidden={hidden} seed={seed:<5d}  "
        f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
    )
    return ok


# =============== hc_head ===============

def hc_head_ref(x, fn, scale, base, norm_eps=1e-6, hc_eps=1e-6):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


def hc_head_fused(x, fn, scale, base, norm_eps=1e-6, hc_eps=1e-6):
    from sglang.srt.layers.mhc_head import fused_hc_head
    return fused_hc_head(x.contiguous(), fn, scale, base, norm_eps, hc_eps)


def test_hc_head():
    print("=" * 70)
    print("hc_head — fused triton vs eager-torch reference")
    print("=" * 70)
    failures = 0
    for hidden in HIDDENS:
        for T in TS:
            for seed in SEEDS:
                torch.manual_seed(seed)
                x = torch.randn(T, HC_MULT, hidden, dtype=torch.bfloat16, device="cuda") * 0.5
                fn = torch.randn(HC_MULT, HC_MULT * hidden, dtype=torch.float32, device="cuda") * (
                    1.0 / (HC_MULT * hidden) ** 0.5
                )
                scale = torch.tensor([0.7], dtype=torch.float32, device="cuda")
                base = torch.randn(HC_MULT, dtype=torch.float32, device="cuda") * 0.1
                y_r = hc_head_ref(x, fn, scale, base)
                y_f = hc_head_fused(x, fn, scale, base)
                if not bf16_check("hc_head", T, hidden, seed, y_r, y_f):
                    failures += 1
    return failures


# =============== mhc_pre + RMSNorm ===============

def mhc_ref(x, fn, hc_scale, hc_base, rms):
    from sglang.srt.layers.mhc import mhc_pre
    post, comb, y = mhc_pre(
        residual=x, fn=fn, hc_scale=hc_scale, hc_base=hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0, sinkhorn_repeat=20,
    )
    return post, comb, rms(y)


def mhc_fused(x, fn, hc_scale, hc_base, rms):
    from sglang.srt.layers.mhc import mhc_pre
    return mhc_pre(
        residual=x, fn=fn, hc_scale=hc_scale, hc_base=hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0, sinkhorn_repeat=20,
        norm_weight=rms.weight.data, norm_eps=rms.variance_epsilon,
    )


def test_mhc():
    print("=" * 70)
    print("mhc_pre + RMSNorm — fused TileLang vs (mhc_pre + sgl_kernel RMSNorm)")
    print("=" * 70)
    failures = 0
    hc_mult3 = HC_MULT * (2 + HC_MULT)
    for hidden in HIDDENS:
        rms = RMSNorm(hidden, eps=1e-6).to("cuda").to(torch.bfloat16)
        for T in TS:
            for seed in SEEDS:
                torch.manual_seed(seed)
                x = torch.randn(T, HC_MULT, hidden, dtype=torch.bfloat16, device="cuda") * 0.5
                fn = torch.randn(hc_mult3, HC_MULT * hidden, dtype=torch.float32, device="cuda") * (
                    1.0 / (HC_MULT * hidden) ** 0.5
                )
                hc_scale = torch.tensor([0.7, 0.5, 0.3], dtype=torch.float32, device="cuda")
                hc_base = torch.randn(hc_mult3, dtype=torch.float32, device="cuda") * 0.1
                rms.weight.data.copy_(
                    torch.randn(hidden, device="cuda") * 0.1 + 1.0
                )
                post_r, comb_r, y_r = mhc_ref(x, fn, hc_scale, hc_base, rms)
                post_f, comb_f, y_f = mhc_fused(x, fn, hc_scale, hc_base, rms)
                # post_mix and comb_mix are produced by the same code path on both
                # sides — must be byte-identical.
                if not torch.equal(post_r, post_f):
                    print(f"  [FAIL] mhc post_mix divergence T={T} hidden={hidden} seed={seed}")
                    failures += 1
                    continue
                if not torch.equal(comb_r, comb_f):
                    print(f"  [FAIL] mhc comb_mix divergence T={T} hidden={hidden} seed={seed}")
                    failures += 1
                    continue
                if not bf16_check("mhc_pre+norm", T, hidden, seed, y_r, y_f):
                    failures += 1
    return failures


def main():
    n_hc = test_hc_head()
    n_mhc = test_mhc()
    total = n_hc + n_mhc
    n_hc_total = len(HIDDENS) * len(TS) * len(SEEDS)
    n_mhc_total = n_hc_total
    print()
    print("=" * 70)
    print(f"hc_head:      {n_hc_total - n_hc}/{n_hc_total} pass")
    print(f"mhc_pre+norm: {n_mhc_total - n_mhc}/{n_mhc_total} pass")
    print("=" * 70)
    if total == 0:
        print("All correctness checks PASSED.")
        sys.exit(0)
    else:
        print(f"{total} FAILURES.")
        sys.exit(1)


if __name__ == "__main__":
    main()
