"""A2 (gate-1 adjudication): which coefficient form is canonical?

Compares, at the S5 combine boundary where coefficients apply:
  form1  s * FP32(w)   — scaling at the combine, full-precision weight
  form2  s * BF16(w)   — weight rounded to BF16 first (Step-1 declared form)
  form3  BF16(s * w)   — pre-folded coefficient rounded once
against what the PRODUCTION kernel (post_reorder_deepgemm) physically
computes, with (a) realistic softmax-like weights and (b) adversarial weights
sitting adjacent to BF16 rounding boundaries, under non-unit routed scaling.
"""

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm

dev = "cuda:0"
torch.cuda.set_device(dev)
torch.manual_seed(7)


def run_production(down_out, s2d, ids, w, T, K, H, s, delta):
    out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    post_reorder_deepgemm(
        down_out, out, s2d, ids, w, K, T, H, float(s), pair_delta=delta
    )
    return out


def fp32_combine(base_pair, delta_pair, w_form, s):
    # fixed-order sum over k of coeff * (base+delta), scaling applied per form
    vals = base_pair.float() + delta_pair.float()
    return (vals * w_form.unsqueeze(-1)).sum(dim=1) * s


report = []
for wmode in ("softmax", "boundary"):
    for s in (1.0, 2.5, 0.4375):
        T, K, E, H = 256, 8, 32, 512
        m_max = ((T) // 256 + 1) * 256
        base = torch.randn(E, m_max, H, dtype=torch.bfloat16, device=dev)
        delta = torch.randn(T, K, H, dtype=torch.bfloat16, device=dev) * 0.1
        ids = torch.randint(0, E, (T, K), dtype=torch.int32, device=dev)
        # simple src2dst: place pair (t,k) at expert slot
        s2d = torch.zeros(T * K, dtype=torch.int32, device=dev)
        counter = torch.zeros(E, dtype=torch.int64)
        idc = ids.cpu()
        for t in range(T):
            for k in range(K):
                e = int(idc[t, k])
                s2d[t * K + k] = e * m_max + counter[e]
                counter[e] += 1
        if wmode == "softmax":
            w = torch.softmax(torch.randn(T, K, device=dev), dim=-1)
        else:
            # adversarial: values adjacent to BF16 rounding boundaries such
            # that bf16(s*w) and s*bf16(w) round differently
            base_w = torch.rand(T, K, device=dev)
            wb16 = base_w.to(torch.bfloat16).float()
            w = wb16 + wb16 * (2**-9) * 1.001  # just past the half-ULP boundary
        base_pair = base.reshape(E * m_max, H)[s2d.long()].reshape(T, K, H)
        prod = run_production(base, s2d, ids, w, T, K, H, s, delta)
        f1 = fp32_combine(base_pair, delta, w, s)
        f2 = fp32_combine(base_pair, delta, w.to(torch.bfloat16).float(), s)
        f3 = fp32_combine(base_pair, delta, (w * s).to(torch.bfloat16).float(), 1.0)
        S = float(
            (f1 - fp32_combine(base_pair, torch.zeros_like(delta), w, s)).abs().max()
        )
        q = 2**-8  # bf16 half-ULP quantum scale
        row = dict(wmode=wmode, s=s, S=S)
        for name, ref in (("s*FP32(w)", f1), ("s*BF16(w)", f2), ("BF16(s*w)", f3)):
            err = float((prod.float() - ref).abs().max())
            row[name] = err
        # inter-form distinguishability at fp32 (no bf16 store noise)
        row["|f1-f2|"] = float((f1 - f2).abs().max())
        row["|f1-f3|"] = float((f1 - f3).abs().max())
        report.append(row)
print(
    f"{'weights':>9}{'s':>8}{'S':>9} | prod-vs: {'s*FP32(w)':>11}{'s*BF16(w)':>11}{'BF16(s*w)':>11} | {'|f1-f2|':>9}{'|f1-f3|':>9}"
)
for r in report:
    print(
        f"{r['wmode']:>9}{r['s']:>8}{r['S']:>9.3f} |          "
        f"{r['s*FP32(w)']:>11.5f}{r['s*BF16(w)']:>11.5f}{r['BF16(s*w)']:>11.5f} | "
        f"{r['|f1-f2|']:>9.5f}{r['|f1-f3|']:>9.5f}"
    )
print("\nbf16 output quantum at max|y| ~", 2**-8, "* max|y|")
