"""SM70 sigmoid-gated RMSNorm must equal fla's Triton layer_norm_gated_fwd
bitwise (fp16, head dim 128), over every fp16 gate value and x scales from
denormal to near-overflow. Also times both, warm, under a CUDA graph.

Not a CI test:

  CUDA_VISIBLE_DEVICES=0 python test/manual/layers/test_sm70_norm_gate.py
"""

import torch

from sglang.kernels.ops.attention.fla.fused_norm_gate import layer_norm_gated_fwd
from sglang.kernels.ops.elementwise.sm70_norm_gate import (
    sm70_norm_gate,
    sm70_norm_gate_covered,
)

DEV = "cuda"
D = 128


def triton_ref(x, g, w, eps):
    y, _, _, _ = layer_norm_gated_fwd(
        x.clone(), g, w, None, activation="sigmoid", eps=eps, is_rms_norm=True
    )
    return y


def case(name, x, g, w, eps) -> bool:
    assert sm70_norm_gate_covered(x, g, w, "sigmoid")
    ref = triton_ref(x, g, w, eps)
    got = sm70_norm_gate(x.clone(), g, w, eps)
    rb, gb = ref.view(torch.int16), got.view(torch.int16)
    both_nan = ref.isnan() & got.isnan()
    bad = (rb != gb) & ~both_nan
    print(f"{name:44s} rows={x.size(0):6d} eps={eps:.0e}  {'equal' if not bad.any() else f'DIFFER ({int(bad.sum())})'}")
    if bad.any():
        r, c = bad.nonzero()[0].tolist()
        print("   x", x[r, c].item(), "g", g[r, c].item(), "w", w[c].item(), "ref", ref[r, c].item(), "got", got[r, c].item())
    return not bad.any()


def all_halves() -> torch.Tensor:
    return torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16).view(torch.float16).to(DEV)


def main() -> None:
    gen = torch.Generator(device=DEV).manual_seed(0)
    ok = True
    w_rand = (torch.randn(D, device=DEV, generator=gen) * 0.5 + 1).half()
    w_special = w_rand.clone()
    w_special[::17] = 0.0
    w_special[1::17] = -0.0
    w_special[2::17] = 65504.0
    ws = {"w randn": w_rand, "w zeros/extremes": w_special}
    for eps in (1e-5, 1e-6):
        for wname, w in ws.items():
            # Every fp16 value as the gate, over decode-like x.
            g = all_halves().view(-1, D)
            x = (torch.randn(g.shape, device=DEV, generator=gen) * 0.3).half()
            ok &= case(f"every gate, {wname}", x, g, w, eps)
            for scale in (1e-7, 1e-4, 0.05, 1.0, 30.0, 3000.0):
                for rows in (1, 7, 8, 9, 16, 17, 64, 300):
                    x = (torch.randn(rows, D, device=DEV, generator=gen) * scale).half()
                    g = (torch.randn(rows, D, device=DEV, generator=gen) * 3).half()
                    ok &= case(f"x*{scale:g}, {wname}", x, g, w, eps)
            bits = torch.randint(-(2**15), 2**15, (2, 4096, D), device=DEV, generator=gen, dtype=torch.int32)
            x, g = bits.to(torch.int16).view(torch.float16)
            ok &= case(f"random bit patterns (inf/nan), {wname}", x.contiguous(), g.contiguous(), w, eps)
    timing()
    assert ok, "sm70 norm_gate differs from Triton"
    print("ALL OK")


def graph_us(fn, reps=50):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    best = 1e9
    for _ in range(20):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) / reps * 1e3)
    return best


def timing() -> None:
    x = torch.randn(8, D, device=DEV).half()
    g = torch.randn(8, D, device=DEV).half()
    w = torch.randn(D, device=DEV).half()
    t = graph_us(lambda: layer_norm_gated_fwd(x, g, w, None, activation="sigmoid", eps=1e-5, is_rms_norm=True))
    k = graph_us(lambda: sm70_norm_gate(x, g, w, 1e-5))
    print(f"8 heads x 128: triton {t:.1f} us  sm70 {k:.1f} us (warm)")


if __name__ == "__main__":
    main()
