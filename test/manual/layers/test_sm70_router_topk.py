"""SM70 warp router must match _router_triton_kernel bitwise (ids and weights).

Not a CI test:

  CUDA_VISIBLE_DEVICES=0 python test/manual/layers/test_sm70_router_topk.py
"""

import torch

import sglang.kernels.ops.moe.sm70_router_topk as sm70
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate


def triton_route(scores, bias, renormalize, scale, apply_scale):
    covered = sm70.sm70_router_covered
    sm70.sm70_router_covered = lambda *a: False
    try:
        return moe_fused_gate(
            scores,
            bias,
            8,
            scoring_func="sigmoid",
            renormalize=renormalize,
            routed_scaling_factor=scale,
            apply_routed_scaling_factor_on_output=apply_scale,
        )
    finally:
        sm70.sm70_router_covered = covered


def case(name, scores, bias, renormalize=True, scale=2.5, apply_scale=False):
    ref_w, ref_i = triton_route(scores, bias, renormalize, scale, apply_scale)
    w, i = sm70.sm70_router_topk(scores, bias, renormalize, scale, apply_scale)
    ids_ok = torch.equal(i, ref_i)
    w_ok = torch.equal(w.view(torch.int32), ref_w.view(torch.int32))
    print(f"{name:44s} ids {'equal' if ids_ok else 'DIFFER'}  weights {'equal' if w_ok else 'DIFFER'}")
    if not (ids_ok and w_ok):
        bad = (i != ref_i).any(-1) | (w != ref_w).any(-1)
        r = int(bad.nonzero()[0])
        print("  row", r, "ref", ref_i[r].tolist(), ref_w[r].tolist(), "\n  got", i[r].tolist(), w[r].tolist())
    return ids_ok and w_ok


def main() -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    dev = "cuda"
    ok = True
    assert not sm70.sm70_router_covered(torch.zeros(4, 300, device=dev), torch.zeros(300, device=dev), 8)
    for n in (288, 304, 400, 512):
        for m in (1, 3, 4, 64, 4096):
            scores = torch.randn(m, n, device=dev, generator=g) * 2
            bias = torch.randn(n, device=dev, generator=g) * 0.1
            ok &= case(f"N={n} M={m} randn", scores, bias)
        # GLM-style correction bias: large offset, spread of a few ulps.
        scores = torch.randn(4096, n, device=dev, generator=g)
        bias = 8.0 + torch.randint(-4, 5, (n,), device=dev, generator=g).float() * 2**-20
        ok &= case(f"N={n} offset bias", scores, bias)
        # Exact ties: coarse logits and zero bias.
        scores = torch.randint(-3, 4, (4096, n), device=dev, generator=g).float()
        ok &= case(f"N={n} ties", scores, torch.zeros(n, device=dev))
        scores = torch.randn(64, n, device=dev, generator=g)
        scores[::3, ::7] = float("nan")
        ok &= case(f"N={n} nan", scores, torch.randn(n, device=dev, generator=g))
        for renorm, apply in ((False, False), (True, True), (False, True)):
            scores = torch.randn(256, n, device=dev, generator=g) * 4
            bias = torch.randn(n, device=dev, generator=g)
            ok &= case(f"N={n} renorm={renorm} apply_scale={apply}", scores, bias, renorm, 2.5, apply)
    # Strided rows (a view into a wider buffer).
    wide = torch.randn(64, 320, device=dev, generator=g)
    ok &= case("N=288 strided rows", wide[:, :288], torch.randn(288, device=dev, generator=g))
    assert ok, "sm70 router differs from the Triton router"

    scores = torch.randn(1, 288, device=dev)
    bias = torch.randn(288, device=dev)
    for name, fn in (("triton", lambda: triton_route(scores, bias, True, 2.5, False)),
                     ("sm70", lambda: sm70.sm70_router_topk(scores, bias, True, 2.5, False))):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(50):
                fn()
        best = 1e9
        for _ in range(20):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            graph.replay()
            b.record()
            torch.cuda.synchronize()
            best = min(best, a.elapsed_time(b) / 50 * 1e3)
        print(f"{name}: {best:.1f} us per M=1 call (graph)")


if __name__ == "__main__":
    main()
