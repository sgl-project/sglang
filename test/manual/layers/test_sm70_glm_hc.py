"""SM70 GLM mHC kernels against a reference build of the kernel source.

Not a CI test. The reference is any earlier copy of sm70_glm_hc.cuh; pre,
pre_fused and post must match it bitwise. pre_fused must also match pre fed
with fp64-exact mixes, and sit closer to them than the torch fp32 mix chain does.
pre_fused_norm must equal pre_fused followed by sgl_kernel rmsnorm, bitwise:

  CUDA_VISIBLE_DEVICES=0 python test/manual/layers/test_sm70_glm_hc.py /path/to/old/sm70_glm_hc.cuh
"""

import sys

import torch

from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.elementwise.sm70_glm_hc import _module, glm_hc_pre_fused

RMS_EPS = 1e-6


def _run_pre(mod, mixes, residual, hc_scale, hc_base):
    tokens = residual.shape[0]
    pre = torch.empty(tokens, 4, device="cuda")
    post = torch.empty(tokens, 4, device="cuda")
    comb = torch.empty(tokens, 4, 4, device="cuda")
    y = torch.empty(tokens, 4096, device="cuda", dtype=torch.float16)
    mod.pre(pre, post, comb, y, mixes, residual, hc_scale, hc_base, 20, 1e-6)
    return post, comb, y


def _max_err(outs, ref):
    return [(a.double() - b.double()).abs().max().item() for a, b in zip(outs, ref)]


def check_fused(new) -> None:
    torch.manual_seed(1)
    for tokens in (1, 2, 3, 8, 17, 64, 300):
        residual = torch.randn(tokens, 4 * 4096, device="cuda", dtype=torch.float16) * 4
        fn = torch.randn(24, 4 * 4096, device="cuda") * 0.02
        hc_scale = torch.rand(3, device="cuda") + 0.5
        hc_base = torch.randn(24, device="cuda")
        x64 = residual.double()
        exact = (x64 @ fn.double().T) * torch.rsqrt(x64.square().mean(-1, keepdim=True) + RMS_EPS)
        x32 = residual.float()
        torch_mixes = torch.nn.functional.linear(x32, fn) * torch.rsqrt(
            x32.square().mean(-1, keepdim=True) + RMS_EPS)
        ref = _run_pre(new, exact.float(), residual, hc_scale, hc_base)
        via_torch = _run_pre(new, torch_mixes, residual, hc_scale, hc_base)
        fused = glm_hc_pre_fused(residual.view(tokens, 4, 4096), fn, hc_scale, hc_base, RMS_EPS, 20, 1e-6)
        fused = (fused[0].squeeze(-1), fused[1], fused[2])
        e_fused = _max_err(fused, ref)
        e_torch = _max_err(via_torch, ref)
        print(f"tokens={tokens}: max err vs fp64 mixes (post, comb, y) "
              f"fused={['%.2e' % e for e in e_fused]} torch={['%.2e' % e for e in e_torch]}")
        assert all(f <= t for f, t in zip(e_fused, e_torch)), "fused path less accurate than torch"


def time_fused() -> None:
    residual = torch.randn(1, 4, 4096, device="cuda", dtype=torch.float16)
    fn = torch.randn(24, 4 * 4096, device="cuda") * 0.02
    hc_scale, hc_base = torch.rand(3, device="cuda"), torch.randn(24, device="cuda")
    for _ in range(10):
        glm_hc_pre_fused(residual, fn, hc_scale, hc_base, RMS_EPS, 20, 1e-6)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(200):
        glm_hc_pre_fused(residual, fn, hc_scale, hc_base, RMS_EPS, 20, 1e-6)
    e.record()
    torch.cuda.synchronize()
    print(f"pre_fused: {s.elapsed_time(e) / 200 * 1e3:.1f} us per decode call (incl. allocs)")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def check_fused_and_post_vs_ref(ref, new) -> None:
    torch.manual_seed(2)
    for tokens in (1, 2, 3, 17, 64):
        for splits in (64, 16, 1):
            residual = torch.randn(tokens, 4 * 4096, device="cuda", dtype=torch.float16) * 4
            fn = torch.randn(24, 4 * 4096, device="cuda") * 0.02
            hc_scale = torch.rand(3, device="cuda") + 0.5
            hc_base = torch.randn(24, device="cuda")
            outs = []
            for mod in (ref, new):
                bufs = (torch.empty(tokens, 4, device="cuda"), torch.empty(tokens, 4, device="cuda"),
                        torch.empty(tokens, 4, 4, device="cuda"),
                        torch.empty(tokens, 4096, device="cuda", dtype=torch.float16))
                partials = torch.empty(tokens, splits, 25, device="cuda", dtype=torch.float64)
                mod.pre_fused(*bufs, partials, residual, fn, hc_scale, hc_base, 20, 1e-6, RMS_EPS)
                outs.append(bufs)
            for name, a, b in zip(("pre", "post", "comb", "y"), *outs):
                assert torch.equal(_bits(a), _bits(b)), f"pre_fused {name} differs at tokens={tokens} splits={splits}"
        print(f"tokens={tokens}: pre_fused pre/post/comb/y bitwise equal (splits 64, 16, 1)")
    for tokens in (1, 3, 17, 2048):
        x = torch.randn(tokens, 4096, device="cuda", dtype=torch.float16) * 3
        residual = torch.randn(tokens, 4 * 4096, device="cuda", dtype=torch.float16) * 4
        post_mix = torch.rand(tokens, 4, device="cuda") * 2
        comb = torch.rand(tokens, 4, 4, device="cuda")
        a = torch.empty_like(residual)
        b = torch.empty_like(residual)
        ref.post(a, x, residual, post_mix, comb)
        new.post(b, x, residual, post_mix, comb)
        assert torch.equal(_bits(a), _bits(b)), f"post differs at tokens={tokens}"
        print(f"tokens={tokens}: post bitwise equal")


def _norm_weights(gen):
    w = torch.randn(4096, device="cuda", generator=gen, dtype=torch.float32)
    special = w.half()
    special[::97] = 0.0
    special[1::97] = -0.0
    special[2::97] = 65504.0
    special[3::97] = -1e-7
    return {
        "randn": (w * 0.3 + 1.0).half(),
        "signed zeros, extremes": special,
        "all ones": torch.ones(4096, device="cuda", dtype=torch.float16),
    }


def check_fused_norm(new) -> None:
    from sgl_kernel import rmsnorm

    gen = torch.Generator(device="cuda").manual_seed(3)
    weights = _norm_weights(gen)
    for scale in (4.0, 1e-3, 1e-6, 300.0):
        for tokens in (1, 2, 3, 4, 17, 64):
            for splits in (64, 16, 1):
                residual = (torch.randn(tokens, 4 * 4096, device="cuda", generator=gen) * scale).half()
                fn = torch.randn(24, 4 * 4096, device="cuda", generator=gen) * 0.02
                hc_scale = torch.rand(3, device="cuda", generator=gen) + 0.5
                hc_base = torch.randn(24, device="cuda", generator=gen)
                for wname, w in weights.items():
                    for norm_eps in (1e-5, 1e-6):
                        bufs = [(torch.empty(tokens, 4, device="cuda"), torch.empty(tokens, 4, device="cuda"),
                                 torch.empty(tokens, 4, 4, device="cuda"),
                                 torch.empty(tokens, 4096, device="cuda", dtype=torch.float16)) for _ in range(2)]
                        partials = torch.empty(tokens, splits, 25, device="cuda", dtype=torch.float64)
                        new.pre_fused(*bufs[0], partials, residual, fn, hc_scale, hc_base, 20, 1e-6, RMS_EPS)
                        new.pre_fused_norm(*bufs[1], partials, residual, fn, hc_scale, hc_base, w, 20, 1e-6,
                                           RMS_EPS, norm_eps)
                        want = (*bufs[0][:3], rmsnorm(bufs[0][3], w, norm_eps))
                        for name, a, b in zip(("pre", "post", "comb", "y"), want, bufs[1]):
                            assert torch.equal(_bits(a), _bits(b)), (
                                f"pre_fused_norm {name} differs: scale={scale} tokens={tokens} splits={splits} "
                                f"weight={wname} eps={norm_eps} "
                                f"({int((_bits(a) != _bits(b)).sum())} elements)")
        print(f"scale={scale}: pre_fused_norm == pre_fused + rmsnorm bitwise "
              f"(tokens 1-64, splits 64/16/1, {len(weights)} weights, eps 1e-5/1e-6)")


def time_fused_norm(new) -> None:
    from sgl_kernel import rmsnorm

    residual = torch.randn(1, 4 * 4096, device="cuda", dtype=torch.float16)
    fn = torch.randn(24, 4 * 4096, device="cuda") * 0.02
    hc_scale, hc_base = torch.rand(3, device="cuda"), torch.randn(24, device="cuda")
    w = torch.randn(4096, device="cuda", dtype=torch.float16)
    bufs = (torch.empty(1, 4, device="cuda"), torch.empty(1, 4, device="cuda"),
            torch.empty(1, 4, 4, device="cuda"), torch.empty(1, 4096, device="cuda", dtype=torch.float16))
    partials = torch.empty(1, 64, 25, device="cuda", dtype=torch.float64)

    def unfused():
        new.pre_fused(*bufs, partials, residual, fn, hc_scale, hc_base, 20, 1e-6, RMS_EPS)
        rmsnorm(bufs[3], w, 1e-5)

    def fused():
        new.pre_fused_norm(*bufs, partials, residual, fn, hc_scale, hc_base, w, 20, 1e-6, RMS_EPS, 1e-5)

    for name, fn_ in (("pre_fused + rmsnorm", unfused), ("pre_fused_norm", fused)):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn_()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(50):
                fn_()
        best = 1e9
        for _ in range(20):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            graph.replay()
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e) / 50 * 1e3)
        print(f"{name}: {best:.1f} us per decode call (graph, warm)")


def main(ref_path: str) -> None:
    ref = load_jit(
        "sm70_glm_hc_reference",
        cuda_files=[ref_path],
        cuda_wrappers=[
            ("pre", "sm70_glm_hc::pre"),
            ("pre_fused", "sm70_glm_hc::pre_fused"),
            ("post", "sm70_glm_hc::post"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )
    new = _module()
    check_fused_and_post_vs_ref(ref, new)
    torch.manual_seed(0)
    for tokens in (1, 3, 17, 2048):
        residual = torch.randn(tokens, 4 * 4096, device="cuda", dtype=torch.float16)
        mixes = torch.randn(tokens, 24, device="cuda") * 3
        hc_scale = torch.rand(3, device="cuda") + 0.5
        hc_base = torch.randn(24, device="cuda")
        outs = []
        for mod in (ref, new):
            pre = torch.empty(tokens, 4, device="cuda")
            post = torch.empty(tokens, 4, device="cuda")
            comb = torch.empty(tokens, 4, 4, device="cuda")
            y = torch.empty(tokens, 4096, device="cuda", dtype=torch.float16)
            mod.pre(pre, post, comb, y, mixes, residual, hc_scale, hc_base, 20, 1e-6)
            outs.append((pre, post, comb, y))
        for name, a, b in zip(("pre", "post", "comb", "y"), *outs):
            assert torch.equal(a.view(torch.int32) if a.dtype == torch.float32 else a.view(torch.int16),
                               b.view(torch.int32) if b.dtype == torch.float32 else b.view(torch.int16)), (
                f"{name} differs at tokens={tokens}")
        print(f"tokens={tokens}: pre/post/comb/y bitwise equal")

    residual = torch.randn(1, 4 * 4096, device="cuda", dtype=torch.float16)
    mixes = torch.randn(1, 24, device="cuda")
    args = (torch.rand(3, device="cuda"), torch.randn(24, device="cuda"), 20, 1e-6)
    bufs = (torch.empty(1, 4, device="cuda"), torch.empty(1, 4, device="cuda"),
            torch.empty(1, 4, 4, device="cuda"), torch.empty(1, 4096, device="cuda", dtype=torch.float16))
    for name, mod in (("reference", ref), ("new", new)):
        for _ in range(10):
            mod.pre(*bufs, mixes, residual, *args)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(200):
            mod.pre(*bufs, mixes, residual, *args)
        e.record()
        torch.cuda.synchronize()
        print(f"{name}: {s.elapsed_time(e) / 200 * 1e3:.1f} us per decode call")

    check_fused(new)
    time_fused()
    check_fused_norm(new)
    time_fused_norm(new)


if __name__ == "__main__":
    main(sys.argv[1])
