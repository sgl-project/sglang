"""Split-K SM70 GLM NVFP4 HMMA GEMV: exactness vs the single-warp kernels,
error vs an fp64 reference, and timing on the GLM-5.3-Flash TP8 shapes, with
experts whole (EP8) or sliced along the intermediate dim (TP-MoE)."""

import torch

from sglang.srt.environ import envs

if not envs.SGLANG_SM70_GLM_NVFP4_BUILD_DIR.is_set():
    envs.SGLANG_SM70_GLM_NVFP4_BUILD_DIR.set("/tmp/glm_hmma_splitk_build")
from sglang.kernels.ops.gemm.sm70_glm_nvfp4_gemv import _load

ext = _load()
assert ext is not None
dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)


def rand_marlin(K, N):
    w = torch.randint(-(2**31), 2**31 - 1, (K // 16, N * 2), dtype=torch.int32, device=dev, generator=g)
    # Encoded E4M3 block scales as the V100 path stores them (byte << 7 = fp16 bits).
    s = torch.randint(0x30, 0x48, (K // 16, N), dtype=torch.uint8, device=dev, generator=g)
    return w, s


def dequant(packed, scales, K, N):
    """Exact dequantized [K, N] from one-hot rows (scale*fp4 is exact in fp16)."""
    one = torch.ones(1, dtype=torch.float32, device=dev)
    out = torch.empty(K, N, dtype=torch.float64, device=dev)
    y = torch.empty(4, N, dtype=torch.float16, device=dev)
    for k0 in range(0, K, 4):
        x = torch.zeros(4, K, dtype=torch.float16, device=dev)
        x[torch.arange(4), torch.arange(k0, k0 + 4)] = 1
        ext.gemv_hmma(x, packed, scales, one, y)
        out[k0:k0 + 4] = y.double()
    return out


def rel(a, ref):
    return ((a.double() - ref).abs().max() / ref.abs().max()).item()


def timeit(fn, iters=200):
    for _ in range(10):
        fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def test_dense(K, N, M):
    w, s = rand_marlin(K, N)
    packed = ext.repack(w)
    gs = torch.tensor([0.37], dtype=torch.float32, device=dev)
    x = torch.randn(M, K, dtype=torch.float16, device=dev, generator=g)
    old = torch.empty(M, N, dtype=torch.float16, device=dev)
    new = torch.empty_like(old)
    ext.gemv_hmma(x, packed, s, gs, old)
    ext.gemv_hmma_splitk(x, packed, s, gs, new, 1)
    assert torch.equal(old, new), "split=1 must equal the single-warp kernel"
    ext.gemv_hmma_splitk(x, packed, s, gs, new, 0)
    ref = x.double() @ dequant(packed, s, K, N) * 0.37
    t_old = timeit(lambda: ext.gemv_hmma(x, packed, s, gs, old))
    t_new = timeit(lambda: ext.gemv_hmma_splitk(x, packed, s, gs, new, 0))
    print(f"dense K={K:5d} N={N:5d} M={M} err old={rel(old, ref):.2e} new={rel(new, ref):.2e} "
          f"differ={int((old != new).sum())}/{old.numel()}  {t_old:7.1f}us -> {t_new:6.1f}us  "
          f"({K * N / 2 / t_new / 1e3:.0f} GB/s)")
    assert rel(new, ref) <= max(2 * rel(old, ref), 1e-3)


def test_moe(K, N, experts, valid, per_route):
    ws = [rand_marlin(K, N) for _ in range(experts)]
    packed = torch.stack([ext.repack(w) for w, _ in ws]).contiguous()
    scales = torch.stack([s for _, s in ws]).contiguous()
    gs = torch.rand(experts, dtype=torch.float32, device=dev, generator=g) + 0.5
    routes = 8
    ids = torch.full((routes,), -1, dtype=torch.int32, device=dev)
    chosen = torch.randperm(experts, device=dev)[:valid]
    ids[torch.randperm(routes, device=dev)[:valid]] = chosen.int()
    tw = torch.rand(routes, dtype=torch.float32, device=dev, generator=g)
    x = torch.randn(routes if per_route else 1, K, dtype=torch.float16, device=dev, generator=g)
    old = torch.empty(routes, N, dtype=torch.float16, device=dev)
    new = torch.empty_like(old)
    args = (ids, tw)
    ext.moe_hmma(x, packed, scales, gs, *args, old, per_route, per_route)
    ext.moe_hmma_splitk(x, packed, scales, gs, *args, new, per_route, per_route, 1, False)
    assert torch.equal(old, new), "split=1 must equal the single-warp kernel"
    new.fill_(1)
    ext.moe_hmma_splitk(x, packed, scales, gs, *args, new, per_route, per_route, 0, False)
    ref = torch.zeros(routes, N, dtype=torch.float64, device=dev)
    for r in range(routes):
        e = int(ids[r])
        if e < 0:
            continue
        xr = x[r if per_route else 0].double()
        ref[r] = xr @ dequant(packed[e], scales[e], K, N) * gs[e].double()
        if per_route:
            ref[r] *= tw[r].double()
    assert torch.equal(new[ids < 0], torch.zeros_like(new[ids < 0])), "invalid routes must be zero"
    t_old = timeit(lambda: ext.moe_hmma(x, packed, scales, gs, *args, old, per_route, per_route))
    t_new = timeit(lambda: ext.moe_hmma_splitk(x, packed, scales, gs, *args, new, per_route, per_route, 0, False))
    print(f"moe   K={K:5d} N={N:5d} valid={valid} err old={rel(old, ref):.2e} new={rel(new, ref):.2e} "
          f"{t_old:7.1f}us -> {t_new:6.1f}us  ({valid * K * N / 2 / t_new / 1e3:.0f} GB/s)")
    assert rel(new, ref) <= max(2 * rel(old, ref), 1e-3)


def test_shared_fold(experts, M, swiglu_limit, routed_scale):
    """Shared-expert rows folded into the routed launches must be bitwise the
    dense split-K GEMV, and the fused sum the routed sum followed by `+= shared`."""
    from sglang.kernels.ops.moe.sm70_glm_nvfp4_moe_decode import (
        Sm70SharedExpertPack,
        sm70_glm_nvfp4_moe_decode,
    )
    from sglang.kernels.ops.moe.sm70_moe_glue import swiglu_clamp

    H, I = 4096, 256
    w13 = [rand_marlin(H, 2 * I) for _ in range(experts)]
    w2 = [rand_marlin(I, H) for _ in range(experts)]
    w13_packed = torch.stack([ext.repack(w) for w, _ in w13]).contiguous()
    w13_scales = torch.stack([s for _, s in w13]).contiguous()
    w2_packed = torch.stack([ext.repack(w) for w, _ in w2]).contiguous()
    w2_scales = torch.stack([s for _, s in w2]).contiguous()
    w13_gs = torch.rand(experts, dtype=torch.float32, device=dev, generator=g) * 0.02 + 0.01
    w2_gs = torch.rand(experts, dtype=torch.float32, device=dev, generator=g) * 0.02 + 0.01
    (sw13, ss13), (sw2, ss2) = rand_marlin(H, 2 * I), rand_marlin(I, H)
    pack = Sm70SharedExpertPack(
        w13_packed=ext.repack(sw13), w13_scales=ss13,
        w13_global=torch.tensor([0.013], dtype=torch.float32, device=dev),
        w2_packed=ext.repack(sw2), w2_scales=ss2,
        w2_global=torch.tensor([0.021], dtype=torch.float32, device=dev),
    )
    x = torch.randn(M, H, dtype=torch.float16, device=dev, generator=g)
    ids = torch.stack([torch.randperm(experts, device=dev)[:8] for _ in range(M)]).int().reshape(-1)
    if M > 1:
        ids[3] = -1  # a padded route must still contribute zero
    tw = torch.rand(M * 8, dtype=torch.float32, device=dev, generator=g)
    routes = M * 8

    gate = torch.full((routes + M, 2 * I), 7.0, dtype=torch.float16, device=dev)
    ext.moe_hmma_splitk_shared(x, w13_packed, w13_scales, w13_gs, ids, tw, gate, False, 0,
                               pack.w13_packed, pack.w13_scales, pack.w13_global, M, False)
    ref_routed = torch.empty(routes, 2 * I, dtype=torch.float16, device=dev)
    ext.moe_hmma_splitk(x, w13_packed, w13_scales, w13_gs, ids, tw, ref_routed, False, False, 0, False)
    ref_shared = torch.empty(M, 2 * I, dtype=torch.float16, device=dev)
    ext.gemv_hmma_splitk(x, pack.w13_packed, pack.w13_scales, pack.w13_global, ref_shared, 0)
    assert torch.equal(gate[:routes], ref_routed), "routed gate_up rows changed"
    assert torch.equal(gate[routes:], ref_shared), "shared gate_up rows differ from the dense GEMV"

    act = torch.empty(routes + M, I, dtype=torch.float16, device=dev)
    swiglu_clamp(act, gate, swiglu_limit)
    down = torch.full((routes + M, H), 7.0, dtype=torch.float16, device=dev)
    ext.moe_hmma_splitk_shared(act, w2_packed, w2_scales, w2_gs, ids, tw, down, True, 0,
                               pack.w2_packed, pack.w2_scales, pack.w2_global, M, False)
    ref_down = torch.empty(routes, H, dtype=torch.float16, device=dev)
    ext.moe_hmma_splitk(act[:routes].contiguous(), w2_packed, w2_scales, w2_gs, ids, tw, ref_down,
                        True, True, 0, False)
    ref_sdown = torch.empty(M, H, dtype=torch.float16, device=dev)
    ext.gemv_hmma_splitk(act[routes:].contiguous(), pack.w2_packed, pack.w2_scales, pack.w2_global,
                         ref_sdown, 0)
    assert torch.equal(down[:routes], ref_down), "routed down rows changed"
    assert torch.equal(down[routes:], ref_sdown), "shared down rows differ from the dense GEMV"

    args = (w13_packed, w2_packed, w13_scales, w2_scales, w13_gs, w2_gs, ids, tw, swiglu_limit,
            routed_scale)
    fused = sm70_glm_nvfp4_moe_decode(x, *args, shared=pack)
    ref = sm70_glm_nvfp4_moe_decode(x, *args)
    ref += ref_sdown
    assert torch.equal(fused, ref), "fused decode differs from routed decode + shared expert"
    t_fused = timeit(lambda: sm70_glm_nvfp4_moe_decode(x, *args, shared=pack))
    t_routed = timeit(lambda: sm70_glm_nvfp4_moe_decode(x, *args))
    print(f"shared fold M={M} limit={swiglu_limit} scale={routed_scale}: bitwise equal; "
          f"routed {t_routed:.1f}us, routed+shared {t_fused:.1f}us")


def test_grouped_rows(experts, M, pool, swiglu_limit=7.0, routed_scale=2.5):
    """A batch of M tokens (MTP verify) must give each token bitwise the batch-1
    decode output; `pool` limits the experts drawn so tokens share some."""
    from sglang.kernels.ops.moe.sm70_glm_nvfp4_moe_decode import (
        Sm70SharedExpertPack,
        sm70_glm_nvfp4_moe_decode,
    )

    H, I = 4096, 256
    w13 = [rand_marlin(H, 2 * I) for _ in range(experts)]
    w2 = [rand_marlin(I, H) for _ in range(experts)]
    w13_packed = torch.stack([ext.repack(w) for w, _ in w13]).contiguous()
    w13_scales = torch.stack([s for _, s in w13]).contiguous()
    w2_packed = torch.stack([ext.repack(w) for w, _ in w2]).contiguous()
    w2_scales = torch.stack([s for _, s in w2]).contiguous()
    w13_gs = torch.rand(experts, dtype=torch.float32, device=dev, generator=g) * 0.02 + 0.01
    w2_gs = torch.rand(experts, dtype=torch.float32, device=dev, generator=g) * 0.02 + 0.01
    (sw13, ss13), (sw2, ss2) = rand_marlin(H, 2 * I), rand_marlin(I, H)
    pack = Sm70SharedExpertPack(
        w13_packed=ext.repack(sw13), w13_scales=ss13,
        w13_global=torch.tensor([0.013], dtype=torch.float32, device=dev),
        w2_packed=ext.repack(sw2), w2_scales=ss2,
        w2_global=torch.tensor([0.021], dtype=torch.float32, device=dev),
    )
    x = torch.randn(M, H, dtype=torch.float16, device=dev, generator=g)
    chosen = torch.randperm(experts, device=dev)[:pool]
    ids = torch.stack([chosen[torch.randperm(pool, device=dev)[:8]] for _ in range(M)]).int()
    if M > 1:
        ids[1, 5] = -1
    tw = torch.rand(M, 8, dtype=torch.float32, device=dev, generator=g)
    args = (w13_packed, w2_packed, w13_scales, w2_scales, w13_gs, w2_gs)
    for shared in (None, pack):
        out = sm70_glm_nvfp4_moe_decode(x, *args, ids, tw, swiglu_limit, routed_scale, shared=shared)
        for t in range(M):
            one = sm70_glm_nvfp4_moe_decode(x[t:t + 1].contiguous(), *args, ids[t:t + 1], tw[t:t + 1],
                                            swiglu_limit, routed_scale, shared=shared)
            assert torch.equal(out[t:t + 1], one), f"token {t} differs from batch-1 (shared={shared is not None})"
    distinct = len(set(ids[ids >= 0].tolist()))
    t_grouped = timeit(lambda: sm70_glm_nvfp4_moe_decode(x, *args, ids, tw, swiglu_limit, routed_scale,
                                                         shared=pack))
    t_one = timeit(lambda: sm70_glm_nvfp4_moe_decode(x[:1].contiguous(), *args, ids[:1], tw[:1],
                                                     swiglu_limit, routed_scale, shared=pack))
    print(f"grouped M={M} distinct={distinct:2d}/{int((ids >= 0).sum())}: each row bitwise batch-1; "
          f"batch {t_grouped:.1f}us vs one token {t_one:.1f}us")


def test_unpack_experts(K, N, experts):
    ws = [rand_marlin(K, N)[0] for _ in range(experts)]
    packed = torch.stack([ext.repack(w) for w in ws]).contiguous()
    out = torch.empty(experts, K // 16, N * 2, dtype=torch.int32, device=dev)
    ext.unpack_experts_into(packed, out)
    assert torch.equal(out, torch.stack(ws)), "unpack must restore the Marlin words"
    t = timeit(lambda: ext.unpack_experts_into(packed, out), 20)
    print(f"unpack K={K} N={N} experts={experts}: exact, {t:.0f}us")


if __name__ == "__main__":
    test_unpack_experts(4096, 4096, 36)
    test_unpack_experts(2048, 4096, 36)
    for K, N in [(4096, 512), (256, 4096), (4096, 3072), (1536, 4096), (4096, 4096)]:
        for M in (1, 4):
            test_dense(K, N, M)
    for valid in (1, 3):
        test_moe(4096, 4096, 36, valid, False)
        test_moe(2048, 4096, 36, valid, True)
    # TP-MoE slices: I = 2048 / 8 per rank, all eight routes local.
    test_unpack_experts(4096, 512, 32)
    test_unpack_experts(256, 4096, 32)
    test_moe(4096, 512, 32, 8, False)
    test_moe(256, 4096, 32, 8, True)
    for M in (1, 2, 4):
        test_shared_fold(32, M, 10.0, 1.0)
    test_shared_fold(32, 1, 7.0, 2.5)
    for M, pool in [(2, 288), (3, 16), (4, 288), (4, 12), (4, 8)]:
        test_grouped_rows(288, M, pool)
    print("OK")
