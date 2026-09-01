"""v3 tuning bench: sweep compressed-loop num_stages (NS_C) and BLOCK_KC to
measure how much of the fp8 dequant we can hide behind MFMA via deeper software
pipelining + the bf16-direct dequant (no f32 intermediate).

Reports us/iter vs the bf16 oracle. Also reports the split-K auto path.
"""
import time
import torch
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode_soa_v3 import (
    sparse_attn_v4_paged_decode_soa_v3,
)
from poc_v3_common import quant_soa


def _bench(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def _mono(T, kv_len, swa_pages, C, rc, dev, seed):
    torch.manual_seed(seed + 7)
    cl = int(round(kv_len * rc)); sl = kv_len - cl
    indptr = torch.arange(0, (T + 1) * kv_len, kv_len, dtype=torch.int32, device=dev)
    parts = []
    for _t in range(T):
        s = torch.randint(0, swa_pages, (sl,), device=dev, dtype=torch.int32)
        c = swa_pages + torch.randint(0, C, (cl,), device=dev, dtype=torch.int32)
        parts.append(torch.cat([s, c]))
    return torch.cat(parts).to(torch.int32), indptr, torch.full((T,), sl, dtype=torch.int32, device=dev)


def _refmax(a, b):
    return (a.float() - b.float()).abs().max().item()


def _case(*, T, H, D, swa_pages, C, kv_len, rc, seed=0):
    torch.manual_seed(seed); dev = "cuda"; dt = torch.bfloat16
    q = torch.randn(T, H, D, device=dev, dtype=dt) * 0.5
    swa = torch.randn(swa_pages, D, device=dev, dtype=dt) * 0.5
    comp = torch.randn(C, D, device=dev, dtype=dt) * 0.5
    sink = torch.randn(H, device=dev, dtype=torch.float32); sc = 1.0 / (D ** 0.5)
    nope_fp8, rope_bf16, scale_f32, comp_dq = quant_soa(comp)
    unified = torch.cat([swa, comp_dq], 0).contiguous()
    idx, indptr, swa_len = _mono(T, kv_len, swa_pages, C, rc, dev, seed)

    oracle = sparse_attn_v4_paged_decode(q, unified, idx, indptr, sink, sc)
    ta = _bench(lambda: sparse_attn_v4_paged_decode(q, unified, idx, indptr, sink, sc))

    def v3(ns_c, bk, ks, ns_a=3):
        return sparse_attn_v4_paged_decode_soa_v3(
            q, swa, nope_fp8, rope_bf16, scale_f32, idx, indptr, swa_len, sink, sc,
            swa_pages=swa_pages, block_k=bk, ns_a=ns_a, ns_c=ns_c, kv_splits=ks)

    print(f"--- T={T} H={H} kv_len={kv_len} rc={rc} | bf16 oracle={ta:7.1f}us")
    err = _refmax(oracle, v3(4, 32, None))
    print(f"    max|v3-oracle|={err:.4f}  (tol 2e-2)")
    # PRODUCTION PATH: split-K auto, now with pipelined tl.range loops.
    # Sweep compressed-loop pipelining depth NS_C x tile width BLOCK_K.
    best = None
    for ns_c in (1, 2, 3, 4):
        for bk in (16, 32):
            t = _bench(lambda: v3(ns_c, bk, None))
            tag = f"split-K ns_c={ns_c} bk={bk:>2}"
            print(f"    {tag} : {t:7.1f}us  [{t/ta:5.2f}x]")
            if best is None or t < best[0]:
                best = (t, tag)
    print(f"    >>> BEST: {best[1]}  {best[0]:.1f}us  [{best[0]/ta:.2f}x]")


def main():
    assert torch.cuda.is_available()
    print(f"device={torch.cuda.get_device_name(0)}")
    for i, c in enumerate([
        dict(T=1, H=128, kv_len=2048, rc=1.0),
        dict(T=1, H=128, kv_len=8192, rc=1.0),
        dict(T=16, H=128, kv_len=4096, rc=0.5),
        dict(T=32, H=128, kv_len=4096, rc=0.5),
        dict(T=32, H=128, kv_len=16384, rc=1.0),
        dict(T=128, H=128, kv_len=8192, rc=0.9),
    ]):
        _case(D=512, swa_pages=2048, C=32768, seed=i, **c)
    print("BENCH DONE")


if __name__ == "__main__":
    main()
