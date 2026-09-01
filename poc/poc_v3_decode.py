"""v3 SoA fp8 decode correctness vs bf16 oracle (monotone [swa|comp] indices)."""
import torch
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode_soa_v3 import (
    sparse_attn_v4_paged_decode_soa_v3,
)
from poc_v3_common import quant_soa


def _mono(T, swa_pages, C, rc, dev, seed):
    torch.manual_seed(seed + 1)
    tot = torch.randint(4, 48, (T,), device=dev)
    cl = (tot.float() * rc).round().to(torch.int64).clamp(max=tot)
    sl = tot - cl
    indptr = torch.zeros(T + 1, dtype=torch.int32, device=dev)
    indptr[1:] = torch.cumsum(tot, 0)
    idx = torch.empty(int(indptr[-1]), dtype=torch.int32, device=dev)
    off = 0
    for t in range(T):
        s, c = int(sl[t]), int(cl[t])
        if s: idx[off:off+s] = torch.randint(0, swa_pages, (s,), device=dev, dtype=torch.int32)
        if c: idx[off+s:off+s+c] = swa_pages + torch.randint(0, C, (c,), device=dev, dtype=torch.int32)
        off += s + c
    return idx, indptr, sl.to(torch.int32)


def _run(*, T, H, D, swa_pages, C, seed=0, rc=0.5):
    torch.manual_seed(seed)
    dev = "cuda"; dt = torch.bfloat16
    q = torch.randn(T, H, D, device=dev, dtype=dt) * 0.5
    swa = torch.randn(swa_pages, D, device=dev, dtype=dt) * 0.5
    comp = torch.randn(C, D, device=dev, dtype=dt) * 0.5
    sink = torch.randn(H, device=dev, dtype=torch.float32)
    sc = 1.0 / (D ** 0.5)
    nope_fp8, rope_bf16, scale_f32, comp_dq = quant_soa(comp)
    unified = torch.cat([swa, comp_dq], 0).contiguous()
    idx, indptr, swa_len = _mono(T, swa_pages, C, rc, dev, seed)
    ref = sparse_attn_v4_paged_decode(q, unified, idx, indptr, sink, sc)
    got = sparse_attn_v4_paged_decode_soa_v3(
        q, swa, nope_fp8, rope_bf16, scale_f32, idx, indptr, swa_len, sink, sc,
        swa_pages=swa_pages,
    )
    d = (got.float() - ref.float()).abs()
    ma = d.max().item(); rel = ma / (ref.float().abs().max().item() + 1e-6)
    ok = torch.allclose(got, ref, atol=2e-2, rtol=2e-2)
    print(f"[T={T} H={H} swa={swa_pages} C={C} rc={rc}] max_abs={ma:.3e} rel={rel:.3e} -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    assert torch.cuda.is_available()
    cases = [
        dict(T=1, H=16, D=512, swa_pages=256, C=1024, rc=1.0),
        dict(T=1, H=16, D=512, swa_pages=256, C=1024, rc=0.0),
        dict(T=1, H=128, D=512, swa_pages=512, C=4096, rc=0.5),
        dict(T=16, H=128, D=512, swa_pages=512, C=4096, rc=0.5),
        dict(T=32, H=64, D=512, swa_pages=1024, C=8192, rc=0.7),
    ]
    ok = True
    for i, c in enumerate(cases):
        ok &= _run(seed=i, **c)
    print("ALL OK" if ok else "SOME FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
