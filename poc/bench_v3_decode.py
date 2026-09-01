"""v3 SoA fp8 decode perf vs bf16 baseline AND the existing clean-SoA QUANT_KV."""
import time
import torch
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    sparse_attn_v4_paged_decode, _FP8_DTYPE, _FP8_GROUP_SIZE,
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


def _quant512(kv):
    N, D = kv.shape; G = _FP8_GROUP_SIZE
    x = kv.float().view(N, D // G, G)
    fmax = torch.finfo(_FP8_DTYPE).max
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-6) / fmax
    q = (x / s).clamp(-fmax, fmax).to(_FP8_DTYPE).view(N, D)
    return q.contiguous(), s.squeeze(-1).contiguous().float()


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


def _case(*, T, H, D, swa_pages, C, kv_len, rc, seed=0):
    torch.manual_seed(seed); dev = "cuda"; dt = torch.bfloat16
    q = torch.randn(T, H, D, device=dev, dtype=dt) * 0.5
    swa = torch.randn(swa_pages, D, device=dev, dtype=dt) * 0.5
    comp = torch.randn(C, D, device=dev, dtype=dt) * 0.5
    sink = torch.randn(H, device=dev, dtype=torch.float32); sc = 1.0 / (D ** 0.5)
    nope_fp8, rope_bf16, scale_f32, comp_dq = quant_soa(comp)
    unified = torch.cat([swa, comp_dq], 0).contiguous()
    fp8_kv, kv_scales = _quant512(unified)
    idx, indptr, swa_len = _mono(T, kv_len, swa_pages, C, rc, dev, seed)
    A = lambda: sparse_attn_v4_paged_decode(q, unified, idx, indptr, sink, sc)
    B = lambda: sparse_attn_v4_paged_decode(q, fp8_kv, idx, indptr, sink, sc, kv_scales=kv_scales)
    Cc = lambda: sparse_attn_v4_paged_decode_soa_v3(q, swa, nope_fp8, rope_bf16, scale_f32, idx, indptr, swa_len, sink, sc, swa_pages=swa_pages)
    ta, tb, tc = _bench(A), _bench(B), _bench(Cc)
    print(f"T={T:>3} H={H:>3} kv_len={kv_len:>5} rc={rc:.1f} | (A)bf16={ta:7.1f}  (B)fp8-512-SoA={tb:8.1f}[{tb/ta:5.2f}x]  (C)v3-nopeFp8={tc:7.1f}[{tc/ta:5.2f}x]")


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
