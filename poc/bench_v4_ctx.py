"""DSV4 real-shape long-context decode bench.
Shapes: D=512 (448 nope-fp8 + 64 rope-bf16), H=128 heads.
kv_len in {128K, 256K}; bs in {1,16,32,64,128}; rc=0.9 (~swa_full_tokens_ratio 0.075).
Step 1: PRECISION (v3 vs bf16 oracle over identical dequant KV). Step 2: PERF table.
"""
import time, torch
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode_soa_v3 import (
    sparse_attn_v4_paged_decode_soa_v3,
)
from poc_v3_common import quant_soa

D, H = 512, 128
KVS = {"128K": 131072, "256K": 262144}
BS = [1, 16, 32, 64, 128]
RC = 0.9


def _bench(fn, iters, warmup):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def _build(T, kv_len, seed):
    dev = "cuda"; dt = torch.bfloat16
    torch.manual_seed(seed)
    cl = int(round(kv_len * RC)); sl = kv_len - cl
    swa_pages = max(sl + 1024, kv_len // 8)
    C = cl + 1024
    q = torch.randn(T, H, D, device=dev, dtype=dt) * 0.5
    swa = torch.randn(swa_pages, D, device=dev, dtype=dt) * 0.5
    comp = torch.randn(C, D, device=dev, dtype=dt) * 0.5
    sink = torch.randn(H, device=dev, dtype=torch.float32); sc = 1.0 / (D ** 0.5)
    nope_fp8, rope_bf16, scale_f32, comp_dq = quant_soa(comp)
    unified = torch.cat([swa, comp_dq], 0).contiguous()
    indptr = torch.arange(0, (T + 1) * kv_len, kv_len, dtype=torch.int32, device=dev)
    parts = []
    for _t in range(T):
        s = torch.randint(0, swa_pages, (sl,), device=dev, dtype=torch.int32)
        c = swa_pages + torch.randint(0, C, (cl,), device=dev, dtype=torch.int32)
        parts.append(torch.cat([s, c]))
    idx = torch.cat(parts).to(torch.int32)
    swa_len = torch.full((T,), sl, dtype=torch.int32, device=dev)
    return dict(q=q, swa=swa, unified=unified, nope_fp8=nope_fp8, rope_bf16=rope_bf16,
                scale_f32=scale_f32, sink=sink, sc=sc, idx=idx, indptr=indptr,
                swa_len=swa_len, swa_pages=swa_pages)


def _oracle(d):
    return sparse_attn_v4_paged_decode(d["q"], d["unified"], d["idx"], d["indptr"], d["sink"], d["sc"])

def _v3(d):
    return sparse_attn_v4_paged_decode_soa_v3(
        d["q"], d["swa"], d["nope_fp8"], d["rope_bf16"], d["scale_f32"],
        d["idx"], d["indptr"], d["swa_len"], d["sink"], d["sc"], swa_pages=d["swa_pages"])


def main():
    assert torch.cuda.is_available()
    print(f"device={torch.cuda.get_device_name(0)}  D={D} H={H} rc={RC}")

    print("\n########## STEP 1: PRECISION (v3 vs bf16 oracle) ##########")
    ok = True
    for name, L in KVS.items():
        for T in (1, 32, 128):
            d = _build(T, L, seed=hash((name, T)) & 0xffff)
            a = _oracle(d).float(); b = _v3(d).float()
            mabs = (a - b).abs().max().item()
            rel = ((a - b).abs().max() / a.abs().max().clamp(min=1e-6)).item()
            flag = "OK" if mabs < 2e-2 else "FAIL"
            if mabs >= 2e-2: ok = False
            print(f"  kv={name} bs={T:>3}: max_abs={mabs:.3e} rel={rel:.3e} -> {flag}")
            del d, a, b; torch.cuda.empty_cache()
    print("PRECISION", "ALL OK" if ok else "FAILED")

    print("\n########## STEP 2: PERFORMANCE (us/iter, ratio vs bf16) ##########")
    print(f"{'kv_len':>7} {'bs':>4} | {'bf16(us)':>10} {'v3-fp8(us)':>11} {'ratio':>7}")
    rows = {}
    for name, L in KVS.items():
        for T in BS:
            d = _build(T, L, seed=hash((name, T, 'p')) & 0xffff)
            # heavier cases -> fewer iters
            big = (L * T)
            iters = 30 if big <= 4_000_000 else (15 if big <= 16_000_000 else 8)
            warm = max(3, iters // 4)
            ta = _bench(lambda: _oracle(d), iters, warm)
            tb = _bench(lambda: _v3(d), iters, warm)
            rows[(name, T)] = (ta, tb)
            print(f"{name:>7} {T:>4} | {ta:10.1f} {tb:11.1f} {tb/ta:6.2f}x")
            del d; torch.cuda.empty_cache()
    print("\n########## SUMMARY TABLE ##########")
    print(f"{'bs':>4} | " + " | ".join(f"{n:>16}" for n in KVS))
    for T in BS:
        cells = []
        for n in KVS:
            ta, tb = rows[(n, T)]
            cells.append(f"{tb/ta:5.2f}x ({tb:6.0f}/{ta:6.0f})")
        print(f"{T:>4} | " + " | ".join(f"{c:>16}" for c in cells))
    print("BENCH DONE")


if __name__ == "__main__":
    main()
