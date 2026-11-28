"""GPU check of the fused Kimi-K3.1 DSA decode / verify prologue (dsa_prologue_hip) against
SGLang's stock ROCm launches on TP8 / DCP8 shapes: exactness / max abs diff and per-call
time (CUDA-graph replay). One gfx950 GPU, nothing else running on it:

    python test/manual/kernels/test_dsa_prologue.py [--rows 8,24,48,96,192,288] [--only gemm,index,fp4,kv,all]

Per layer and TP8 rank the stock prologue it replaces is:
  gemm   fused_qkv_a (N 2112) + [wk | weights_proj] (N 160) + g_proj (N 1536), K 7168, bf16
  index  k_norm LayerNorm (rms_norm_gated) + AITER indexer_k_quant_and_cache (FP8, DCP8 rows)
         + indexer_q_quant_head_gate                    (fp4: + store_fp4_index_k / q fp4 gate)
  kv     quantize_k_cache_separate + set_mla_kv_buffer_dcp_sharded (fp8_ds_mla rows)
The fused side is the merged GEMM (N 3808) and one launch for each of index / kv.
"""

import argparse
import json
import sys

import torch

H = 7168
Q_LORA, KV_A, KEY, IDX_HEADS, GATE = 1536, 576, 128, 32, 12 * 128
DCP, RANK = 8, 3
PAGE = 64
EPS = 1e-6


def graph_us(fn, iters=50, reps=5):
    """Per-call us of ``fn`` replayed in a CUDA graph of ``iters`` calls (min of reps)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    best = float("inf")
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) * 1000 / iters)
    return best


def _loc(rows, pages):
    # DCP8 widened slots spread over every rank, a few padding (-1) rows
    loc = torch.randperm(pages * PAGE * DCP, device="cuda")[:rows].to(torch.int64)
    loc[rows - 1 :: 7] = -1
    return loc


def case_gemm(m, out):
    from types import SimpleNamespace

    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    method = UnquantizedLinearMethod()

    def mm(x, w):
        # the route ReplicatedLinear / ColumnParallelLinear bf16 weights take
        return method.apply(SimpleNamespace(weight=w), x)

    x = torch.randn(m, H, device="cuda").bfloat16()
    ws = [
        (torch.randn(n, H, device="cuda") * H**-0.5).bfloat16()
        for n in (Q_LORA + KV_A, KEY + IDX_HEADS, GATE)
    ]
    merged = torch.cat(ws)

    def stock():
        return [mm(x, w) for w in ws]

    def fused():
        return mm(x, merged)

    ref = torch.cat(stock(), dim=-1)
    got = fused()
    diff = (got.float() - ref.float()).abs().max().item()
    out.update(
        gemm_maxabs=diff,
        gemm_ref_absmax=ref.float().abs().max().item(),
        gemm_stock_us=graph_us(stock),
        gemm_fused_us=graph_us(fused),
    )


def _index_inputs(m):
    front = (torch.randn(m, 3808, device="cuda") * 3).bfloat16()
    key = front[:, 2112 : 2112 + KEY]
    gate = front[:, 2240 : 2240 + IDX_HEADS]
    query = (torch.randn(m, IDX_HEADS, 128, device="cuda") * 2).bfloat16()
    w = 1 + 0.1 * torch.randn(128, device="cuda")
    b = 0.1 * torch.randn(128, device="cuda")
    return query, gate, key, w, b


def _stock_layer_norm(key, w, b):
    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated

    return rms_norm_gated(x=key, weight=w, bias=b, eps=EPS, is_rms_norm=False)


def _local_loc(loc):
    owned = (loc >= 0) & (loc % DCP == RANK)
    return torch.where(owned, loc // DCP, -1)


def case_index_fp8(m, out):
    from aiter.ops.cache import indexer_k_quant_and_cache

    from sglang.kernels.ops.attention.dsa import dsa_prologue_hip as fused_ops
    from sglang.kernels.ops.attention.dsa.head_gate import indexer_q_quant_head_gate

    query, gate, key, w, b = _index_inputs(m)
    pages = max(8, m)
    loc = _loc(m, pages)
    local = _local_loc(loc)  # stock: memoized once per forward, not per layer
    caches = [
        torch.zeros(pages, PAGE * 132, dtype=torch.uint8, device="cuda")
        for _ in range(2)
    ]
    sm = 128**-0.5

    def stock():
        kv = caches[0].view(-1, PAGE, 132).view(torch.float8_e4m3fn)
        indexer_k_quant_and_cache(
            _stock_layer_norm(key, w, b), kv, local, 128, "ue8m0", preshuffle=True
        )
        return indexer_q_quant_head_gate(query, gate, sm, "ue8m0")

    def fused():
        return fused_ops.index_qk_prep_fp8(
            query=query,
            gate=gate,
            softmax_scale=sm,
            scale_fmt="ue8m0",
            key=key,
            norm_weight=w,
            norm_bias=b,
            eps=EPS,
            index_k_cache=caches[1],
            loc=loc,
            dcp_size=DCP,
            dcp_rank=RANK,
        )

    q_ref, w_ref = stock()
    q_got, w_got = fused()
    out.update(
        idx8_cache_bytes_diff=int((caches[0] != caches[1]).sum()),
        idx8_q_codes_diff=int(
            (q_ref.view(torch.uint8) != q_got.view(torch.uint8)).sum()
        ),
        idx8_weights_maxabs=(w_ref - w_got).abs().max().item(),
        idx8_stock_us=graph_us(stock),
        idx8_fused_us=graph_us(fused),
    )


def case_index_fp4(m, out):
    from sglang.kernels.ops.attention.dsa import dsa_prologue_hip as fused_ops
    from sglang.kernels.ops.attention.dsa.fp4_index_hip import (
        indexer_q_fp4_head_gate,
        store_fp4_index_k,
    )

    query, gate, key, w, b = _index_inputs(m)
    pages = max(8, m)
    loc = _loc(m, pages)
    local = _local_loc(loc)
    bufs = [
        (
            torch.zeros(pages, 4096, dtype=torch.uint8, device="cuda"),
            torch.zeros(pages, 256, dtype=torch.uint8, device="cuda"),
        )
        for _ in range(2)
    ]
    sm = 128**-0.5

    def stock():
        store_fp4_index_k(_stock_layer_norm(key, w, b), *bufs[0], local)
        return indexer_q_fp4_head_gate(query, gate, sm)

    def fused():
        return fused_ops.index_qk_prep_fp4(
            query=query,
            gate=gate,
            softmax_scale=sm,
            key=key,
            norm_weight=w,
            norm_bias=b,
            eps=EPS,
            payload=bufs[1][0],
            scale=bufs[1][1],
            loc=loc,
            dcp_size=DCP,
            dcp_rank=RANK,
        )

    (q_ref, w_ref), (q_got, w_got) = stock(), fused()
    out.update(
        idx4_cache_bytes_diff=int((bufs[0][0] != bufs[1][0]).sum())
        + int((bufs[0][1] != bufs[1][1]).sum()),
        idx4_q_bytes_diff=int((q_ref.payload != q_got.payload).sum())
        + int((q_ref.scale != q_got.scale).sum()),
        idx4_weights_maxabs=(w_ref - w_got).abs().max().item(),
        idx4_stock_us=graph_us(stock),
        idx4_fused_us=graph_us(fused),
    )


def case_kv(m, out):
    from sglang.kernels.ops.attention.dsa import dsa_prologue_hip as fused_ops
    from sglang.kernels.ops.kvcache.mla_buffer import _set_mla_kv_buffer_impl
    from sglang.kernels.ops.quantization.dsa_quant_k_cache import (
        quantize_k_cache_separate,
    )

    latent = (torch.randn(m, 3808, device="cuda") * 4).bfloat16()
    nope = latent[:, 1536:2048].unsqueeze(1)
    rope = latent[:, 2048:2112].unsqueeze(1)
    rows = max(64, m) * 2
    loc = torch.randperm(rows * DCP, device="cuda")[:m]
    bufs = [
        torch.zeros(rows, 1, 656, dtype=torch.uint8, device="cuda") for _ in range(2)
    ]

    def stock():
        n, r = quantize_k_cache_separate(nope, rope, scale_ue8m0=True)
        _set_mla_kv_buffer_impl(
            bufs[0], loc, n, r, reserved_skip_index=0, dcp_world_size=DCP, dcp_rank=RANK
        )

    def fused():
        fused_ops.fp8_ds_mla_store(
            bufs[1], loc, nope, rope, scale_ue8m0=True, dcp_size=DCP, dcp_rank=RANK
        )

    stock()
    fused()
    out.update(
        kv_bytes_diff=int((bufs[0] != bufs[1]).sum()),
        kv_stock_us=graph_us(stock),
        kv_fused_us=graph_us(fused),
    )


CASES = {
    "gemm": case_gemm,
    "index": case_index_fp8,
    "fp4": case_index_fp4,
    "kv": case_kv,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="8,24,48,96,192,288")
    p.add_argument("--only", default="all")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    torch.manual_seed(0)
    names = list(CASES) if args.only == "all" else args.only.split(",")
    results = []
    for m in [int(r) for r in args.rows.split(",")]:
        out = {"rows": m}
        for name in names:
            try:
                CASES[name](m, out)
            except Exception as e:  # report and keep going: one case per line
                out[f"{name}_error"] = repr(e)[:300]
        stock = sum(
            v for k, v in out.items() if k.endswith("_stock_us") and "idx4" not in k
        )
        fused = sum(
            v for k, v in out.items() if k.endswith("_fused_us") and "idx4" not in k
        )
        out["prologue_fp8_saved_us_per_layer"] = stock - fused
        results.append(out)
        print(json.dumps(out), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    bad = [
        r
        for r in results
        if any(
            r.get(k, 0)
            for k in (
                "idx8_cache_bytes_diff",
                "idx4_cache_bytes_diff",
                "kv_bytes_diff",
                "idx8_q_codes_diff",
                "idx4_q_bytes_diff",
            )
        )
    ]
    if bad:
        print(
            f"NOTE: {len(bad)} row counts differ in stored bytes; see README (LayerNorm "
            "reduction order can move a key by 1 bf16 ulp)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
