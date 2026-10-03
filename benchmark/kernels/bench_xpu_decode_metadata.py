"""Benchmark: decode page-table / metadata preparation on Intel XPU.

Compares the current PyTorch sequence in
``XPUAttentionBackend.init_forward_metadata_out_graph`` (python/sglang/srt/layers/
attention/xpu_backend.py, non-encoder-decoder branch) against the fused Triton
helper ``normal_decode_set_metadata`` (python/sglang/kernels/ops/attention/
metadata.py) that the NVIDIA FlashAttention backend already uses.

Checks exact equality of cache_seqlens, cu_seqlens_k and all *live* page-table
entries, then reports host-inclusive latency per call.

Run (inside an XPU container with torch+xpu and triton-xpu):
    python3 benchmark/kernels/bench_xpu_decode_metadata.py [--iters 200] [--markdown]
"""

import argparse
import importlib.util
import os
import sys
import time

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HELPER_PATH = os.path.join(
    REPO_ROOT, "python", "sglang", "kernels", "ops", "attention", "metadata.py"
)


def load_helper():
    """Load the fused helper straight from the repo file.
    Called with keyword args so both the older signature (with strided_indices)
    and the current one work. (avoids importing the
    whole sglang package, which needs sgl_kernel)."""
    spec = importlib.util.spec_from_file_location("sgl_attn_metadata", HELPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sgl_attn_metadata"] = mod
    spec.loader.exec_module(mod)
    return mod.normal_decode_set_metadata


def xpu_reference(
    cache_seqlens,
    cu_seqlens_k,
    page_table,
    req_to_token,
    req_pool_indices,
    strided_indices,
    seq_lens,
    page_size,
    max_len,
):
    """Verbatim logic of xpu_backend.py init_forward_metadata_out_graph
    (cache_seqlens / cu_seqlens_k fill + the non-encoder-decoder page-table block)."""
    bs = req_pool_indices.shape[0]
    cache_seqlens.copy_(seq_lens.to(torch.int32))
    cu_seqlens_k[0] = 0
    cu_seqlens_k[1 : bs + 1].copy_(torch.cumsum(seq_lens.to(torch.int32), dim=0))
    raw_page = req_to_token[
        req_pool_indices[:, None],
        strided_indices[: ((max_len + page_size - 1) // page_size)][None, :],
    ]
    if page_size > 1:
        raw_page = raw_page // page_size
    page_table[:bs, : raw_page.shape[1]].copy_(raw_page.to(torch.int32))
    page_table[:bs, raw_page.shape[1] :].zero_()


def timeit(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def make_inputs(bs, max_ctx, page_size, dev, pool_size=64):
    max_num_pages = (max_ctx + page_size - 1) // page_size
    req_to_token = torch.randint(
        0, 1 << 20, (pool_size, max_ctx), dtype=torch.int64, device=dev
    )
    req_pool_indices = torch.randperm(pool_size, device=dev)[:bs]
    seq_lens = torch.randint(1, max_ctx + 1, (bs,), dtype=torch.int64, device=dev)
    seq_lens[0] = max_ctx  # one request at full width
    strided = torch.arange(0, max_ctx, page_size, device=dev)
    return max_num_pages, req_to_token, req_pool_indices, seq_lens, strided


def bufs(bs, max_num_pages, dev, fill):
    return (
        torch.zeros(bs, dtype=torch.int32, device=dev),
        torch.zeros(bs + 1, dtype=torch.int32, device=dev),
        torch.full((bs, max_num_pages), fill, dtype=torch.int32, device=dev),
    )


def run_case(fused, bs, max_ctx, page_size, iters, dev):
    mnp, r2t, rpi, sl, strided = make_inputs(bs, max_ctx, page_size, dev)
    max_len = int(sl.max().item())

    # ---- correctness ----
    cs_r, cu_r, pt_r = bufs(bs, mnp, dev, -7)
    xpu_reference(cs_r, cu_r, pt_r, r2t, rpi, strided, sl, page_size, max_len)
    cs_f, cu_f, pt_f = bufs(bs, mnp, dev, -7)
    fused(
        cache_seqlens_int32=cs_f,
        cu_seqlens_k=cu_f,
        page_table=pt_f,
        req_to_token=r2t,
        req_pool_indices=rpi,
        max_seq_pages=mnp,
        seq_lens=sl,
        seq_len_delta=0,
        page_size=page_size,
    )
    torch.xpu.synchronize()
    live = (
        torch.arange(mnp, device=dev)[None, :]
        < ((sl + page_size - 1) // page_size)[:, None]
    )
    ok = (
        torch.equal(cs_r, cs_f)
        and torch.equal(cu_r, cu_f)
        and torch.equal(pt_r[live], pt_f[live])
    )
    tail_untouched = bool((pt_f[~live] == -7).all())

    # ---- timing ----
    t_ref = timeit(
        lambda: xpu_reference(
            cs_r, cu_r, pt_r, r2t, rpi, strided, sl, page_size, max_len
        ),
        iters,
    )
    # The real XPU path derives max_len from seq_lens on device each step (.item()),
    # unless seq_lens_cpu is available. Time that variant too.
    t_ref_sync = timeit(
        lambda: xpu_reference(
            cs_r, cu_r, pt_r, r2t, rpi, strided, sl, page_size, int(sl.max().item())
        ),
        iters,
    )
    t_fused = timeit(
        lambda: fused(
            cache_seqlens_int32=cs_f,
            cu_seqlens_k=cu_f,
            page_table=pt_f,
            req_to_token=r2t,
            req_pool_indices=rpi,
            max_seq_pages=mnp,
            seq_lens=sl,
            seq_len_delta=0,
            page_size=page_size,
        ),
        iters,
    )
    return ok, tail_untouched, t_ref, t_ref_sync, t_fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096, 8192])
    ap.add_argument("--page-sizes", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--markdown", action="store_true", help="print a markdown table")
    args = ap.parse_args()

    assert torch.xpu.is_available(), "XPU not available"
    dev = torch.device("xpu")
    torch.manual_seed(args.seed)
    import triton

    print(f"device: {torch.xpu.get_device_name(0)}")
    print(f"torch {torch.__version__}  triton {triton.__version__}  iters={args.iters}")

    fused = load_helper()

    # Global warm-up: JIT-compile both Triton variants and init torch XPU ops
    # so the first measured configuration is not polluted by one-time costs.
    for ps in sorted(set(args.page_sizes)):
        run_case(fused, 4, 1024, ps, iters=5, dev=dev)

    rows = []
    all_ok = True
    for ps in args.page_sizes:
        for bs in args.batch_sizes:
            for ctx in args.contexts:
                ok, tail, t_ref, t_ref_sync, t_fused = run_case(
                    fused, bs, ctx, ps, args.iters, dev
                )
                all_ok &= ok
                rows.append((bs, ctx, ps, ok, tail, t_ref, t_ref_sync, t_fused))

    if args.markdown:
        print(
            "\n| bs | ctx | page | live-eq | pytorch (us) | pytorch+.item sync (us) | fused triton (us) | speedup |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for bs, ctx, ps, ok, tail, a, b, c in rows:
            print(
                f"| {bs} | {ctx} | {ps} | {ok} | {a:.1f} | {b:.1f} | {c:.1f} | {a / c:.1f}x to {b / c:.1f}x |"
            )
    else:
        print(
            f"\n{'bs':>3} {'ctx':>5} {'page':>4} {'live-eq':>7} {'tail-untouched':>14} | "
            f"{'pytorch':>9} {'pytorch+sync':>12} {'fused':>8} | speedup"
        )
        for bs, ctx, ps, ok, tail, a, b, c in rows:
            print(
                f"{bs:>3} {ctx:>5} {ps:>4} {str(ok):>7} {str(tail):>14} | "
                f"{a:>7.1f}us {b:>10.1f}us {c:>6.1f}us | {a / c:.1f}x to {b / c:.1f}x"
            )

    print(f"\nALL LIVE-ENTRY + METADATA CHECKS PASSED: {all_ok}")
    print(
        "note: fused helper leaves tail columns untouched (by contract); "
        "the PyTorch block zeroes them."
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
