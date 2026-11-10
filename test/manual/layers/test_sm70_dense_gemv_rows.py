"""Two to four verification rows on the GLM-5.3 TP8 projection shapes must be
bitwise the one-row SM70 GEMV, row by row. Prints time against cuBLAS.

Not a CI test:

  SGLANG_SM70_DENSE_GEMV=1 CUDA_VISIBLE_DEVICES=0 python test/manual/layers/test_sm70_dense_gemv_rows.py
"""

import torch

from sglang.kernels.ops.gemm import sm70_dense_gemv as g

SHAPES = sorted(g._ROW_EXACT)


def bench(fn, iters=200):
    for _ in range(10):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


def main() -> None:
    torch.manual_seed(0)
    ok = True
    for n, k in SHAPES:
        w = torch.randn(n, k, device="cuda").half() * 0.05
        for m in (2, 3, 4):
            x = torch.randn(m, k, device="cuda").half()
            assert g.supported(x, w), (m, n, k)
            got = g.linear(x, w)
            ref = torch.cat([g.linear(x[i : i + 1].contiguous(), w) for i in range(m)])
            same = torch.equal(got.view(torch.int16), ref.view(torch.int16))
            ok &= same
            t_rows = bench(lambda: g.linear(x, w))
            t_one = bench(lambda: g.linear(x[:1].contiguous(), w))
            t_blas = bench(lambda: torch.matmul(x, w.t()))
            print(
                f"M={m} N={n:5d} K={k:5d} {'bitwise' if same else 'DIFFER':8s} "
                f"rows {t_rows:6.1f} us  one-row {t_one:6.1f} us  cuBLAS {t_blas:6.1f} us"
            )
    assert ok, "multi-row GEMV differs from the one-row GEMV"


if __name__ == "__main__":
    main()
