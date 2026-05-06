"""Bench fast triton FP8 quant vs sgl_kernel's at the W_o shape."""
import time, torch
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.fast_fp8_quant import fast_per_token_group_quant_fp8_128

T, G, D = 8192, 4, 4096
M, K = T * G, D


def bench_graph(func, n_iter=300, n_warm=20):
    for _ in range(n_warm):
        func()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        func()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n_iter):
            g.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6 / n_iter)
    return min(times)


def main():
    torch.manual_seed(0)
    x = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous()

    # Correctness
    q_a, s_a = sglang_per_token_group_quant_fp8(x, group_size=128)
    q_b, s_b = fast_per_token_group_quant_fp8_128(x)
    # Compare in fp32: max abs diff between (q*s) reconstructions
    rec_a = q_a.float() * s_a.repeat_interleave(128, dim=-1)
    rec_b = q_b.float() * s_b.repeat_interleave(128, dim=-1)
    err = (rec_a - rec_b).abs().max().item()
    err_x = (rec_b - x.float()).abs().max().item()
    print(f"correctness: max|reconstruct(sgl) - reconstruct(fast)| = {err:.4e}")
    print(f"             max|x - reconstruct(fast)|                = {err_x:.4e}")

    t_sgl = bench_graph(lambda: sglang_per_token_group_quant_fp8(x, group_size=128))
    t_fast = bench_graph(lambda: fast_per_token_group_quant_fp8_128(x))

    print(f"shape: M={M} K={K} group=128")
    print(f"  sgl_kernel: {t_sgl:7.2f} us")
    print(f"  fast tri:   {t_fast:7.2f} us  ({100*(t_sgl-t_fast)/t_sgl:+.1f}%)")


if __name__ == "__main__":
    main()
