"""Bench W_o (bf16 vs fp8+fast quant) at decode shapes.

DSV4-Pro decode (TP=12, DP=12, dp_attention=True): attn_tp_size=1 typically,
so n_local_groups = n_groups = 16. K = n_heads*head_dim/n_groups = 4096.
Decode T = batch_size in [1, 1024].
"""
import time, torch
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.fast_fp8_quant import fast_per_token_group_quant_fp8_128

D, R = 4096, 1024
device = "cuda"


def bench(func, n_iter=300, n_warm=20):
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


def run(T, G):
    torch.manual_seed(0)
    o = torch.randn(T, G, D, dtype=torch.bfloat16, device=device) * 0.1
    wo_a_bf16 = torch.randn(G, R, D, dtype=torch.bfloat16, device=device) * 0.01

    out_bf = torch.empty(T, G, R, dtype=torch.bfloat16, device=device)

    def bf16_einsum():
        out_bf.copy_(torch.einsum("tgd,grd->tgr", o, wo_a_bf16))

    import deep_gemm
    wo_a_fp8 = (torch.randn(G, R, D, dtype=torch.float32, device=device) * 0.01).to(torch.float8_e4m3fn)
    wo_a_scales = torch.ones(G, R, D // 128, dtype=torch.float32, device=device)
    out_fp8 = torch.empty(T, G, R, dtype=torch.bfloat16, device=device)

    def fp8_path_fast():
        o_fp8, o_s = fast_per_token_group_quant_fp8_128(o.reshape(T * G, D).contiguous())
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
            (wo_a_fp8.view(G, R, D), wo_a_scales),
            out_fp8,
            recipe=(1, 1, 128),
        )

    def fp8_path_sgl():
        o_fp8, o_s = sglang_per_token_group_quant_fp8(
            o.reshape(T * G, D).contiguous(), group_size=128
        )
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
            (wo_a_fp8.view(G, R, D), wo_a_scales),
            out_fp8,
            recipe=(1, 1, 128),
        )

    t_bf = bench(bf16_einsum)
    t_fp8_fast = bench(fp8_path_fast)
    t_fp8_sgl = bench(fp8_path_sgl)
    return t_bf, t_fp8_sgl, t_fp8_fast


def main():
    print(f"{'T':>5} {'G':>3} {'bf16':>8} {'fp8+sgl':>9} {'fp8+fast':>10} {'fast vs bf16':>14}")
    for T in (128, 256, 400, 512, 768, 1024):
        for G in (16,):  # decode n_local_groups=16 with attn_tp=1
            t_bf, t_fp8_sgl, t_fp8_fast = run(T, G)
            delta_pct = 100 * (t_bf - t_fp8_fast) / t_bf
            print(
                f"{T:>5} {G:>3} {t_bf:>8.2f} {t_fp8_sgl:>9.2f} {t_fp8_fast:>10.2f} "
                f"{delta_pct:>+12.1f}%"
            )


if __name__ == "__main__":
    main()
