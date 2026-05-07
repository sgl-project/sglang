"""Bench W_o (wo_a) projection: bf16 einsum vs FP8 quant + fp8_einsum.
Production shapes for DSV4-Pro at TP=4: T=8192, G=4 (n_groups/4),
D=128*512/16=4096, R=o_lora_rank=1024.
"""
import time, torch
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

T, G, D, R = 8192, 4, 4096, 1024
device = "cuda"
torch.manual_seed(0)


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
    o = torch.randn(T, G, D, dtype=torch.bfloat16, device=device) * 0.1
    wo_a_bf16 = torch.randn(G, R, D, dtype=torch.bfloat16, device=device) * 0.01

    # bf16 path
    out_bf16 = torch.empty(T, G, R, dtype=torch.bfloat16, device=device)

    def bf16_einsum():
        out_bf16.copy_(torch.einsum("tgd,grd->tgr", o, wo_a_bf16))

    # fp8 path: pre-quantize o, then fp8_einsum
    import deep_gemm

    # Build fp8 weights with scales (E4M3 + per-128-channel ue8m0 scales)
    # Mimic what the model loader produces.
    wo_a_fp8 = torch.randn(G, R, D, dtype=torch.float32, device=device) * 0.01
    wo_a_fp8_q = wo_a_fp8.to(torch.float8_e4m3fn)
    # weight_scale_inv format: per-128 channel along D
    wo_a_scales = torch.ones(G, R, D // 128, dtype=torch.float32, device=device)

    out_fp8 = torch.empty(T, G, R, dtype=torch.bfloat16, device=device)

    from sglang.srt.layers.fast_fp8_quant import fast_per_token_group_quant_fp8_128

    def fp8_quant_sgl():
        return sglang_per_token_group_quant_fp8(
            o.reshape(T * G, D).contiguous(), group_size=128, enable_v2=True
        )

    def fp8_quant_fast():
        return fast_per_token_group_quant_fp8_128(
            o.reshape(T * G, D).contiguous()
        )

    def fp8_full():
        o_fp8, o_s = fast_per_token_group_quant_fp8_128(
            o.reshape(T * G, D).contiguous()
        )
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
            (wo_a_fp8_q.view(G, R, D), wo_a_scales),
            out_fp8,
            recipe=(1, 1, 128),
        )

    # Bench
    t_bf16 = bench_graph(bf16_einsum)
    t_quant_sgl = bench_graph(fp8_quant_sgl)
    t_quant_fast = bench_graph(fp8_quant_fast)
    t_fp8 = bench_graph(fp8_full)

    print(f"shapes: T={T} G={G} K=D={D} N=R={R}  (per-group GEMM)")
    print(f"  bf16 einsum:               {t_bf16:7.2f} us")
    print(f"  fp8 quant (sgl_kernel):    {t_quant_sgl:7.2f} us")
    print(f"  fp8 quant (fast triton):   {t_quant_fast:7.2f} us  ({100*(t_quant_sgl-t_quant_fast)/t_quant_sgl:+.1f}% vs sgl)")
    print(f"  fp8 GEMM only:             {t_fp8 - t_quant_fast:7.2f} us")
    print(f"  fp8 full path (fast):      {t_fp8:7.2f} us")
    print(f"  delta(bf16 - fp8 fast):    {t_bf16 - t_fp8:+7.2f} us  ({100*(t_bf16-t_fp8)/t_bf16:+.1f}%)")
    print()
    print(f"per-forward at T={T}, 60 layers: ~{(t_bf16 - t_fp8)*60/1000:.2f} ms saved")


if __name__ == "__main__":
    main()
