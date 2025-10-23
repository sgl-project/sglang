import argparse
import itertools

import torch
import triton
from sgl_kernel import scaled_fp4_grouped_quant, silu_and_mul_scaled_fp4_grouped_quant
from sgl_kernel.elementwise import silu_and_mul

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_post_quant_fwd


def _test_accuracy_once(E, M, K, input_dtype, device):
    x = torch.randn(E, M, K, device=device, dtype=input_dtype)
    glb_scales = torch.ones((E,), dtype=torch.float32, device=device)
    masks = torch.full((E,), M, dtype=torch.int32, device=device)
    out, blk_scales = silu_and_mul_scaled_fp4_grouped_quant(x, glb_scales, masks)
    out1, blk_scales1 = scaled_fp4_grouped_quant(
        silu_and_mul(x),
        glb_scales,
        masks,
    )

    torch.testing.assert_close(out, out1)
    torch.testing.assert_close(blk_scales, blk_scales1)
    print(f"E: {E}, M: {M}, K: {K}, type: {input_dtype} OK")


NUM_RANKS = 48
M_PER_RANKs = [128, 256, 512, 1024]
Ms = [M_PER_RANK * NUM_RANKS for M_PER_RANK in M_PER_RANKs]
Ks = [2048, 4096, 7168]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "K"],
        x_vals=list(itertools.product(Ms, Ks)),
        x_log=False,
        line_arg="provider",
        line_vals=["triton_fp8", "cuda_unfused_fp4", "cuda_fused_fp4"],
        line_names=["triton_fp8", "cuda_unfused_fp4", "cuda_fused_fp4"],
        styles=[("blue", "-"), ("orange", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="fp4 quant",
        args={},
    )
)
def benchmark(M, K, provider):
    E = 6
    device = "cuda"
    x = torch.randn(E, M, K, device=device, dtype=torch.bfloat16)
    glb_scales = torch.ones((E,), dtype=torch.float32, device=device)
    masks = torch.randint(1, 4096, (E,), dtype=torch.int32, device=device)
    fp8_out = torch.empty(
        (
            x.shape[0],
            x.shape[1],
            x.shape[2] // 2,
        ),
        device=x.device,
        dtype=torch.float8_e4m3fn,
    )
    scale_block_size = 128
    fp8_scales = torch.empty(
        (
            x.shape[0],
            x.shape[1],
            x.shape[2] // 2 // scale_block_size,
        ),
        device=x.device,
        dtype=torch.float32,
    )

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton_fp8":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: silu_and_mul_masked_post_quant_fwd(
                x,
                fp8_out,
                fp8_scales,
                scale_block_size,
                masks,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            ),
            quantiles=quantiles,
        )
    if provider == "cuda_unfused_fp4":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: scaled_fp4_grouped_quant(
                silu_and_mul(x),
                glb_scales,
                masks,
            ),
            quantiles=quantiles,
        )
    if provider == "cuda_fused_fp4":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: silu_and_mul_scaled_fp4_grouped_quant(
                x,
                glb_scales,
                masks,
            ),
            quantiles=quantiles,
        )

    return ms, min_ms, max_ms


def test_accuracy():
    E = 6
    N_RANKS = 48
    Ms = [128, 256, 512, 1024]
    Ks = [2048, 4096, 7168]
    input_dtype = torch.bfloat16
    for M in Ms:
        for K in Ks:
            _test_accuracy_once(E, N_RANKS * M, K, input_dtype, "cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./bench_fp4_quant_res",
        help="Path to save fp4 quant benchmark results",
    )
    args = parser.parse_args()

    test_accuracy()

    benchmark.run(print_data=True, show_plots=True, save_path=args.save_path)
