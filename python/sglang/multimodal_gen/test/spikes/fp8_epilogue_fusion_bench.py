# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: is fusing (FP8 GEMM + col-scale + GELU + FP8-cast) into one
CUTLASS call actually worth it vs. reusing a stock scaled GEMM + separate
elementwise epilogue kernels?

This isolates the *ceiling* on the benefit of the OmniDreams custom fused
linear (cosmos_block.cu: cutlass_linear_layer_rcr_fp8_colscale_gelu_fp8) by
comparing, on the real OmniDreams DiT shapes:

  FUSED  : one launch  -> scaled_mm produces the final tensor directly
  REUSE  : scaled_mm (GEMM+scale, like sgl-kernel fp8_blockwise) THEN a
           separate GELU launch THEN a separate FP8 requant launch.

The delta (REUSE - FUSED-proxy) = extra kernel launches + extra HBM
round-trips of the activation. That delta is the most a fully-fused kernel can
ever save. If it is a few percent, reusing sgl-kernel's GEMM is nearly free; if
it is large, the custom fusion earns its keep.

No sglang imports -- pure torch so it runs on any box.
"""

import argparse
import statistics

import torch


def _supports_fp8() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        m, _ = torch.cuda.get_device_capability(0)
        return m >= 8  # e4m3 scaled_mm on sm89+
    except Exception:
        return False


FP8 = torch.float8_e4m3fn


def _bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return statistics.median(times) * 1000.0  # us


def make_fp8(M, K, N, device):
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16).clamp(-3, 3)
    b = torch.randn(N, K, device=device, dtype=torch.bfloat16).clamp(-3, 3)
    a8 = a.to(FP8)
    b8 = b.to(FP8)  # (N,K); pass b8.t() so it's (K,N) col-major
    sa = torch.tensor(1.0, device=device)
    sb = torch.tensor(1.0, device=device)
    return a8, b8, sa, sb


def run_shape(name, M, K, N, with_gelu, device, iters):
    a8, b8, sa, sb = make_fp8(M, K, N, device)
    bt = b8.t()  # (K, N) column-major view

    # Static (delayed) fp8 scale -- the standard for FP8 inference. NO runtime
    # amax reduction; requant is a single read+write elementwise pass.
    static_scale = torch.tensor(64.0, device=device)

    def gemm_only():
        return torch._scaled_mm(a8, bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

    # FUSED proxy: the single-launch ideal == just the GEMM producing final out.
    t_fused_proxy = _bench(gemm_only, iters)

    # REUSE path: GEMM (+scale) -> [GELU] -> static requant to fp8 (feeds next GEMM)
    def reuse_path():
        c = torch._scaled_mm(a8, bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        if with_gelu:
            c = torch.nn.functional.gelu(c)
        return (c * static_scale).to(FP8)

    t_reuse = _bench(reuse_path, iters)

    # isolate the epilogue-only cost on the output tensor.
    c = torch._scaled_mm(a8, bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

    # (a) eager: gelu + mul + cast = 3 separate torch kernels (naive upper bound)
    def epi_eager():
        x = torch.nn.functional.gelu(c) if with_gelu else c
        return (x * static_scale).to(FP8)

    t_epi_eager = _bench(epi_eager, iters)

    # (b) compiled: Inductor fuses the pointwise chain into ONE kernel == the
    #     fair "reuse with a fused gelu+quant epilogue" (what sgl-kernel/jit
    #     would provide). This is the realistic reuse epilogue cost.
    def _epi(cc):
        x = torch.nn.functional.gelu(cc) if with_gelu else cc
        return (x * static_scale).to(FP8)

    try:
        epi_compiled = torch.compile(_epi, fullgraph=True)
        epi_compiled(c)  # warm compile outside timing
        t_epi_fused = _bench(lambda: epi_compiled(c), iters)
    except Exception as e:
        t_epi_fused = float("nan")
        print(f"  [compile failed: {type(e).__name__}: {e}]")

    # --- DEFINITIVE end-to-end fused-vs-reuse for the no-gelu case ---
    # FUSED real  = scaled_mm writing fp8 directly (one kernel, GEMM+cast).
    # REUSE real  = scaled_mm -> bf16, then compiled cast to fp8 (two kernels).
    true_delta = float("nan")
    if not with_gelu:
        sr = torch.tensor(1.0 / 64.0, device=device)
        try:
            def fused_real():
                return torch._scaled_mm(
                    a8, bt, scale_a=sa, scale_b=sb, scale_result=sr, out_dtype=FP8
                )

            fused_real()
            t_fused_real = _bench(fused_real, iters)

            cast_c = torch.compile(lambda cc: (cc * static_scale).to(FP8), fullgraph=True)
            cast_c(c)

            def reuse_real():
                bf = torch._scaled_mm(a8, bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                return cast_c(bf)

            t_reuse_real = _bench(reuse_real, iters)
            true_delta = (t_reuse_real - t_fused_real) / t_fused_real * 100.0
        except Exception as e:
            print(f"  [fp8-out scaled_mm unavailable: {type(e).__name__}: {e}]")

    ceil_real = t_epi_fused / t_fused_proxy * 100.0
    extra = f" TRUE_fused_gain={true_delta:5.1f}%" if not with_gelu else ""
    print(
        f"{name:24s} M={M:6d} N={N:5d} gelu={int(with_gelu)} | "
        f"gemm={t_fused_proxy:8.1f}us  epi_fused(1k)={t_epi_fused:7.1f}us | "
        f"fusion_ceiling={ceil_real:5.1f}%{extra}"
    )
    return ceil_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn", type=int, default=8192)
    args = ap.parse_args()

    if not _supports_fp8():
        print("FP8 scaled_mm not supported on this device; aborting.")
        return
    device = "cuda"
    H, F = args.hidden, args.ffn
    cap = torch.cuda.get_device_capability(0)
    print(f"# device={torch.cuda.get_device_name(0)} cap={cap} torch={torch.__version__}")
    print(f"# hidden={H} ffn={F}  (OmniDreams DiT: 28 blocks, 7 GEMMs/block)")
    print("# fusion_ceiling = max % a fully-fused kernel can save vs reuse(GEMM)+sep epilogue\n")

    # M = tokens per AR chunk (sweep latency-bound -> compute-bound)
    for M in (256, 1024, 4096, 16384):
        run_shape("qkv/out proj (no gelu)", M, H, H, False, device, args.iters)
        run_shape("FFN gemm1 (+gelu+fp8)", M, H, F, True, device, args.iters)
        run_shape("FFN gemm2 (no gelu)", M, F, H, False, device, args.iters)
        print()


if __name__ == "__main__":
    main()
