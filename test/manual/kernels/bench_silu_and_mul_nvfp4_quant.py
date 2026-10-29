"""SiLU-and-mul followed by NVFP4 activation quantization: silu_and_mul + fp4_quantize vs
FlashInfer's fused kernel (used by SiluAndMul.fuse_output_quant), at the down-projection widths
of common dense models.

Requires SM100+ (NVFP4). Times are medians of triton.testing.do_bench; the fused output is
checked against the unfused one first (packed FP4 bytes, <= 2e-3 mismatch: the FP32 SiLU may
round to a neighbouring BF16 value now and then).
"""

import argparse

import torch
import triton

# Down-projection input widths (intermediate sizes).
MODELS = {
    "Llama-3.1-8B": 14336,
    "Qwen3-8B": 12288,
    "Llama-3.2-3B": 8192,
}


def unfused(x, global_scale):
    from sglang.srt.layers.activation import silu_and_mul
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    act = torch.empty(x.shape[0], x.shape[1] // 2, dtype=x.dtype, device=x.device)
    silu_and_mul(x, act)
    return fp4_quantize(act, global_scale)


def fused(x, global_scale):
    from flashinfer.quantization import silu_and_mul_nvfp4_quantize

    return silu_and_mul_nvfp4_quantize(x, global_scale, nvfp4_4over6=None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokens", type=int, nargs="+", default=[16, 256, 2048, 8192, 32768]
    )
    args = p.parse_args()

    print(
        f"{'model':14s} {'d':>6s} {'tokens':>7s} {'unfused us':>11s} {'fused us':>9s} {'speedup':>8s} {'mismatch':>9s}"
    )
    torch.manual_seed(0)
    for name, d in MODELS.items():
        for m in args.tokens:
            x = torch.randn(m, 2 * d, dtype=torch.bfloat16, device="cuda") * 3
            global_scale = torch.tensor([448.0 * 6.0 / 20.0], device="cuda")
            ref_fp4, _ = unfused(x, global_scale)
            fp4, _ = fused(x, global_scale)
            mismatch = (ref_fp4.view(torch.uint8) != fp4).float().mean().item()
            assert mismatch <= 2e-3, mismatch
            t_u = triton.testing.do_bench(lambda: unfused(x, global_scale)) * 1e3
            t_f = triton.testing.do_bench(lambda: fused(x, global_scale)) * 1e3
            print(
                f"{name:14s} {d:6d} {m:7d} {t_u:11.1f} {t_f:9.1f} {t_u / t_f:7.2f}x {mismatch:9.1e}"
            )


if __name__ == "__main__":
    main()
