import argparse

import torch

from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    sglang_fused_rms_fp8_group_quant,
    sglang_per_token_group_quant_fp8,
)


def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    w_fp32 = w.to(torch.float32)
    rms = torch.rsqrt((x_fp32 * x_fp32).mean(dim=-1, keepdim=True) + eps)
    return x_fp32 * rms * w_fp32


def stats(name: str, a: torch.Tensor, b: torch.Tensor):
    d = (a - b).abs().to(torch.float32)
    print(
        f"[{name}] max={d.max().item():.6g}, mean={d.mean().item():.6g}, p99={torch.quantile(d.flatten(), 0.99).item():.6g}"
    )


def classify_fp8_diff(tag: str, qa: torch.Tensor, qb: torch.Tensor):
    a_code = qa.view(torch.uint8)
    b_code = qb.view(torch.uint8)
    total = a_code.numel()

    same_code = a_code == b_code
    diff_code = ~same_code

    a_f16 = qa.to(torch.float16)
    b_f16 = qb.to(torch.float16)
    same_value = a_f16 == b_f16

    diff_code_same_value = (diff_code & same_value).sum().item()
    diff_code_diff_value = (diff_code & (~same_value)).sum().item()

    abs_err = (a_f16 - b_f16).abs().to(torch.float32)

    print(f"\n[{tag}] fp8逐元素分类")
    print(f"  total                      : {total}")
    print(
        f"  same fp8 code             : {same_code.sum().item()} ({same_code.float().mean().item() * 100:.4f}%)"
    )
    print(
        f"  diff fp8 code             : {diff_code.sum().item()} ({diff_code.float().mean().item() * 100:.4f}%)"
    )
    print(f"  diff code but same fp16   : {diff_code_same_value}")
    print(f"  diff code and diff fp16   : {diff_code_diff_value}")
    print(
        f"  abs(f16) max/mean/p99     : {abs_err.max().item():.6g} / {abs_err.mean().item():.6g} / {torch.quantile(abs_err.flatten(), 0.99).item():.6g}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--column-major-scales", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("CUDA FP8 不可用")

    torch.manual_seed(args.seed)
    device = "cuda"

    x = torch.randn(args.m, args.n, device=device, dtype=torch.bfloat16)
    w = torch.randn(args.n, device=device, dtype=torch.bfloat16)

    # 路径 A: fused kernel（内部 fp32 归一化 + 量化）
    (q_fused, s_fused), u_fused, _, _ = sglang_fused_rms_fp8_group_quant(
        x,
        w,
        args.eps,
        group_size=args.group_size,
        dtype_quant=torch.float8_e4m3fn,
        res1=None,
        column_major_scales=args.column_major_scales,
        scale_tma_aligned=False,
        output_unquantized_inp1=True,
    )

    # 路径 B: 用 fused 输出的未量化结果（通常是 bf16）再次做 sglang group quant
    q_requant_u, s_requant_u = sglang_per_token_group_quant_fp8(
        u_fused,
        group_size=args.group_size,
        column_major_scales=args.column_major_scales,
        scale_tma_aligned=False,
    )

    # 路径 C: 参考实现，先 fp32 RMSNorm，再量化（Triton raw path）
    y_ref_fp32 = rmsnorm_ref(x, w, args.eps)
    q_ref_raw, s_ref_raw = per_token_group_quant_fp8(
        y_ref_fp32,
        group_size=args.group_size,
        column_major_scales=args.column_major_scales,
        scale_tma_aligned=False,
    )

    # 路径 D: 用 sglang path 量化 fp32 参考，隔离“量化实现路径”影响
    q_ref_sgl, s_ref_sgl = sglang_per_token_group_quant_fp8(
        y_ref_fp32,
        group_size=args.group_size,
        column_major_scales=args.column_major_scales,
        scale_tma_aligned=False,
    )

    print("=== 基本信息 ===")
    print(
        f"x dtype={x.dtype}, u_fused dtype={u_fused.dtype}, y_ref_fp32 dtype={y_ref_fp32.dtype}"
    )
    print(
        f"shape={tuple(x.shape)}, group_size={args.group_size}, column_major_scales={args.column_major_scales}"
    )

    print("\n=== scale 误差 ===")
    stats(
        "s_fused vs s_requant_u",
        s_fused.to(torch.float32),
        s_requant_u.to(torch.float32),
    )
    stats(
        "s_fused vs s_ref_raw", s_fused.to(torch.float32), s_ref_raw.to(torch.float32)
    )
    stats(
        "s_fused vs s_ref_sgl", s_fused.to(torch.float32), s_ref_sgl.to(torch.float32)
    )

    print("\n=== q(转fp16) 误差 ===")
    stats(
        "q_fused vs q_requant_u",
        q_fused.to(torch.float16),
        q_requant_u.to(torch.float16),
    )
    stats(
        "q_fused vs q_ref_raw", q_fused.to(torch.float16), q_ref_raw.to(torch.float16)
    )
    stats(
        "q_fused vs q_ref_sgl", q_fused.to(torch.float16), q_ref_sgl.to(torch.float16)
    )

    classify_fp8_diff("fused vs requant(u_fused)", q_fused, q_requant_u)
    classify_fp8_diff("fused vs ref_raw(fp32->raw)", q_fused, q_ref_raw)
    classify_fp8_diff("fused vs ref_sgl(fp32->sgl)", q_fused, q_ref_sgl)

    print("\n=== 中间量误差（定位来源）===")
    # 关键：u_fused 常为 bf16，和 y_ref_fp32 之间存在一次精度截断
    stats("u_fused(bf16) vs y_ref_fp32", u_fused.to(torch.float32), y_ref_fp32)


if __name__ == "__main__":
    main()
