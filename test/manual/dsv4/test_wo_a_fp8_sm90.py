"""Manual correctness check for DeepSeek-V4 fp8 wo_a (deep_gemm.fp8_einsum path).

Mirrors models/deepseek_v4.py MQALayer wo_a: quantize the token-major attention
output [T, G, D] per-token-group(128) to fp8, then run the grouped matmul over the
group/head dim via deep_gemm.fp8_einsum("bhr,hdr->bhd") -> [T, G, R], and compare
against FP32 references before and after quantization. Exercises FP32 storage
with power-of-two values + recipe (1,128,128) on SM90 and SM12x. Covers the Flash
(G=8) and Pro (G=16) shapes for both prefill (T=1024) and decode (small T).

    CUDA_VISIBLE_DEVICES=0 python3 test/manual/dsv4/test_wo_a_fp8_sm90.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WoACase:
    name: str
    groups: int
    tokens: int
    k: int = 4096
    n: int = 1024


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def quantize_weight_by_group(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block (128x128) fp8 cast of the per-group wo_a weight [G, N, K]."""
    from deep_gemm.utils import per_block_cast_to_fp8

    groups, n, k = weight.shape
    weight_fp8 = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty(
        (groups, n // 128, k // 128), device=weight.device, dtype=torch.float32
    )
    for group in range(groups):
        weight_fp8[group], weight_scale[group] = per_block_cast_to_fp8(
            weight[group], use_ue8m0=True, gran_k=128
        )
    return weight_fp8, weight_scale


def run_wo_a_einsum(
    o: torch.Tensor,  # [T, G, D] bf16
    weight_fp8: torch.Tensor,  # [G, N, K] fp8
    weight_scale: torch.Tensor,  # [G, N/128, K/128] fp32
    *,
    scale_ue8m0: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror models/deepseek_v4.py MQALayer wo_a fp8 einsum path."""
    import deep_gemm

    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    T, G, D = o.shape
    _, R, _ = weight_fp8.shape
    o_fp8, o_s = sglang_per_token_group_quant_fp8(
        o.reshape(T * G, D).contiguous(),
        group_size=128,
        scale_ue8m0=scale_ue8m0,
    )

    assert o_s.dtype == torch.float32
    assert o_s.shape == (T * G, D // 128)
    assert o_s.stride() == (D // 128, 1)
    assert torch.isfinite(o_s).all() and (o_s > 0).all()
    assert (o_s >= torch.finfo(torch.float32).tiny).all()
    if scale_ue8m0:
        assert (o_s.view(torch.int32) & 0x7FFFFF).eq(0).all(), (
            "fp8_einsum activation scales must have power-of-two values"
        )
    o_dequant = (o_fp8.float().unflatten(-1, (-1, 128)) * o_s.unsqueeze(-1)).flatten(-2)
    torch.testing.assert_close(
        o_dequant,
        o.reshape(T * G, D).float(),
        rtol=0.063,
        atol=o_s.max().item() * 2**-10,
    )

    recipe = (1, 128, 128)
    output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
        (weight_fp8, weight_scale),
        output,
        recipe=recipe,
    )
    weight_dequant = weight_fp8.float() * weight_scale.repeat_interleave(
        128, dim=-2
    ).repeat_interleave(128, dim=-1)
    ref = torch.einsum("tgd,grd->tgr", o_dequant.view(T, G, D), weight_dequant)
    return output, ref


def relative_errors(actual: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Global max and L2 relative errors, stable when individual outputs are zero."""
    diff = actual.float() - ref.float()
    return (
        (diff.abs().max() / ref.abs().max().clamp_min(1e-12)).item(),
        (diff.norm() / ref.norm().clamp_min(1e-12)).item(),
    )


def check(case: WoACase, args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    o = (
        torch.randn(
            case.tokens, case.groups, case.k, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    weight = (
        torch.randn(case.groups, case.n, case.k, device=device, dtype=torch.bfloat16)
        * 0.05
    )
    weight_fp8, weight_scale = quantize_weight_by_group(weight)

    out, dequant_ref = run_wo_a_einsum(o, weight_fp8, weight_scale)
    fp32_ref = torch.einsum("tgd,grd->tgr", o.float(), weight.float())
    bf16_ref = fp32_ref.to(torch.bfloat16)
    torch.cuda.synchronize()

    cb = cosine(out, bf16_ref)
    kernel_error = relative_errors(out, dequant_ref)
    total_error = relative_errors(out, fp32_ref)
    floor_error = relative_errors(dequant_ref.bfloat16(), dequant_ref)
    bf16_error = relative_errors(bf16_ref, fp32_ref)
    print(
        f"{case.name}: G={case.groups} T={case.tokens} cos_bf16={cb:.6f}\n"
        f"  max_relative,l2_relative: fixed_kernel={kernel_error} "
        f"fixed_total={total_error} fp8_bf16_floor={floor_error} "
        f"bf16_rounding={bf16_error}"
    )
    if args.compare_legacy:
        old_out, old_ref = run_wo_a_einsum(
            o, weight_fp8, weight_scale, scale_ue8m0=False
        )
        print(f"  old_kernel={relative_errors(old_out, old_ref)}")
    if kernel_error[0] >= 0.01 or kernel_error[1] >= 0.005:
        raise AssertionError(f"{case.name}: excessive kernel error {kernel_error}")
    if cb <= args.cos_gate:
        raise AssertionError(f"{case.name} cos_bf16 {cb} <= {args.cos_gate}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cos-gate", type=float, default=0.999)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--prefill-tokens", type=int, default=1024)
    parser.add_argument(
        "--compare-legacy",
        action="store_true",
        help="Also run the old non-power-of-two path (may assert on some DeepGEMM layouts)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        print("SKIP: FP8 einsum requires an SM90 or newer CUDA GPU.")
        return

    cases = [
        WoACase("flash prefill", groups=8, tokens=args.prefill_tokens),
        WoACase("flash decode", groups=8, tokens=args.decode_tokens),
        WoACase("pro prefill", groups=16, tokens=args.prefill_tokens),
        WoACase("pro decode", groups=16, tokens=args.decode_tokens),
    ]
    for case in cases:
        check(case, args)
    print("ALL OK")


if __name__ == "__main__":
    main()
