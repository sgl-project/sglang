"""Manual correctness check for DeepSeek-V4 fp8 wo_a (deep_gemm.fp8_einsum path).

Mirrors models/deepseek_v4.py MQALayer wo_a: quantize the token-major attention
output [T, G, D] per-token-group(128) to fp8, then run the grouped matmul over the
group/head dim via deep_gemm.fp8_einsum("bhr,hdr->bhd") -> [T, G, R], and compare
against a bf16 einsum reference. sm100 (Blackwell) uses ue8m0 scales + recipe
(1,1,128); sm90 (Hopper) uses fp32 scales + recipe (1,128,128). Covers the Flash
(G=8) and Pro (G=16) shapes for both prefill (T=1024) and decode (small T).

    CUDA_VISIBLE_DEVICES=0 python3 test/manual/dsv4/wo_a_fp8_sm90_correctness.py
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
            weight[group], use_ue8m0=False, gran_k=128
        )
    return weight_fp8, weight_scale


def run_wo_a_einsum(
    o: torch.Tensor,  # [T, G, D] bf16
    weight_fp8: torch.Tensor,  # [G, N, K] fp8
    weight_scale: torch.Tensor,  # [G, N/128, K/128] fp32
) -> torch.Tensor:
    """Mirror models/deepseek_v4.py MQALayer wo_a fp8 einsum path."""
    import deep_gemm

    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    T, G, D = o.shape
    _, R, _ = weight_fp8.shape
    o_fp8, o_s = sglang_per_token_group_quant_fp8(
        o.reshape(T * G, D).contiguous(),
        group_size=128,
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
    return output


def check(case: WoACase, args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    o = (
        torch.randn(case.tokens, case.groups, case.k, device=device, dtype=torch.bfloat16)
        * 0.1
    )
    weight = (
        torch.randn(case.groups, case.n, case.k, device=device, dtype=torch.bfloat16)
        * 0.05
    )
    weight_fp8, weight_scale = quantize_weight_by_group(weight)

    out = run_wo_a_einsum(o, weight_fp8, weight_scale)
    bf16_ref = torch.einsum("tgd,grd->tgr", o.float(), weight.float()).to(torch.bfloat16)
    torch.cuda.synchronize()

    cb = cosine(out, bf16_ref)
    print(f"{case.name}: G={case.groups} T={case.tokens} cos_bf16={cb:.6f}")
    if cb <= args.cos_gate:
        raise AssertionError(f"{case.name} cos_bf16 {cb} <= {args.cos_gate}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cos-gate", type=float, default=0.999)
    parser.add_argument("--decode-tokens", type=int, default=16)
    args = parser.parse_args()

    from sglang.srt.layers import deep_gemm_wrapper

    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        print(
            "SKIP: this manual test validates the sm90 fp32-scale wo_a einsum path; "
            "the sm100/Blackwell production path uses ue8m0 scales + a weight-scale "
            "transform not reproduced here."
        )
        return

    cases = [
        WoACase("flash prefill", groups=8, tokens=1024),
        WoACase("flash decode", groups=8, tokens=args.decode_tokens),
        WoACase("pro prefill", groups=16, tokens=1024),
        WoACase("pro decode", groups=16, tokens=args.decode_tokens),
    ]
    for case in cases:
        check(case, args)
    print("ALL OK")


if __name__ == "__main__":
    main()
