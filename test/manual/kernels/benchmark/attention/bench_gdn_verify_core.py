"""Check bytes and time the projection-unpack through FP8-output sequence."""

import argparse
import json

import torch

from bench_gdn_verify_tiles import benchmark, make_case, snapshot

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)
from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
    qwen3_5_gdn_prefill_projection_views,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, nargs="+", default=[4, 8, 2])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    for tp in args.tp:
        for batch in (1, 2):
            case = make_case(tp, batch, 7, 7)
            tokens, h, hv = batch * 7, 16 // tp, 64 // tp
            dim = 2 * h * 128 + hv * 128
            qkvz = torch.randn(
                tokens, dim + hv * 128, device="cuda", dtype=torch.bfloat16
            )
            ba = torch.randn(tokens, 2 * hv, device="cuda", dtype=torch.bfloat16)
            conv_state = torch.randn(4, dim, 3, device="cuda", dtype=torch.bfloat16)
            initial_conv_state = conv_state.clone()
            weight = torch.randn(dim, 4, device="cuda", dtype=torch.bfloat16)
            norm_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
            scale = torch.tensor(0.02, device="cuda")
            intermediate = torch.zeros(
                4, 7, dim, 3, device="cuda", dtype=torch.bfloat16
            )

            def run(
                views=False, conv_tile=256, tiled=False, fused_quant=False, direct=False
            ):
                split = (
                    qwen3_5_gdn_prefill_projection_views
                    if views
                    else fused_qkvzba_split_reshape_cat_contiguous
                )
                mixed, z, b, a = split(qkvz, ba, h, hv, 128, 128)
                processed = (
                    causal_conv1d_update(
                        mixed.view(batch, 7, dim).transpose(1, 2),
                        conv_state,
                        weight,
                        activation="silu",
                        conv_state_indices=case["ssm_state_indices"],
                        intermediate_conv_window=intermediate,
                        intermediate_state_indices=case["replay_indices"],
                        token_major_out=views,
                        block_n=conv_tile,
                    )
                    .transpose(1, 2)
                    .view(tokens, dim)
                )
                q, k, v = processed.split((h * 128, h * 128, hv * 128), dim=-1)
                inputs = dict(
                    case,
                    q=q.view(tokens, h, 128),
                    k=k.view(tokens, h, 128),
                    v=v.view(tokens, hv, 128),
                    a=a,
                    b=b,
                )
                opts = dict(block_v=32, num_warps=2) if tiled else {}
                gdn_replayssm_spec_decode(**inputs, direct_solve=direct, **opts)
                norm_z = z if views else z.reshape(-1, 128)
                result, _, _ = _layer_norm_fwd(
                    case["out"].view(-1, 128),
                    norm_weight,
                    None,
                    1e-6,
                    z=norm_z,
                    is_rms_norm=True,
                    static_quant_scale=scale if fused_quant else None,
                )
                if fused_quant:
                    quantized = result[0]
                else:
                    quantized, _ = static_quant_fp8(
                        result.view(tokens, hv * 128), scale
                    )
                return processed, quantized.view(tokens, hv * 128)

            expected_conv, expected_quant = run()
            expected_conv, expected_quant = (
                expected_conv.clone(),
                expected_quant.clone(),
            )
            expected_rings = snapshot(case)
            expected_intermediate = intermediate.clone()
            expected_conv_state = conv_state.clone()
            for opts in (
                {},
                dict(tiled=True),
                dict(views=True),
                dict(conv_tile=128),
                dict(fused_quant=True),
                dict(views=True, conv_tile=128, tiled=True, fused_quant=True),
                dict(
                    views=True, conv_tile=128, tiled=True, fused_quant=True, direct=True
                ),
            ):
                conv_state.copy_(initial_conv_state)
                conv, quant = run(**opts)
                equal = dict(
                    conv=bool(
                        torch.equal(
                            conv.view(torch.uint8), expected_conv.view(torch.uint8)
                        )
                    ),
                    quant=bool(
                        torch.equal(
                            quant.view(torch.uint8), expected_quant.view(torch.uint8)
                        )
                    ),
                    intermediate=bool(
                        torch.equal(
                            intermediate.view(torch.uint8),
                            expected_intermediate.view(torch.uint8),
                        )
                    ),
                )
                equal["conv_state"] = bool(
                    torch.equal(
                        conv_state.view(torch.uint8),
                        expected_conv_state.view(torch.uint8),
                    )
                )
                equal.update(
                    {
                        key: bool(
                            torch.equal(
                                case[key].view(torch.uint8), value.view(torch.uint8)
                            )
                        )
                        for key, value in expected_rings.items()
                    }
                )
                timing = (
                    {} if args.check_only else benchmark(lambda: run(**opts), calls=45)
                )
                result = dict(
                    tp=tp,
                    batch=batch,
                    options=opts,
                    bitwise=equal,
                    accuracy="PENDING",
                    **timing,
                )
                if not opts.get("direct"):
                    assert all(equal.values()), result
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
