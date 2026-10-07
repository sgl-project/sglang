"""MXFP4 expert LoRA parity with GPT-OSS activation, biases, and tile padding."""

import sys
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import fused_experts_none_to_marlin
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from sglang.srt.layers.quantization.mxfp4_marlin_moe import build_marlin_moe_quant_info
from sglang.srt.lora.lora_moe_runner_marlin import MarlinLoraRunnerCore
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Marlin LoRA requires Hopper or newer",
)
@pytest.mark.parametrize("M", [1, 7])
def test_mxfp4_marlin_lora(M):
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", moe_runner_backend="marlin")
    )
    torch.manual_seed(42)
    device = "cuda"
    E, K, N, KP, NP, R = 4, 288, 160, 512, 256, 8
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=device
    )

    def weight(rows, cols):
        packed = torch.randint(
            0, 256, (E, rows, cols // 2), device=device, dtype=torch.uint8
        )
        unpacked = (
            lut[torch.stack((packed & 15, packed >> 4), dim=-1).long()].flatten(-2)
            * 0.03125
        )
        return packed.view(torch.int8), unpacked.to(torch.bfloat16)

    w1, ref1 = weight(2 * NP, KP)
    w2, ref2 = weight(KP, NP)
    # Padding must contain zeros, including gate/up halves separately.
    ref1[:, N:NP] = 0
    ref1[:, NP + N :] = 0
    ref1[:, :, K:] = 0
    w1[:, N:NP] = 0
    w1[:, NP + N :] = 0
    w1[:, :, K // 2 :] = 0
    ref2[:, K:] = 0
    ref2[:, :, N:] = 0
    w2[:, K:] = 0
    w2[:, :, N // 2 :] = 0
    b1 = torch.randn(E, 2 * NP, device=device, dtype=torch.bfloat16) * 0.1
    b2 = torch.randn(E, KP, device=device, dtype=torch.bfloat16) * 0.1
    # Exercise both sides of the GPT-OSS activation clamp.
    b1[0, 0] = 9
    b1[1, NP] = -9
    b1[:, N:NP] = 0
    b1[:, NP + N :] = 0
    b2[:, K:] = 0
    layer = torch.nn.Module()
    for name, tensor in {
        "w13_weight": w1,
        "w2_weight": w2,
        "w13_weight_scale": torch.full(
            (E, 2 * NP, KP // 32), 122, device=device, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
        "w2_weight_scale": torch.full(
            (E, KP, NP // 32), 122, device=device, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
        "w13_weight_bias": b1,
        "w2_weight_bias": b2,
    }.items():
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.dispatcher = SimpleNamespace(local_expert_mapping=None)
    prepare_moe_mxfp4_layer_for_marlin(layer)
    quant = build_marlin_moe_quant_info(layer)
    config = MoeRunnerConfig(
        intermediate_size_per_partition=N,
        activation="silu",
        gemm1_alpha=1.702,
        gemm1_clamp_limit=7.0,
    )
    a1 = torch.randn(2, E, R, K, device=device, dtype=torch.bfloat16) * 0.03
    bb1 = torch.randn(2, E, 2 * N, R, device=device, dtype=torch.bfloat16) * 0.03
    a2 = torch.randn(2, E, R, N, device=device, dtype=torch.bfloat16) * 0.03
    bb2 = torch.randn(2, E, K, R, device=device, dtype=torch.bfloat16) * 0.03
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    ids = torch.stack(
        [
            torch.arange(M, device=device) % E,
            (torch.arange(M, device=device) + 1) % E,
        ],
        dim=1,
    ).int()
    weights = torch.softmax(torch.randn(M, 2, device=device), dim=1)
    slots = torch.arange(M, device=device) % 2

    def gate_hook(h, out, tw, ti):
        for m in range(M):
            for k in range(2):
                e = int(ti[m, k])
                s = int(slots[m])
                out[m, k].add_(
                    (bb1[s, e].float() @ (a1[s, e].float() @ h[m].float())).to(
                        out.dtype
                    )
                )

    def down_hook(h, out, tw, ti):
        for m in range(M):
            for k in range(2):
                e = int(ti[m, k])
                s = int(slots[m])
                out[m, k].add_(
                    (
                        bb2[s, e].float()
                        @ (a2[s, e].float() @ h[m * 2 + k].float())
                        * tw[m, k]
                    ).to(out.dtype)
                )

    dispatch = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=weights,
            topk_ids=ids,
            router_logits=torch.zeros(M, E, device=device),
        ),
    )
    for active in (False, True):
        hooks = SimpleNamespace(
            after_gate_up=gate_hook if active else None,
            after_down=down_hook if active else None,
        )
        out = (
            MarlinLoraRunnerCore(config)
            .run_from_dispatch(dispatch, quant, config, hooks)
            .hidden_states
        )
        expected = torch.zeros_like(x, dtype=torch.float32)
        for m in range(M):
            for k in range(2):
                e = int(ids[m, k])
                s = int(slots[m])
                gateup = (ref1[e, :, :K].float() @ x[m].float() + b1[e].float()).to(
                    torch.bfloat16
                )
                gateup = torch.cat([gateup[:N], gateup[NP : NP + N]])
                if active:
                    gateup.add_(
                        (bb1[s, e].float() @ (a1[s, e].float() @ x[m].float())).to(
                            torch.bfloat16
                        )
                    )
                gate = gateup[:N].clamp(max=7)
                up = gateup[N:].clamp(-7, 7)
                activation = gate * torch.sigmoid(gate * 1.702) * (up + 1)
                down = (
                    (ref2[e, :K, :N].float() @ activation.float() + b2[e, :K].float())
                    * weights[m, k]
                ).to(torch.bfloat16)
                if active:
                    down.add_(
                        (
                            bb2[s, e].float()
                            @ (a2[s, e].float() @ activation.float())
                            * weights[m, k]
                        ).to(torch.bfloat16)
                    )
                expected[m] += down.float()
        error = (out.float() - expected).abs()
        print(
            "KERNEL",
            M,
            active,
            "max",
            error.max().item(),
            "mean",
            error.mean().item(),
            flush=True,
        )
        if not active:
            padded = dispatch._replace(
                hidden_states=torch.nn.functional.pad(x, (0, KP - K))
            )
            stock = fused_experts_none_to_marlin(padded, quant, config).hidden_states[
                :, :K
            ]
            print(
                "STOCK_DIFF",
                float((stock.float() - expected).abs().max()),
                "PATCH_VS_STOCK",
                float((stock - out).abs().max()),
                flush=True,
            )
            torch.testing.assert_close(out, stock, atol=0.001, rtol=0.001)
        relative_l2 = torch.linalg.vector_norm(
            out.float() - expected
        ) / torch.linalg.vector_norm(expected)
        print("RELATIVE_L2", relative_l2.item(), flush=True)
        assert relative_l2 < 0.015
    print(
        "PASS MXFP4 Marlin expert LoRA with GPT-OSS activation, biases, padding and two adapters",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
