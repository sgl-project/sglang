"""Single-GPU compute fixtures: no EP bindings, communicator, or model download."""

from dataclasses import dataclass
from typing import Annotated

import torch
import torch.nn.functional as F


def configure_compute():
    from sglang.srt.arg_groups.arg_utils import NS
    from sglang.srt.runtime_context import get_context

    @dataclass
    class ComputeConfig:
        enable_fused_moe_sum_all_reduce: Annotated[bool, NS("exec.moe")] = False
        enable_deterministic_inference: Annotated[bool, NS("exec.deterministic")] = (
            False
        )

    get_context().set_server_args(ComputeConfig())


def make_compute_fixture(*, hidden=256, intermediate=128, experts=4, capacity=8):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLDispatchOutput

    rng = torch.Generator().manual_seed(32774)
    received = (torch.randn(experts, capacity, hidden, generator=rng) * 2).to(
        torch.float8_e4m3fn
    )
    scales = torch.full((experts, capacity, hidden // 128), 0.25)
    counts = torch.tensor([capacity // 2] * experts, dtype=torch.int32)
    w13 = torch.randn(experts, 2 * intermediate, hidden, generator=rng).to(
        torch.float8_e4m3fn
    )
    w2 = torch.randn(experts, hidden, intermediate, generator=rng).to(
        torch.float8_e4m3fn
    )
    quant = TritonMoeQuantInfo(
        w13.cuda(),
        w2.cuda(),
        use_fp8_w8a8=True,
        w13_scale=torch.full(
            (experts, 2 * intermediate // 128, hidden // 128),
            hidden**-0.5,
            device="cuda",
        ),
        w2_scale=torch.full(
            (experts, hidden // 128, intermediate // 128),
            intermediate**-0.5,
            device="cuda",
        ),
        block_shape=[128, 128],
    )
    # Source routing belongs to combine and is deliberately unrelated to the
    # receiver's local slot IDs. The adapter must preserve it verbatim.
    ids = torch.tensor([[7, 3], [-1, -1]], dtype=torch.int64, device="cuda")
    weights = torch.tensor([[0.25, 0.75], [0, 0]], device="cuda")
    dispatched = DeepEPLLDispatchOutput(
        received.cuda(), scales.cuda(), ids, weights, counts.cuda(), 1
    )
    config = MoeRunnerConfig(
        num_experts=2 * experts,
        num_local_experts=experts,
        hidden_size=hidden,
        top_k=2,
        params_dtype=torch.bfloat16,
        routed_scaling_factor=2.5,
    )
    return dispatched, quant, config


def _quantize_rows_cpu(x):
    grouped = x.float().reshape(*x.shape[:-1], -1, 128)
    amax = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    # The CUDA quantizer rounds reciprocal(amax) before multiplying by 448;
    # dividing by the dequant scale has different FP8 midpoint rounding.
    scales = amax * (1.0 / 448.0)
    quant = (
        (grouped * (amax.reciprocal() * 448.0)).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    return (quant.float() * scales).reshape(x.shape)


def _gemm_cpu(x, weight, weight_scale):
    grouped = x.float().reshape(x.shape[0], -1, 128)
    amax = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = amax * (1.0 / 448.0)
    quant = (
        (grouped * (amax.reciprocal() * 448.0))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .float()
    )
    weight = weight.float()
    expanded_scale = weight_scale.repeat_interleave(128, 0)
    result = torch.zeros(x.shape[0], weight.shape[0])
    # Block-scaled FP8 accumulates each unscaled dot before applying scales.
    # Dequantizing both full matrices before matmul changes rounding order.
    for group in range(grouped.shape[1]):
        dot = quant[:, group] @ weight[:, group * 128 : (group + 1) * 128].T
        result += dot * (scale[:, group] * expanded_scale[:, group])
    return result.bfloat16().float()


def expert_reference(dispatched, quant):
    """FP32 CPU arithmetic with the BF16/FP8 rounding stages of this adapter."""
    received, scales, _, _, counts, _ = dispatched
    received, scales, counts = received.cpu().float(), scales.cpu(), counts.cpu()
    w13, w2 = quant.w13_weight.cpu(), quant.w2_weight.cpu()
    s13, s2 = quant.w13_scale.cpu(), quant.w2_scale.cpu()
    result = torch.zeros_like(received)
    for expert, count in enumerate(counts.tolist()):
        if not count:
            continue
        x = (
            received[expert, :count] * scales[expert, :count].repeat_interleave(128, -1)
        ).bfloat16()
        first = _gemm_cpu(x, w13[expert], s13[expert])
        gate, up = first.chunk(2, -1)
        activated = (F.silu(gate) * up).bfloat16()
        result[expert, :count] = _gemm_cpu(activated, w2[expert], s2[expert])
    return result


def check_expert_output(actual, dispatched, quant):
    expected = expert_reference(dispatched, quant)
    for expert, count in enumerate(dispatched.masked_m.cpu().tolist()):
        torch.testing.assert_close(
            actual[expert, :count].cpu().float(),
            expected[expert, :count],
            rtol=0.02,
            atol=0.02,
        )
