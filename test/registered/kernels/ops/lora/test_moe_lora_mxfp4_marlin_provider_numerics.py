"""MXFP4 Marlin provider against exact dequantization, using the serving repack.
Checkpoint format: packed E2M1 (even index in low nibble) with group-32 E8M0
scales, as FP8 checkpoints with FP4 experts load them. Compare raw pair-major
outputs, then drive the shared-outer decode plan through the runner.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.lora.moe.activation import ActivationFn
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.plan import resolve_plans
from sglang.srt.lora.moe.runner import MoeLoraBatch, MoeLoraRunner
from sglang.srt.lora.utils import Phase, architecture_for_capability
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_E = 4
_H = 256
_I = 128
_TOPK = 2
_T = 32

_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Marlin MXFP4 needs SM90+ here")


def _quant_mxfp4(weight: torch.Tensor):
    """[E, N, K] bf16 -> (packed uint8 [E,N,K/2], e8m0 [E,N,K/32]) plus the
    exact dequantized weights."""
    e, n, k = weight.shape
    groups = weight.float().view(e, n, k // 32, 32)
    # OCP MX shared exponent: the block maximum lands in E2M1's top binade.
    exponent = (torch.floor(torch.log2(groups.abs().amax(dim=-1))) - 2).clamp(
        min=-127, max=127
    )
    scale = torch.exp2(exponent)
    scaled = (groups / scale[..., None]).clamp(-6.0, 6.0)
    table = _E2M1_VALUES.to(weight.device)
    idx = (scaled.abs().unsqueeze(-1) - table).abs().argmin(dim=-1)
    sign = scaled < 0
    codes = (idx | (sign.long() << 3)).to(torch.uint8).view(e, n, k)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    e8m0 = (exponent + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    dequant = (torch.where(sign, -table[idx], table[idx]) * scale[..., None]).view(
        e, n, k
    )
    return packed, e8m0, dequant


def _fake_layer(w13, w2, gated=True):
    w13_q, w13_scale, w13_eff = _quant_mxfp4(w13)
    w2_q, w2_scale, w2_eff = _quant_mxfp4(w2)
    layer = SimpleNamespace(
        w13_weight=SimpleNamespace(data=w13_q.view(torch.int8)),
        w2_weight=SimpleNamespace(data=w2_q.view(torch.int8)),
        w13_weight_scale_inv=SimpleNamespace(data=w13_scale),
        w2_weight_scale_inv=SimpleNamespace(data=w2_scale),
        moe_runner_config=SimpleNamespace(is_gated=gated),
        num_local_experts=w2.shape[0],
        intermediate_size_per_partition=w2.shape[2],
        hidden_size=w2.shape[1],
        params_dtype=torch.bfloat16,
    )
    return layer, w13_eff, w2_eff


def _marlin_quant_info(layer):
    """Repack the fake layer in place and bind it as serving does."""
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )
    from sglang.srt.lora.moe.quant_info import MoeLoraMxfp4MarlinQuantInfo

    prepare_moe_mxfp4_layer_for_marlin(layer)
    return MoeLoraMxfp4MarlinQuantInfo.from_layer(layer)


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


def _standalone_output_allocation(runner, *, num_tokens, dtype, device):
    return torch.empty((num_tokens, runner.hidden_size), dtype=dtype, device=device)


@pytest.mark.parametrize("slices,intermediate", [(2, _I), (1, _I), (2, 96), (1, 96)])
def test_mxfp4_marlin_gateup_and_down_match_reference(slices, intermediate):
    """Padded widths (96 -> 64-aligned 128) pad every gate/up slice on its own."""
    _skip_unless_supported()
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(11)

    def rand(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16).to(device)

    w13 = rand(_E, slices * intermediate, _H)
    w2 = rand(_E, _H, intermediate)
    hidden = rand(_T, _H)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(_T)]
    ).to(device=device, dtype=torch.int32)

    layer, w13_eff, w2_eff = _fake_layer(w13, w2, gated=slices == 2)
    quant_info = _marlin_quant_info(layer)
    assert (quant_info.gate_up_slices, quant_info.intermediate_size) == (
        slices,
        intermediate,
    )
    assert layer.w2_weight.shape[1] * 16 == (128 if intermediate == 96 else _I)

    provider = select_provider_cls("route_major", "mxfp4", "marlin")(quant_info)

    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, _TOPK)
        gateup = torch.zeros(
            provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
        )
        assert gateup.shape[1] == slices * intermediate
        provider.gateup(state, gateup)

        flat_ids = topk_ids.flatten()
        pair_idx = torch.nonzero(flat_ids >= 0, as_tuple=True)[0]
        experts = flat_ids[pair_idx].long()
        tokens = (pair_idx // _TOPK).long()

        ref_gateup = torch.einsum(
            "ph,pnh->pn", hidden[tokens].float(), w13_eff[experts]
        )
        got_gateup = gateup[pair_idx].float()
        assert _rel_l2(got_gateup, ref_gateup) < 2e-2

        act_pairs = (torch.randn_like(ref_gateup[:, :intermediate]) * 0.1).to(
            torch.bfloat16
        )
        act = torch.zeros(
            provider.act_out_shape(state), dtype=torch.bfloat16, device=device
        )
        act[pair_idx] = act_pairs
        down = torch.zeros(
            provider.down_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.down(state, act, down)

        ref_down = torch.einsum("pi,phi->ph", act_pairs.float(), w2_eff[experts])
        got_down = down[pair_idx].float()
        assert _rel_l2(got_down, ref_down) < 2e-2

        if intermediate == _I:
            return

        from sglang.srt.lora.workspace import LoraWorkspace

        rank = 16
        a13 = rand(_E, rank, _H).float()[experts]
        b13 = rand(_E, slices * intermediate, rank).float()[experts]
        a2 = rand(_E, rank, intermediate).float()[experts]
        b2 = rand(_E, _H, rank).float()[experts]
        weights = torch.tensor([0.65, 0.35], device=device).repeat(_T, 1)
        output = torch.empty((_T, _H), dtype=torch.bfloat16, device=device)
        act_pairs = torch.empty(
            (_T, _TOPK, intermediate), dtype=torch.bfloat16, device=device
        )
        workspace = LoraWorkspace()
        workspace.begin_forward(graph_mode=True)
        activation = "silu" if slices == 2 else "relu2"

        def delta(a, b, x):
            return torch.bmm(b, torch.bmm(a, x.unsqueeze(-1))).squeeze(-1)

        def run():
            state = provider.prepare(hidden, topk_ids, _TOPK, workspace=workspace)
            provider.gateup(state, gateup)
            gate_delta = delta(a13, b13, hidden[tokens].float()).to(torch.bfloat16)
            provider.act_with_delta(
                state,
                gateup,
                gate_delta.view(_T, _TOPK, slices * intermediate),
                topk_ids,
                act,
                act_pairs,
                activation=activation,
            )
            # Padding must be cleared even when its reusable storage is poisoned.
            state.padded_act[:, intermediate:].fill_(float("nan"))
            provider.down(state, act, down)
            down_delta = delta(a2, b2, act_pairs.flatten(0, 1).float()).to(
                torch.bfloat16
            )
            provider.finalize(
                state,
                down,
                topk_ids,
                weights,
                0.75,
                output,
                lora_delta=down_delta.view(_T, _TOPK, _H),
            )

        def reference():
            inputs = hidden[tokens].float()
            gate_up = torch.einsum("ph,pnh->pn", inputs, w13_eff[experts])
            gate_up = gate_up + delta(a13, b13, inputs)
            if slices == 2:
                activation_out = (
                    F.silu(gate_up[:, :intermediate]) * gate_up[:, intermediate:]
                )
            else:
                activation_out = gate_up.relu().square()
            pairs = torch.einsum("pi,phi->ph", activation_out, w2_eff[experts])
            pairs += delta(a2, b2, activation_out)
            return (pairs.view(_T, _TOPK, _H) * weights[..., None]).sum(1) * 0.75

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        torch.cuda.synchronize()
        assert _rel_l2(output.float(), reference()) < 3e-2
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for _ in range(2):
            hidden.mul_(-0.5)
            graph.replay()
            torch.cuda.synchronize()
            assert _rel_l2(output.float(), reference()) < 3e-2


@pytest.mark.parametrize("slices", (2, 1), ids=("gated-silu", "relu2"))
def test_mxfp4_shared_outer_decode_plan_matches_reference(
    monkeypatch: pytest.MonkeyPatch, slices: int
) -> None:
    """Run the architecture's shared-outer decode row end to end with Marlin base
    GEMMs: eager first, then a graph replayed with changed adapter slots,
    routing, and inputs. sm100 selects the Marlin vendor row.
    """
    _skip_unless_supported()
    device = torch.device("cuda")
    architecture = architecture_for_capability(
        torch.cuda.get_device_capability(device)[0]
    )
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    g = torch.Generator(device="cpu").manual_seed(23)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    tokens, slots, rank = 8, 2, 16
    activation = ActivationFn.SILU if slices == 2 else ActivationFn.RELU2
    layer, w13_eff, w2_eff = _fake_layer(
        rand(_E, slices * _I, _H), rand(_E, _H, _I), gated=slices == 2
    )
    quant_info = _marlin_quant_info(layer)
    selected = resolve_plans(
        architecture=architecture,
        is_shared_outer=True,
        physical_rank=rank,
        activation=activation,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family="mxfp4",
        vendor="marlin",
    )[Phase.DECODE]
    assert selected.name == {
        "sm100": "decode.shared.marlin",
        "sm90": "decode.shared",
    }.get(architecture, selected.name)
    provider = select_provider_cls(selected.base_gemm_rows, "mxfp4", "marlin")(
        quant_info
    )
    runner = MoeLoraRunner(
        providers={selected.base_gemm_rows: provider},
        top_k=_TOPK,
        routed_scaling_factor=0.75,
        activation=activation,
        is_gated=slices == 2,
    )
    runner.validate_plan(selected.plan, base_gemm_rows=selected.base_gemm_rows)
    runner.plans = {Phase.DECODE: selected}

    hidden = rand(tokens, _H, scale=0.2)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[3, 1] = -1  # an EP hole on an adapter token
    weights = torch.tensor([0.65, 0.35], device=device).repeat(tokens, 1)
    # Outer factors (gate/up-A, down-B) per slot; inner ones per (slot, expert).
    gate_up_a = rand(slots, 1, slices * rank, _H, scale=0.1)
    gate_up_b = rand(slots, _E, slices * _I, rank, scale=0.1)
    down_a = rand(slots, _E, rank, _I, scale=0.1)
    down_b = rand(slots, 1, _H, rank, scale=0.1)
    mapping = torch.tensor([0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.int32, device=device)

    def reference(mapping):
        pair_tokens = torch.arange(tokens, device=device).repeat_interleave(_TOPK)
        experts = topk_ids.flatten().long()
        routed = (experts >= 0).float()[:, None]
        experts = experts.clamp(min=0)
        slot = mapping[pair_tokens].long()
        adapted = (slot >= 0).float()[:, None]
        slot = slot.clamp(min=0)
        inputs = hidden.float()[pair_tokens]
        gate_up = torch.einsum("ph,pnh->pn", inputs, w13_eff[experts])
        outer = torch.einsum("ph,prh->pr", inputs, gate_up_a.float()[slot, 0])
        inner = gate_up_b.float()[slot, experts]
        delta = torch.cat(
            [
                torch.einsum(
                    "pir,pr->pi",
                    inner[:, s * _I : (s + 1) * _I],
                    outer[:, s * rank : (s + 1) * rank],
                )
                for s in range(slices)
            ],
            dim=1,
        )
        gate_up = gate_up + adapted * delta
        if slices == 2:
            act = F.silu(gate_up[:, :_I]) * gate_up[:, _I:]
        else:
            act = gate_up.relu().square()
        pairs = torch.einsum("pi,phi->ph", act, w2_eff[experts])
        down_rank = torch.einsum("pi,pri->pr", act, down_a.float()[slot, experts])
        pairs = pairs + adapted * torch.einsum(
            "pr,phr->ph", down_rank, down_b.float()[slot, 0]
        )
        pairs = pairs * routed
        return (pairs.view(tokens, _TOPK, _H) * weights[..., None]).sum(1) * 0.75

    dispatch = StandardDispatchOutput(
        hidden_states=hidden,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=weights,
            topk_ids=topk_ids,
            router_logits=torch.zeros((tokens, _E), device=device),
        ),
    )
    batch = MoeLoraBatch(
        gate_up_lora_a=gate_up_a,
        gate_up_lora_b=gate_up_b,
        down_lora_a=down_a,
        down_lora_b=down_b,
        token_lora_mapping=mapping,
    )

    def forward():
        with get_context().override_server_args():
            return runner.run(dispatch, batch).hidden_states

    output = forward()
    torch.cuda.synchronize()
    assert _rel_l2(output.float(), reference(mapping)) < 3e-2
    # LoRA math must be present, not bypassed within tolerance.
    assert _rel_l2(reference(torch.full_like(mapping, -1)), reference(mapping)) > 0.1

    batch.use_cuda_graph = True
    # Workspace scratch is stream-local: warm up on the capture stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward()
    torch.cuda.synchronize()
    runner.workspace.routes.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = forward()
    graph.replay()
    torch.cuda.synchronize()
    assert _rel_l2(captured.float(), reference(mapping)) < 3e-2

    # Replay over new adapter slots, routing, and inputs in the captured buffers.
    mapping.copy_(mapping.roll(1))
    topk_ids.copy_(topk_ids.flip(1))
    hidden.mul_(-0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert _rel_l2(captured.float(), reference(mapping)) < 3e-2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
