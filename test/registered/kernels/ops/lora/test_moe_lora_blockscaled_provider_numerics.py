"""MX block-scaled CuTeDSL providers against references built from the same
quantization rule as the row quantizer, so the GEMMs are compared exactly;
then the shared-outer rows end to end through the runner with nonzero adapters.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.lora.moe.activation import ActivationFn
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.plan import resolve_plans
from sglang.srt.lora.moe.runner import MoeLoraBatch, MoeLoraRunner
from sglang.srt.lora.utils import Phase
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_E = 4
_H = 512
_I = 256
_TOPK = 2
_T = 40
_VEC = 32

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _e2m1_codes(magnitude: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 code; ties go to the even code, as the hardware cvt.rn does."""
    mid = _E2M1_MID.to(magnitude.device)
    idx = torch.bucketize(magnitude, mid)  # a tie lands on the lower code
    tie = (idx < 7) & (magnitude == mid[idx.clamp(max=6)]) & (idx % 2 == 1)
    return idx + tie.long()


def _skip_unless_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("block-scaled tcgen05 MMA needs SM100")


def _quant_mx(x: torch.Tensor, elem: str):
    """[..., K] -> (data, E8M0 bytes [..., K/32], exact dequant fp32)."""
    *lead, k = x.shape
    groups = x.float().reshape(*lead, k // _VEC, _VEC)
    amax = groups.abs().amax(dim=-1)
    elem_max, elem_max_log2 = (6.0, 2) if elem == "e2m1" else (448.0, 8)
    # Smallest power of two with amax <= elem_max * 2**e, as the row quantizer.
    floor_log2 = torch.frexp(amax)[1].float() - 1
    exponent = floor_log2 - elem_max_log2
    exponent = exponent + (amax > elem_max * torch.exp2(exponent)).float()
    exponent = torch.where(amax > 0, exponent, torch.zeros_like(amax))
    exponent = exponent.clamp(-126.0, 127.0)
    scale = torch.exp2(exponent)
    scaled = groups / scale[..., None]
    if elem == "e2m1":
        table = _E2M1.to(x.device)
        idx = _e2m1_codes(scaled.abs())
        sign = scaled < 0
        values = torch.where(sign, -table[idx], table[idx])
        codes = (idx | (sign.long() << 3)).to(torch.uint8).reshape(*lead, k)
        data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    else:
        q = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        values = q.float()
        data = q.reshape(*lead, k)
    return (
        data,
        (exponent + 127).to(torch.uint8),
        (values * scale[..., None]).reshape(*lead, k),
    )


def _quant_nvfp4(x: torch.Tensor, global_scale: float):
    """E2M1 pairs with E4M3 scales per 16 under one global scale (the row rule)."""
    *lead, k = x.shape
    groups = x.float().reshape(*lead, k // 16, 16)
    amax = groups.abs().amax(dim=-1)
    block = (amax / (6.0 * global_scale)).clamp(2.0**-6, 448.0).to(torch.float8_e4m3fn)
    denom = block.float() * global_scale
    scaled = (groups * (1.0 / denom)[..., None]).clamp(-6.0, 6.0)
    table = _E2M1.to(x.device)
    idx = _e2m1_codes(scaled.abs())
    sign = scaled < 0
    values = torch.where(sign, -table[idx], table[idx])
    codes = (idx | (sign.long() << 3)).to(torch.uint8).reshape(*lead, k)
    data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return data, block.view(torch.uint8), (values * denom[..., None]).reshape(*lead, k)


_W13_GLOBAL = 0.02
_W2_GLOBAL = 0.01
_A13_GLOBAL = 0.25
_A2_GLOBAL = 0.05


def _quant_info(family: str, w13: torch.Tensor, w2: torch.Tensor):
    from sglang.srt.lora.moe.quant_info import (
        MoeLoraMxfp4QuantInfo,
        MoeLoraMxfp8QuantInfo,
        MoeLoraNvFp4QuantInfo,
    )

    if family == "nvfp4":
        # Distinct gate/up global scales exercise the alpha half selection.
        w13_gs = torch.tensor([[_W13_GLOBAL, 2 * _W13_GLOBAL]] * _E, device=w13.device)
        w13_q, w13_s, w13_eff = [], [], []
        for half in range(2):
            q, s, eff = _quant_nvfp4(
                w13[:, half * _I : (half + 1) * _I], w13_gs[0, half].item()
            )
            w13_q.append(q), w13_s.append(s), w13_eff.append(eff)
        w2_q, w2_s, w2_eff = _quant_nvfp4(w2, _W2_GLOBAL)
        info = MoeLoraNvFp4QuantInfo(
            w13_weight=torch.cat(w13_q, dim=1),
            w13_scale=torch.cat(w13_s, dim=1),
            w2_weight=w2_q,
            w2_scale=w2_s,
            elements_per_byte=2,
            num_local_experts=_E,
            intermediate_size=_I,
            hidden_size=_H,
            w13_global_scale=w13_gs,
            w2_global_scale=torch.full((_E,), _W2_GLOBAL, device=w2.device),
            w13_input_scale=_A13_GLOBAL,
            w2_input_scale=_A2_GLOBAL,
        )
        return info, torch.cat(w13_eff, dim=1), w2_eff, "nvfp4"

    elem = "e2m1" if family == "mxfp4" else "e4m3"
    w13_q, w13_s, w13_eff = _quant_mx(w13, elem)
    w2_q, w2_s, w2_eff = _quant_mx(w2, elem)
    cls = MoeLoraMxfp4QuantInfo if family == "mxfp4" else MoeLoraMxfp8QuantInfo
    if family == "mxfp4":
        # The loader keeps FP4 pairs as int8.
        w13_q, w2_q = w13_q.view(torch.int8), w2_q.view(torch.int8)
    info = cls(
        w13_weight=w13_q,
        w13_scale=w13_s,
        w2_weight=w2_q,
        w2_scale=w2_s,
        elements_per_byte=2 if family == "mxfp4" else 1,
        num_local_experts=_E,
        intermediate_size=_I,
        hidden_size=_H,
    )
    return info, w13_eff, w2_eff, elem


def _gather_pairs(slab_or_rows, pair_to_row, pair_idx):
    rows = (
        slab_or_rows.reshape(-1, slab_or_rows.shape[-1])
        if slab_or_rows.ndim == 3
        else slab_or_rows
    )
    return rows[pair_to_row[pair_idx].long()]


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


def _quant_activation(x: torch.Tensor, elem: str, stage: str):
    """Dequantized activation rows as the provider quantizes them."""
    if elem == "nvfp4":
        gs = _A13_GLOBAL if stage == "gemm1" else _A2_GLOBAL
        return _quant_nvfp4(x, gs)[2]
    return _quant_mx(x, elem)[2]


@pytest.mark.parametrize("family", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("rows", ["expert_major", "route_major"])
def test_blockscaled_gateup_and_down_match_reference(family, rows):
    _skip_unless_sm100()
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(7)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    w13, w2 = rand(_E, 2 * _I, _H), rand(_E, _H, _I)
    hidden = rand(_T, _H, scale=0.5)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(_T)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[1, 0] = -1  # an EP hole
    quant_info, w13_eff, w2_eff, elem = _quant_info(family, w13, w2)

    provider = select_provider_cls(rows, family, "cutedsl")(quant_info)
    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, _TOPK)
        gateup = torch.zeros(
            provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.gateup(state, gateup)

    flat_ids = topk_ids.flatten()
    pair_idx = torch.nonzero(flat_ids >= 0, as_tuple=True)[0]
    experts = flat_ids[pair_idx].long()
    tokens = (pair_idx // _TOPK).long()
    pair_to_row = state.pair_to_row

    # The provider quantizes its rows with the same rule, so references are exact.
    hidden_eff = _quant_activation(hidden[tokens], elem, "gemm1")
    ref_gateup = torch.einsum("ph,pnh->pn", hidden_eff, w13_eff[experts])
    got_gateup = _gather_pairs(gateup, pair_to_row, pair_idx).float()
    assert _rel_l2(got_gateup, ref_gateup) < 1.5e-2

    act_pairs = (torch.randn_like(ref_gateup[:, :_I]) * 0.1).to(torch.bfloat16)
    act = torch.zeros(
        provider.act_out_shape(state), dtype=torch.bfloat16, device=device
    )
    act.reshape(-1, _I)[pair_to_row[pair_idx].long()] = act_pairs
    down = torch.zeros(
        provider.down_out_shape(state), dtype=torch.bfloat16, device=device
    )
    with get_context().override_server_args():
        provider.down(state, act, down)
    act_eff = _quant_activation(act_pairs, elem, "gemm2")
    ref_down = torch.einsum("pi,phi->ph", act_eff, w2_eff[experts])
    got_down = _gather_pairs(down, pair_to_row, pair_idx).float()
    assert _rel_l2(got_down, ref_down) < 1.5e-2


_ROUTED_SCALING = 0.75
# Measured 0.2-0.3% rel L2 on GB300 for every family and phase.
_RUNNER_TOL = 1e-2


def _standalone_output_allocation(runner, *, num_tokens, dtype, device):
    return torch.empty((num_tokens, runner.hidden_size), dtype=dtype, device=device)


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """A BF16 stage store between FP32 computations."""
    return x.to(torch.bfloat16).float()


@pytest.mark.parametrize("family", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize(
    "phase", [Phase.DECODE, Phase.PREFILL], ids=["decode", "prefill"]
)
def test_blockscaled_shared_outer_plans_match_reference(
    monkeypatch: pytest.MonkeyPatch, family: str, phase: Phase
) -> None:
    """Run the sm100 shared-outer decode (expert slabs) and prefill (flat
    segments) rows end to end on the CuTeDSL vendor with nonzero adapters.
    The reference quantizes both GEMM inputs by the row rule and rounds where
    the runner stores BF16. Decode then replays a graph over changed adapter
    slots, routing, and inputs.
    """
    _skip_unless_sm100()
    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    g = torch.Generator(device="cpu").manual_seed(29)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    tokens = 8 if phase is Phase.DECODE else 40
    slots, rank = 2, 16
    quant_info, w13_eff, w2_eff, elem = _quant_info(
        family, rand(_E, 2 * _I, _H), rand(_E, _H, _I)
    )
    selected = resolve_plans(
        architecture="sm100",
        is_shared_outer=True,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family=family,
        vendor="cutedsl",
    )[phase]
    assert selected.name == (
        "decode.shared" if phase is Phase.DECODE else "prefill.shared"
    )
    provider = select_provider_cls(selected.base_gemm_rows, family, "cutedsl")(
        quant_info
    )
    runner = MoeLoraRunner(
        providers={selected.base_gemm_rows: provider},
        top_k=_TOPK,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFn.SILU,
    )
    runner.validate_plan(selected.plan, base_gemm_rows=selected.base_gemm_rows)
    runner.plans = {phase: selected}

    hidden = rand(tokens, _H, scale=0.5)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[3, 1] = -1  # an EP hole on an adapter token
    weights = torch.tensor([0.65, 0.35], device=device).repeat(tokens, 1)
    # Outer factors (gate/up-A, down-B) per slot; inner ones per (slot, expert).
    gate_up_a = rand(slots, 1, 2 * rank, _H, scale=0.1)
    gate_up_b = rand(slots, _E, 2 * _I, rank, scale=0.1)
    down_a = rand(slots, _E, rank, _I, scale=0.1)
    down_b = rand(slots, 1, _H, rank, scale=0.1)
    # Slots 0, 1 and the base path (-1) in turn.
    mapping = ((torch.arange(tokens, device=device) + 1) % 3 - 1).to(torch.int32)

    def reference(mapping):
        pair_tokens = torch.arange(tokens, device=device).repeat_interleave(_TOPK)
        experts = topk_ids.flatten().long()
        routed = (experts >= 0).float()[:, None]
        experts = experts.clamp(min=0)
        slot = mapping[pair_tokens].long()
        adapted = (slot >= 0).float()[:, None]
        slot = slot.clamp(min=0)
        inputs = hidden.float()[pair_tokens]
        # The base GEMMs read row-quantized inputs and store BF16.
        inputs_q = _quant_activation(inputs, elem, "gemm1")
        gate_up = _bf16(torch.einsum("ph,pnh->pn", inputs_q, w13_eff[experts]))
        outer = _bf16(torch.einsum("ph,prh->pr", inputs, gate_up_a.float()[slot, 0]))
        inner = gate_up_b.float()[slot, experts]
        delta = torch.cat(
            [
                torch.einsum(
                    "pir,pr->pi",
                    inner[:, s * _I : (s + 1) * _I],
                    outer[:, s * rank : (s + 1) * rank],
                )
                for s in range(2)
            ],
            dim=1,
        )
        # The decode row stores the materialized delta in BF16; the prefill row's
        # fused LoRA-B activation adds it in FP32.
        gate_up = gate_up + adapted * (_bf16(delta) if phase is Phase.DECODE else delta)
        act = _bf16(F.silu(gate_up[:, :_I]) * gate_up[:, _I:])
        act_q = _quant_activation(act, elem, "gemm2")
        pairs = _bf16(torch.einsum("pi,phi->ph", act_q, w2_eff[experts]))
        down_rank = _bf16(
            torch.einsum("pi,pri->pr", act, down_a.float()[slot, experts])
        )
        pairs = pairs + adapted * _bf16(
            torch.einsum("pr,phr->ph", down_rank, down_b.float()[slot, 0])
        )
        pairs = pairs * routed
        return (pairs.view(tokens, _TOPK, _H) * weights[..., None]).sum(
            1
        ) * _ROUTED_SCALING

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
        is_prefill=phase is Phase.PREFILL,
    )

    def forward():
        with get_context().override_server_args():
            return runner.run(dispatch, batch).hidden_states

    output = forward()
    torch.cuda.synchronize()
    assert _rel_l2(output.float(), reference(mapping)) < _RUNNER_TOL
    # LoRA math must be present, not bypassed within tolerance.
    assert _rel_l2(reference(torch.full_like(mapping, -1)), reference(mapping)) > 0.1
    if phase is Phase.PREFILL:
        return

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
    assert _rel_l2(captured.float(), reference(mapping)) < _RUNNER_TOL

    # Replay over new adapter slots, routing, and inputs in the captured buffers.
    mapping.copy_(mapping.roll(1))
    topk_ids.copy_(topk_ids.flip(1))
    hidden.mul_(-0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert _rel_l2(captured.float(), reference(mapping)) < _RUNNER_TOL


def test_block_scaled_admission():
    from types import SimpleNamespace

    from sglang.srt.lora.moe.base_gemm_provider import resolve_vendor
    from sglang.srt.lora.moe.quant_info import MoeLoraMxfp8QuantInfo

    assert resolve_vendor("mxfp8", "triton") == "cutedsl"
    assert resolve_vendor("mxfp4", "cutedsl") == "cutedsl"
    assert resolve_vendor("mxfp4", "marlin") == "marlin"
    layer = SimpleNamespace(
        params_dtype=torch.bfloat16,
        w13_weight=torch.zeros((2, 128, 192), dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.zeros((2, 128, 6), dtype=torch.uint8),
        w2_weight=torch.zeros((2, 192, 64), dtype=torch.float8_e4m3fn),
        w2_weight_scale_inv=torch.zeros((2, 192, 2), dtype=torch.uint8),
        num_local_experts=2,
        intermediate_size_per_partition=64,
        hidden_size=192,
    )
    with pytest.raises(NotImplementedError, match="hidden_size % 128"):
        MoeLoraMxfp8QuantInfo.from_layer(layer)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
