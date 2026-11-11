"""Marlin W8A16 provider for the FP8 families, using the serving repack, against
exact dequantization and against the native W8A8 provider of each family;
then the shared-outer decode plan through the runner.
Checkpoint layouts: 128-block FP32 scales (``fp8``), one FP32 scale per output
row (``fp8_channel``), one FP32 scale per output shard (``fp8_tensor``: the
loader's separate gate and up scales, one scale for down) and E8M0 scales per
32 along K (``mxfp8``).
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
from sglang.test.kernels.moe_lora import (
    cutedsl_ready,
    rel_l2,
    standalone_output_allocation,
)

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# The MXFP8 native comparison needs the Blackwell block-scaled provider.
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_E = 4
_H = 256
_I = 128
_TOPK = 2
_T = 32
_FAMILIES = ("fp8", "fp8_channel", "fp8_tensor", "mxfp8")
# In this geometry the down GEMM's K (128) would hold a single 128-wide group,
# which the kernel reads as channel-wise, so the repack serves every family
# with 32-wide groups (marlin_fp8_moe_group_size).
_GROUP_SIZE = {"fp8": 32, "fp8_channel": 32, "fp8_tensor": 32, "mxfp8": 32}
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

# W8A16 Marlin keeps the activations exact and rounds each scale once to BF16,
# so its gap to the FP32 reference stays far inside the W4A16 bound (2e-2) the
# NVFP4 / MXFP4 Marlin tests use; the native W8A8 providers quantize the
# activations too, and their own tests bound them at 4e-2 (per 128-group) and
# 6e-2 (per token) against the same reference. The native per-tensor provider
# folds the gate and up scales into one, so fp8_tensor has no like-for-like
# native comparison and is checked against the exact product only.
_MARLIN_TOL = 2e-2
_NATIVE_TOL = {"fp8": 4e-2, "fp8_channel": 6e-2, "mxfp8": 4e-2}


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("the Marlin FP8 MoE LoRA provider needs SM90+ here")


def _quantize(family: str, w: torch.Tensor, *, shards: int = 1):
    """[E, N, K] bf16 -> (e4m3 weights, checkpoint scales, Marlin FP32 group
    scales, dequantized FP32 weights) in the family's checkpoint layout;
    ``shards`` is the number of output shards with their own per-tensor scale
    (2 for the loader's separate gate and up scales)."""
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        marlin_moe_fp8_block_scales,
        marlin_moe_fp8_channel_scales,
        marlin_moe_fp8_tensor_scales,
        marlin_moe_mxfp8_scales,
    )

    e, n, k = w.shape
    w = w.float()
    if family == "fp8_channel":
        scale = w.abs().amax(dim=2, keepdim=True).clamp(min=1e-6) / _FP8_MAX
        q = (w / scale).to(torch.float8_e4m3fn)
        return q, scale, marlin_moe_fp8_channel_scales(scale), q.float() * scale
    if family == "fp8_tensor":
        parts = w.view(e, shards, n // shards, k)
        scale = parts.abs().amax(dim=(2, 3)).clamp(min=1e-6) / _FP8_MAX
        # Deliberately unequal shard scales (1x, 3x, ...): each shard must keep
        # its own instead of being requantized to a shared one.
        scale = scale * (1 + 2 * torch.arange(shards, device=w.device).float())
        q = (parts / scale[:, :, None, None]).to(torch.float8_e4m3fn)
        deq = (q.float() * scale[:, :, None, None]).view(e, n, k)
        checkpoint = scale if shards > 1 else scale[:, 0]
        return (
            q.view(e, n, k),
            checkpoint,
            marlin_moe_fp8_tensor_scales(checkpoint, size_n=n),
            deq,
        )
    if family == "fp8":
        nb, kb = n // 128, k // 128
        blocks = w.view(e, nb, 128, kb, 128)
        scale = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-6) / _FP8_MAX
        q = (blocks / scale[:, :, None, :, None]).to(torch.float8_e4m3fn)
        deq = (q.float() * scale[:, :, None, :, None]).view(e, n, k)
        return (
            q.view(e, n, k),
            scale,
            marlin_moe_fp8_block_scales(scale, size_n=n, block_n=128),
            deq,
        )
    assert family == "mxfp8"
    groups = w.view(e, n, k // 32, 32)
    # The smallest power of two that brings the group's maximum into E4M3.
    exponent = torch.ceil(
        torch.log2(groups.abs().amax(dim=-1).clamp(min=2.0**-120) / _FP8_MAX)
    )
    scale = torch.exp2(exponent)
    q = (groups / scale[..., None]).to(torch.float8_e4m3fn)
    deq = (q.float() * scale[..., None]).view(e, n, k)
    e8m0 = (exponent + 127).to(torch.uint8)
    return q.view(e, n, k), e8m0, marlin_moe_mxfp8_scales(e8m0), deq


def _native_quant_info(family, w13_q, w13_s, w2_q, w2_s):
    from sglang.srt.lora.moe.quant_info import (
        MoeLoraFp8ChannelQuantInfo,
        MoeLoraFp8QuantInfo,
        MoeLoraMxfp8QuantInfo,
    )

    sizes = dict(num_local_experts=_E, intermediate_size=_I, hidden_size=_H)
    if family == "fp8_tensor":
        return None
    if family == "fp8":
        return MoeLoraFp8QuantInfo(
            w13_weight=w13_q,
            w13_scale=w13_s,
            w2_weight=w2_q,
            w2_scale=w2_s,
            block_shape=(128, 128),
            **sizes,
        )
    if family == "fp8_channel":
        return MoeLoraFp8ChannelQuantInfo(
            w13_weight=w13_q, w13_scale=w13_s, w2_weight=w2_q, w2_scale=w2_s, **sizes
        )
    return MoeLoraMxfp8QuantInfo(
        w13_weight=w13_q,
        w13_scale=w13_s,
        w2_weight=w2_q,
        w2_scale=w2_s,
        elements_per_byte=1,
        **sizes,
    )


def _marlin_layer(family: str, w13: torch.Tensor, w2: torch.Tensor, gated=True):
    """Repack the experts as serving does and bind them; also return the
    dequantized weights and the native provider's quant info."""
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        prepare_moe_fp8_layer_for_marlin,
    )
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8MarlinQuantInfo

    w13_q, w13_s, w13_m, w13_eff = _quantize(family, w13, shards=2)
    w2_q, w2_s, w2_m, w2_eff = _quantize(family, w2)
    native = _native_quant_info(family, w13_q, w13_s, w2_q, w2_s) if gated else None
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(w13_q.clone(), requires_grad=False),
        w2_weight=torch.nn.Parameter(w2_q.clone(), requires_grad=False),
        params_dtype=torch.bfloat16,
        moe_runner_config=SimpleNamespace(
            is_gated=gated,
            activation="silu" if gated else "relu2",
            apply_router_weight_on_input=False,
        ),
        num_local_experts=_E,
        intermediate_size_per_partition=w2.shape[2],
        hidden_size=w2.shape[1],
    )
    prepare_moe_fp8_layer_for_marlin(
        layer, w13_scales=w13_m, w2_scales=w2_m, params_dtype=torch.bfloat16
    )
    return MoeLoraFp8MarlinQuantInfo.from_layer(layer), native, w13_eff, w2_eff


def _gather_pairs(rows, pair_to_row, pair_idx):
    return rows.reshape(-1, rows.shape[-1])[pair_to_row[pair_idx].long()]


def _native_vendor(family: str) -> str | None:
    """The native provider to compare with on this device, if it runs here."""
    if family == "fp8_tensor":
        return None
    if family == "mxfp8":
        return (
            "cutedsl"
            if cutedsl_ready() and torch.cuda.get_device_capability()[0] == 10
            else None
        )
    return "triton"


@pytest.mark.parametrize("family", _FAMILIES)
def test_fp8_marlin_gateup_and_down_match_reference(family):
    _skip_unless_supported()
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(11)

    def rand(*shape):
        return (
            (torch.randn(*shape, generator=g, device="cpu") * 0.05)
            .to(torch.bfloat16)
            .to(device)
        )

    w13, w2, hidden = rand(_E, 2 * _I, _H), rand(_E, _H, _I), rand(_T, _H)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g, device="cpu")[:_TOPK] for _ in range(_T)]
    ).to(device=device, dtype=torch.int32)
    quant_info, native_info, w13_eff, w2_eff = _marlin_layer(family, w13, w2)
    assert quant_info.group_size == _GROUP_SIZE[family]

    flat_ids = topk_ids.flatten()
    pair_idx = torch.nonzero(flat_ids >= 0, as_tuple=True)[0]
    experts = flat_ids[pair_idx].long()
    tokens = (pair_idx // _TOPK).long()
    ref_gateup = torch.einsum("ph,pnh->pn", hidden[tokens].float(), w13_eff[experts])
    act_pairs = (torch.randn_like(ref_gateup[:, :_I]) * 0.1).to(torch.bfloat16)
    ref_down = torch.einsum("pi,phi->ph", act_pairs.float(), w2_eff[experts])

    def run(provider):
        with get_context().override_server_args():
            state = provider.prepare(hidden, topk_ids, _TOPK)
            gateup = torch.zeros(
                provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
            )
            provider.gateup(state, gateup)
            act = torch.zeros(
                provider.act_out_shape(state), dtype=torch.bfloat16, device=device
            )
            act.reshape(-1, _I)[state.pair_to_row[pair_idx].long()] = act_pairs
            down = torch.zeros(
                provider.down_out_shape(state), dtype=torch.bfloat16, device=device
            )
            provider.down(state, act, down)
        return (
            _gather_pairs(gateup, state.pair_to_row, pair_idx).float(),
            _gather_pairs(down, state.pair_to_row, pair_idx).float(),
        )

    marlin = select_provider_cls("route_major", family, "marlin")(quant_info)
    got_gateup, got_down = run(marlin)
    marlin_err = (rel_l2(got_gateup, ref_gateup), rel_l2(got_down, ref_down))
    assert max(marlin_err) < _MARLIN_TOL, marlin_err

    vendor = _native_vendor(family)
    if vendor is None:
        pytest.skip(f"no like-for-like native {family} provider on this device")
    native = select_provider_cls("route_major", family, vendor)(native_info)
    nat_gateup, nat_down = run(native)
    assert rel_l2(got_gateup, nat_gateup) < _NATIVE_TOL[family]
    assert rel_l2(got_down, nat_down) < _NATIVE_TOL[family]
    # Exact activations beat quantized ones against the same weights.
    native_err = (rel_l2(nat_gateup, ref_gateup), rel_l2(nat_down, ref_down))
    assert marlin_err[0] < native_err[0] and marlin_err[1] < native_err[1], (
        marlin_err,
        native_err,
    )


@pytest.mark.parametrize("family", _FAMILIES)
def test_fp8_marlin_shared_outer_decode_plan_matches_reference(
    monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    """Run the architecture's shared-outer decode row end to end with Marlin
    W8A16 base GEMMs: eager first, then a graph replayed with changed adapter
    slots, routing, and inputs. sm100 selects the Marlin vendor row.
    """
    _skip_unless_supported()
    device = torch.device("cuda")
    architecture = architecture_for_capability(
        torch.cuda.get_device_capability(device)[0]
    )
    monkeypatch.setattr(MoeLoraRunner, "_allocate_output", standalone_output_allocation)
    g = torch.Generator(device="cpu").manual_seed(23)

    def rand(*shape, scale=0.05):
        return (
            (torch.randn(*shape, generator=g, device="cpu") * scale)
            .to(torch.bfloat16)
            .to(device)
        )

    tokens, slots, rank = 8, 2, 16
    quant_info, _, w13_eff, w2_eff = _marlin_layer(
        family, rand(_E, 2 * _I, _H), rand(_E, _H, _I)
    )
    selected = resolve_plans(
        architecture=architecture,
        is_shared_outer=True,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family=family,
        vendor="marlin",
    )[Phase.DECODE]
    assert selected.name == {
        "sm100": "decode.shared.marlin",
        "sm90": "decode.shared",
    }.get(architecture, selected.name)
    provider = select_provider_cls(selected.base_gemm_rows, family, "marlin")(
        quant_info
    )
    runner = MoeLoraRunner(
        providers={selected.base_gemm_rows: provider},
        top_k=_TOPK,
        routed_scaling_factor=0.75,
        activation=ActivationFn.SILU,
        is_gated=True,
    )
    runner.validate_plan(selected.plan, base_gemm_rows=selected.base_gemm_rows)
    runner.plans = {Phase.DECODE: selected}

    hidden = rand(tokens, _H, scale=0.2)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g, device="cpu")[:_TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[3, 1] = -1  # an EP hole on an adapter token
    weights = torch.tensor([0.65, 0.35], device=device).repeat(tokens, 1)
    # Outer factors (gate/up-A, down-B) per slot; inner ones per (slot, expert).
    gate_up_a = rand(slots, 1, 2 * rank, _H, scale=0.1)
    gate_up_b = rand(slots, _E, 2 * _I, rank, scale=0.1)
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
                for s in range(2)
            ],
            dim=1,
        )
        gate_up = gate_up + adapted * delta
        act = F.silu(gate_up[:, :_I]) * gate_up[:, _I:]
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
    assert rel_l2(output.float(), reference(mapping)) < _MARLIN_TOL
    # LoRA math must be present, not bypassed within tolerance.
    assert rel_l2(reference(torch.full_like(mapping, -1)), reference(mapping)) > 0.1

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
    assert rel_l2(captured.float(), reference(mapping)) < _MARLIN_TOL

    # Replay over new adapter slots, routing, and inputs in the captured buffers.
    mapping.copy_(mapping.roll(1))
    topk_ids.copy_(topk_ids.flip(1))
    hidden.mul_(-0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert rel_l2(captured.float(), reference(mapping)) < _MARLIN_TOL


def _lora_reference(
    *,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    weights: torch.Tensor,
    routed_scaling: float,
    w13_eff: torch.Tensor,
    w2_eff: torch.Tensor,
    gate_up_a: torch.Tensor,
    gate_up_b: torch.Tensor,
    down_a: torch.Tensor,
    down_b: torch.Tensor,
    mapping: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """FP32 MoE + LoRA on the dequantized experts; outer factors are per
    expert ([L, E, ...]) or shared across experts ([L, 1, ...])."""
    tokens, intermediate = hidden.shape[0], w2_eff.shape[2]
    pair_tokens = torch.arange(tokens, device=hidden.device).repeat_interleave(_TOPK)
    experts = topk_ids.flatten().long()
    routed = (experts >= 0).float()[:, None]
    experts = experts.clamp(min=0)
    slot = mapping[pair_tokens].long()
    adapted = (slot >= 0).float()[:, None]
    slot = slot.clamp(min=0)
    a_expert = experts if gate_up_a.shape[1] > 1 else torch.zeros_like(experts)
    b_expert = experts if down_b.shape[1] > 1 else torch.zeros_like(experts)
    inputs = hidden.float()[pair_tokens]
    gate_up = torch.einsum("ph,pnh->pn", inputs, w13_eff[experts])
    outer = torch.einsum("ph,prh->pr", inputs, gate_up_a.float()[slot, a_expert])
    inner = gate_up_b.float()[slot, experts]
    delta = torch.cat(
        [
            torch.einsum(
                "pir,pr->pi",
                inner[:, s * intermediate : (s + 1) * intermediate],
                outer[:, s * rank : (s + 1) * rank],
            )
            for s in range(2)
        ],
        dim=1,
    )
    gate_up = gate_up + adapted * delta
    act = F.silu(gate_up[:, :intermediate]) * gate_up[:, intermediate:]
    pairs = torch.einsum("pi,phi->ph", act, w2_eff[experts])
    down_rank = torch.einsum("pi,pri->pr", act, down_a.float()[slot, experts])
    pairs = pairs + adapted * torch.einsum(
        "pr,phr->ph", down_rank, down_b.float()[slot, b_expert]
    )
    pairs = pairs * routed
    return (pairs.view(tokens, _TOPK, -1) * weights[..., None]).sum(1) * routed_scaling


@pytest.mark.parametrize(
    "phase", (Phase.PREFILL, Phase.DECODE), ids=("prefill", "decode")
)
@pytest.mark.parametrize(
    "is_shared_outer", (False, True), ids=("per_expert", "shared_outer")
)
@pytest.mark.parametrize("family", _FAMILIES)
def test_fp8_marlin_plans_replay_mixed_adapter_ranks(
    monkeypatch: pytest.MonkeyPatch, family: str, is_shared_outer: bool, phase: Phase
) -> None:
    """The architecture's selected row for each adapter layout and phase, with
    Marlin W8A16 base GEMMs: a batch mixing base tokens, a rank-16 adapter and
    a rank-8 adapter (zero-padded to the physical rank) matches the FP32
    reference eagerly, then from a CUDA graph replayed over changed slots,
    routing and inputs. The decode batch is smaller than a Marlin M block, so
    the sorted rows are padded. The per-tensor family runs with unequal gate
    and up shard scales, so a merged scale would miss the reference."""
    _skip_unless_supported()
    device = torch.device("cuda")
    architecture = architecture_for_capability(
        torch.cuda.get_device_capability(device)[0]
    )
    monkeypatch.setattr(MoeLoraRunner, "_allocate_output", standalone_output_allocation)
    g = torch.Generator(device="cpu").manual_seed(31)

    def rand(*shape, scale=0.05):
        return (
            (torch.randn(*shape, generator=g, device="cpu") * scale)
            .to(torch.bfloat16)
            .to(device)
        )

    tokens = 48 if phase is Phase.PREFILL else 6
    slots, rank = 2, 16
    quant_info, _, w13_eff, w2_eff = _marlin_layer(
        family, rand(_E, 2 * _I, _H), rand(_E, _H, _I)
    )
    selected = resolve_plans(
        architecture=architecture,
        is_shared_outer=is_shared_outer,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family=family,
        vendor="marlin",
    )[phase]
    provider = select_provider_cls(selected.base_gemm_rows, family, "marlin")(
        quant_info
    )
    runner = MoeLoraRunner(
        providers={selected.base_gemm_rows: provider},
        top_k=_TOPK,
        routed_scaling_factor=0.75,
        activation=ActivationFn.SILU,
        is_gated=True,
    )
    runner.validate_plan(selected.plan, base_gemm_rows=selected.base_gemm_rows)
    runner.plans = {phase: selected}

    hidden = rand(tokens, _H, scale=0.2)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g, device="cpu")[:_TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[1, 1] = -1  # an EP hole on an adapter token
    weights = torch.tensor([0.65, 0.35], device=device).repeat(tokens, 1)
    outer_experts = 1 if is_shared_outer else _E
    gate_up_a = rand(slots, outer_experts, 2 * rank, _H, scale=0.1)
    gate_up_b = rand(slots, _E, 2 * _I, rank, scale=0.1)
    down_a = rand(slots, _E, rank, _I, scale=0.1)
    down_b = rand(slots, outer_experts, _H, rank, scale=0.1)
    # Slot 1 holds a rank-8 adapter: the upper half of every rank axis is padding.
    half = rank // 2
    gate_up_a[1, :, half:rank] = 0
    gate_up_a[1, :, rank + half :] = 0
    gate_up_b[1, :, :, half:] = 0
    down_a[1, :, half:] = 0
    down_b[1, :, :, half:] = 0
    mapping = torch.tensor([0, 1, -1] * (tokens // 3), dtype=torch.int32, device=device)

    def reference(mapping):
        return _lora_reference(
            hidden=hidden,
            topk_ids=topk_ids,
            weights=weights,
            routed_scaling=0.75,
            w13_eff=w13_eff,
            w2_eff=w2_eff,
            gate_up_a=gate_up_a,
            gate_up_b=gate_up_b,
            down_a=down_a,
            down_b=down_b,
            mapping=mapping,
            rank=rank,
        )

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
    assert rel_l2(output.float(), reference(mapping)) < _MARLIN_TOL, selected.name
    # Both adapters must matter: all-base and slot-1-as-slot-0 differ clearly.
    assert rel_l2(reference(torch.full_like(mapping, -1)), reference(mapping)) > 0.1
    assert rel_l2(reference(mapping.clamp(max=0)), reference(mapping)) > 0.02

    batch.use_cuda_graph = True
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
    assert rel_l2(captured.float(), reference(mapping)) < _MARLIN_TOL, selected.name

    mapping.copy_(mapping.roll(1))
    topk_ids.copy_(topk_ids.flip(1))
    hidden.mul_(-0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert rel_l2(captured.float(), reference(mapping)) < _MARLIN_TOL, selected.name


@pytest.mark.parametrize("intermediate", (320, 448))
@pytest.mark.parametrize("family", ("fp8_channel", "fp8_tensor"))
def test_fp8_marlin_provider_serves_64_aligned_intermediates(family, intermediate):
    """Channel-wise and per-tensor experts whose intermediate size is a multiple
    of 64 but not of 128 were served before; they now take 32-wide groups and
    must still repack, bind and match the exact product in both GEMMs."""
    _skip_unless_supported()
    device = torch.device("cuda")
    hidden = 512
    g = torch.Generator(device="cpu").manual_seed(17)

    def rand(*shape):
        return (
            (torch.randn(*shape, generator=g, device="cpu") * 0.05)
            .to(torch.bfloat16)
            .to(device)
        )

    w13, w2, hid = (
        rand(_E, 2 * intermediate, hidden),
        rand(_E, hidden, intermediate),
        rand(_T, hidden),
    )
    quant_info, _, w13_eff, w2_eff = _marlin_layer(family, w13, w2)
    assert quant_info.group_size == 32
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g, device="cpu")[:_TOPK] for _ in range(_T)]
    ).to(device=device, dtype=torch.int32)
    pair_idx = torch.arange(topk_ids.numel(), device=device)
    experts = topk_ids.flatten().long()
    tokens = pair_idx // _TOPK
    act_pairs = (
        (torch.randn((_T * _TOPK, intermediate), generator=g, device="cpu") * 0.1)
        .to(torch.bfloat16)
        .to(device)
    )
    ref_gateup = torch.einsum("ph,pnh->pn", hid[tokens].float(), w13_eff[experts])
    ref_down = torch.einsum("pi,phi->ph", act_pairs.float(), w2_eff[experts])
    provider = select_provider_cls("route_major", family, "marlin")(quant_info)
    with get_context().override_server_args():
        state = provider.prepare(hid, topk_ids, _TOPK)
        gateup = torch.zeros(
            provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.gateup(state, gateup)
        act = torch.zeros(
            provider.act_out_shape(state), dtype=torch.bfloat16, device=device
        )
        act.reshape(-1, intermediate)[state.pair_to_row[pair_idx].long()] = act_pairs
        down = torch.zeros(
            provider.down_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.down(state, act, down)
    got_gateup = _gather_pairs(gateup, state.pair_to_row, pair_idx).float()
    got_down = _gather_pairs(down, state.pair_to_row, pair_idx).float()
    assert rel_l2(got_gateup, ref_gateup) < _MARLIN_TOL
    assert rel_l2(got_down, ref_down) < _MARLIN_TOL


def test_fp8_marlin_channel_scales_are_applied_before_accumulation() -> None:
    """Channel-wise FP8 experts with codes of 1 and scales of 1: BF16 inputs of
    2**-60 must give exactly 2**-52 over the gate/up GEMM's K of 256 and 2**-53
    over the down GEMM's K of 128, on both reduction paths. The kernel's
    single-group (channel-wise) path scales the FP32 accumulators after the K
    loop, where these 2**-120-shifted products have already underflowed to 0;
    the repack hands the kernel the scale repeated over K groups instead."""
    _skip_unless_supported()
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        marlin_moe_fp8_channel_scales,
        prepare_moe_fp8_layer_for_marlin,
    )
    from sglang.srt.lora.moe.base_gemm_provider.marlin_fp8 import (
        MarlinFp8ContiguousProvider,
    )
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8MarlinQuantInfo

    device = torch.device("cuda")
    ones = lambda *shape: torch.ones(shape, device=device).to(torch.float8_e4m3fn)
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(ones(_E, 2 * _I, _H), requires_grad=False),
        w2_weight=torch.nn.Parameter(ones(_E, _H, _I), requires_grad=False),
        params_dtype=torch.bfloat16,
        moe_runner_config=SimpleNamespace(
            is_gated=True, activation="silu", apply_router_weight_on_input=False
        ),
        num_local_experts=_E,
        intermediate_size_per_partition=_I,
        hidden_size=_H,
    )
    prepare_moe_fp8_layer_for_marlin(
        layer,
        w13_scales=marlin_moe_fp8_channel_scales(
            torch.ones((_E, 2 * _I, 1), device=device)
        ),
        w2_scales=marlin_moe_fp8_channel_scales(torch.ones((_E, _H, 1), device=device)),
        params_dtype=torch.bfloat16,
    )
    quant_info = MoeLoraFp8MarlinQuantInfo.from_layer(layer)
    assert quant_info.group_size == 32
    tokens = 4
    hidden = torch.full((tokens, _H), 2.0**-60, dtype=torch.bfloat16, device=device)
    topk_ids = torch.stack(
        [torch.randperm(_E, device=device)[:_TOPK] for _ in range(tokens)]
    ).int()
    pair_idx = torch.arange(topk_ids.numel(), device=device)
    for atomic in (True, False):
        provider_cls = type(
            "Provider", (MarlinFp8ContiguousProvider,), {"_use_atomic_add": atomic}
        )
        provider = provider_cls(quant_info)
        with get_context().override_server_args():
            state = provider.prepare(hidden, topk_ids, _TOPK)
            gateup = torch.zeros(
                provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
            )
            provider.gateup(state, gateup)
            act = torch.full(
                provider.act_out_shape(state),
                2.0**-60,
                dtype=torch.bfloat16,
                device=device,
            )
            down = torch.zeros(
                provider.down_out_shape(state), dtype=torch.bfloat16, device=device
            )
            provider.down(state, act, down)
        torch.cuda.synchronize()
        rows = state.pair_to_row[pair_idx].long()
        got_gateup = gateup.reshape(-1, 2 * _I)[rows]
        got_down = down.reshape(-1, _H)[rows]
        assert torch.equal(got_gateup, torch.full_like(got_gateup, 2.0**-52)), atomic
        assert torch.equal(got_down, torch.full_like(got_down, 2.0**-53)), atomic


def test_fp8_marlin_split_k_reduction_paths() -> None:
    """A decode-shaped gate/up GEMM with a long K is split along K across
    thread blocks. The kernel's atomic path adds each block's BF16-rounded
    partial into the output in completion order, so repeats may differ by BF16
    units of the running sums: bounded here on a cancelling input whose
    partials are large and whose exact product is zero. The FP32 reduction
    path sums the partials in slice order and is bitwise repeatable across
    repeats, a fresh provider and a graph replay. Both stay within the
    provider tolerance of the exact product on a random input."""
    _skip_unless_supported()
    from sglang.srt.lora.moe.base_gemm_provider.marlin_fp8 import (
        MarlinFp8ContiguousProvider,
    )

    class _Atomic(MarlinFp8ContiguousProvider):
        _use_atomic_add = True

    class _Fp32Reduce(MarlinFp8ContiguousProvider):
        _use_atomic_add = False

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(41)
    hidden_size, intermediate, tokens = 4096, 128, 4

    def rand(*shape, scale=0.05):
        return (
            (torch.randn(*shape, generator=g, device="cpu") * scale)
            .to(torch.bfloat16)
            .to(device)
        )

    w13 = rand(_E, 2 * intermediate, hidden_size)
    # The second half of K mirrors the first with the opposite sign: a mirrored
    # input then cancels exactly while every partial sum stays large.
    w13[:, :, hidden_size // 2 :] = -w13[:, :, : hidden_size // 2]
    quant_info, _, w13_eff, _ = _marlin_layer(
        "fp8_channel", w13, rand(_E, hidden_size, intermediate)
    )
    hidden = rand(tokens, hidden_size, scale=0.2)
    mirrored = hidden.clone()
    mirrored[:, hidden_size // 2 :] = mirrored[:, : hidden_size // 2]
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g, device="cpu")[:_TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    pair_idx = torch.arange(topk_ids.numel(), device=device)
    experts = topk_ids.flatten().long()
    pair_tokens = pair_idx // _TOPK

    def gateup(provider, inputs):
        with get_context().override_server_args():
            state = provider.prepare(inputs, topk_ids, _TOPK)
            out = torch.zeros(
                provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
            )
            provider.gateup(state, out)
        torch.cuda.synchronize()
        return _gather_pairs(out, state.pair_to_row, pair_idx).float()

    def graphed_gateup(provider, inputs):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gateup(provider, inputs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with get_context().override_server_args():
                state = provider.prepare(inputs, topk_ids, _TOPK)
                out = torch.zeros(
                    provider.gateup_out_shape(state),
                    dtype=torch.bfloat16,
                    device=device,
                )
                provider.gateup(state, out)
        graph.replay()
        torch.cuda.synchronize()
        return _gather_pairs(out, state.pair_to_row, pair_idx).float()

    bf16_unit = 2.0**-8
    report = {}
    for label, inputs in (("random", hidden), ("cancelling", mirrored)):
        exact = torch.einsum(
            "ph,pnh->pn", inputs[pair_tokens].float(), w13_eff[experts]
        )
        # The running sums never exceed the sum of the absolute products.
        partials = torch.einsum(
            "ph,pnh->pn", inputs[pair_tokens].float().abs(), w13_eff[experts].abs()
        )
        atomic_runs = [gateup(_Atomic(quant_info), inputs) for _ in range(3)]
        atomic_runs.append(graphed_gateup(_Atomic(quant_info), inputs))
        fp32_runs = [gateup(_Fp32Reduce(quant_info), inputs) for _ in range(3)]
        fp32_runs.append(graphed_gateup(_Fp32Reduce(quant_info), inputs))
        for run in fp32_runs[1:]:
            assert torch.equal(run, fp32_runs[0]), label
        scatter = max(
            (a - b).abs().max().item() for a in atomic_runs for b in atomic_runs
        )
        # At most 16 BF16 units of the largest running sum.
        for a in atomic_runs:
            assert bool(
                ((a - atomic_runs[0]).abs() <= 16 * bf16_unit * partials).all()
            ), label
        report[label] = dict(
            atomic_scatter=scatter,
            atomic_err=(atomic_runs[0] - exact).abs().max().item(),
            fp32_err=(fp32_runs[0] - exact).abs().max().item(),
            partials=partials.max().item(),
        )
        if label == "random":
            assert rel_l2(atomic_runs[0], exact) < _MARLIN_TOL, report
            assert rel_l2(fp32_runs[0], exact) < _MARLIN_TOL, report
    print("split-K reduction paths:", report)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
