"""NVFP4 provider against exact dequantization, using the serving repack helper.
Checkpoint format: packed E2M1 (even index in low nibble), group-16 E4M3 scales,
and an FP32 per-expert global scale. Compare raw pair-major outputs, then drive
the sm100 shared-outer decode plan through the runner.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.plan import ActivationFn, AFamily, BFamily, resolve_plans
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
        pytest.skip("Marlin NVFP4 needs SM90+ here")


def _quant_nvfp4(weight: torch.Tensor):
    """[E, N, K] bf16 -> (packed uint8 [E,N,K/2], e4m3 [E,N,K/16], fp32 [E])
    plus the exact dequantized weights."""
    e, n, k = weight.shape
    w = weight.float()
    global_scale = (w.abs().amax(dim=(1, 2)) / (448.0 * 6.0)).clamp(min=1e-12)
    groups = w.view(e, n, k // 16, 16)
    amax16 = groups.abs().amax(dim=-1)
    block_scale = (
        (amax16 / (6.0 * global_scale[:, None, None]))
        .clamp(min=2**-6)
        .to(torch.float8_e4m3fn)
    )
    denom = block_scale.float() * global_scale[:, None, None]
    scaled = groups / denom[..., None]
    table = _E2M1_VALUES.to(weight.device)
    idx = (scaled.abs().unsqueeze(-1) - table).abs().argmin(dim=-1)
    sign = scaled < 0
    codes = (idx | (sign.long() << 3)).to(torch.uint8).view(e, n, k)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    dequant = (torch.where(sign, -table[idx], table[idx]) * denom[..., None]).view(
        e, n, k
    )
    return packed, block_scale, global_scale, dequant


def _fake_layer(w13, w2, gated=True):
    w13_q, w13_bs, w13_gs, w13_eff = _quant_nvfp4(w13)
    w2_q, w2_bs, w2_gs, w2_eff = _quant_nvfp4(w2)
    layer = SimpleNamespace(
        w13_weight=SimpleNamespace(data=w13_q),
        w2_weight=SimpleNamespace(data=w2_q),
        w13_weight_scale=SimpleNamespace(data=w13_bs),
        w2_weight_scale=SimpleNamespace(data=w2_bs),
        w13_weight_scale_2=SimpleNamespace(data=w13_gs),
        w2_weight_scale_2=SimpleNamespace(data=w2_gs),
        quant_config=SimpleNamespace(group_size=16),
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
        prepare_moe_nvfp4_layer_for_marlin,
    )
    from sglang.srt.lora.moe.quant_info import MoeLoraNvFp4MarlinQuantInfo

    prepare_moe_nvfp4_layer_for_marlin(layer)
    # The fake layer has no quant_method: use the repacked tensors directly.
    return MoeLoraNvFp4MarlinQuantInfo.from_layer(layer)


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


def _standalone_output_allocation(runner, *, num_tokens, dtype, device):
    return torch.empty((num_tokens, runner.hidden_size), dtype=dtype, device=device)


@pytest.mark.parametrize("slices,intermediate", [(2, _I), (1, _I), (1, 64)])
def test_nvfp4_marlin_gateup_and_down_match_reference(slices, intermediate):
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

    provider = select_provider_cls("route_major", "nvfp4", "marlin")(quant_info)

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

        assert layer.w2_weight.shape[1] * 16 == 128
        rank = 16
        a13 = rand(_E, rank, _H).float()[experts]
        b13 = rand(_E, intermediate, rank).float()[experts]
        a2 = rand(_E, rank, intermediate).float()[experts]
        b2 = rand(_E, _H, rank).float()[experts]
        weights = torch.tensor([0.65, 0.35], device=device).repeat(_T, 1)
        output = torch.empty((_T, _H), dtype=torch.bfloat16, device=device)
        act_pairs = torch.empty(
            (_T, _TOPK, intermediate), dtype=torch.bfloat16, device=device
        )
        workspace = LoraWorkspace()
        workspace.begin_forward(graph_mode=True)

        def delta(a, b, x):
            return torch.bmm(b, torch.bmm(a, x.unsqueeze(-1))).squeeze(-1)

        def run():
            state = provider.prepare(hidden, topk_ids, _TOPK, workspace=workspace)
            provider.gateup(state, gateup)
            gate_delta = delta(a13, b13, hidden[tokens].float()).to(torch.bfloat16)
            provider.act_with_delta(
                state,
                gateup,
                gate_delta.view(_T, _TOPK, intermediate),
                topk_ids,
                act,
                act_pairs,
                activation="relu2",
            )
            # Padding must be cleared even when its reusable storage is poisoned.
            state.padded_intermediate[:, intermediate:].fill_(float("nan"))
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
            gate = torch.einsum("ph,pih->pi", inputs, w13_eff[experts])
            activation = (gate + delta(a13, b13, inputs)).relu().square()
            pairs = torch.einsum("pi,phi->ph", activation, w2_eff[experts])
            pairs += delta(a2, b2, activation)
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
def test_nvfp4_shared_outer_decode_plan_matches_reference(
    monkeypatch: pytest.MonkeyPatch, slices: int
) -> None:
    """Run the sm100 ``decode.shared.nvfp4`` row end to end: token_dense gate/up-A
    planes selected by per_row B, Marlin base GEMMs, per_row down-A, and the
    shared one-pass finalize. Eager first, then a graph replayed with changed
    adapter slots, routing, and inputs.
    """
    _skip_unless_supported()
    device = torch.device("cuda")
    major = torch.cuda.get_device_capability(device)[0]
    if major != 10:
        pytest.skip("decode.shared.nvfp4 is an sm100 plan row")
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
        architecture=architecture_for_capability(major),
        is_shared_outer=True,
        physical_rank=rank,
        activation=activation,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family="nvfp4",
    )[Phase.DECODE]
    assert (
        selected.name,
        selected.plan.gate_up_a.family,
        selected.plan.gate_up_b.family,
    ) == ("decode.shared.nvfp4", AFamily.TOKEN_DENSE, BFamily.PER_ROW)
    provider = select_provider_cls(selected.base_gemm_rows, "nvfp4", "marlin")(
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


def test_single_token_align_matches_general_path_with_invalid_ids():
    """Single-token alignment matches the general path with invalid EP ids excluded,
    and all-valid ids match the upstream CUDA warp kernel. ``align_rows`` takes
    the single-token path for one token and the general path for more; both
    write -1 into pair_to_row for invalid pairs.
    """
    _skip_unless_supported()
    from sglang.kernels.ops.lora.moe.align_rows import (
        moe_align_single_token,
    )
    from sglang.kernels.ops.moe.moe_align_single_token import (
        moe_align_single_token as moe_align_single_token_cuda,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.lora.moe.base_gemm_provider.input_alignment import align_rows

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(7)
    num_experts, topk = 64, 8

    def random_ids(tokens, n_invalid):
        ids = torch.stack(
            [torch.randperm(num_experts, generator=g)[:topk] for _ in range(tokens)]
        )
        if n_invalid:
            ids[0, torch.randperm(topk, generator=g)[:n_invalid]] = -1
        return ids.to(device=device, dtype=torch.int32)

    def assert_aligned(got, ids, block_size):
        sorted_got, experts_got, post_got = got
        sorted_ref, experts_ref, post_ref = moe_align_block_size(
            ids, block_size, num_experts, ignore_invalid_expert=True
        )
        post = int(post_ref.item())
        assert int(post_got.item()) == post
        assert torch.equal(sorted_got[:post], sorted_ref[:post])
        blocks = post // block_size
        assert torch.equal(experts_got[:blocks], experts_ref[:blocks])

    for block_size in (8, 16, 48, 256):
        for n_invalid in (0, 3, topk):
            ids = random_ids(1, n_invalid)
            fast = moe_align_single_token(ids, block_size)
            assert_aligned(fast, ids, block_size)
            if n_invalid == 0:
                cuda = moe_align_single_token_cuda(ids, block_size)
                assert all(torch.equal(a, b) for a, b in zip(fast, cuda))

    for tokens in (1, 2):
        ids = random_ids(tokens, 1)
        pair_to_row = torch.full_like(ids.flatten(), -777)
        got = align_rows(ids, 16, num_experts, pair_to_row_out=pair_to_row)
        expected_rows = torch.arange(ids.numel(), device=device, dtype=torch.int32)
        expected_rows[ids.flatten() < 0] = -1
        assert torch.equal(pair_to_row, expected_rows)
        assert_aligned(got, ids, 16)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
