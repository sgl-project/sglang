"""NVFP4 and MXFP4 Marlin repack, provider numerics, and decode replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.lora.moe.activation import ActivationFn
from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.plan import resolve_plans
from sglang.srt.lora.moe.runner import MoeLoraBatch
from sglang.srt.lora.utils import Phase, architecture_for_capability
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.moe_lora import check_marlin_lora_replay, rel_l2
from sglang.test.quant_ref_utils import quantize_mx_ref, quantize_nvfp4_static_ref

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_E = 4
_H = 256
_I = 128
_TOPK = 2
_T = 32


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Marlin FP4 needs SM90+ here")


def _marlin_quant_info(family, w13, w2, gated):
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
        prepare_moe_nvfp4_layer_for_marlin,
    )
    from sglang.srt.lora.moe.quant_info import (
        MoeLoraMxfp4MarlinQuantInfo,
        MoeLoraNvFp4MarlinQuantInfo,
    )

    fields, dequant = {}, []
    for name, w in (("w13", w13), ("w2", w2)):
        if family == "mxfp4":
            q, scale, w_eff = quantize_mx_ref(w, 32, "e2m1")
            fields[f"{name}_weight"] = q.view(torch.int8)
            fields[f"{name}_weight_scale_inv"] = scale.view(torch.float8_e8m0fnu)
        else:
            gs = (w.float().abs().amax(dim=(1, 2)) / (448.0 * 6.0)).clamp(min=1e-12)
            q, scale, w_eff = quantize_nvfp4_static_ref(w, gs)
            fields[f"{name}_weight"] = q
            fields[f"{name}_weight_scale"] = scale.view(torch.float8_e4m3fn)
            fields[f"{name}_weight_scale_2"] = gs
        dequant.append(w_eff)
    layer = SimpleNamespace(
        **{key: SimpleNamespace(data=value) for key, value in fields.items()},
        moe_runner_config=SimpleNamespace(is_gated=gated),
        num_local_experts=w2.shape[0],
        intermediate_size_per_partition=w2.shape[2],
        hidden_size=w2.shape[1],
        params_dtype=torch.bfloat16,
    )
    if family == "mxfp4":
        prepare_moe_mxfp4_layer_for_marlin(layer)
        return MoeLoraMxfp4MarlinQuantInfo.from_layer(layer), *dequant
    prepare_moe_nvfp4_layer_for_marlin(layer, group_size=16)
    return MoeLoraNvFp4MarlinQuantInfo.from_layer(layer), *dequant


# Exercise gated MXFP4 and non-gated NVFP4 intermediate padding.
@pytest.mark.parametrize(
    "family,slices,intermediate",
    [("mxfp4", 2, 96), ("nvfp4", 2, _I), ("nvfp4", 1, 64)],
)
def test_fp4_marlin_gateup_and_down_match_reference(family, slices, intermediate):
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

    quant_info, w13_eff, w2_eff = _marlin_quant_info(family, w13, w2, gated=slices == 2)
    assert (quant_info.gate_up_slices, quant_info.intermediate_size) == (
        slices,
        intermediate,
    )
    assert quant_info.w2_qweight.shape[1] * 16 == _I

    provider = select_provider_cls("route_major", family, "marlin")(quant_info)

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
        assert rel_l2(got_gateup, ref_gateup) < 2e-2

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
        assert rel_l2(got_down, ref_down) < 2e-2

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
        assert rel_l2(output.float(), reference()) < 3e-2
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for _ in range(2):
            hidden.mul_(-0.5)
            graph.replay()
            torch.cuda.synchronize()
            assert rel_l2(output.float(), reference()) < 3e-2


@pytest.mark.parametrize(
    "family,slices",
    [("mxfp4", 1), ("nvfp4", 2), ("nvfp4", 1)],
    ids=("mxfp4-relu2", "nvfp4-gated-silu", "nvfp4-relu2"),
)
def test_fp4_marlin_shared_outer_decode_plan_matches_reference(
    monkeypatch: pytest.MonkeyPatch, family: str, slices: int
) -> None:
    _skip_unless_supported()
    device = torch.device("cuda")
    architecture = architecture_for_capability(
        torch.cuda.get_device_capability(device)[0]
    )
    if family == "nvfp4" and architecture != "sm100":
        pytest.skip("the NVFP4 case covers the sm100 decode.shared.marlin row")
    g = torch.Generator(device="cpu").manual_seed(23)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    tokens, slots, rank = 8, 2, 16
    activation = ActivationFn.SILU if slices == 2 else ActivationFn.RELU2
    quant_info, w13_eff, w2_eff = _marlin_quant_info(
        family, rand(_E, slices * _I, _H), rand(_E, _H, _I), gated=slices == 2
    )
    selected = resolve_plans(
        architecture=architecture,
        is_shared_outer=True,
        physical_rank=rank,
        activation=activation,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family=family,
        vendor="marlin",
    )[Phase.DECODE]
    rows = {"sm90": "decode.shared", "sm100": "decode.shared.marlin"}
    assert selected.name == rows[architecture]
    provider = select_provider_cls(selected.base_gemm_rows, family, "marlin")(
        quant_info
    )
    hidden = rand(tokens, _H, scale=0.2)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[3, 1] = -1  # an EP hole on an adapter token
    # Outer factors (gate/up-A, down-B) per slot; inner ones per (slot, expert).
    gate_up_a = rand(slots, 1, slices * rank, _H, scale=0.1)
    gate_up_b = rand(slots, _E, slices * _I, rank, scale=0.1)
    down_a = rand(slots, _E, rank, _I, scale=0.1)
    down_b = rand(slots, 1, _H, rank, scale=0.1)
    mapping = torch.tensor([0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.int32, device=device)

    batch = MoeLoraBatch(
        gate_up_lora_a=gate_up_a,
        gate_up_lora_b=gate_up_b,
        down_lora_a=down_a,
        down_lora_b=down_b,
        token_lora_mapping=mapping,
    )
    check_marlin_lora_replay(
        monkeypatch,
        selected,
        provider,
        hidden,
        topk_ids,
        batch,
        w13_eff,
        w2_eff,
        tol=3e-2,
    )


def test_single_token_align_matches_general_path_with_invalid_ids():
    """Both alignment paths exclude dropped expert-parallel routes."""
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
