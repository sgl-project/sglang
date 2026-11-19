"""FP8 Marlin repack, provider numerics, and graph replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.lora.moe.activation import ActivationFn
from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.plan import resolve_plans
from sglang.srt.lora.moe.runner import MoeLoraBatch
from sglang.srt.lora.utils import Phase, architecture_for_capability
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.moe_lora import (
    check_marlin_lora_replay,
    gather_pairs,
    rel_l2,
)
from sglang.test.quant_ref_utils import quantize_mx_ref

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, stage="nightly", runner_config="4-gpu-b200")

_E = 4
_H = 256
_I = 128
_TOPK = 2
_T = 32
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_MARLIN_TOL = 2e-2


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("the Marlin FP8 MoE LoRA provider needs SM90+ here")


def _rand(g: torch.Generator, *shape: int, scale: float = 0.05) -> torch.Tensor:
    return (
        (torch.randn(*shape, generator=g, device="cpu") * scale)
        .to(torch.bfloat16)
        .to("cuda")
    )


def _route(g: torch.Generator, tokens: int, experts: int = _E, top_k: int = _TOPK):
    ids = [
        torch.randperm(experts, generator=g, device="cpu")[:top_k]
        for _ in range(tokens)
    ]
    return torch.stack(ids).to(device="cuda", dtype=torch.int32)


def _quantize(family: str, w: torch.Tensor):
    """Produce checkpoint weights, Marlin scales, and the FP32 reference."""
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        marlin_moe_fp8_block_scales,
        marlin_moe_fp8_channel_scales,
        marlin_moe_mxfp8_scales,
    )

    e, n, k = w.shape
    w = w.float()
    if family == "fp8_channel":
        scale = w.abs().amax(dim=2, keepdim=True).clamp(min=1e-6) / _FP8_MAX
        q = (w / scale).to(torch.float8_e4m3fn)
        return q, marlin_moe_fp8_channel_scales(scale), q.float() * scale
    if family == "fp8":
        nb, kb = n // 128, k // 128
        blocks = w.view(e, nb, 128, kb, 128)
        scale = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-6) / _FP8_MAX
        q = (blocks / scale[:, :, None, :, None]).to(torch.float8_e4m3fn)
        deq = (q.float() * scale[:, :, None, :, None]).view(e, n, k)
        return (
            q.view(e, n, k),
            marlin_moe_fp8_block_scales(scale, size_n=n, block_n=128),
            deq,
        )
    assert family == "mxfp8"
    q, e8m0, deq = quantize_mx_ref(w, 32, "e4m3")
    return q, marlin_moe_mxfp8_scales(e8m0), deq


def _bind(w13_q, w2_q, w13_scales, w2_scales):
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        prepare_moe_fp8_layer_for_marlin,
    )
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8MarlinQuantInfo

    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(w13_q, requires_grad=False),
        w2_weight=torch.nn.Parameter(w2_q, requires_grad=False),
        moe_runner_config=SimpleNamespace(
            is_gated=True, activation="silu", apply_router_weight_on_input=False
        ),
        num_local_experts=w13_q.shape[0],
        intermediate_size_per_partition=w2_q.shape[2],
        hidden_size=w2_q.shape[1],
    )
    prepare_moe_fp8_layer_for_marlin(layer, w13_scales=w13_scales, w2_scales=w2_scales)
    return MoeLoraFp8MarlinQuantInfo.from_layer(layer)


def _marlin_layer(family: str, w13: torch.Tensor, w2: torch.Tensor):
    w13_q, w13_scales, w13_eff = _quantize(family, w13)
    w2_q, w2_scales, w2_eff = _quantize(family, w2)
    return _bind(w13_q, w2_q, w13_scales, w2_scales), w13_eff, w2_eff


def _run_gemms(provider, hidden, topk_ids, act_pairs):
    pair_idx = torch.arange(topk_ids.numel(), device=hidden.device)
    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, topk_ids.shape[1])
        gateup = hidden.new_zeros(provider.gateup_out_shape(state))
        provider.gateup(state, gateup)
        act = hidden.new_zeros(provider.act_out_shape(state))
        rows = state.pair_to_row[pair_idx].long()
        act.reshape(-1, act_pairs.shape[1])[rows] = act_pairs
        down = hidden.new_zeros(provider.down_out_shape(state))
        provider.down(state, act, down)
    return (
        state,
        gather_pairs(gateup, state.pair_to_row, pair_idx),
        gather_pairs(down, state.pair_to_row, pair_idx),
    )


def _check_gemms(gateup, down, hidden, topk_ids, act_pairs, w13_eff, w2_eff):
    experts = topk_ids.flatten().long()

    def reference(rows, weights):
        out = rows.new_empty((rows.shape[0], weights.shape[1]), dtype=torch.float32)
        for expert in torch.unique(experts).tolist():
            mask = experts == expert
            out[mask] = rows[mask].float() @ weights[expert].T
        return out

    inputs = hidden.repeat_interleave(topk_ids.shape[1], dim=0)
    assert rel_l2(gateup.float(), reference(inputs, w13_eff)) < _MARLIN_TOL
    assert rel_l2(down.float(), reference(act_pairs, w2_eff)) < _MARLIN_TOL


def test_fp8_marlin_gateup_and_down_match_reference():
    _skip_unless_supported()
    family = "fp8_channel"
    # 1,024 routed pairs exercise general alignment. I=320 needs 32-wide groups.
    tokens, hidden_size, intermediate = 512, 512, 320
    g = torch.Generator(device="cpu").manual_seed(11)
    w13 = _rand(g, _E, 2 * intermediate, hidden_size)
    w2 = _rand(g, _E, hidden_size, intermediate)
    hidden, topk_ids = _rand(g, tokens, hidden_size), _route(g, tokens)
    quant_info, w13_eff, w2_eff = _marlin_layer(family, w13, w2)
    assert quant_info.group_size == 32
    act_pairs = _rand(g, tokens * _TOPK, intermediate, scale=0.1)
    provider = select_provider_cls("route_major", family, "marlin")(quant_info)
    _, gateup, down = _run_gemms(provider, hidden, topk_ids, act_pairs)
    _check_gemms(gateup, down, hidden, topk_ids, act_pairs, w13_eff, w2_eff)


def test_fp8_marlin_block_16_at_the_measured_shape():
    _skip_unless_supported()
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("the early block-16 rule is SM90-only")
    tokens = 13  # 6.5 pairs per expert, the new block-16 boundary.
    g = torch.Generator(device="cpu").manual_seed(17)
    hidden_size, intermediate = 2048, 512
    quant_info, w13_eff, w2_eff = _marlin_layer(
        "fp8",
        _rand(g, _E, 2 * intermediate, hidden_size),
        _rand(g, _E, hidden_size, intermediate),
    )
    assert quant_info.group_size == 128
    provider = select_provider_cls("route_major", "fp8", "marlin")(quant_info)
    hidden, topk_ids = _rand(g, tokens, hidden_size), _route(g, tokens)
    act_pairs = _rand(g, tokens * _TOPK, intermediate, scale=0.1)

    state, gateup, down = _run_gemms(provider, hidden, topk_ids, act_pairs)
    assert state.block_size_m == 16
    _check_gemms(gateup, down, hidden, topk_ids, act_pairs, w13_eff, w2_eff)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _run_gemms(provider, hidden, topk_ids, act_pairs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        state, gateup, down = _run_gemms(provider, hidden, topk_ids, act_pairs)
    assert state.block_size_m == 16
    # Replay over new routing and inputs in the captured buffers.
    topk_ids.copy_(_route(g, tokens))
    hidden.copy_(_rand(g, tokens, hidden_size))
    act_pairs.copy_(_rand(g, tokens * _TOPK, intermediate, scale=0.1))
    graph.replay()
    torch.cuda.synchronize()
    _check_gemms(gateup, down, hidden, topk_ids, act_pairs, w13_eff, w2_eff)


@pytest.mark.parametrize(
    "family, is_shared_outer, phase",
    (
        ("fp8_channel", False, Phase.PREFILL),
        ("fp8", True, Phase.PREFILL),
        ("mxfp8", True, Phase.DECODE),
    ),
)
def test_fp8_marlin_plans_replay_mixed_adapter_ranks(
    monkeypatch: pytest.MonkeyPatch, family: str, is_shared_outer: bool, phase: Phase
) -> None:
    """Replay mixed base and LoRA traffic after changing routing and slots."""
    _skip_unless_supported()
    device = torch.device("cuda")
    architecture = architecture_for_capability(
        torch.cuda.get_device_capability(device)[0]
    )
    g = torch.Generator(device="cpu").manual_seed(31)
    tokens = 48 if phase is Phase.PREFILL else 6
    slots, rank = 2, 16
    quant_info, w13_eff, w2_eff = _marlin_layer(
        family, _rand(g, _E, 2 * _I, _H), _rand(g, _E, _H, _I)
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
    if is_shared_outer and phase is Phase.DECODE:
        # sm100 selects the Marlin vendor row.
        expected = {"sm100": "decode.shared.marlin", "sm90": "decode.shared"}
        assert selected.name == expected.get(architecture, selected.name)
    provider = select_provider_cls(selected.base_gemm_rows, family, "marlin")(
        quant_info
    )
    hidden = _rand(g, tokens, _H, scale=0.2)
    topk_ids = _route(g, tokens)
    topk_ids[1, 1] = -1  # an EP hole on an adapter token
    outer_experts = 1 if is_shared_outer else _E
    gate_up_a = _rand(g, slots, outer_experts, 2 * rank, _H, scale=0.1)
    gate_up_b = _rand(g, slots, _E, 2 * _I, rank, scale=0.1)
    down_a = _rand(g, slots, _E, rank, _I, scale=0.1)
    down_b = _rand(g, slots, outer_experts, _H, rank, scale=0.1)
    # Slot 1 holds a rank-8 adapter: the upper half of every rank axis is padding.
    half = rank // 2
    gate_up_a[1, :, half:rank] = 0
    gate_up_a[1, :, rank + half :] = 0
    gate_up_b[1, :, :, half:] = 0
    down_a[1, :, half:] = 0
    down_b[1, :, :, half:] = 0
    mapping = torch.tensor([0, 1, -1] * (tokens // 3), dtype=torch.int32, device=device)

    batch = MoeLoraBatch(
        gate_up_lora_a=gate_up_a,
        gate_up_lora_b=gate_up_b,
        down_lora_a=down_a,
        down_lora_b=down_b,
        token_lora_mapping=mapping,
        is_prefill=phase is Phase.PREFILL,
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
        tol=_MARLIN_TOL,
        check_distinct_slots=True,
        check_base_only=family in ("fp8", "mxfp8"),
    )


@pytest.mark.parametrize(
    "hidden_size, intermediate, group_size", ((256, 128, 32), (512, 256, 128))
)
def test_fp8_marlin_channel_scales_are_applied_before_accumulation(
    hidden_size: int, intermediate: int, group_size: int
) -> None:
    # The old post-accumulation scale underflows on these inputs.
    _skip_unless_supported()
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        marlin_moe_fp8_channel_scales,
    )
    from sglang.srt.lora.moe.base_gemm_provider.marlin import (
        MarlinFp8PairProvider,
    )

    def weight(n, k):
        codes = torch.ones((_E, n, k), device="cuda")
        scales = torch.ones((_E, n, 1), device="cuda")
        if group_size == 32:
            codes[:, 0] = _FP8_MAX
            scales[:, 0] = 255.0
        return codes.to(torch.float8_e4m3fn), marlin_moe_fp8_channel_scales(scales)

    w13, s13 = weight(2 * intermediate, hidden_size)
    w2, s2 = weight(hidden_size, intermediate)
    quant_info = _bind(w13, w2, s13, s2)
    assert quant_info.group_size == group_size
    tokens = 4
    values = torch.full((tokens, 1), 2.0**-60, device="cuda", dtype=torch.bfloat16)
    if group_size == 32:
        # Also exercise the maximum scale at a normal-sized activation.
        values[0] = 2.0**-10
    pair_values = values.repeat_interleave(_TOPK, dim=0)
    hidden = values.expand(-1, hidden_size).contiguous()
    act_pairs = pair_values.expand(-1, intermediate).contiguous()
    topk_ids = _route(torch.Generator().manual_seed(0), tokens)
    for atomic in (True, False):
        provider = type(
            "Provider", (MarlinFp8PairProvider,), {"_use_atomic_add": atomic}
        )(quant_info)
        _, gateup, down = _run_gemms(provider, hidden, topk_ids, act_pairs)
        for got, k in ((gateup, hidden_size), (down, intermediate)):
            expected = pair_values.float() * k
            if group_size == 32:
                torch.testing.assert_close(
                    got[:, :1].float(),
                    expected * _FP8_MAX * 255.0,
                    rtol=_MARLIN_TOL,
                    atol=0,
                )
                got = got[:, 1:]
            assert torch.equal(got, expected.to(got.dtype).expand_as(got)), atomic


def _cancellation_gateup(atomic: bool) -> torch.Tensor:
    """Split-K partials 1 + 2^-9 and -1 + 2^-9 must sum to 2^-8.
    Storing a partial in BF16 leaves 2^-9; BF16 atomic adds leave 0."""
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        marlin_moe_fp8_block_scales,
    )
    from sglang.srt.lora.moe.base_gemm_provider.marlin import (
        MarlinFp8PairProvider,
    )

    hidden_size, tokens = 4096, 4
    w13 = torch.zeros(_E, 2 * _I, hidden_size, device="cuda")
    w13[..., 0], w13[..., 1], w13[..., -2], w13[..., -1] = 1.0, 2.0**-9, 2.0**-9, -1.0
    w2 = torch.zeros(_E, hidden_size, _I, device="cuda")

    def unit_scales(n, k):
        ones = torch.ones(_E, n // 128, k // 128, device="cuda")
        return marlin_moe_fp8_block_scales(ones, size_n=n, block_n=128)

    quant_info = _bind(
        w13.to(torch.float8_e4m3fn),
        w2.to(torch.float8_e4m3fn),
        unit_scales(2 * _I, hidden_size),
        unit_scales(hidden_size, _I),
    )
    provider = type("Provider", (MarlinFp8PairProvider,), {"_use_atomic_add": atomic})(
        quant_info
    )
    hidden = torch.zeros(tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    hidden[:, [0, 1, -2, -1]] = 1.0
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], device="cuda")
    pair_idx = torch.arange(topk_ids.numel(), device="cuda")
    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids.int(), _TOPK)
        out = hidden.new_zeros(provider.gateup_out_shape(state))
        provider.gateup(state, out)
    return gather_pairs(out, state.pair_to_row, pair_idx)


def test_fp8_marlin_split_k_reduction_paths():
    _skip_unless_supported()
    from sglang.srt.lora.moe.base_gemm_provider.marlin import (
        MarlinFp8PairProvider,
    )

    g = torch.Generator(device="cpu").manual_seed(41)
    hidden_size, tokens = 4096, 4
    quant_info, w13_eff, _ = _marlin_layer(
        "fp8_channel", _rand(g, _E, 2 * _I, hidden_size), _rand(g, _E, hidden_size, _I)
    )
    hidden, topk_ids = _rand(g, tokens, hidden_size, scale=0.2), _route(g, tokens)
    pair_idx = torch.arange(topk_ids.numel(), device=hidden.device)
    ref = torch.einsum(
        "ph,pnh->pn",
        hidden[pair_idx // _TOPK].float(),
        w13_eff[topk_ids.flatten().long()],
    )

    def gateup(provider):
        with get_context().override_server_args():
            state = provider.prepare(hidden, topk_ids, _TOPK)
            out = hidden.new_zeros(provider.gateup_out_shape(state))
            provider.gateup(state, out)
        return gather_pairs(out, state.pair_to_row, pair_idx)

    for atomic in (True, False):
        provider = type(
            "Provider", (MarlinFp8PairProvider,), {"_use_atomic_add": atomic}
        )(quant_info)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            eager = gateup(provider)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = gateup(provider)
        graph.replay()
        torch.cuda.synchronize()
        for out in (eager, captured):
            assert rel_l2(out.float(), ref) < _MARLIN_TOL, atomic
        if not atomic:
            assert torch.equal(eager, captured)
    # The FP32 reduction keeps the residual (production FP8 adds atomically).
    residual = _cancellation_gateup(atomic=False)
    assert torch.equal(residual, torch.full_like(residual, 2.0**-8))


def test_online_mxfp8_experts_match_a_serialized_reload():
    _skip_unless_supported()
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8MarlinQuantInfo
    from sglang.srt.runtime_context import get_flags

    g = torch.Generator(device="cpu").manual_seed(2)
    w13, w2 = _rand(g, _E, 2 * _I, _H), _rand(g, _E, _H, _I)
    w13[0, 0] = 0

    q13, s13, ref13 = quantize_mx_ref(w13, 32, "e4m3")
    q2, s2, ref2 = quantize_mx_ref(w2, 32, "e4m3")
    # The online checkpoint format encodes zero groups with exponent byte 0.
    s13[0, 0] = 0

    def load(w13, w2, w13_scale, w2_scale, *, serialized):
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(w13, requires_grad=False),
            w2_weight=torch.nn.Parameter(w2, requires_grad=False),
            w13_weight_scale_inv=torch.nn.Parameter(w13_scale, requires_grad=False),
            w2_weight_scale_inv=torch.nn.Parameter(w2_scale, requires_grad=False),
            hidden_size=_H,
            intermediate_size_per_partition=_I,
            num_local_experts=_E,
            moe_runner_config=SimpleNamespace(
                is_gated=True, activation="silu", apply_router_weight_on_input=False
            ),
        )
        method = Fp8MoEMethod(
            Fp8Config(is_checkpoint_fp8_serialized=serialized, use_mxfp8=True)
        )
        method.process_weights_after_loading(layer)
        return MoeLoraFp8MarlinQuantInfo.from_layer(layer)

    with get_flags().moe.override(runner_backend=MoeRunnerBackend.LORA_MARLIN):
        online = load(
            w13, w2, torch.zeros_like(s13), torch.zeros_like(s2), serialized=False
        )
        serialized = load(q13, q2, s13, s2, serialized=True)
    for name in ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales"):
        assert torch.equal(getattr(online, name), getattr(serialized, name)), name
    assert online.group_size == 32

    hidden = _rand(g, _T, _H)
    topk_ids = _route(g, _T)
    act_pairs = _rand(g, _T * _TOPK, _I)
    provider = select_provider_cls("route_major", "mxfp8", "marlin")(online)
    _, gateup, down = _run_gemms(provider, hidden, topk_ids, act_pairs)
    _check_gemms(gateup, down, hidden, topk_ids, act_pairs, ref13, ref2)


def test_compressed_tensors_marlin_keeps_separate_gate_and_up_scales():
    _skip_unless_supported()
    from compressed_tensors.quantization import QuantizationArgs

    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
        CompressedTensorsW8A8Fp8MoE,
    )
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8MarlinQuantInfo
    from sglang.srt.runtime_context import get_flags

    hidden_size, intermediate = 512, 448
    g = torch.Generator(device="cpu").manual_seed(3)
    codes13 = _rand(g, _E, 2 * intermediate, hidden_size, scale=8).to(
        torch.float8_e4m3fn
    )
    codes2 = _rand(g, _E, hidden_size, intermediate, scale=8).to(torch.float8_e4m3fn)
    gate_up_scale = torch.tensor(
        [[0.01, 0.03], [0.02, 0.01], [0.04, 0.01], [0.01, 0.07]], device="cuda"
    )
    down_scale = gate_up_scale[:, 0].contiguous()
    layer = torch.nn.Module()
    layer.moe_tp_size = 1
    layer.num_local_experts = _E
    layer.hidden_size = hidden_size
    layer.intermediate_size_per_partition = intermediate
    layer.params_dtype = torch.bfloat16
    layer.moe_runner_config = SimpleNamespace(
        is_gated=True, activation="silu", apply_router_weight_on_input=False
    )
    with get_flags().moe.override(runner_backend=MoeRunnerBackend.LORA_MARLIN):
        scheme = CompressedTensorsW8A8Fp8MoE(
            QuantizationArgs(num_bits=8, type="float", strategy="tensor"),
            QuantizationArgs(
                num_bits=8, type="float", strategy="tensor", dynamic=False
            ),
        )
        scheme.create_weights(layer, _E, hidden_size, intermediate, torch.bfloat16)
        layer.cuda()
        layer.w13_weight.data.copy_(codes13)
        layer.w2_weight.data.copy_(codes2)
        layer.w13_weight_scale.data.copy_(gate_up_scale)
        layer.w2_weight_scale.data.copy_(down_scale)
        # W8A16 must ignore the checkpoint's static input scales.
        layer.w13_input_scale.data.fill_(1000)
        layer.w2_input_scale.data.fill_(1000)
        scheme.process_weights_after_loading(layer)

    info = MoeLoraFp8MarlinQuantInfo.from_layer(layer)
    assert info.group_size == 32
    row_scale = gate_up_scale.repeat_interleave(intermediate, dim=1)
    expected = _bind(
        codes13,
        codes2,
        row_scale[:, None],
        down_scale[:, None, None].expand(-1, 1, hidden_size),
    )
    assert torch.equal(info.w13_qweight, expected.w13_qweight)
    assert torch.equal(
        info.w13_scales.view(torch.int16), expected.w13_scales.view(torch.int16)
    )
    ref13 = codes13.float() * row_scale.unsqueeze(-1)
    ref2 = codes2.float() * down_scale[:, None, None]
    hidden = _rand(g, _T, hidden_size)
    topk_ids = _route(g, _T)
    act_pairs = _rand(g, _T * _TOPK, intermediate)
    provider = select_provider_cls("route_major", "fp8_tensor", "marlin")(info)
    _, gateup, down = _run_gemms(provider, hidden, topk_ids, act_pairs)
    _check_gemms(gateup, down, hidden, topk_ids, act_pairs, ref13, ref2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
