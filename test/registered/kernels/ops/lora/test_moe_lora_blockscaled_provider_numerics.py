"""Numerical regressions for the CuTeDSL block-scaled MoE providers."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from msgspec.structs import replace as struct_replace

from sglang.kernels.ops.lora.moe.activation import ActivationFn
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.plan import (
    BFamily,
    BSpec,
    DownOverlap,
    SelectedPlan,
    Site,
    resolve_plans,
)
from sglang.srt.lora.moe.quant_info import (
    NvFp4ActivationRecipe,
    resolve_nvfp4_activation_recipe,
)
from sglang.srt.lora.moe.runner import MoeLoraBatch, MoeLoraRunner
from sglang.srt.lora.utils import Phase
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.moe_lora import (
    rel_l2,
    standalone_output_allocation,
)
from sglang.test.quant_ref_utils import (
    quantize_mx_ref,
    quantize_nvfp4_per_token_ref,
    quantize_nvfp4_static_ref,
)

# SM100 only (block-scaled tcgen05 MMA); one GPU of the 4-gpu-b200 pool.
register_cuda_ci(est_time=60, stage="nightly", runner_config="4-gpu-b200")

_E = 4
_H = 512
_I = 256
_TOPK = 2
_VEC = 32


def _skip_unless_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("block-scaled tcgen05 MMA needs SM100")


_W13_GLOBAL = 0.02
_W2_GLOBAL = 0.01
_A13_GLOBAL = 0.25
_A2_GLOBAL = 0.05


def _family_and_mode(family: str) -> tuple[str, str]:
    if family == "nvfp4_pt":
        return "nvfp4", "per_token"
    return family, "static"


def _quant_info(
    family: str,
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    w2_global: tuple | None = None,
    a2_global: float | None = None,
):
    from sglang.srt.lora.moe.quant_info import (
        MoeLoraMxfp4QuantInfo,
        MoeLoraMxfp8QuantInfo,
        MoeLoraNvFp4QuantInfo,
    )

    if family in ("nvfp4", "nvfp4_pt"):
        # Distinct gate/up global scales exercise the alpha half selection.
        w13_gs = torch.tensor([[_W13_GLOBAL, 2 * _W13_GLOBAL]] * _E, device=w13.device)
        w13_q, w13_s, w13_eff = [], [], []
        for half in range(2):
            q, s, eff = quantize_nvfp4_static_ref(
                w13[:, half * _I : (half + 1) * _I], w13_gs[0, half].item()
            )
            w13_q.append(q), w13_s.append(s), w13_eff.append(eff)
        if w2_global is None:
            w2_q, w2_s, w2_eff = quantize_nvfp4_static_ref(w2, _W2_GLOBAL)
            w2_gs = torch.full((_E,), _W2_GLOBAL, device=w2.device)
        else:
            w2_gs = torch.tensor(w2_global, dtype=torch.float32, device=w2.device)
            scaled = w2.float() * (w2_gs / _W2_GLOBAL)[:, None, None]
            w2_q, w2_s, w2_eff = (
                torch.stack(part)
                for part in zip(
                    *(
                        quantize_nvfp4_static_ref(scaled[e], w2_global[e])
                        for e in range(_E)
                    )
                )
            )
        per_token = family == "nvfp4_pt"
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
            w2_global_scale=w2_gs,
            w13_input_scale=None if per_token else _A13_GLOBAL,
            w2_input_scale=None if per_token else (a2_global or _A2_GLOBAL),
            act_scale_mode="per_token" if per_token else "static",
        )
        return info, torch.cat(w13_eff, dim=1), w2_eff, family

    elem = "e2m1" if family == "mxfp4" else "e4m3"
    w13_q, w13_s, w13_eff = quantize_mx_ref(w13, _VEC, elem)
    w2_q, w2_s, w2_eff = quantize_mx_ref(w2, _VEC, elem)
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


def _quant_activation(
    x: torch.Tensor,
    elem: str,
    stage: str,
    a2_global: float = _A2_GLOBAL,
    recipe: NvFp4ActivationRecipe | None = None,
):
    """Dequantized activation rows as the provider quantizes them."""
    if elem == "nvfp4":
        gs = _A13_GLOBAL if stage == "gemm1" else a2_global
        return quantize_nvfp4_static_ref(x, gs)[2]
    if elem == "nvfp4_pt":
        return quantize_nvfp4_per_token_ref(x, recipe)[3].float()
    return quantize_mx_ref(x, _VEC, elem)[2]


def _seeded(seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to("cuda")

    return g, rand


def _topk(g: torch.Generator, tokens: int, hole=(3, 1)) -> torch.Tensor:
    """Use distinct experts per token; hole marks a dropped pair (-1)."""
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(tokens)]
    ).to(device="cuda", dtype=torch.int32)
    topk_ids[hole] = -1
    return topk_ids


_ROUTED_SCALING = 0.75
_WEIGHTS = (0.65, 0.35)  # every token's two routing weights
# Measured 0.2-0.3% rel L2 on GB300 for every family and phase.
_RUNNER_TOL = 1e-2


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """A BF16 stage store between FP32 computations."""
    return x.to(torch.bfloat16).float()


def _adapters(rand, slots: int, rank: int, outer: int = _E) -> tuple:
    """An outer axis of length one shares gate/up-A and down-B across experts."""
    return (
        rand(slots, outer, 2 * rank, _H, scale=0.1),
        rand(slots, _E, 2 * _I, rank, scale=0.1),
        rand(slots, _E, rank, _I, scale=0.1),
        rand(slots, outer, _H, rank, scale=0.1),
    )


def _runner(family: str, quant_info, phase: Phase, rank: int, shared_outer=False):
    quant_family, act_scale = _family_and_mode(family)
    selected = resolve_plans(
        architecture="sm100",
        is_shared_outer=shared_outer,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family=quant_family,
        vendor="cutedsl",
        act_scale=act_scale,
    )[phase]
    rows = selected.base_gemm_rows
    provider = select_provider_cls(rows, quant_family, "cutedsl")(quant_info)
    runner = MoeLoraRunner(
        providers={rows: provider},
        top_k=_TOPK,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFn.SILU,
    )
    runner.validate_plan(selected.plan, base_gemm_rows=rows)
    down_b = selected.plan.down_b
    if down_b is not None and down_b.family is BFamily.EPILOGUE:
        provider.enable_epilogue_down_lora(rank)
    runner.plans = {phase: selected}
    return selected, runner


def _down_b_plan(
    selected: SelectedPlan, family: BFamily, overlap: DownOverlap = DownOverlap.NONE
):
    return replace(selected.plan, down_b=BSpec(Site.DOWN, family), down_overlap=overlap)


def _dispatch(hidden: torch.Tensor, topk_ids: torch.Tensor) -> StandardDispatchOutput:
    tokens = hidden.shape[0]
    return StandardDispatchOutput(
        hidden_states=hidden,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.tensor(_WEIGHTS, device=hidden.device).repeat(tokens, 1),
            topk_ids=topk_ids,
            router_logits=torch.zeros((tokens, _E), device=hidden.device),
        ),
    )


def _batch(adapters, mapping: torch.Tensor, phase: Phase) -> MoeLoraBatch:
    gate_up_a, gate_up_b, down_a, down_b = adapters
    return MoeLoraBatch(
        gate_up_lora_a=gate_up_a,
        gate_up_lora_b=gate_up_b,
        down_lora_a=down_a,
        down_lora_b=down_b,
        token_lora_mapping=mapping,
        is_prefill=phase is Phase.PREFILL,
    )


def _forward(runner: MoeLoraRunner, dispatch, batch: MoeLoraBatch, plan=None):
    # Rebuild routing and adapter slots for each eager batch.
    runner.workspace.routes.clear()
    if plan is not None:
        phase = Phase.PREFILL if batch.is_prefill else Phase.DECODE
        runner.plans = {phase: replace(runner.plans[phase], plan=plan)}
    with get_context().override_server_args():
        out = runner.run(dispatch, batch).hidden_states
    torch.cuda.synchronize()
    return out


def _check_replay(runner, dispatch, batch, plan, load, states, expected):
    load(states[0])
    # Workspace scratch is stream-local, so warm up on the capture stream.
    batch.use_cuda_graph = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _forward(runner, dispatch, batch, plan)
    torch.cuda.synchronize()
    runner.workspace.routes.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), get_context().override_server_args():
        captured = runner.run(dispatch, batch).hidden_states
    for state in states:
        load(state)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, expected[state]), state


def _reference(
    quant,
    adapters,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    mapping: torch.Tensor,
    *,
    decode: bool,
    down: str = "each",
    a2_global: float = _A2_GLOBAL,
    recipe: NvFp4ActivationRecipe | None = None,
    detail: bool = False,
):
    """Reference with BF16 rounding at the runner's intermediate stores."""
    w13_eff, w2_eff, elem = quant
    # A shared outer factor (one per slot) serves every expert.
    gate_up_a, gate_up_b, down_a, down_b = (
        adapter.float().expand(-1, _E, -1, -1) for adapter in adapters
    )
    tokens, rank = hidden.shape[0], gate_up_b.shape[-1]
    pair_tokens = torch.arange(tokens, device=hidden.device).repeat_interleave(_TOPK)
    experts = topk_ids.flatten().long()
    routed = (experts >= 0).float()[:, None]
    experts = experts.clamp(min=0)
    slot = mapping[pair_tokens].long()
    adapted = (slot >= 0).float()[:, None]
    slot = slot.clamp(min=0)
    inputs = hidden.float()[pair_tokens]
    inputs_q = _quant_activation(inputs, elem, "gemm1", recipe=recipe)
    gate_up = _bf16(torch.einsum("ph,pnh->pn", inputs_q, w13_eff[experts]))
    bridge = _bf16(torch.einsum("ph,prh->pr", inputs, gate_up_a[slot, experts]))
    inner = gate_up_b[slot, experts]
    delta = torch.cat(
        [
            torch.einsum(
                "pir,pr->pi",
                inner[:, s * _I : (s + 1) * _I],
                bridge[:, s * rank : (s + 1) * rank],
            )
            for s in range(2)
        ],
        dim=1,
    )
    gate_up = gate_up + adapted * (_bf16(delta) if decode else delta)
    act = _bf16(F.silu(gate_up[:, :_I]) * gate_up[:, _I:])
    act_q = _quant_activation(act, elem, "gemm2", a2_global, recipe)
    base = torch.einsum("pi,phi->ph", act_q, w2_eff[experts])
    down_rank = _bf16(torch.einsum("pi,pri->pr", act, down_a[slot, experts]))
    delta = torch.einsum("pr,phr->ph", down_rank, down_b[slot, experts])
    if down == "each":
        pairs = _bf16(base) + adapted * _bf16(delta)
    else:
        pairs = base + adapted * delta
        if down == "once":
            pairs = _bf16(pairs)
    pairs = pairs * routed
    weights = torch.tensor(_WEIGHTS, device=hidden.device)[:, None]
    out = (pairs.view(tokens, _TOPK, _H) * weights).sum(1) * _ROUTED_SCALING
    if detail:
        return out, down_rank, experts, (adapted * routed).flatten() > 0
    return out


def test_blockscaled_shared_outer_decode_matches_reference(monkeypatch):
    family, phase = "mxfp8", Phase.DECODE
    _skip_unless_sm100()
    monkeypatch.setattr(MoeLoraRunner, "_allocate_output", standalone_output_allocation)
    g, rand = _seeded(29)
    tokens, slots, rank = 8, 2, 16
    quant_info, *quant = _quant_info(family, rand(_E, 2 * _I, _H), rand(_E, _H, _I))
    selected, runner = _runner(family, quant_info, phase, rank, shared_outer=True)
    assert selected.name == "decode.shared"
    hidden = rand(tokens, _H, scale=0.5)
    topk_ids = _topk(g, tokens)
    # Outer factors (gate/up-A, down-B) per slot; inner ones per (slot, expert).
    adapters = _adapters(rand, slots, rank, outer=1)
    # Slots 0, 1 and the base path (-1) in turn.
    mapping = ((torch.arange(tokens, device="cuda") + 1) % 3 - 1).to(torch.int32)
    dispatch, batch = _dispatch(hidden, topk_ids), _batch(adapters, mapping, phase)
    ref = _reference(quant, adapters, hidden, topk_ids, mapping, decode=True)
    assert rel_l2(_forward(runner, dispatch, batch).float(), ref) < _RUNNER_TOL
    base = _reference(
        quant, adapters, hidden, topk_ids, torch.full_like(mapping, -1), decode=True
    )
    assert rel_l2(base, ref) > 0.1


# Both the minimum normal and minimum subnormal BF16 amax give FP32-subnormal scales.
_TINY_TOKENS = {1: 2.0**-133, 2: 2.0**-126}
_TINY_SLOT = 7  # adapters large enough to show a tiny token's delta


def _per_token_mapping(name: str, tokens: int) -> torch.Tensor:
    if name == "base":
        return torch.full((tokens,), -1, dtype=torch.int32)
    mapping = torch.tensor(
        [(1, -1, 4, -1)[t % 4] for t in range(tokens)], dtype=torch.int32
    )
    mapping[list(_TINY_TOKENS)] = _TINY_SLOT
    return mapping


@pytest.mark.parametrize(
    "phase", [Phase.DECODE, Phase.PREFILL], ids=["decode", "prefill"]
)
def test_blockscaled_per_expert_per_token_plans_match_reference(monkeypatch, phase):
    """Check per-token 4over6 decode and fused prefill, including graph replay."""
    _skip_unless_sm100()
    device = torch.device("cuda")
    monkeypatch.setattr(MoeLoraRunner, "_allocate_output", standalone_output_allocation)
    g, rand = _seeded(37)
    decode = phase is Phase.DECODE
    tokens, slots, rank = (8 if decode else 40), 8, 16
    quant_info, *quant = _quant_info("nvfp4_pt", rand(_E, 2 * _I, _H), rand(_E, _H, _I))
    if decode:
        with monkeypatch.context() as env:
            env.setenv("FLASHINFER_NVFP4_4OVER6", "1")
            for suffix in ("E4M3_USE_256", "ERR_MODE", "ERR_USE_FAST_MATH"):
                env.delenv(f"FLASHINFER_NVFP4_4OVER6_{suffix}", raising=False)
            recipe = resolve_nvfp4_activation_recipe()
        assert recipe == NvFp4ActivationRecipe(
            e4m3_max=448, err_mode="MAE", err_use_fast_math=False
        )
    else:
        recipe = NvFp4ActivationRecipe(
            e4m3_max=256, err_mode="MSE", err_use_fast_math=True
        )
    quant_info = struct_replace(quant_info, recipe=recipe)
    selected, runner = _runner("nvfp4_pt", quant_info, phase, rank)
    assert selected.name == (
        "decode.per_expert" if decode else "prefill.per_expert.cutedsl_blockscaled"
    )
    if not decode:
        assert selected.plan.down_b.family is BFamily.EPILOGUE
        # The packaged prefill.per_expert row's separate down-B, same tiles.
        unfused_plan = _down_b_plan(selected, BFamily.GROUPED, DownOverlap.DOWN_A)
        runner.validate_plan(unfused_plan, base_gemm_rows=selected.base_gemm_rows)

    hidden0 = rand(tokens, _H, scale=0.5)
    hidden0[0].zero_()  # a zero token: per-token scale 0
    for t, amax in _TINY_TOKENS.items():
        row = (torch.rand(_H, generator=g) * 2 - 1) * amax
        row[0] = amax
        hidden0[t] = row.to(torch.bfloat16).to(device)
    topk0 = _topk(g, tokens)
    adapters = _adapters(rand, slots, rank)
    # Amplify LoRA on tiny rows so its delta survives base underflow.
    for adapter, scale in zip(adapters, (2.0**40, 2.0**40, 2.0**20, 2.0**20)):
        adapter[_TINY_SLOT] = (adapter[_TINY_SLOT].float() * scale).to(torch.bfloat16)
    hidden, topk_ids = hidden0.clone(), topk0.clone()
    mapping = torch.empty(tokens, dtype=torch.int32, device=device)
    dispatch, batch = _dispatch(hidden, topk_ids), _batch(adapters, mapping, phase)
    reference = partial(
        _reference,
        quant,
        adapters,
        hidden,
        topk_ids,
        decode=decode,
        recipe=quant_info.recipe,
    )
    down = "each" if decode else "once"
    states = [("mixed", False), ("base", False), ("mixed", True)]

    def load(state):
        name, flipped = state
        mapping.copy_(_per_token_mapping(name, tokens))
        topk_ids.copy_(topk0.flip(1) if flipped else topk0)
        hidden.copy_(hidden0 * -0.5 if flipped else hidden0)

    eager = {}
    for state in states:
        load(state)
        out = eager[state] = _forward(runner, dispatch, batch, selected.plan)
        ref = reference(mapping, down=down)
        assert bool(torch.isfinite(out).all()), state
        err = float(rel_l2(out.float(), ref))
        assert err < _RUNNER_TOL, (state, err)
        if not decode and state == states[0]:
            unfused = _forward(runner, dispatch, batch, unfused_plan)
            err = float(rel_l2(unfused.float(), reference(mapping)))
            assert err < _RUNNER_TOL, (state, err)
            assert float(rel_l2(out.float(), unfused.float())) < 4e-3, state
            base_tokens = mapping < 0
            assert torch.equal(out[base_tokens], unfused[base_tokens]), state
        if state[0] != "base":
            without = reference(torch.full_like(mapping, -1), down=down)
            assert rel_l2(without, ref) > 0.1, state
        for t in _TINY_TOKENS:
            row, ref_row = out[t].double(), ref[t].double()
            if ref_row.norm() == 0:
                assert bool((row == 0).all()), (state, t)
            elif int(mapping[t]) >= 0:
                # The large slot's delta is the row: visible and reproduced.
                without = mapping.clone()
                without[t] = -1
                base_norm = float(reference(without, down=down)[t].double().norm())
                assert base_norm < 1e-3 * float(ref_row.norm()), (state, t)
                err = float(rel_l2(row, ref_row))
                assert err < 2.5e-2, (state, t, err)
            else:
                assert torch.allclose(row, ref_row, rtol=0.1, atol=2.0**-130), (
                    state,
                    t,
                )

    _check_replay(
        runner, dispatch, batch, selected.plan, load, states + states[:1], eager
    )


def test_per_token_provider_bounds_the_weight_global_scales() -> None:
    _skip_unless_sm100()
    device = torch.device("cuda")
    _, rand = _seeded(41)
    quant_info, *_ = _quant_info("nvfp4_pt", rand(_E, 2 * _I, _H), rand(_E, _H, _I))
    cls = select_provider_cls("expert_major", "nvfp4", "cutedsl")
    fp32_max = torch.finfo(torch.float32).max
    bound_down, bound_gate_up = (fp32_max / (k * 2688.0**2) for k in (_I, _H))

    def w2(*values):
        return torch.tensor(values, dtype=torch.float32, device=device)

    tiny = torch.finfo(torch.float32).tiny
    cls(
        struct_replace(
            quant_info, w2_global_scale=w2(0.0, 2e-37, tiny, 0.99 * bound_down)
        )
    )
    with pytest.raises(ValueError, match="subnormal fp32"):
        cls(struct_replace(quant_info, w2_global_scale=w2(1e-3, 1e-39, 1e-3, 1e-3)))
    for invalid in (1.01 * bound_down, float("nan")):
        with pytest.raises(ValueError, match="gemm2 weight global scales"):
            cls(
                struct_replace(
                    quant_info, w2_global_scale=w2(1e-3, 1e-3, 1e-3, invalid)
                )
            )
    w13 = quant_info.w13_global_scale.clone()
    w13[2, 1] = 1.01 * bound_gate_up
    with pytest.raises(ValueError, match="gemm1 weight global scales"):
        cls(struct_replace(quant_info, w13_global_scale=w13))


def test_mxfp4_prefill_down_b_in_the_epilogue(monkeypatch):
    _skip_unless_sm100()
    monkeypatch.setattr(MoeLoraRunner, "_allocate_output", standalone_output_allocation)
    g, rand = _seeded(37)
    tokens, rank = 40, 8
    quant_info, *quant = _quant_info("mxfp4", rand(_E, 2 * _I, _H), rand(_E, _H, _I))
    phase = Phase.PREFILL
    selected, runner = _runner("mxfp4", quant_info, phase, rank)
    assert selected.plan.down_b.family is BFamily.EPILOGUE
    separate_plan = _down_b_plan(selected, BFamily.GROUPED)
    hidden, topk_ids = rand(tokens, _H, scale=0.5), _topk(g, tokens)
    adapters = _adapters(rand, 2, rank)
    mapping = torch.arange(tokens, device="cuda", dtype=torch.int32) % 3 - 1
    dispatch, batch = _dispatch(hidden, topk_ids), _batch(adapters, mapping, phase)
    ref = _reference(
        quant, adapters, hidden, topk_ids, mapping, decode=False, down="once"
    )
    fused = _forward(runner, dispatch, batch, selected.plan)
    separate = _forward(runner, dispatch, batch, separate_plan)
    assert rel_l2(fused.float(), ref) < _RUNNER_TOL
    assert rel_l2(fused.float(), separate.float()) < 4e-3
    assert torch.equal(fused[mapping < 0], separate[mapping < 0])


# 122B NVFP4 snapshot 98915d83, layer 6, expert 188: 1/alpha overflows FP32.
# The other scales cover two BF16 bridge overflows and the checkpoint median.
_STATIC_A2_GLOBAL = 0.008603050373494625
_STATIC_W2_GLOBAL = (2.0711090409231365e-37, 1.75e-36, 1.47e-35, 2.07e-5)
_STATIC_LARGE_SLOT = 4  # its down-A is 512 times larger: bridges in the hundreds


def _row_rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.double(), b.double()
    return (a - b).norm(dim=1) / b.norm(dim=1).clamp(min=1e-300)


def test_fused_static_epilogue_plan_at_the_checkpoint_scale_extremes(monkeypatch):
    """The 122B checkpoint's small alphas overflow an inverse-scaled LoRA bridge."""
    _skip_unless_sm100()
    from types import SimpleNamespace

    from sglang.kernels.ops.lora.moe.cutedsl.api import GroupedGemmConfig
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.runtime_context import get_flags

    device = torch.device("cuda")
    monkeypatch.setattr(MoeLoraRunner, "_allocate_output", standalone_output_allocation)
    g, rand = _seeded(41)
    tokens, slots, rank, phase = 40, 8, 16, Phase.PREFILL
    quant_info, *quant = _quant_info(
        "nvfp4",
        rand(_E, 2 * _I, _H),
        rand(_E, _H, _I),
        w2_global=_STATIC_W2_GLOBAL,
        a2_global=_STATIC_A2_GLOBAL,
    )
    layer = SimpleNamespace(
        params_dtype=torch.bfloat16,
        quant_method=SimpleNamespace(
            quant_config=SimpleNamespace(use_per_token_activation=False)
        ),
        w13_weight=quant_info.w13_weight,
        w2_weight=quant_info.w2_weight,
        w13_weight_scale=quant_info.w13_scale,
        w2_weight_scale=quant_info.w2_scale,
        w13_weight_scale_2=quant_info.w13_global_scale,
        w2_weight_scale_2=quant_info.w2_global_scale,
        w13_input_scale=torch.tensor(_A13_GLOBAL, device=device),
        w2_input_scale=torch.tensor(_STATIC_A2_GLOBAL, device=device),
        num_local_experts=_E,
        intermediate_size_per_partition=_I,
        hidden_size=_H,
        moe_runner_config=MoeRunnerConfig(
            top_k=_TOPK,
            routed_scaling_factor=_ROUTED_SCALING,
        ),
    )
    original_scales = layer.w2_weight_scale.clone()
    monkeypatch.setattr(MoeLoraRunner, "_admit", lambda *_: "nvfp4")
    with (
        get_context().override_server_args(),
        get_flags().moe.override(runner_backend=MoeRunnerBackend.LORA_CUTEDSL),
    ):
        runner = MoeLoraRunner.from_layer(
            layer, is_shared_outer=False, physical_rank=rank
        )
    masked = runner.providers["expert_major"]
    contiguous = runner.providers["route_major"]
    assert masked.quant_info is contiguous.quant_info
    assert masked._sf_w13 is contiguous._sf_w13
    assert masked._sf_w2 is contiguous._sf_w2
    assert masked._alpha["gemm2"] is contiguous._alpha["gemm2"]
    source_scales = layer.w2_weight_scale
    assert source_scales is quant_info.w2_scale
    assert torch.equal(
        source_scales.view(torch.uint8), original_scales.view(torch.uint8)
    )

    selected = runner.plans[phase]
    assert selected.name == "prefill.per_expert.cutedsl_blockscaled"
    assert selected.plan.down_b.family is BFamily.EPILOGUE
    # The packaged prefill.per_expert row's separate down-B, same tiles.
    unfused_plan = _down_b_plan(selected, BFamily.GROUPED, DownOverlap.DOWN_A)
    runner.validate_plan(unfused_plan, base_gemm_rows=selected.base_gemm_rows)
    assert selected.base_gemm_rows == "route_major"
    inverse = 1.0 / contiguous._alpha["gemm2"].reshape(-1)
    assert not bool(torch.isfinite(inverse[0])) and bool(
        torch.isfinite(inverse[1:]).all()
    )

    hidden0 = rand(tokens, _H, scale=0.5)
    topk0 = _topk(g, tokens)
    adapters = _adapters(rand, slots, rank)
    down_a = adapters[2]
    down_a[_STATIC_LARGE_SLOT] = (down_a[_STATIC_LARGE_SLOT].float() * 512).to(
        torch.bfloat16
    )
    hidden, topk_ids = hidden0.clone(), topk0.clone()
    mapping = torch.empty(tokens, dtype=torch.int32, device=device)
    dispatch, batch = _dispatch(hidden, topk_ids), _batch(adapters, mapping, phase)
    reference = partial(
        _reference,
        quant,
        adapters,
        hidden,
        topk_ids,
        decode=False,
        down="once",
        a2_global=_STATIC_A2_GLOBAL,
    )
    states = ("base", "mixed", "base")

    def load(state):
        flipped = state == "mixed"
        mapping.copy_(_per_token_mapping(state, tokens))
        topk_ids.copy_(topk0.flip(1) if flipped else topk0)
        hidden.copy_(hidden0 * -0.5 if flipped else hidden0)

    fused = {}
    hazards = torch.zeros(_E, dtype=torch.bool, device=device)
    for state in states[:2]:
        load(state)
        tag = (state, rank)
        unfused = _forward(runner, dispatch, batch, unfused_plan)
        out = fused[state] = _forward(runner, dispatch, batch, selected.plan)
        ref, bridge, experts, adapted = reference(mapping, detail=True)
        assert bool(torch.isfinite(out).all()), tag
        assert bool(torch.isfinite(unfused).all()), tag
        compensated = (bridge * inverse[experts][:, None]).to(torch.bfloat16)
        bad = ~torch.isfinite(compensated).all(dim=1) & adapted
        hazards[experts[bad]] = True
        for slot in sorted(set(mapping.tolist())):
            rows = mapping == slot
            assert float(rel_l2(out[rows].float(), ref[rows])) < _RUNNER_TOL, (
                *tag,
                slot,
            )
        assert float(rel_l2(out.float(), unfused.float())) < 4e-3, tag
        assert float(_row_rel(out, unfused).max()) < 1.5e-2, tag
        base_tokens = mapping < 0
        assert torch.equal(out[base_tokens], unfused[base_tokens]), tag
        if state == "mixed":
            assert rel_l2(reference(torch.full_like(mapping, -1)), ref) > 0.1, tag
    # An inverse-alpha bridge was not finite on the three near-dead experts.
    assert hazards.tolist() == [True, True, True, False], rank
    load(states[-1])
    assert torch.equal(_forward(runner, dispatch, batch, selected.plan), fused["base"])

    _check_replay(runner, dispatch, batch, selected.plan, load, states, fused)

    with pytest.raises(ValueError, match="multiples of 8"):
        masked.enable_epilogue_down_lora(12)
    monkeypatch.setitem(
        masked._tile_configs, 256, GroupedGemmConfig(mma_tiler_mn=(128, 256))
    )
    with pytest.raises(ValueError, match="256"):
        masked.enable_epilogue_down_lora(16)
    assert not any("gemm2:lora" in calls for calls in masked._compiled.values())


def _row_buffers(rows, slab, k, dtype, vec, *, per_token=False, global_scale=None):
    from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
    from sglang.kernels.ops.lora.moe.quant_blockscaled import BlockScaledRows

    width = k // 2 if dtype == torch.uint8 else k
    return BlockScaledRows(
        data=torch.full((rows, width), 0x55, dtype=torch.uint8, device="cuda").view(
            dtype
        ),
        scales=torch.full(
            (scale_atom_bytes(slab, k, vec, rows // slab),),
            0x55,
            dtype=torch.uint8,
            device="cuda",
        ),
        sf_vec_size=vec,
        rows_per_slab=slab,
        global_scale=global_scale,
        row_scales=torch.full((rows,), -1.0, device="cuda") if per_token else None,
        row_denominator=6 * 448.0 if per_token else 0.0,
    )


def _check_row_buffers(target, dst, codes, scales, row_scales=None):
    assert dst.unique().numel() == dst.numel()
    data = target.data.view(torch.uint8)
    expected_data = torch.full_like(data, 0x55)
    expected_data[dst] = codes.view(torch.uint8)
    assert torch.equal(data, expected_data)

    # Independent tcgen05 scale-atom indexing, including unused bytes in each slab.
    groups = scales.shape[1]
    slab, row = dst // target.rows_per_slab, dst % target.rows_per_slab
    row_atom = slab * (-(-target.rows_per_slab // 128)) + row // 128
    column = torch.arange(groups, device=data.device)
    atom = row_atom[:, None] * (-(-groups // 4)) + column // 4
    byte = ((row % 32) * 16 + (row // 32 % 4) * 4)[:, None] + column % 4
    expected_scales = torch.full_like(target.scales, 0x55)
    expected_scales[atom * 512 + byte] = scales
    assert torch.equal(target.scales, expected_scales)

    if row_scales is not None:
        expected_rows = torch.full_like(target.row_scales, -1.0)
        expected_rows[dst] = row_scales
        assert torch.equal(target.row_scales, expected_rows)


@pytest.mark.parametrize(
    "fmt,k", [("mxfp8", 640), ("mxfp4", 5120), ("nvfp4", 640), ("nvfp4_pt", 5120)]
)
def test_blockscaled_row_movers_match_row_quantizer(fmt, k):
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
        contiguous_m_pad_ceiling,
        dispatch_fill_rows_contiguous_blockscaled,
        dispatch_layout_contiguous,
    )
    from sglang.kernels.ops.lora.moe.dispatch_masked import (
        dispatch_fill_masked_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import quantize_rows

    dtype = torch.float8_e4m3fn if fmt == "mxfp8" else torch.uint8
    vec = 16 if fmt.startswith("nvfp4") else 32
    per_token, crosses_atom = fmt == "nvfp4_pt", fmt == "mxfp4"
    num_tokens = 132 if crosses_atom else 12
    top_k = num_experts = 2
    global_scale = 0.3 if fmt == "nvfp4" else None
    rows = partial(
        _row_buffers,
        k=k,
        dtype=dtype,
        vec=vec,
        per_token=per_token,
        global_scale=global_scale,
    )

    g = torch.Generator().manual_seed(0x5B1F)
    x = torch.randn((num_tokens, k), generator=g).mul_(0.3).to(torch.bfloat16)
    x[0].zero_()
    x[9 if per_token else 3, ::128] = 32
    if per_token:
        for i, amax in enumerate((2.0**-133, 2.0**-126, 2.0**-120, 2.0**-100), 1):
            x[i] = (torch.randn(k, generator=g) * amax).to(torch.bfloat16)
            x[i, 0] = amax
        x[5] = torch.finfo(torch.bfloat16).max
        x[5, 1::2] *= -1
        x[6:8].zero_()
        x[6:8, 0] = 1.0
        x[6, 16:32] = 2.0**-20
        x[7, 16:32] = 2.0**-18
        x[6:8, 17].neg_()
        x[8] = 0.75
    elif crosses_atom:
        x[4, :32].zero_()
        x[4, :8] = (
            torch.tensor([0.75, -0.25, 1.25, -1.75, 2.5, -3.5, 5.0, 6.0]) * 2.0**-3
        )
    x = x.cuda()

    row_scales = None
    if per_token:
        codes, scales, row_scales, _, _ = quantize_nvfp4_per_token_ref(x)
        assert 0 < row_scales[1] < torch.finfo(torch.float32).tiny
        assert scales[6, 1] == 0 and 0 < scales[7, 1] < 8
        assert (scales[8] == 0x7E).all()
    elif global_scale is not None:
        codes, scales, _ = quantize_nvfp4_static_ref(
            x, x.new_tensor([global_scale], dtype=torch.float32)
        )
    else:
        codes, scales, _ = quantize_mx_ref(x, vec, "e4m3" if fmt == "mxfp8" else "e2m1")
        if crosses_atom:
            assert codes[4, :4].tolist() == [0x82, 0xC2, 0xE4, 0x76]

    direct = rows(num_tokens, num_tokens)
    direct_ids = torch.arange(num_tokens, device=x.device, dtype=torch.int32)
    direct_ids[-2] = -1
    quantize_rows(x, direct, direct_ids)
    live = direct_ids[direct_ids >= 0].long()
    _check_row_buffers(
        direct, live, codes[live], scales[live], row_scales[live] if per_token else None
    )

    topk_ids = torch.arange(top_k, device=x.device, dtype=torch.int32).repeat(
        num_tokens, 1
    )
    topk_ids[-2] = -1
    topk_ids[-1, 1] = -1
    valid = topk_ids.flatten() >= 0
    experts = topk_ids.flatten()[valid].long()
    tokens = torch.arange(num_tokens, device=x.device).repeat_interleave(top_k)[valid]
    pair_to_row = torch.empty_like(topk_ids.flatten())
    counts = torch.empty(num_experts, dtype=torch.int32, device=x.device)
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=x.device)

    m_max = 160 if crosses_atom else 32
    masked = rows(num_experts * m_max, m_max)
    dispatch_fill_masked_blockscaled(
        x,
        topk_ids,
        top_k,
        masked_m_out=counts,
        pair_to_row_out=pair_to_row,
        rows_out=masked,
    )
    _check_row_buffers(
        masked,
        pair_to_row[valid].long(),
        codes[tokens],
        scales[tokens],
        row_scales[tokens] if per_token else None,
    )
    assert (pair_to_row[~valid] == -1).all()
    assert counts.tolist() == [num_tokens - 1, num_tokens - 2]
    assert torch.equal(pair_to_row[valid].long() // m_max, experts)
    if crosses_atom:
        expert_zero_rows = pair_to_row[topk_ids.flatten() == 0]
        assert (expert_zero_rows == 127).any() and (expert_zero_rows == 128).any()

    m_pad = contiguous_m_pad_ceiling(pair_to_row.numel(), num_experts, 128)
    contiguous = rows(m_pad, m_pad)
    dispatch_layout_contiguous(
        x,
        topk_ids,
        num_experts,
        top_k,
        128,
        seg_counts_out=counts,
        seg_offsets_out=offsets,
        pair_to_row_out=pair_to_row,
    )
    dispatch_fill_rows_contiguous_blockscaled(
        x,
        topk_ids,
        pair_to_row,
        rows_out=contiguous,
        token_scales=torch.empty(num_tokens, device=x.device) if per_token else None,
    )
    dst = pair_to_row[valid].long()
    starts = offsets[experts]
    assert ((starts <= dst) & (dst < starts + counts[experts])).all()
    _check_row_buffers(
        contiguous,
        dst,
        codes[tokens],
        scales[tokens],
        row_scales[tokens] if per_token else None,
    )
    assert (pair_to_row[~valid] == -1).all()


def test_triton_row_quantizer_addresses_rows_past_2_31_elements():
    """Offsets past 2**31 elements must address the intended expert rows."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from sglang.kernels.ops.lora.moe.quant_blockscaled import (
        BlockScaledRows,
        quantize_rows,
    )

    device = torch.device("cuda")
    k, vec = 128, 32
    rows = 2**31 // k + 2048  # the last rows' offsets pass 2**31 elements
    if torch.cuda.mem_get_info(device)[0] < 8 * 2**30:
        pytest.skip("needs 8 GiB of free device memory")
    x = torch.zeros((rows, k), dtype=torch.bfloat16, device=device)
    generator = torch.Generator(device="cpu").manual_seed(3)
    pattern = (torch.randn(k, generator=generator) * 0.3).to(torch.bfloat16)
    probes = torch.tensor(
        [0, 2**31 // k - 1, 2**31 // k, rows - 1], dtype=torch.int32, device=device
    )
    x[probes.long()] = pattern.to(device)
    out = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=device)
    row_atoms = -(-rows // 128)  # k // vec = 4 groups: one atom column
    sf = torch.zeros(row_atoms * 512, dtype=torch.uint8, device=device)
    target = BlockScaledRows(data=out, scales=sf, sf_vec_size=vec, rows_per_slab=rows)
    quantize_rows(x, target, probes)
    torch.cuda.synchronize()

    codes = out.view(torch.uint8)[probes.long()]
    assert codes[0].any()
    assert torch.equal(codes, codes[:1].expand_as(codes))

    def scale_bytes(row: int) -> torch.Tensor:
        base = (row // 128) * 512 + (row % 32) * 16 + (row // 32 % 4) * 4
        return sf[base : base + 4]

    reference = scale_bytes(0)
    assert reference.any()
    for row in probes.tolist()[1:]:
        assert torch.equal(scale_bytes(row), reference), row


def test_per_token_row_quantizer_selects_4over6_scales():
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.quant_blockscaled import quantize_rows

    recipe = NvFp4ActivationRecipe(e4m3_max=256, err_mode="MSE", err_use_fast_math=True)
    x = torch.zeros((3, 128), device="cuda", dtype=torch.bfloat16)
    x[0, :3] = torch.tensor([1536.0, 576.0, 1152.0], device=x.device)
    x[1, 0] = 1.0
    x[1, 16:19] = torch.tensor([3.0, -3.0, 1.5], device=x.device) * 2**-20
    pair_to_row = torch.tensor([1, -1, 0], dtype=torch.int32, device=x.device)
    target = struct_replace(
        _row_buffers(3, 3, 128, torch.uint8, 16, per_token=True),
        row_denominator=6 * recipe.e4m3_max,
        use_4over6=True,
        err_mode=recipe.err_mode,
        err_use_fast_math=recipe.err_use_fast_math,
    )
    quantize_rows(x, target, pair_to_row)
    codes, scales, row_scales, _, _ = quantize_nvfp4_per_token_ref(x, recipe)
    assert scales[0, 0] == 0x7C  # The 4-based scale lowers the error.
    assert scales[1, 1] == 0x01  # The 6-based candidate rounds to zero.
    live = pair_to_row[pair_to_row >= 0].long()
    _check_row_buffers(target, live, codes[live], scales[live], row_scales[live])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
