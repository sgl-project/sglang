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
from sglang.srt.lora.moe.plan import (
    BFamily,
    BSpec,
    DownOverlap,
    SelectedPlan,
    Site,
    resolve_plans,
)
from sglang.srt.lora.moe.runner import MoeLoraBatch, MoeLoraRunner
from sglang.srt.lora.utils import Phase
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Every test but the admission test needs SM100 (block-scaled tcgen05 MMA) and
# skips on the H100 lane, so the file is also registered on the SM100 lane. It
# runs on one GPU: 4-gpu-b200 is the kernel lane's only SM100 pool, so its four
# GPUs are allocation granularity, not multi-GPU coverage.
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

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
# The default per-token recipe: D = 6 * e4m3_max with the 448 reference.
_PER_TOKEN_DENOMINATOR = 6.0 * 448.0


def _quant_nvfp4_per_token(x: torch.Tensor):
    """Per-token NVFP4 as the kernels evaluate it: the stored row scale
    fl(row_amax / D), block scales fl(amax / (6 s)) saturated at 448, codes
    e2m1((x / s) / dec(bs)); zero rows and blocks whose scale rounds to zero
    quantize to zero. Returns the dequantized values."""
    *lead, k = x.shape
    xf = x.float()
    # A tensor divisor: the correctly rounded division the kernels use.
    s = xf.abs().amax(dim=-1) / torch.tensor(_PER_TOKEN_DENOMINATOR, device=x.device)
    s_safe = torch.where(s > 0, s, torch.ones_like(s))
    groups = xf.reshape(*lead, k // 16, 16)
    amax = groups.abs().amax(dim=-1)
    block = (amax / (6.0 * s_safe)[..., None]).clamp(max=448.0)
    decoded = block.to(torch.float8_e4m3fn).float()
    live = decoded > 0
    denom = torch.where(live, decoded, torch.ones_like(decoded))
    scaled = (groups / s_safe[..., None, None]) / denom[..., None]
    scaled = torch.where(live[..., None], scaled, torch.zeros_like(scaled))
    scaled = scaled.clamp(-6.0, 6.0)
    table = _E2M1.to(x.device)
    idx = _e2m1_codes(scaled.abs())
    values = torch.where(torch.signbit(scaled), -table[idx], table[idx])
    return (values * denom[..., None] * s[..., None, None]).reshape(*lead, k)


def _family_and_mode(family: str) -> tuple[str, str]:
    """``nvfp4_pt`` is the NVFP4 family in per-token activation-scale mode."""
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
    """``w2_global`` (one down weight global scale per expert) and
    ``a2_global`` (the static down activation scale) replace the constants;
    the down weights then scale with their expert's global scale, so their
    codes sit at the same magnitudes under each."""
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
            q, s, eff = _quant_nvfp4(
                w13[:, half * _I : (half + 1) * _I], w13_gs[0, half].item()
            )
            w13_q.append(q), w13_s.append(s), w13_eff.append(eff)
        if w2_global is None:
            w2_q, w2_s, w2_eff = _quant_nvfp4(w2, _W2_GLOBAL)
            w2_gs = torch.full((_E,), _W2_GLOBAL, device=w2.device)
        else:
            w2_gs = torch.tensor(w2_global, dtype=torch.float32, device=w2.device)
            scaled = w2.float() * (w2_gs / _W2_GLOBAL)[:, None, None]
            w2_q, w2_s, w2_eff = (
                torch.stack(part)
                for part in zip(
                    *(_quant_nvfp4(scaled[e], w2_global[e]) for e in range(_E))
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


def _quant_activation(
    x: torch.Tensor, elem: str, stage: str, a2_global: float = _A2_GLOBAL
):
    """Dequantized activation rows as the provider quantizes them."""
    if elem == "nvfp4":
        gs = _A13_GLOBAL if stage == "gemm1" else a2_global
        return _quant_nvfp4(x, gs)[2]
    if elem == "nvfp4_pt":
        return _quant_nvfp4_per_token(x)
    return _quant_mx(x, elem)[2]


@pytest.mark.parametrize("family", ["mxfp8", "mxfp4", "nvfp4", "nvfp4_pt"])
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

    provider = select_provider_cls(rows, _family_and_mode(family)[0], "cutedsl")(
        quant_info
    )
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


@pytest.mark.parametrize("family", ["mxfp8", "mxfp4", "nvfp4", "nvfp4_pt"])
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
    quant_family, act_scale = _family_and_mode(family)
    selected = resolve_plans(
        architecture="sm100",
        is_shared_outer=True,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family=quant_family,
        vendor="cutedsl",
        act_scale=act_scale,
    )[phase]
    assert selected.name == (
        "decode.shared" if phase is Phase.DECODE else "prefill.shared"
    )
    provider = select_provider_cls(selected.base_gemm_rows, quant_family, "cutedsl")(
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


# Tokens whose BF16 rows sit at the subnormal end (token -> row amax): subnormal
# inputs, and stored per-token scales that are fp32 subnormals.
_TINY_TOKENS = {1: 2.0**-133, 2: 2.0**-126}
_TINY_SLOT = 7  # adapters large enough to show a tiny token's delta
_PER_TOKEN_PATTERNS = {
    "mixed": (1, -1, 4, -1),
    "base": (-1,),
    "slot1": (1,),
    "slot4": (4, -1),
    "high": (4, 1, 4),
}


def _per_token_mapping(name: str, tokens: int) -> torch.Tensor:
    """Slots per token under one pattern: the tiny tokens take the large slot
    when the pattern adapts them, the base path otherwise."""
    pattern = _PER_TOKEN_PATTERNS[name]
    mapping = [pattern[t % len(pattern)] for t in range(tokens)]
    for t in _TINY_TOKENS:
        mapping[t] = -1 if name in ("base", "slot1") else _TINY_SLOT
    return torch.tensor(mapping, dtype=torch.int32)


@pytest.mark.parametrize("rank", [16, 8])
@pytest.mark.parametrize(
    "phase", [Phase.DECODE, Phase.PREFILL], ids=["decode", "prefill"]
)
def test_blockscaled_per_expert_per_token_plans_match_reference(
    monkeypatch: pytest.MonkeyPatch, phase: Phase, rank: int
) -> None:
    """NVFP4 in per-token mode through the sm100 per-expert rows: decode
    (expert slabs, the materialized activation's two-pass quantization) and
    prefill (flat segments, the fused LoRA-B activation with the separate
    row quantizer, the down-B fused into the base GEMM epilogue by the
    packaged table's epilogue row). Eight
    resident slots with adapters in slots 1, 4 and 7 beside base tokens, an
    EP hole, a zero token and two tiny tokens (BF16 amax 2^-133 and 2^-126:
    subnormal inputs and stored row scales) whose slot-7 adapters are large
    enough (2^40 at gate/up, 2^20 at down) that their delta sits in the BF16
    range of the output and must come through, nothing on the path flushing
    it. Every pattern against the reference that quantizes both GEMM inputs
    per token; then one captured graph per phase replayed over the slot,
    base and mixed patterns and over changed routing and inputs, bitwise
    with the eager runs."""
    _skip_unless_sm100()
    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    g = torch.Generator(device="cpu").manual_seed(37)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    tokens = 8 if phase is Phase.DECODE else 40
    slots = 8
    quant_info, w13_eff, w2_eff, elem = _quant_info(
        "nvfp4_pt", rand(_E, 2 * _I, _H), rand(_E, _H, _I)
    )
    selected = resolve_plans(
        architecture="sm100",
        is_shared_outer=False,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family="nvfp4",
        vendor="cutedsl",
        act_scale="per_token",
    )[phase]
    assert selected.name == (
        "decode.per_expert"
        if phase is Phase.DECODE
        else "prefill.per_expert.cutedsl_blockscaled"
    )
    provider = select_provider_cls(selected.base_gemm_rows, "nvfp4", "cutedsl")(
        quant_info
    )
    if phase is Phase.PREFILL:
        assert selected.plan.down_b.family is BFamily.EPILOGUE
        provider.enable_epilogue_down_lora(rank)
    runner = MoeLoraRunner(
        providers={selected.base_gemm_rows: provider},
        top_k=_TOPK,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFn.SILU,
    )
    runner.validate_plan(selected.plan, base_gemm_rows=selected.base_gemm_rows)
    runner.plans = {phase: selected}

    hidden0 = rand(tokens, _H, scale=0.5)
    hidden0[0].zero_()  # a zero token: per-token scale 0
    for t, amax in _TINY_TOKENS.items():
        row = (torch.rand(_H, generator=g) * 2 - 1) * amax
        row[0] = amax
        hidden0[t] = row.to(torch.bfloat16).to(device)
    topk0 = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk0[3, 1] = -1  # an EP hole on an adapter token
    weights = torch.tensor([0.65, 0.35], device=device).repeat(tokens, 1)
    gate_up_a = rand(slots, _E, 2 * rank, _H, scale=0.1)
    gate_up_b = rand(slots, _E, 2 * _I, rank, scale=0.1)
    down_a = rand(slots, _E, rank, _I, scale=0.1)
    down_b = rand(slots, _E, _H, rank, scale=0.1)
    # Slot 7 serves the tiny tokens only: through 2^40 at gate/up and 2^20
    # at down, a 2^-133 token's delta lands near 2^-50 and a 2^-126 token's
    # near 2^-36, while silu(gate) * up of either base row is far below BF16.
    for adapters, scale in (
        (gate_up_a, 2.0**40),
        (gate_up_b, 2.0**40),
        (down_a, 2.0**20),
        (down_b, 2.0**20),
    ):
        adapters[_TINY_SLOT] = (adapters[_TINY_SLOT].float() * scale).to(torch.bfloat16)
    hidden, topk_ids = hidden0.clone(), topk0.clone()
    mapping = torch.empty(tokens, dtype=torch.int32, device=device)
    # The slot patterns on the first inputs, then the mixed pattern with
    # flipped routing and rescaled inputs (the 2^-133 row rounds to zero).
    states = [(name, False) for name in _PER_TOKEN_PATTERNS] + [("mixed", True)]

    def load(state):
        name, flipped = state
        mapping.copy_(_per_token_mapping(name, tokens))
        topk_ids.copy_(topk0.flip(1) if flipped else topk0)
        hidden.copy_(hidden0 * -0.5 if flipped else hidden0)

    def reference(mapping):
        pair_tokens = torch.arange(tokens, device=device).repeat_interleave(_TOPK)
        experts = topk_ids.flatten().long()
        routed = (experts >= 0).float()[:, None]
        experts = experts.clamp(min=0)
        slot = mapping[pair_tokens].long()
        adapted = (slot >= 0).float()[:, None]
        slot = slot.clamp(min=0)
        inputs = hidden.float()[pair_tokens]
        inputs_q = _quant_activation(inputs, elem, "gemm1")
        gate_up = _bf16(torch.einsum("ph,pnh->pn", inputs_q, w13_eff[experts]))
        bridge = _bf16(
            torch.einsum("ph,prh->pr", inputs, gate_up_a.float()[slot, experts])
        )
        inner = gate_up_b.float()[slot, experts]
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
        gate_up = gate_up + adapted * (_bf16(delta) if phase is Phase.DECODE else delta)
        act = _bf16(F.silu(gate_up[:, :_I]) * gate_up[:, _I:])
        act_q = _quant_activation(act, elem, "gemm2")
        pairs = _bf16(torch.einsum("pi,phi->ph", act_q, w2_eff[experts]))
        down_rank = _bf16(
            torch.einsum("pi,pri->pr", act, down_a.float()[slot, experts])
        )
        pairs = pairs + adapted * _bf16(
            torch.einsum("pr,phr->ph", down_rank, down_b.float()[slot, experts])
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
        # A new batch, as the serving backend prepares one: the routes (the
        # slot bound among them) rebuild from the mapping.
        runner.workspace.routes.clear()
        with get_context().override_server_args():
            out = runner.run(dispatch, batch).hidden_states
        torch.cuda.synchronize()
        return out

    eager = {}
    for state in states:
        load(state)
        out = eager[state] = forward()
        ref = reference(mapping)
        assert bool(torch.isfinite(out).all()), state
        err = float(_rel_l2(out.float(), ref))
        assert err < _RUNNER_TOL, (state, err)
        if state[0] != "base":
            assert _rel_l2(reference(torch.full_like(mapping, -1)), ref) > 0.1, state
        for t in _TINY_TOKENS:
            row, ref_row = out[t].double(), ref[t].double()
            if ref_row.norm() == 0:
                assert bool((row == 0).all()), (state, t)
            elif int(mapping[t]) >= 0:
                # The large slot's delta is the row: visible and reproduced.
                without = mapping.clone()
                without[t] = -1
                base_norm = float(reference(without)[t].double().norm())
                assert base_norm < 1e-3 * float(ref_row.norm()), (state, t)
                err = float(_rel_l2(row, ref_row))
                assert err < 2.5e-2, (state, t, err)
            else:
                assert torch.allclose(row, ref_row, rtol=0.1, atol=2.0**-130), (
                    state,
                    t,
                )

    # One graph per phase, captured under the first pattern and replayed
    # over the rest in place: bitwise with the eager runs.
    load(states[0])
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
        with get_context().override_server_args():
            captured = runner.run(dispatch, batch).hidden_states
    for state in states + states[:1]:
        load(state)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager[state]), state


def test_per_token_provider_bounds_the_weight_global_scales() -> None:
    """The per-token epilogue multiplies alpha into the fp32 accumulator
    before the row scale, so binding refuses weight global scales whose
    product with the largest accumulator (K times 2688 squared) leaves the
    fp32 range, per GEMM, and subnormal scales, below which the product can
    round to zero for a nonzero accumulator; zero (an all-zero expert), the
    smallest normal fp32, the checkpoints' smallest scale (2e-37) and a scale
    just under the bound bind."""
    _skip_unless_sm100()
    from msgspec.structs import replace as struct_replace

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(41)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

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
    with pytest.raises(ValueError, match="gemm2 weight global scales"):
        cls(
            struct_replace(
                quant_info, w2_global_scale=w2(1e-3, 1e-3, 1e-3, 1.01 * bound_down)
            )
        )
    with pytest.raises(ValueError, match="gemm2 weight global scales"):
        cls(
            struct_replace(
                quant_info, w2_global_scale=w2(1e-3, float("nan"), 1e-3, 1e-3)
            )
        )
    w13 = quant_info.w13_global_scale.clone()
    w13[2, 1] = 1.01 * bound_gate_up
    with pytest.raises(ValueError, match="gemm1 weight global scales"):
        cls(struct_replace(quant_info, w13_global_scale=w13))


@pytest.mark.parametrize("family", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("rank", [12, 9])
def test_blockscaled_per_expert_rank_between_multiples_keeps_separate_b(
    family: str, rank: int
) -> None:
    """A pool rank inside the epilogue row's 8..16 window that is not a
    multiple of eight selects the separate down-B row, and the provider it
    gets never has the epilogue enabled: the epilogue's own contract still
    refuses such a rank, so the selector is what keeps the server up."""
    _skip_unless_sm100()
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(31)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    quant_info, _, _, _ = _quant_info(family, rand(_E, 2 * _I, _H), rand(_E, _H, _I))
    selected = resolve_plans(
        architecture="sm100",
        is_shared_outer=False,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family=family,
        vendor="cutedsl",
    )[Phase.PREFILL]
    assert selected.name == "prefill.per_expert"
    assert selected.plan.down_b is not None
    assert selected.plan.down_b.family is BFamily.GROUPED
    provider = select_provider_cls("route_major", family, "cutedsl")(quant_info)
    assert provider.epilogue_down_lora_rank() == 0
    with pytest.raises(ValueError, match="multiples of 8"):
        provider.enable_epilogue_down_lora(rank)
    assert provider.epilogue_down_lora_rank() == 0


@pytest.mark.parametrize("family", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("rank", [16, 8])
def test_blockscaled_per_expert_epilogue_down_b_matches_separate_b(
    monkeypatch: pytest.MonkeyPatch, family: str, rank: int
) -> None:
    """The sm100 per-expert prefill row with its down-B fused into the base
    GEMM epilogue, against the same row with the separate grouped down-B and
    against the FP32 reference, with nonzero adapters over mixed slots, base
    tokens and an EP hole. Zero down-B weights make both paths bitwise equal
    (the epilogue adds exact zeros); the epilogue path also replays under a
    prefill CUDA graph over changed slots, routing and inputs.
    """
    _skip_unless_sm100()
    from dataclasses import replace

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    g = torch.Generator(device="cpu").manual_seed(31)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    tokens, slots = 40, 2
    quant_info, w13_eff, w2_eff, elem = _quant_info(
        family, rand(_E, 2 * _I, _H), rand(_E, _H, _I)
    )
    selected = resolve_plans(
        architecture="sm100",
        is_shared_outer=False,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family=family,
        vendor="cutedsl",
    )[Phase.PREFILL]
    assert selected.base_gemm_rows == "route_major"
    separate_plan = replace(
        selected.plan,
        down_b=BSpec(Site.DOWN, BFamily.GROUPED),
        down_overlap=DownOverlap.NONE,
    )
    epilogue_plan = replace(
        selected.plan,
        down_b=BSpec(Site.DOWN, BFamily.EPILOGUE),
        down_overlap=DownOverlap.NONE,
    )
    provider = select_provider_cls("route_major", family, "cutedsl")(quant_info)
    provider.enable_epilogue_down_lora(rank)
    runner = MoeLoraRunner(
        providers={"route_major": provider},
        top_k=_TOPK,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFn.SILU,
    )
    for plan in (separate_plan, epilogue_plan):
        runner.validate_plan(plan, base_gemm_rows="route_major")

    def select(plan):
        runner.plans = {
            Phase.PREFILL: SelectedPlan(
                name="test",
                base_gemm_rows="route_major",
                plan=plan,
                tiles=selected.tiles,
            )
        }

    hidden = rand(tokens, _H, scale=0.5)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[3, 1] = -1  # an EP hole on an adapter token
    weights = torch.tensor([0.65, 0.35], device=device).repeat(tokens, 1)
    gate_up_a = rand(slots, _E, 2 * rank, _H, scale=0.1)
    gate_up_b = rand(slots, _E, 2 * _I, rank, scale=0.1)
    down_a = rand(slots, _E, rank, _I, scale=0.1)
    down_b = rand(slots, _E, _H, rank, scale=0.1)
    # Slots 0, 1 and the base path (-1) in turn.
    mapping = ((torch.arange(tokens, device=device) + 1) % 3 - 1).to(torch.int32)

    def reference(mapping, down_b):
        pair_tokens = torch.arange(tokens, device=device).repeat_interleave(_TOPK)
        experts = topk_ids.flatten().long()
        routed = (experts >= 0).float()[:, None]
        experts = experts.clamp(min=0)
        slot = mapping[pair_tokens].long()
        adapted = (slot >= 0).float()[:, None]
        slot = slot.clamp(min=0)
        inputs = hidden.float()[pair_tokens]
        inputs_q = _quant_activation(inputs, elem, "gemm1")
        gate_up = _bf16(torch.einsum("ph,pnh->pn", inputs_q, w13_eff[experts]))
        bridge = _bf16(
            torch.einsum("ph,prh->pr", inputs, gate_up_a.float()[slot, experts])
        )
        inner = gate_up_b.float()[slot, experts]
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
        # The prefill row's fused LoRA-B activation adds the delta in FP32.
        gate_up = gate_up + adapted * delta
        act = _bf16(F.silu(gate_up[:, :_I]) * gate_up[:, _I:])
        act_q = _quant_activation(act, elem, "gemm2")
        base = torch.einsum("pi,phi->ph", act_q, w2_eff[experts])
        down_rank = _bf16(
            torch.einsum("pi,pri->pr", act, down_a.float()[slot, experts])
        )
        delta_d = torch.einsum("pr,phr->ph", down_rank, down_b.float()[slot, experts])
        # Unrounded base + delta: the separate path rounds both to BF16 first,
        # the epilogue path rounds their FP32 sum once.
        pairs = (base + adapted * delta_d) * routed
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

    def batch_for(down_b):
        return MoeLoraBatch(
            gate_up_lora_a=gate_up_a,
            gate_up_lora_b=gate_up_b,
            down_lora_a=down_a,
            down_lora_b=down_b,
            token_lora_mapping=mapping,
            is_prefill=True,
        )

    def forward(batch):
        with get_context().override_server_args():
            out = runner.run(dispatch, batch).hidden_states
        torch.cuda.synchronize()
        return out

    select(separate_plan)
    separate = forward(batch_for(down_b))
    select(epilogue_plan)
    fused = forward(batch_for(down_b))
    ref = reference(mapping, down_b)
    err_separate = _rel_l2(separate.float(), ref)
    err_fused = _rel_l2(fused.float(), ref)
    assert err_separate < _RUNNER_TOL and err_fused < _RUNNER_TOL, (
        err_separate,
        err_fused,
    )
    # The two paths are within 4e-3 relative L2 of each other over the batch
    # (a gate on the whole output, not an elementwise BF16-ulp bound) and the
    # LoRA math is present (not bypassed within tolerance).
    assert _rel_l2(fused.float(), separate.float()) < 4e-3
    assert _rel_l2(reference(torch.full_like(mapping, -1), down_b), ref) > 0.1

    # Zero down-B: the epilogue adds exact zeros, so both paths coincide.
    zero_b = torch.zeros_like(down_b)
    select(separate_plan)
    separate_zero = forward(batch_for(zero_b))
    select(epilogue_plan)
    fused_zero = forward(batch_for(zero_b))
    assert torch.equal(separate_zero, fused_zero)

    # Prefill graph replay of the epilogue path.
    batch = batch_for(down_b)
    batch.use_cuda_graph = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward(batch)
    torch.cuda.synchronize()
    runner.workspace.routes.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        with get_context().override_server_args():
            captured = runner.run(dispatch, batch).hidden_states
    graph.replay()
    torch.cuda.synchronize()
    assert _rel_l2(captured.float(), ref) < _RUNNER_TOL
    mapping.copy_(mapping.roll(1))
    topk_ids.copy_(topk_ids.flip(1))
    hidden.mul_(-0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert _rel_l2(captured.float(), reference(mapping, down_b)) < _RUNNER_TOL


# Token slot patterns of an eight-slot pool: one adapter alone in slot 7 or 4,
# slots 1/4/7 mixed with base tokens, all eight, two low slots, and no adapter.
_SLOT_PATTERNS = {
    "slot7": [7, -1],
    "slot4": [4, -1],
    "mixed": [1, 4, 7, -1],
    "eight": [0, 1, 2, 3, 4, 5, 6, 7, -1],
    "low": [0, 1, -1],
    "base": [-1],
}


@pytest.mark.parametrize("family", ["mxfp8", "mxfp4", "nvfp4"])
def test_blockscaled_per_expert_epilogue_down_b_slot_bounds(
    monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    """The epilogue down-B over the resident-slot bounds an eight-slot pool
    presents (the bound is the highest slot in the batch plus one, computed
    on the device), eager and under one captured prefill graph replayed
    across them in place: high slots, low slots, no adapter, high again.
    Every bound matches the reference and the separate grouped down-B, and
    an adapter's rows are bitwise the same wherever the pool holds it.
    """
    _skip_unless_sm100()
    from dataclasses import replace

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    g = torch.Generator(device="cpu").manual_seed(37)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    tokens, slots, rank = 320, 8, 16  # ~160 rows per expert: two tiles each
    quant_info, w13_eff, w2_eff, elem = _quant_info(
        family, rand(_E, 2 * _I, _H), rand(_E, _H, _I)
    )
    selected = resolve_plans(
        architecture="sm100",
        is_shared_outer=False,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family=family,
        vendor="cutedsl",
    )[Phase.PREFILL]
    separate_plan = replace(
        selected.plan,
        down_b=BSpec(Site.DOWN, BFamily.GROUPED),
        down_overlap=DownOverlap.NONE,
    )
    epilogue_plan = replace(
        selected.plan,
        down_b=BSpec(Site.DOWN, BFamily.EPILOGUE),
        down_overlap=DownOverlap.NONE,
    )
    provider = select_provider_cls("route_major", family, "cutedsl")(quant_info)
    provider.enable_epilogue_down_lora(rank)
    runner = MoeLoraRunner(
        providers={"route_major": provider},
        top_k=_TOPK,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFn.SILU,
    )

    def select(plan):
        runner.plans = {
            Phase.PREFILL: SelectedPlan(
                name="test",
                base_gemm_rows="route_major",
                plan=plan,
                tiles=selected.tiles,
            )
        }

    hidden = rand(tokens, _H, scale=0.5)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[3, 1] = -1  # an EP hole on an adapter token
    weights = torch.tensor([0.65, 0.35], device=device).repeat(tokens, 1)
    # Distinct adapters in all eight slots.
    gate_up_a = rand(slots, _E, 2 * rank, _H, scale=0.1)
    gate_up_b = rand(slots, _E, 2 * _I, rank, scale=0.1)
    down_a = rand(slots, _E, rank, _I, scale=0.1)
    down_b = rand(slots, _E, _H, rank, scale=0.1)
    mapping = torch.empty(tokens, dtype=torch.int32, device=device)

    def set_mapping(name):
        pattern = torch.tensor(_SLOT_PATTERNS[name], dtype=torch.int32)
        mapping.copy_(pattern[torch.arange(tokens) % len(pattern)])

    def reference(mapping):
        pair_tokens = torch.arange(tokens, device=device).repeat_interleave(_TOPK)
        experts = topk_ids.flatten().long()
        routed = (experts >= 0).float()[:, None]
        experts = experts.clamp(min=0)
        slot = mapping[pair_tokens].long()
        adapted = (slot >= 0).float()[:, None]
        slot = slot.clamp(min=0)
        inputs = hidden.float()[pair_tokens]
        inputs_q = _quant_activation(inputs, elem, "gemm1")
        gate_up = _bf16(torch.einsum("ph,pnh->pn", inputs_q, w13_eff[experts]))
        bridge = _bf16(
            torch.einsum("ph,prh->pr", inputs, gate_up_a.float()[slot, experts])
        )
        inner = gate_up_b.float()[slot, experts]
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
        gate_up = gate_up + adapted * delta
        act = _bf16(F.silu(gate_up[:, :_I]) * gate_up[:, _I:])
        act_q = _quant_activation(act, elem, "gemm2")
        base = torch.einsum("pi,phi->ph", act_q, w2_eff[experts])
        down_rank = _bf16(
            torch.einsum("pi,pri->pr", act, down_a.float()[slot, experts])
        )
        delta_d = torch.einsum("pr,phr->ph", down_rank, down_b.float()[slot, experts])
        pairs = (base + adapted * delta_d) * routed
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

    def batch_for(adapters=(gate_up_a, gate_up_b, down_a, down_b), mapping=mapping):
        return MoeLoraBatch(
            gate_up_lora_a=adapters[0],
            gate_up_lora_b=adapters[1],
            down_lora_a=adapters[2],
            down_lora_b=adapters[3],
            token_lora_mapping=mapping,
            is_prefill=True,
        )

    def forward(plan, batch):
        # A new batch, as the serving backend prepares one: routes (the slot
        # bound among them) rebuild from the mapping.
        runner.workspace.routes.clear()
        select(plan)
        with get_context().override_server_args():
            out = runner.run(dispatch, batch).hidden_states
        torch.cuda.synchronize()
        return out

    fused = {}
    for name in _SLOT_PATTERNS:
        set_mapping(name)
        ref = reference(mapping)
        separate = forward(separate_plan, batch_for())
        fused[name] = forward(epilogue_plan, batch_for())
        err_separate = _rel_l2(separate.float(), ref)
        err_fused = _rel_l2(fused[name].float(), ref)
        assert err_separate < _RUNNER_TOL and err_fused < _RUNNER_TOL, (
            family,
            name,
            err_separate,
            err_fused,
        )
        assert _rel_l2(fused[name].float(), separate.float()) < 4e-3, (family, name)
        if name != "base":
            assert _rel_l2(reference(torch.full_like(mapping, -1)), ref) > 0.1, (
                family,
                name,
            )

    # Slot 7's adapter served from slot 0 gives its tokens the same rows, bitwise.
    moved = tuple(t.clone() for t in (gate_up_a, gate_up_b, down_a, down_b))
    for t in moved:
        t[0] = t[7]
    set_mapping("slot7")
    low_mapping = torch.where(mapping == 7, 0, mapping)
    assert torch.equal(
        forward(epilogue_plan, batch_for(moved, low_mapping)), fused["slot7"]
    ), family

    # One prefill graph, replayed while the mapping changes the bound in place.
    set_mapping("slot7")
    batch = batch_for()
    batch.use_cuda_graph = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward(epilogue_plan, batch)
    torch.cuda.synchronize()
    runner.workspace.routes.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        with get_context().override_server_args():
            captured = runner.run(dispatch, batch).hidden_states
    for name in ("slot7", "low", "base", "mixed", "slot4", "eight", "slot7"):
        set_mapping(name)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, fused[name]), (family, name)


@pytest.mark.parametrize("family", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("rank", [16, 8])
def test_blockscaled_per_expert_rows_depend_only_on_their_own_token(
    monkeypatch: pytest.MonkeyPatch, family: str, rank: int
) -> None:
    """Isolation at the runner level, for the epilogue row and the separate
    grouped down-B row alike: a token's output depends on its own hidden
    state, routing and slot only. Over an eight-slot pool with base tokens,
    bitwise: the batch permuted gives the permuted outputs; a subset of the
    tokens run alone gives their rows as in the full batch; the other
    tokens' slots changed (all base, all on slot 0, or a bound of two
    instead of eight) leave a token's rows; base tokens of a mixed batch
    are the base-only batch's rows; and after the bound sequence eight,
    two, base-only, the eight-slot batch repeats bitwise (no stale state).
    Every batch also lands on its own per-token reference.
    """
    _skip_unless_sm100()
    from dataclasses import replace

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    g = torch.Generator(device="cpu").manual_seed(41)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    tokens, slots = 320, 8
    quant_info, w13_eff, w2_eff, elem = _quant_info(
        family, rand(_E, 2 * _I, _H), rand(_E, _H, _I)
    )
    selected = resolve_plans(
        architecture="sm100",
        is_shared_outer=False,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family=family,
        vendor="cutedsl",
    )[Phase.PREFILL]
    plans = {
        "separate": replace(
            selected.plan,
            down_b=BSpec(Site.DOWN, BFamily.GROUPED),
            down_overlap=DownOverlap.NONE,
        ),
        "epilogue": replace(
            selected.plan,
            down_b=BSpec(Site.DOWN, BFamily.EPILOGUE),
            down_overlap=DownOverlap.NONE,
        ),
    }
    provider = select_provider_cls("route_major", family, "cutedsl")(quant_info)
    provider.enable_epilogue_down_lora(rank)
    runner = MoeLoraRunner(
        providers={"route_major": provider},
        top_k=_TOPK,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFn.SILU,
    )

    hidden = rand(tokens, _H, scale=0.5)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[3, 1] = -1  # an EP hole on an adapter token
    weights = torch.tensor([0.65, 0.35], device=device).repeat(tokens, 1)
    gate_up_a = rand(slots, _E, 2 * rank, _H, scale=0.1)
    gate_up_b = rand(slots, _E, 2 * _I, rank, scale=0.1)
    down_a = rand(slots, _E, rank, _I, scale=0.1)
    down_b = rand(slots, _E, _H, rank, scale=0.1)
    # All eight slots and the base path in turn.
    mapping = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, -1], dtype=torch.int32)[
        torch.arange(tokens) % 9
    ].to(device)

    def reference(hidden, topk_ids, weights, mapping):
        n = hidden.shape[0]
        pair_tokens = torch.arange(n, device=device).repeat_interleave(_TOPK)
        experts = topk_ids.flatten().long()
        routed = (experts >= 0).float()[:, None]
        experts = experts.clamp(min=0)
        slot = mapping[pair_tokens].long()
        adapted = (slot >= 0).float()[:, None]
        slot = slot.clamp(min=0)
        inputs = hidden.float()[pair_tokens]
        inputs_q = _quant_activation(inputs, elem, "gemm1")
        gate_up = _bf16(torch.einsum("ph,pnh->pn", inputs_q, w13_eff[experts]))
        bridge = _bf16(
            torch.einsum("ph,prh->pr", inputs, gate_up_a.float()[slot, experts])
        )
        inner = gate_up_b.float()[slot, experts]
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
        gate_up = gate_up + adapted * delta
        act = _bf16(F.silu(gate_up[:, :_I]) * gate_up[:, _I:])
        act_q = _quant_activation(act, elem, "gemm2")
        base = torch.einsum("pi,phi->ph", act_q, w2_eff[experts])
        down_rank = _bf16(
            torch.einsum("pi,pri->pr", act, down_a.float()[slot, experts])
        )
        delta_d = torch.einsum("pr,phr->ph", down_rank, down_b.float()[slot, experts])
        pairs = (base + adapted * delta_d) * routed
        return (pairs.view(n, _TOPK, _H) * weights[..., None]).sum(1) * _ROUTED_SCALING

    def forward(name, hidden, topk_ids, weights, mapping):
        # A new batch, as the serving backend prepares one: routes (the slot
        # bound among them) rebuild from the mapping.
        runner.workspace.routes.clear()
        runner.plans = {
            Phase.PREFILL: SelectedPlan(
                name="test",
                base_gemm_rows="route_major",
                plan=plans[name],
                tiles=selected.tiles,
            )
        }
        dispatch = StandardDispatchOutput(
            hidden_states=hidden,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=weights,
                topk_ids=topk_ids,
                router_logits=torch.zeros((hidden.shape[0], _E), device=device),
            ),
        )
        batch = MoeLoraBatch(
            gate_up_lora_a=gate_up_a,
            gate_up_lora_b=gate_up_b,
            down_lora_a=down_a,
            down_lora_b=down_b,
            token_lora_mapping=mapping,
            is_prefill=True,
        )
        with get_context().override_server_args():
            out = runner.run(dispatch, batch).hidden_states
        torch.cuda.synchronize()
        return out

    def same(actual, expected, what):
        assert torch.equal(actual, expected), (
            family,
            rank,
            what,
            float((actual.float() - expected.float()).abs().max()),
            int((actual != expected).sum()),
        )

    base_mapping = torch.full_like(mapping, -1)
    two_mapping = torch.where(
        mapping <= 1, mapping, base_mapping
    )  # slots 0, 1: bound 2
    for name in ("epilogue", "separate"):
        full = forward(name, hidden, topk_ids, weights, mapping)
        assert _rel_l2(full.float(), reference(hidden, topk_ids, weights, mapping)) < (
            _RUNNER_TOL
        ), (family, rank, name)
        # The batch permuted: the permuted outputs.
        perm = torch.randperm(tokens, generator=g).to(device)
        permuted = forward(
            name, hidden[perm], topk_ids[perm], weights[perm], mapping[perm]
        )
        same(permuted[torch.argsort(perm)], full, f"{name}: permuted batch")
        # A subset of the tokens alone: their rows as in the full batch.
        idx = torch.arange(0, tokens, 3, device=device)
        alone = forward(name, hidden[idx], topk_ids[idx], weights[idx], mapping[idx])
        same(alone, full[idx], f"{name}: subset alone")
        # The other tokens' slots changed: the slot-7 tokens keep their rows.
        own = mapping == 7
        others_base = torch.where(own, mapping, base_mapping)
        others_low = torch.where(
            own | (mapping < 0), mapping, torch.zeros_like(mapping)
        )
        same(
            forward(name, hidden, topk_ids, weights, others_base)[own],
            full[own],
            f"{name}: other tokens base",
        )
        same(
            forward(name, hidden, topk_ids, weights, others_low)[own],
            full[own],
            f"{name}: other tokens on slot 0",
        )
        # The slot-1 tokens under a bound of two and under a bound of eight.
        own1 = mapping == 1
        two = forward(name, hidden, topk_ids, weights, two_mapping)
        assert (
            _rel_l2(two.float(), reference(hidden, topk_ids, weights, two_mapping))
            < _RUNNER_TOL
        ), (family, rank, name)
        same(two[own1], full[own1], f"{name}: slot-1 rows, bound 2 vs 8")
        # Base tokens of the mixed batch: the base-only batch's rows.
        base_only = forward(name, hidden, topk_ids, weights, base_mapping)
        assert (
            _rel_l2(
                base_only.float(), reference(hidden, topk_ids, weights, base_mapping)
            )
            < _RUNNER_TOL
        ), (family, rank, name)
        same(full[mapping < 0], base_only[mapping < 0], f"{name}: base rows")
        # After eight, two, base-only: eight again, bitwise (no stale state).
        same(
            forward(name, hidden, topk_ids, weights, mapping),
            full,
            f"{name}: eight slots again after two and base-only",
        )


def test_block_scaled_admission():
    from types import SimpleNamespace

    from sglang.srt.lora.moe.families import resolve_vendor
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


@pytest.mark.parametrize("family", ["mxfp8", "mxfp4", "nvfp4", "nvfp4_pt"])
def test_layer_providers_share_the_scales_their_layer_derives(family):
    """Both row layouts of a layer bind one quant info, so the packed block
    scales (and NVFP4's alpha) exist once per layer; a second layer binds its
    own; the layer keeps its checkpoint-form tensors as they were."""
    from types import SimpleNamespace
    from unittest import mock

    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.runtime_context import get_flags

    _skip_unless_sm100()
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(11)
    quant_family, mode = _family_and_mode(family)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    def layer_and_providers():
        fields = _quant_info(family, rand(_E, 2 * _I, _H), rand(_E, _H, _I))[0]
        layer = SimpleNamespace(
            params_dtype=torch.bfloat16,
            w13_weight=fields.w13_weight,
            w2_weight=fields.w2_weight,
            num_local_experts=_E,
            intermediate_size_per_partition=_I,
            hidden_size=_H,
            moe_runner_config=SimpleNamespace(
                activation="silu",
                top_k=_TOPK,
                routed_scaling_factor=_ROUTED_SCALING,
                is_gated=True,
            ),
        )
        if quant_family == "nvfp4":
            vars(layer).update(
                w13_weight_scale=fields.w13_scale,
                w2_weight_scale=fields.w2_scale,
                w13_weight_scale_2=fields.w13_global_scale,
                w2_weight_scale_2=fields.w2_global_scale,
                w13_input_scale=torch.tensor(_A13_GLOBAL, device=device),
                w2_input_scale=torch.tensor(_A2_GLOBAL, device=device),
                quant_method=SimpleNamespace(
                    quant_config=SimpleNamespace(
                        use_per_token_activation=mode == "per_token"
                    )
                ),
            )
        else:
            vars(layer).update(
                w13_weight_scale_inv=fields.w13_scale,
                w2_weight_scale_inv=fields.w2_scale,
            )
        attributes = vars(layer).copy()
        snapshot = {
            name: value.clone()
            for name, value in attributes.items()
            if isinstance(value, torch.Tensor)
        }
        # The runner's own construction. Admission is pinned to the family (the
        # stand-in layer has no dispatcher or quant method to recognize it by);
        # at rank 32 the per-expert plans use both row layouts and keep down-B
        # out of the base GEMM's epilogue, so nothing else is compiled.
        with (
            mock.patch.object(MoeLoraRunner, "_admit", return_value=quant_family),
            get_context().override_server_args(),
            get_flags().moe.override(runner_backend=MoeRunnerBackend.LORA_CUTEDSL),
        ):
            runner = MoeLoraRunner.from_layer(
                layer, is_shared_outer=False, physical_rank=32
            )
        assert list(runner.providers) == ["expert_major", "route_major"]
        providers = list(runner.providers.values())
        # One quant info for the layer, bound by both providers.
        (quant_info,) = {id(p.quant_info): p.quant_info for p in providers}.values()
        # The layer still holds every tensor it held, bit for bit.
        for name, value in attributes.items():
            assert getattr(layer, name) is value, name
        for name, value in snapshot.items():
            assert torch.equal(
                getattr(layer, name).reshape(-1).view(torch.uint8),
                value.reshape(-1).view(torch.uint8),
            ), name
        return layer, quant_info, providers

    layer, quant_info, (masked, contiguous) = layer_and_providers()
    scale_attr = "weight_scale" if quant_family == "nvfp4" else "weight_scale_inv"
    for stage, bound in (("w13", "_sf_w13"), ("w2", "_sf_w2")):
        packed = getattr(masked, bound)
        # One packed copy for the layer, bound by both layouts ...
        assert packed is getattr(contiguous, bound)
        # ... beside the checkpoint form, which the quant info reads in place.
        canonical = getattr(layer, f"{stage}_{scale_attr}")
        assert getattr(quant_info, f"{stage}_scale").data_ptr() == canonical.data_ptr()
        assert packed.data_ptr() != canonical.data_ptr()
    if quant_family == "nvfp4":
        for stage in ("gemm1", "gemm2"):
            assert masked._alpha[stage] is contiguous._alpha[stage]
        factor = (_A13_GLOBAL, _A2_GLOBAL) if mode == "static" else (1.0, 1.0)
        assert torch.equal(
            masked._alpha["gemm1"], layer.w13_weight_scale_2.reshape(_E, -1) * factor[0]
        )
        assert torch.equal(
            masked._alpha["gemm2"], layer.w2_weight_scale_2.reshape(-1, 1) * factor[1]
        )

    _, _, (other, _) = layer_and_providers()
    bound = ["_sf_w13", "_sf_w2"]
    for name in bound:
        assert getattr(other, name).data_ptr() != getattr(masked, name).data_ptr()
    if quant_family == "nvfp4":
        for stage in ("gemm1", "gemm2"):
            assert other._alpha[stage].data_ptr() != masked._alpha[stage].data_ptr()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# --- The per-token epilogue down-B through the separate accumulator -----------


def _unfused_plan(selected: SelectedPlan):
    """The selected plan with the separate grouped down-B of the packaged
    table's `prefill.per_expert` row (its down-A overlap included)."""
    from dataclasses import replace

    return replace(
        selected.plan,
        down_b=BSpec(Site.DOWN, BFamily.GROUPED),
        down_overlap=DownOverlap.DOWN_A,
    )


@pytest.mark.parametrize("rank", [16, 8])
def test_fused_per_token_epilogue_plan_matches_reference(
    monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    """NVFP4 per-token prefill through the sm100 per-expert row with its
    down-B fused into the base GEMM epilogue (the second accumulator, the raw
    bridge), against the same row with the separate grouped down-B and
    against the reference: eight resident slots with adapters in slots 1, 4
    and 7 beside base tokens, an EP hole, a zero token, two tiny tokens (BF16
    amax 2^-133 and 2^-126) whose slot-7 adapters make their delta visible,
    padded segment rows. The bitwise gates of design 6.5: the tokens without
    an adapter and the batch without adapters are bitwise the unfused path;
    one captured prefill graph replayed over the slot, base and mixed
    patterns and over changed routing and inputs is bitwise the eager runs."""
    _skip_unless_sm100()
    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    g = torch.Generator(device="cpu").manual_seed(37)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    tokens, slots, phase = 40, 8, Phase.PREFILL
    quant_info, w13_eff, w2_eff, elem = _quant_info(
        "nvfp4_pt", rand(_E, 2 * _I, _H), rand(_E, _H, _I)
    )
    selected = resolve_plans(
        architecture="sm100",
        is_shared_outer=False,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family="nvfp4",
        vendor="cutedsl",
        act_scale="per_token",
    )[phase]
    # The packaged table serves per-token prefill through the epilogue row;
    # the separate grouped down-B is the comparison under the same tiles.
    assert selected.name == "prefill.per_expert.cutedsl_blockscaled"
    assert selected.plan.down_b.family is BFamily.EPILOGUE
    fused_plan, unfused_plan = selected.plan, _unfused_plan(selected)
    provider = select_provider_cls(selected.base_gemm_rows, "nvfp4", "cutedsl")(
        quant_info
    )
    provider.enable_epilogue_down_lora(rank)
    runner = MoeLoraRunner(
        providers={selected.base_gemm_rows: provider},
        top_k=_TOPK,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFn.SILU,
    )
    for plan in (unfused_plan, fused_plan):
        runner.validate_plan(plan, base_gemm_rows=selected.base_gemm_rows)

    def select(plan):
        runner.plans = {
            phase: SelectedPlan(
                name="test",
                base_gemm_rows=selected.base_gemm_rows,
                plan=plan,
                tiles=selected.tiles,
            )
        }

    hidden0 = rand(tokens, _H, scale=0.5)
    hidden0[0].zero_()
    for t, amax in _TINY_TOKENS.items():
        row = (torch.rand(_H, generator=g) * 2 - 1) * amax
        row[0] = amax
        hidden0[t] = row.to(torch.bfloat16).to(device)
    topk0 = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk0[3, 1] = -1  # an EP hole on an adapter token
    weights = torch.tensor([0.65, 0.35], device=device).repeat(tokens, 1)
    gate_up_a = rand(slots, _E, 2 * rank, _H, scale=0.1)
    gate_up_b = rand(slots, _E, 2 * _I, rank, scale=0.1)
    down_a = rand(slots, _E, rank, _I, scale=0.1)
    down_b = rand(slots, _E, _H, rank, scale=0.1)
    for adapters, scale in (
        (gate_up_a, 2.0**40),
        (gate_up_b, 2.0**40),
        (down_a, 2.0**20),
        (down_b, 2.0**20),
    ):
        adapters[_TINY_SLOT] = (adapters[_TINY_SLOT].float() * scale).to(torch.bfloat16)
    hidden, topk_ids = hidden0.clone(), topk0.clone()
    mapping = torch.empty(tokens, dtype=torch.int32, device=device)
    states = [(name, False) for name in _PER_TOKEN_PATTERNS] + [("mixed", True)]

    def load(state):
        name, flipped = state
        mapping.copy_(_per_token_mapping(name, tokens))
        topk_ids.copy_(topk0.flip(1) if flipped else topk0)
        hidden.copy_(hidden0 * -0.5 if flipped else hidden0)

    def reference(mapping, fused):
        pair_tokens = torch.arange(tokens, device=device).repeat_interleave(_TOPK)
        experts = topk_ids.flatten().long()
        routed = (experts >= 0).float()[:, None]
        experts = experts.clamp(min=0)
        slot = mapping[pair_tokens].long()
        adapted = (slot >= 0).float()[:, None]
        slot = slot.clamp(min=0)
        inputs = hidden.float()[pair_tokens]
        inputs_q = _quant_activation(inputs, elem, "gemm1")
        gate_up = _bf16(torch.einsum("ph,pnh->pn", inputs_q, w13_eff[experts]))
        bridge = _bf16(
            torch.einsum("ph,prh->pr", inputs, gate_up_a.float()[slot, experts])
        )
        inner = gate_up_b.float()[slot, experts]
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
        # The prefill row's fused LoRA-B activation adds the delta in FP32.
        gate_up = gate_up + adapted * delta
        act = _bf16(F.silu(gate_up[:, :_I]) * gate_up[:, _I:])
        act_q = _quant_activation(act, elem, "gemm2")
        base = torch.einsum("pi,phi->ph", act_q, w2_eff[experts])
        down_rank = _bf16(
            torch.einsum("pi,pri->pr", act, down_a.float()[slot, experts])
        )
        delta_d = torch.einsum("pr,phr->ph", down_rank, down_b.float()[slot, experts])
        # The epilogue rounds the FP32 sum once; the separate path rounds the
        # base and the delta each.
        if fused:
            pairs = _bf16(base + adapted * delta_d)
        else:
            pairs = _bf16(base) + adapted * _bf16(delta_d)
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
        is_prefill=True,
    )

    def forward(plan):
        runner.workspace.routes.clear()
        select(plan)
        with get_context().override_server_args():
            out = runner.run(dispatch, batch).hidden_states
        torch.cuda.synchronize()
        return out

    fused = {}
    for state in states:
        load(state)
        unfused_out = forward(unfused_plan)
        out = fused[state] = forward(fused_plan)
        ref = reference(mapping, True)
        assert bool(torch.isfinite(out).all()), state
        assert float(_rel_l2(out.float(), ref)) < _RUNNER_TOL, (state, rank)
        assert float(_rel_l2(unfused_out.float(), reference(mapping, False))) < (
            _RUNNER_TOL
        ), (state, rank)
        assert float(_rel_l2(out.float(), unfused_out.float())) < 4e-3, (state, rank)
        base_tokens = mapping < 0
        assert torch.equal(out[base_tokens], unfused_out[base_tokens]), (state, rank)
        if state[0] == "base":
            assert torch.equal(out, unfused_out), rank
        else:
            assert _rel_l2(reference(torch.full_like(mapping, -1), True), ref) > 0.1
        for t in _TINY_TOKENS:
            row, ref_row = out[t].double(), ref[t].double()
            if ref_row.norm() == 0:
                assert bool((row == 0).all()), (state, t)
            elif int(mapping[t]) >= 0:
                without = mapping.clone()
                without[t] = -1
                base_norm = float(reference(without, True)[t].double().norm())
                assert base_norm < 1e-3 * float(ref_row.norm()), (state, t)
                assert float(_rel_l2(row, ref_row)) < 2.5e-2, (state, t)
            else:
                assert torch.allclose(row, ref_row, rtol=0.1, atol=2.0**-130), (
                    state,
                    t,
                )

    load(states[0])
    batch.use_cuda_graph = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward(fused_plan)
    torch.cuda.synchronize()
    runner.workspace.routes.clear()
    select(fused_plan)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        with get_context().override_server_args():
            captured = runner.run(dispatch, batch).hidden_states
    for state in states + states[:1]:
        load(state)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, fused[state]), (state, rank)


@pytest.mark.parametrize("rows", ["expert_major", "route_major"])
def test_fused_per_token_epilogue_down_both_layouts(rows: str) -> None:
    """The per-token providers' down GEMM with the epilogue LoRA on both row
    layouts (expert slabs, flat segments): the raw bridges of an eight-slot
    pool on the planes the provider hands out, slots 1, 4 and 7 beside base
    pairs, an EP hole; against the reference of the per-token quantizer; the
    pairs without an adapter and a bound of zero bitwise the plain down GEMM;
    an adapter's rows bitwise wherever the pool holds it."""
    _skip_unless_sm100()
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(11)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    rank, slots = 16, 8
    hidden = rand(_T, _H, scale=0.5)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(_T)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[1, 0] = -1  # an EP hole
    quant_info, _, w2_eff, elem = _quant_info(
        "nvfp4_pt", rand(_E, 2 * _I, _H), rand(_E, _H, _I)
    )
    provider = select_provider_cls(rows, "nvfp4", "cutedsl")(quant_info)
    provider.enable_epilogue_down_lora(rank)
    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, _TOPK)
    flat_ids = topk_ids.flatten()
    pair_idx = torch.nonzero(flat_ids >= 0, as_tuple=True)[0]
    experts = flat_ids[pair_idx].long()
    tokens = (pair_idx // _TOPK).long()
    pair_to_row = state.pair_to_row
    act_pairs = (torch.randn(len(pair_idx), _I, generator=g) * 0.1).to(
        device=device, dtype=torch.bfloat16
    )
    act = torch.zeros(
        provider.act_out_shape(state), dtype=torch.bfloat16, device=device
    )
    act.reshape(-1, _I)[pair_to_row[pair_idx].long()] = act_pairs
    down_b = rand(slots, _E, _H, rank, scale=0.1)
    bridge = (torch.randn(len(pair_idx), rank, generator=g) * 0.5).to(
        device=device, dtype=torch.bfloat16
    )
    slot_of_token = torch.tensor([1, -1, 4, -1, 7, -1], device=device)[
        torch.arange(_T, device=device) % 6
    ]
    slot_of_pair = slot_of_token[tokens]

    def operands(placement):
        planes = provider.epilogue_down_lora_planes(
            state, None, max_loras=len(placement), rank=rank
        )
        weights = torch.empty(
            (len(placement), _E, _H, rank), dtype=torch.bfloat16, device=device
        )
        for p, s in enumerate(placement):
            weights[p] = down_b[p if s is None else s]
            if s is not None:
                sel = slot_of_pair == s
                planes[p, pair_to_row[pair_idx[sel]].long(), :rank] = bridge[sel]
        bound = torch.tensor([len(placement)], dtype=torch.int32, device=device)
        return weights.flatten(0, 1), planes, bound

    def run(lora):
        down = torch.zeros(
            provider.down_out_shape(state), dtype=torch.bfloat16, device=device
        )
        with get_context().override_server_args():
            provider.down(state, act, down, lora=lora)
        torch.cuda.synchronize()
        return _gather_pairs(down, pair_to_row, pair_idx)

    plain = run(None)
    ref_base = torch.einsum(
        "pi,phi->ph", _quant_activation(act_pairs, elem, "gemm2"), w2_eff[experts]
    )
    assert float(_rel_l2(plain.float(), ref_base)) < 1.5e-2, rows

    def reference(placement):
        placed = torch.zeros_like(slot_of_pair, dtype=torch.bool)
        for s in placement:
            if s is not None:
                placed |= slot_of_pair == s
        delta = torch.einsum(
            "pr,phr->ph",
            bridge.float(),
            down_b.float()[slot_of_pair.clamp(min=0), experts],
        )
        return ref_base + delta * placed.float()[:, None]

    weights, planes, _ = operands([0])
    zero_bound = torch.zeros(1, dtype=torch.int32, device=device)
    assert torch.equal(run((weights, planes, zero_bound)), plain), rows
    alone = {s: run(operands([s])) for s in (1, 4, 7)}
    for s, out in alone.items():
        assert float(_rel_l2(out.float(), reference([s]))) < 1.5e-2, (rows, s)
    full = run(operands(list(range(slots))))
    assert bool(torch.isfinite(full).all())
    assert float(_rel_l2(full.float(), reference(list(range(slots))))) < 1.5e-2, rows
    base = slot_of_pair < 0
    assert torch.equal(full[base], plain[base]), rows
    assert not torch.equal(full[~base], plain[~base]), rows
    for s, out in alone.items():
        sel = slot_of_pair == s
        assert torch.equal(full[sel], out[sel]), (rows, s)


@pytest.mark.parametrize("family", ["nvfp4", "nvfp4_pt"])
def test_fused_nvfp4_epilogue_refuses_wide_tiles(family: str) -> None:
    """An NVFP4 provider compiling a 256-wide token tile cannot take the
    epilogue down-B in either activation-scale mode: the second accumulator
    does not fit in TMEM beside that tile's base accumulator, so plan binding
    is refused with the reason."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import GroupedGemmConfig

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(5)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    quant_info, _, _, _ = _quant_info(family, rand(_E, 2 * _I, _H), rand(_E, _H, _I))
    provider = select_provider_cls("expert_major", "nvfp4", "cutedsl")(quant_info)
    provider._tile_configs[256] = GroupedGemmConfig(
        mma_tiler_mn=(128, 256), persistent_clusters=2
    )
    with pytest.raises(ValueError, match="256"):
        provider.enable_epilogue_down_lora(16)
    assert provider.epilogue_down_lora_rank() == 0


# Down scales of nvidia/Qwen3.5-122B-A10B-NVFP4 (snapshot 98915d83): layer 6's
# activation scale with expert 188's weight global scale as stored (their fp32
# product, the epilogue's alpha, is a subnormal whose reciprocal is infinite),
# weight scales that give the alphas of two more near-dead experts (finite
# reciprocals that overflow a BF16 bridge past |z| = 5 and 43), and the
# checkpoint's median alpha.
_STATIC_A2_GLOBAL = 0.008603050373494625
_STATIC_W2_GLOBAL = (2.0711090409231365e-37, 1.75e-36, 1.47e-35, 2.07e-5)
_STATIC_LARGE_SLOT = 4  # its down-A is 512 times larger: bridges in the hundreds
# Base only, adapters only, base only again, mixed batches, base only.
_STATIC_STATES = ("base", "high", "base", "mixed", "slot4", "slot1", "base")


def _row_rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Relative L2 distance of each row, in FP64."""
    a, b = a.double(), b.double()
    return (a - b).norm(dim=1) / b.norm(dim=1).clamp(min=1e-300)


@pytest.mark.parametrize("rank", [16, 8])
def test_fused_static_epilogue_plan_at_the_checkpoint_scale_extremes(
    monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    """NVFP4 static prefill through the sm100 per-expert row with its down-B
    fused into the base GEMM epilogue, at the down scales of a served
    checkpoint: an expert whose alpha is a subnormal fp32, two whose alpha
    has a finite reciprocal that overflows a BF16 bridge, and an ordinary
    one. Eight resident slots with adapters in slots 1, 4 and 7 beside base
    tokens, one slot with bridges in the hundreds, an EP hole. A bridge divided
    by alpha is not finite on the first three experts; the raw bridge and the
    second accumulator keep every output finite. The numerical gates are
    relative L2 distances, not elementwise BF16-ulp bounds: from the
    reference below 1e-2 per token class (``_RUNNER_TOL``); from the separate
    grouped down-B below 4e-3 over the batch and below 1.5e-2 on every token.
    Base tokens and base-only batches are bitwise the unfused path, also
    after adapter batches; one captured prefill graph replayed over the
    whole sequence is bitwise the eager runs."""
    _skip_unless_sm100()
    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    g = torch.Generator(device="cpu").manual_seed(41)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    tokens, slots, phase = 40, 8, Phase.PREFILL
    quant_info, w13_eff, w2_eff, elem = _quant_info(
        "nvfp4",
        rand(_E, 2 * _I, _H),
        rand(_E, _H, _I),
        w2_global=_STATIC_W2_GLOBAL,
        a2_global=_STATIC_A2_GLOBAL,
    )
    selected = resolve_plans(
        architecture="sm100",
        is_shared_outer=False,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=_H,
        num_local_experts=_E,
        quant_family="nvfp4",
        vendor="cutedsl",
    )[phase]
    assert selected.name == "prefill.per_expert.cutedsl_blockscaled"
    assert selected.plan.down_b.family is BFamily.EPILOGUE
    fused_plan, unfused_plan = selected.plan, _unfused_plan(selected)
    provider = select_provider_cls(selected.base_gemm_rows, "nvfp4", "cutedsl")(
        quant_info
    )
    provider.enable_epilogue_down_lora(rank)
    inverse = 1.0 / provider._alpha["gemm2"].reshape(-1)
    assert not bool(torch.isfinite(inverse[0])) and bool(
        torch.isfinite(inverse[1:]).all()
    )
    runner = MoeLoraRunner(
        providers={selected.base_gemm_rows: provider},
        top_k=_TOPK,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFn.SILU,
    )
    for plan in (unfused_plan, fused_plan):
        runner.validate_plan(plan, base_gemm_rows=selected.base_gemm_rows)

    def select(plan):
        runner.plans = {
            phase: SelectedPlan(
                name="test",
                base_gemm_rows=selected.base_gemm_rows,
                plan=plan,
                tiles=selected.tiles,
            )
        }

    hidden0 = rand(tokens, _H, scale=0.5)
    topk0 = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk0[3, 1] = -1  # an EP hole on an adapter token
    weights = torch.tensor([0.65, 0.35], device=device).repeat(tokens, 1)
    gate_up_a = rand(slots, _E, 2 * rank, _H, scale=0.1)
    gate_up_b = rand(slots, _E, 2 * _I, rank, scale=0.1)
    down_a = rand(slots, _E, rank, _I, scale=0.1)
    down_b = rand(slots, _E, _H, rank, scale=0.1)
    down_a[_STATIC_LARGE_SLOT] = (down_a[_STATIC_LARGE_SLOT].float() * 512).to(
        torch.bfloat16
    )
    hidden, topk_ids = hidden0.clone(), topk0.clone()
    mapping = torch.empty(tokens, dtype=torch.int32, device=device)
    states = [(i, name, i % 2 == 1) for i, name in enumerate(_STATIC_STATES)]

    def load(state):
        _, name, flipped = state
        mapping.copy_(_per_token_mapping(name, tokens))
        topk_ids.copy_(topk0.flip(1) if flipped else topk0)
        hidden.copy_(hidden0 * -0.5 if flipped else hidden0)

    def reference(mapping, fused):
        """The token outputs and, per pair, the BF16 down bridge, the expert
        and whether an adapter applies."""
        pair_tokens = torch.arange(tokens, device=device).repeat_interleave(_TOPK)
        experts = topk_ids.flatten().long()
        routed = (experts >= 0).float()[:, None]
        experts = experts.clamp(min=0)
        slot = mapping[pair_tokens].long()
        adapted = (slot >= 0).float()[:, None]
        slot = slot.clamp(min=0)
        inputs = hidden.float()[pair_tokens]
        inputs_q = _quant_activation(inputs, elem, "gemm1")
        gate_up = _bf16(torch.einsum("ph,pnh->pn", inputs_q, w13_eff[experts]))
        bridge = _bf16(
            torch.einsum("ph,prh->pr", inputs, gate_up_a.float()[slot, experts])
        )
        inner = gate_up_b.float()[slot, experts]
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
        gate_up = gate_up + adapted * delta
        act = _bf16(F.silu(gate_up[:, :_I]) * gate_up[:, _I:])
        act_q = _quant_activation(act, elem, "gemm2", _STATIC_A2_GLOBAL)
        base = torch.einsum("pi,phi->ph", act_q, w2_eff[experts])
        down_rank = _bf16(
            torch.einsum("pi,pri->pr", act, down_a.float()[slot, experts])
        )
        delta_d = torch.einsum("pr,phr->ph", down_rank, down_b.float()[slot, experts])
        if fused:
            pairs = _bf16(base + adapted * delta_d)
        else:
            pairs = _bf16(base) + adapted * _bf16(delta_d)
        pairs = pairs * routed
        out = (pairs.view(tokens, _TOPK, _H) * weights[..., None]).sum(
            1
        ) * _ROUTED_SCALING
        return out, down_rank, experts, (adapted * routed).flatten() > 0

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
        is_prefill=True,
    )

    def forward(plan):
        runner.workspace.routes.clear()
        select(plan)
        with get_context().override_server_args():
            out = runner.run(dispatch, batch).hidden_states
        torch.cuda.synchronize()
        return out

    fused = {}
    hazards = torch.zeros(_E, dtype=torch.bool, device=device)
    for state in states:
        load(state)
        tag = (state, rank)
        unfused_out = forward(unfused_plan)
        out = fused[state] = forward(fused_plan)
        ref, bridge, experts, adapted = reference(mapping, True)
        assert bool(torch.isfinite(out).all()), tag
        assert bool(torch.isfinite(unfused_out).all()), tag
        compensated = (bridge * inverse[experts][:, None]).to(torch.bfloat16)
        bad = ~torch.isfinite(compensated).all(dim=1) & adapted
        hazards |= torch.zeros_like(hazards).index_put_((experts[bad],), bad[bad])
        for slot in sorted(set(mapping.tolist())):
            rows = mapping == slot
            assert float(_rel_l2(out[rows].float(), ref[rows])) < _RUNNER_TOL, (
                *tag,
                slot,
            )
        assert float(_rel_l2(out.float(), unfused_out.float())) < 4e-3, tag
        assert float(_row_rel(out, unfused_out).max()) < 1.5e-2, tag
        base_tokens = mapping < 0
        assert torch.equal(out[base_tokens], unfused_out[base_tokens]), tag
        if state[1] == "base":
            assert torch.equal(out, unfused_out), tag
        else:
            without, _, _, _ = reference(torch.full_like(mapping, -1), True)
            assert _rel_l2(without, ref) > 0.1, tag
    # An inverse-alpha bridge was not finite on the three near-dead experts.
    assert hazards.tolist() == [True, True, True, False], rank
    # A base-only batch is the same batch before and after adapter batches.
    base_states = [state for state in states if state[1] == "base"]
    for state in base_states[1:]:
        if state[2] == base_states[0][2]:
            assert torch.equal(fused[state], fused[base_states[0]]), (state, rank)

    load(states[0])
    batch.use_cuda_graph = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward(fused_plan)
    torch.cuda.synchronize()
    runner.workspace.routes.clear()
    select(fused_plan)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        with get_context().override_server_args():
            captured = runner.run(dispatch, batch).hidden_states
    for state in states + states[:1]:
        load(state)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, fused[state]), (state, rank)


@pytest.mark.parametrize("rows", ["expert_major", "route_major"])
def test_fused_static_epilogue_down_both_layouts_at_the_scale_extremes(
    rows: str,
) -> None:
    """The static NVFP4 providers' down GEMM with the epilogue LoRA on both
    row layouts (expert slabs, flat segments) at the checkpoint's down scales
    (a subnormal alpha, two near it, an ordinary one): the raw bridges of an
    eight-slot pool, slots 1, 4 and 7 beside base pairs, slot 4's bridges 128
    times larger and every other slot-7 bridge zero, an EP hole. Every pair
    is finite and each expert's pairs are within 1.5e-2 relative L2 of the
    reference; the pairs without an adapter, the pairs with a zero bridge,
    zero weights and a bound of zero are bitwise the plain down GEMM; an
    adapter's rows are bitwise the same wherever the pool holds it."""
    _skip_unless_sm100()
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(43)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    rank, slots = 16, 8
    hidden = rand(_T, _H, scale=0.5)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(_T)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[1, 0] = -1  # an EP hole
    quant_info, _, w2_eff, elem = _quant_info(
        "nvfp4",
        rand(_E, 2 * _I, _H),
        rand(_E, _H, _I),
        w2_global=_STATIC_W2_GLOBAL,
        a2_global=_STATIC_A2_GLOBAL,
    )
    provider = select_provider_cls(rows, "nvfp4", "cutedsl")(quant_info)
    provider.enable_epilogue_down_lora(rank)
    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, _TOPK)
    flat_ids = topk_ids.flatten()
    pair_idx = torch.nonzero(flat_ids >= 0, as_tuple=True)[0]
    experts = flat_ids[pair_idx].long()
    tokens = (pair_idx // _TOPK).long()
    pair_to_row = state.pair_to_row
    act_pairs = (torch.randn(len(pair_idx), _I, generator=g) * 0.1).to(
        device=device, dtype=torch.bfloat16
    )
    act = torch.zeros(
        provider.act_out_shape(state), dtype=torch.bfloat16, device=device
    )
    act.reshape(-1, _I)[pair_to_row[pair_idx].long()] = act_pairs
    down_b = rand(slots, _E, _H, rank, scale=0.1)
    slot_of_token = torch.tensor([1, -1, 4, -1, 7, -1], device=device)[
        torch.arange(_T, device=device) % 6
    ]
    slot_of_pair = slot_of_token[tokens]
    bridge = (torch.randn(len(pair_idx), rank, generator=g) * 0.5).to(device)
    bridge[slot_of_pair == 4] *= 128
    zeroed = (slot_of_pair == 7) & (tokens % 12 == 4)
    bridge[zeroed] = 0
    bridge = bridge.to(torch.bfloat16)
    assert bool(zeroed.any()) and bool(((slot_of_pair == 7) & ~zeroed).any())

    def operands(placement, weights_of=down_b):
        planes = provider.epilogue_down_lora_planes(
            state, None, max_loras=len(placement), rank=rank
        )
        weights = torch.empty(
            (len(placement), _E, _H, rank), dtype=torch.bfloat16, device=device
        )
        for p, s in enumerate(placement):
            weights[p] = weights_of[p if s is None else s]
            if s is not None:
                sel = slot_of_pair == s
                planes[p, pair_to_row[pair_idx[sel]].long(), :rank] = bridge[sel]
        bound = torch.tensor([len(placement)], dtype=torch.int32, device=device)
        return weights.flatten(0, 1), planes, bound

    def run(lora):
        down = torch.zeros(
            provider.down_out_shape(state), dtype=torch.bfloat16, device=device
        )
        with get_context().override_server_args():
            provider.down(state, act, down, lora=lora)
        torch.cuda.synchronize()
        return _gather_pairs(down, pair_to_row, pair_idx)

    ref_base = torch.einsum(
        "pi,phi->ph",
        _quant_activation(act_pairs, elem, "gemm2", _STATIC_A2_GLOBAL).double(),
        w2_eff[experts].double(),
    )
    delta = torch.einsum(
        "pr,phr->ph",
        bridge.double(),
        down_b.double()[slot_of_pair.clamp(min=0), experts],
    )

    def check(out, placement, name):
        """Finite, and each expert's pairs within 1.5e-2 relative L2 of the
        reference."""
        assert bool(torch.isfinite(out).all()), (rows, name)
        placed = torch.zeros_like(slot_of_pair, dtype=torch.bool)
        for s in placement:
            if s is not None:
                placed |= slot_of_pair == s
        ref = ref_base + delta * placed.double()[:, None]
        for e in range(_E):
            sel = experts == e
            err = (out.double()[sel] - ref[sel]).norm() / ref[sel].norm()
            assert float(err) < 1.5e-2, (rows, name, e, float(err))

    # The fixture reaches the hazards of an inverse-alpha bridge.
    inverse = 1.0 / provider._alpha["gemm2"].reshape(-1)
    compensated = (bridge.float() * inverse[experts][:, None]).to(torch.bfloat16)
    bad = ~torch.isfinite(compensated).all(dim=1) & (slot_of_pair >= 0)
    assert [bool(bad[experts == e].any()) for e in range(_E)] == [
        True,
        True,
        True,
        False,
    ], rows

    plain = run(None)
    check(plain, [], "plain")
    weights, planes, _ = operands([0])
    zero_bound = torch.zeros(1, dtype=torch.int32, device=device)
    assert torch.equal(run((weights, planes, zero_bound)), plain), rows
    every = list(range(slots))
    full = run(operands(every))
    check(full, every, "full")
    quiet = (slot_of_pair < 0) | zeroed
    assert torch.equal(full[quiet], plain[quiet]), rows
    assert not torch.equal(full[~quiet], plain[~quiet]), rows
    assert torch.equal(run(operands(every, torch.zeros_like(down_b))), plain), rows
    for s in (1, 4, 7):
        out = run(operands([s]))
        check(out, [s], f"alone{s}")
        sel = slot_of_pair == s
        assert torch.equal(full[sel], out[sel]), (rows, s)
        assert torch.equal(out[~sel], plain[~sel]), (rows, s)
