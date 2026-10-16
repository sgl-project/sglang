"""The spec-decode verify (T = 8 and 16) expert-TP4 NVFP4 MoE megakernel and its
post-FFN tail against full-expert references: FP64 GEMMs, FlashInfer's static
FC2 requant recipe, the stock weight-first post-expert norm finalize, and the
stock post-FFN norm, gated residual and next pre-attention norm.

Run with ``python <this file>``; it relaunches itself under torchrun on 4 GPUs.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import os

import msgspec
import pytest
import torch
import torch.distributed as dist

from sglang.srt.layers.quantization.utils import (
    prepare_static_weights_for_trtllm_fp4_moe,
    reorder_w1w3_to_w3w1,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

try:
    import helion
except ModuleNotFoundError as error:
    if error.name != "helion":
        raise
    HELION_MEGAKERNELS = False
else:
    HELION_MEGAKERNELS = (
        "cross_loop_pipeline" in inspect.signature(helion.Config).parameters
    )
    from sglang.kernels.ops.muse_spark.helion_megakernels.moe_tp4_verify_megakernel import (
        MoeTp4VerifyBank,
        MuseMoeTailVerifyMegakernel,
        MuseMoeTp4VerifyMegakernel,
        build_moe_tp4_verify_bank,
    )

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = [
    pytest.mark.skipif(
        not HELION_MEGAKERNELS, reason="requires Helion with cross-loop pipelining"
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="the megakernels are specialized for SM10x",
    ),
]

EXPERTS, HIDDEN, INTERMEDIATE, TOPK = 256, 4096, 4096, 8
MODEL_HIDDEN = 8192
QKV_OUT = 4608  # one rank's Q, K, V and gate rows
NORM_EPS, POST_EPS, PRE_EPS = 1e-8, 1e-8, 1e-5
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FC2_GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)
BF16_MAX = torch.finfo(torch.bfloat16).max
FP32_MAX = torch.finfo(torch.float32).max
ROUTE_SEEDS = {"uniform": 1, "shared": 2, "same8": 3, "disjoint": 4}
REPEATS = 4


class LogicalExperts(msgspec.Struct, frozen=True):
    """Checkpoint-order expert weights for the loaded expert ids."""

    ids: list[int]
    w13: torch.Tensor  # [n, 2I, H/2] u8, rows [w1 (gate); w3 (up)]
    w13_scale: torch.Tensor  # [n, 2I, H/16] e4m3
    w13_scale_2: torch.Tensor  # [n, 2] f32 (gate, up)
    w2: torch.Tensor  # [n, H, I/2] u8
    w2_scale: torch.Tensor  # [n, H, I/16] e4m3
    w2_scale_2: torch.Tensor  # [n] f32


class RankWeights(msgspec.Struct, frozen=True):
    """One rank's FI TRT-LLM layout slices, [256, ...] with the loaded ids set."""

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


class Activation(msgspec.Struct, frozen=True):
    hidden_states: torch.Tensor  # [T, H/2] u8
    hidden_states_scale: torch.Tensor  # [T, H/16] e4m3
    per_token_scale: torch.Tensor  # [T] f32


class RoutedRows(msgspec.Struct, frozen=True):
    """One loaded expert's routes in a draw and their FP32 SwiGLU rows."""

    slot: int  # index into LogicalExperts.ids
    tokens: torch.Tensor  # [n]
    topk_slots: torch.Tensor  # [n]
    act: torch.Tensor  # [n, I] f32, true scale (no FC2 input scale)


class Draw(msgspec.Struct, frozen=True):
    """One T-token routing with its activation and reference SwiGLU rows."""

    kind: str
    packed: torch.Tensor  # [T, 8] i32 PackedScoreIdx
    x: Activation
    rows: list[RoutedRows]


class Case(msgspec.Struct, frozen=True):
    """One bank holding every draw's experts, and the kernel's FC scalars."""

    logical: LogicalExperts
    bank: MoeTp4VerifyBank
    fc2_scale: float
    alphas: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # up, gate, out [E]
    norm_weight: torch.Tensor  # [H] bf16
    draws: list[Draw]


class Boundary(msgspec.Struct, frozen=True):
    """Post-FFN boundary tensors, identical on every rank."""

    unpool: torch.Tensor  # [D, H] bf16
    stream: torch.Tensor  # [T, D] f32
    alpha: torch.Tensor
    beta: torch.Tensor
    post_weight: torch.Tensor
    pre_weight: torch.Tensor
    qkv: torch.Tensor  # [Nq, D] bf16, the next layer's QKV rows


@functools.cache
def _init_group_once() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    atexit.register(dist.destroy_process_group)
    return dist.group.WORLD


@functools.cache
def _megakernel_once(tokens: int) -> MuseMoeTp4VerifyMegakernel:
    # One per T, as served: a T-token window leaves its slot's pad rows alone.
    return MuseMoeTp4VerifyMegakernel(
        torch.device("cuda", torch.cuda.current_device()),
        _init_group_once(),
        tokens=tokens,
        norm_eps=NORM_EPS,
    )


@functools.cache
def _tail_once(tokens: int) -> MuseMoeTailVerifyMegakernel:
    return MuseMoeTailVerifyMegakernel(
        torch.device("cuda", torch.cuda.current_device()),
        _init_group_once(),
        tokens=tokens,
        post_eps=POST_EPS,
        pre_eps=PRE_EPS,
    )


def _routes(kind: str, tokens: int) -> torch.Tensor:
    """[T, 8] distinct experts per token. same8: all tokens route one set of 8;
    shared: subsets of a 20-expert pool; uniform: any 8; disjoint: 128 distinct."""

    g = torch.Generator().manual_seed(ROUTE_SEEDS[kind])
    order = torch.randperm(EXPERTS, generator=g)
    if kind == "disjoint":
        return order[: tokens * TOPK].view(tokens, TOPK)
    pool = order[: {"same8": TOPK, "shared": 20, "uniform": EXPERTS}[kind]]
    picks = [
        pool[torch.randperm(pool.numel(), generator=g)[:TOPK]] for _ in range(tokens)
    ]
    return torch.stack(picks)


def _random_experts(ids: list[int], seed: int) -> LogicalExperts:
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = len(ids)

    def scales(*shape: int) -> torch.Tensor:
        # E4M3 values in [2^-9, 2^-5): a spread of exponents, no zeros.
        exponent = torch.randint(-9, -5, shape, device="cuda", generator=g)
        mantissa = 1 + torch.rand(shape, device="cuda", generator=g)
        return (mantissa * torch.exp2(exponent.float())).to(torch.float8_e4m3fn)

    def nibbles(*shape: int) -> torch.Tensor:
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    return LogicalExperts(
        ids=ids,
        w13=nibbles(n, 2 * INTERMEDIATE, HIDDEN // 2),
        w13_scale=scales(n, 2 * INTERMEDIATE, HIDDEN // 16),
        w13_scale_2=0.5 + torch.rand(n, 2, device="cuda", generator=g),
        w2=nibbles(n, HIDDEN, INTERMEDIATE // 2),
        w2_scale=scales(n, HIDDEN, INTERMEDIATE // 16),
        w2_scale_2=0.5 + torch.rand(n, device="cuda", generator=g),
    )


def _rank_weights(logical: LogicalExperts, rank: int, world: int) -> RankWeights:
    """Expert-TP narrow of the checkpoint tensors, then the served FI prep."""

    local = INTERMEDIATE // world
    gate = slice(rank * local, (rank + 1) * local)
    up = slice(INTERMEDIATE + rank * local, INTERMEDIATE + (rank + 1) * local)
    w13 = torch.cat((logical.w13[:, gate], logical.w13[:, up]), dim=1)
    w13_scale = torch.cat((logical.w13_scale[:, gate], logical.w13_scale[:, up]), dim=1)
    k2 = slice(rank * local // 2, (rank + 1) * local // 2)
    w2 = logical.w2[:, :, k2].contiguous()
    w2_scale = logical.w2_scale[:, :, rank * local // 16 : (rank + 1) * local // 16]
    w13, w13_scale = reorder_w1w3_to_w3w1(w13, w13_scale)
    w13, w13_scale, w2, w2_scale = prepare_static_weights_for_trtllm_fp4_moe(
        w13,
        w2,
        w13_scale,
        w2_scale.contiguous(),
        HIDDEN,
        local,
        len(logical.ids),
        is_gated=True,
    )
    ids = torch.tensor(logical.ids, device="cuda")

    def scatter(values: torch.Tensor, *shape: int) -> torch.Tensor:
        full = torch.empty((EXPERTS, *shape), dtype=torch.uint8, device="cuda")
        full[ids] = values.view(torch.uint8).reshape(len(ids), *shape)
        return full

    e4m3 = torch.float8_e4m3fn
    return RankWeights(
        w13=scatter(w13, 2 * local, HIDDEN // 2),
        w13_scale=scatter(w13_scale, 2 * local, HIDDEN // 16).view(e4m3),
        w2=scatter(w2, HIDDEN, local // 2),
        w2_scale=scatter(w2_scale, HIDDEN, local // 16).view(e4m3),
    )


def _kernel_alphas(
    logical: LogicalExperts, fc2_scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(up, gate, out) FP32 alphas; as served, the static FC2 input scale is
    folded into up and out, so the kernel quantizes z with a unit global scale."""

    ids = torch.tensor(logical.ids, device="cuda")
    up, gate, out = (torch.ones(EXPERTS, device="cuda") for _ in range(3))
    up[ids] = logical.w13_scale_2[:, 1] / fc2_scale
    gate[ids] = logical.w13_scale_2[:, 0]
    out[ids] = logical.w2_scale_2 * fc2_scale
    return up, gate, out


def _random_activation(seed: int, tokens: int) -> Activation:
    g = torch.Generator(device="cuda").manual_seed(seed)
    exponent = torch.randint(-8, -4, (tokens, HIDDEN // 16), device="cuda", generator=g)
    return Activation(
        hidden_states=torch.randint(
            0, 256, (tokens, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=g
        ),
        hidden_states_scale=torch.exp2(exponent.float()).to(torch.float8_e4m3fn),
        per_token_scale=0.5 + torch.rand(tokens, device="cuda", generator=g),
    )


def _pack_routes(topk_ids: torch.Tensor, seed: int) -> torch.Tensor:
    """FlashInfer PackedScoreIdx words: expert id << 16 | BF16 route weight."""

    g = torch.Generator(device="cuda").manual_seed(seed)
    raw = 0.25 + torch.rand(len(topk_ids), TOPK, device="cuda", generator=g)
    scores = (raw / raw.sum(-1, keepdim=True)).bfloat16().view(torch.int16)
    expert = topk_ids.to(torch.int32).cuda()
    return (expert << 16) | (scores.to(torch.int32) & 0xFFFF)


def _boundary(tokens: int) -> Boundary:
    g = torch.Generator(device="cuda").manual_seed(11)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", generator=g)

    return Boundary(
        unpool=(randn(MODEL_HIDDEN, HIDDEN) / 64.0).bfloat16(),
        stream=randn(tokens, MODEL_HIDDEN) * 4.0,
        alpha=1.0 + 0.1 * randn(MODEL_HIDDEN),
        beta=0.5 + 0.1 * randn(MODEL_HIDDEN),
        post_weight=(1.0 + 0.1 * randn(MODEL_HIDDEN)).bfloat16(),
        pre_weight=(1.0 + 0.1 * randn(MODEL_HIDDEN)).bfloat16(),
        qkv=(randn(QKV_OUT, MODEL_HIDDEN) / 90.0).bfloat16(),
    )


def _dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP64 values of NVFP4 bytes [..., K/2] (low nibble first), E4M3 [..., K/16]."""

    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float64).cuda()
    values = torch.stack((lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1)
    values = values.reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return values * scale.view(torch.float8_e4m3fn).double().repeat_interleave(16, -1)


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """e2m1 round-to-nearest-even, saturating at 6."""

    a = x.abs()
    index = sum(
        (a >= b if odd_below else a > b).int()
        for b, odd_below in (
            (0.25, False),
            (0.75, True),
            (1.25, False),
            (1.75, True),
            (2.5, False),
            (3.5, True),
            (5.0, False),
        )
    )
    return torch.tensor(E2M1, device=x.device)[index] * torch.sign(x)


def _quant_unit(act: torch.Tensor) -> torch.Tensor:
    """NVFP4 with a unit global scale (the FC2 input scale is pre-applied),
    dequantized to FP64: the kernel's FC2 input."""

    blocks = act.float().reshape(act.shape[0], -1, 16)
    fp8 = (blocks.abs().amax(-1) / 6.0).clamp(max=448.0)
    fp8 = fp8.to(torch.float8_e4m3fn).float()
    x = blocks * torch.where(fp8 != 0, 1.0 / fp8, 0.0)[..., None]
    return (_round_e2m1(x) * fp8[..., None]).reshape(act.shape).double()


def _quant_full_row(act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer's dynamic BF16 -> NVFP4 FC2 requant of full rows (the B=1
    recipe), dequantized: (FP64 values [R, I], FP32 token scale [R])."""

    act = act.clamp(-BF16_MAX, BF16_MAX).bfloat16()
    blocks = act.float().reshape(act.shape[0], -1, 16)
    block_amax = blocks.abs().amax(-1)
    token_scale = block_amax.amax(-1, keepdim=True) * FC2_GLOBAL_SCALE_INV
    encode = torch.where(token_scale != 0, 1.0 / token_scale, FP32_MAX)
    fp8 = (encode * (block_amax * (1.0 / 6.0))).clamp(max=448.0)
    fp8 = fp8.to(torch.float8_e4m3fn).float()
    out_scale = torch.where(fp8 != 0, 1.0 / (fp8 * (1.0 / encode)), 0.0)
    x = blocks * out_scale[..., None]
    values = (_round_e2m1(x) * fp8[..., None]).reshape(act.shape).double()
    return values, token_scale.view(-1)


def _swiglu_rows(
    logical: LogicalExperts, topk_ids: torch.Tensor, x: Activation
) -> list[RoutedRows]:
    """FP64 W13 GEMMs of each routed expert's tokens, then the FP32 SwiGLU with
    the per-token scale and the unfolded W13 alphas."""

    x_deq = _dequant(x.hidden_states, x.hidden_states_scale)
    topk_ids = topk_ids.cuda()
    out = []
    for slot, expert in enumerate(logical.ids):
        tokens, topk_slots = (topk_ids == expert).nonzero().unbind(-1)
        if tokens.numel() == 0:
            continue
        w13 = _dequant(logical.w13[slot], logical.w13_scale[slot])
        acc = (x_deq[tokens] @ w13.T).float().clamp(-FP32_MAX, FP32_MAX)
        scale = x.per_token_scale[tokens][:, None] * logical.w13_scale_2[slot]
        gate = acc[:, :INTERMEDIATE] * scale[:, 0:1]
        up = acc[:, INTERMEDIATE:] * scale[:, 1:2]
        act = gate / (1.0 + torch.exp(-gate)) * up
        out.append(RoutedRows(slot=slot, tokens=tokens, topk_slots=topk_slots, act=act))
    return out


def _make_case(kinds: tuple[str, ...], group: dist.ProcessGroup, tokens: int) -> Case:
    """Every draw's experts in one bank; the static FC2 input scale is calibrated
    on the routed rows of all draws, as a checkpoint's input scale would be."""

    topk = [_routes(kind, tokens) for kind in kinds]
    ids = sorted(set().union(*(t.unique().tolist() for t in topk)))
    logical = _random_experts(ids, seed=7)
    weights = _rank_weights(logical, group.rank(), group.size())
    bank = build_moe_tp4_verify_bank(
        w13=weights.w13,
        w13_scale=weights.w13_scale,
        w2=weights.w2,
        w2_scale=weights.w2_scale,
        group=group,
    )
    del weights
    draws = []
    for i, (kind, topk_ids) in enumerate(zip(kinds, topk)):
        x = _random_activation(100 + ROUTE_SEEDS[kind], tokens)
        rows = _swiglu_rows(logical, topk_ids, x)
        packed = _pack_routes(topk_ids, seed=5 + i)
        draws.append(Draw(kind=kind, packed=packed, x=x, rows=rows))
    amax = max(r.act.abs().max().item() for d in draws for r in d.rows)
    fc2_scale = amax * FC2_GLOBAL_SCALE_INV
    g = torch.Generator(device="cuda").manual_seed(7)
    norm_weight = 1.0 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=g)
    return Case(
        logical=logical,
        bank=bank,
        fc2_scale=fc2_scale,
        alphas=_kernel_alphas(logical, fc2_scale),
        norm_weight=norm_weight.bfloat16(),
        draws=draws,
    )


def _gemm2_refs(case: Case, draw: Draw) -> dict[str, torch.Tensor]:
    """Routed W2 rows [T * 8, H] BF16 per FC2 input: the kernel's static recipe,
    FlashInfer's full-row dynamic requant, and exact (no requant)."""

    logical, fc2 = case.logical, case.fc2_scale
    tokens = len(draw.packed)
    shape = (tokens, TOPK, HIDDEN)
    out = {
        mode: torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        for mode in ("static", "full_row", "exact")
    }
    for r in draw.rows:
        w2 = _dequant(logical.w2[r.slot], logical.w2_scale[r.slot])
        alpha = float(logical.w2_scale_2[r.slot])
        full_row, token_scale = _quant_full_row(r.act)
        inputs = {
            "static": (_quant_unit(r.act / fc2), fc2),
            "full_row": (full_row, token_scale.double()[:, None]),
            "exact": (r.act.double(), 1.0),
        }
        for mode, (z, scale) in inputs.items():
            g2 = ((z @ w2.T) * (scale * alpha)).clamp(-BF16_MAX, BF16_MAX)
            out[mode][r.tokens, r.topk_slots] = g2.bfloat16()
    return {mode: v.view(tokens * TOPK, HIDDEN) for mode, v in out.items()}


def _finalize(
    gemm2: torch.Tensor, packed: torch.Tensor, norm_weight: torch.Tensor
) -> torch.Tensor:
    """Stock weight-first post-expert RMSNorm per route, then the weighted
    route sum: [T, H] BF16."""

    weights = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16).double()
    rows = gemm2.view(len(packed), TOPK, HIDDEN).float() * norm_weight.float()
    rows = rows.double()
    rstd = torch.rsqrt(rows.pow(2).mean(-1, keepdim=True) + NORM_EPS)
    return (weights[..., None] * rows * rstd).sum(1).float().bfloat16()


def _stock_tail(fin: torch.Tensor, b: Boundary) -> tuple[torch.Tensor, torch.Tensor]:
    """Stock post-FFN per token: BF16 unpool GEMM, input-scale post norm, gate,
    pre norm. Returns (new stream FP32, next attention input BF16)."""

    branch = (fin.double() @ b.unpool.double().T).bfloat16()
    v = (branch.float() * b.post_weight.float()).bfloat16().float()
    normed = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + POST_EPS)).bfloat16()
    stream = b.alpha * b.stream + b.beta * normed.float()
    pre_rstd = torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + PRE_EPS)
    return stream, (stream * pre_rstd * b.pre_weight.float()).bfloat16()


def _run(
    case: Case, draw: Draw, b: Boundary, unpool: torch.Tensor
) -> dict[str, torch.Tensor]:
    """One MoE launch on the input slot's first T rows and one tail launch on
    its output, as served."""

    tokens = len(draw.packed)
    mk = _megakernel_once(tokens)
    inputs = mk.inputs
    inputs.fp4[:tokens] = draw.x.hidden_states
    inputs.scales[:tokens] = draw.x.hidden_states_scale.view(torch.uint8)
    inputs.per_token_scale[:tokens] = draw.x.per_token_scale
    inputs.packed.copy_(draw.packed)
    y = mk(
        bank=case.bank,
        packed_routing=inputs.packed,
        hidden_states=inputs.fp4,
        hidden_states_scale=inputs.scales,
        per_token_scale=inputs.per_token_scale,
        output1_scale_scalar=case.alphas[0],
        output1_scale_gate_scalar=case.alphas[1],
        output2_scale_scalar=case.alphas[2],
        norm_weight=case.norm_weight,
    )
    gemm2 = mk.gemm2_ws.clone()
    stream, attn, qkv = _tail_once(tokens)(
        expert_output=y,
        unpool_weight=unpool,
        stream=b.stream,
        alpha=b.alpha,
        beta=b.beta,
        post_weight=b.post_weight,
        pre_weight=b.pre_weight,
        next_qkv_weight=b.qkv,
    )
    return {"y": y, "gemm2": gemm2, "stream": stream, "attn": attn, "qkv": qkv}


def _rrmse(got: torch.Tensor, expected: torch.Tensor) -> float:
    got, expected = got.double(), expected.double()
    return ((got - expected).norm() / expected.norm().clamp_min(1e-30)).item()


def _cosine(got: torch.Tensor, expected: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        got.double().view(-1), expected.double().view(-1), dim=0
    ).item()


def _metrics(
    case: Case, draw: Draw, out: dict[str, torch.Tensor], b: Boundary
) -> dict[str, float]:
    """Kernel vs reference errors; the tail is checked on the kernel's own y."""

    rank, world = dist.get_rank(), dist.get_world_size()
    refs = _gemm2_refs(case, draw)
    fins = {m: _finalize(g, draw.packed, case.norm_weight) for m, g in refs.items()}
    cols = slice(rank * HIDDEN // world, (rank + 1) * HIDDEN // world)
    tail_stream, tail_attn = _stock_tail(out["y"], b)
    _, e2e_attn = _stock_tail(fins["static"], b)
    return {
        "gemm2": _rrmse(out["gemm2"], refs["static"][:, cols]),
        "y": _rrmse(out["y"], fins["static"]),
        "y_cos": _cosine(out["y"], fins["static"]),
        "y_vs_exact": _rrmse(out["y"], fins["exact"]),
        "full_row_vs_exact": _rrmse(fins["full_row"], fins["exact"]),
        "stream": _rrmse(out["stream"], tail_stream),
        "attn": _rrmse(out["attn"], tail_attn),
        "attn_cos": _cosine(out["attn"], tail_attn),
        "attn_e2e": _rrmse(out["attn"], e2e_attn),
        "qkv": _rrmse(out["qkv"], out["attn"].double() @ b.qkv.double().T),
    }


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _failures(
    draw: Draw, m: dict[str, float], runs: list[dict[str, torch.Tensor]]
) -> list[str]:
    rank, world = dist.get_rank(), dist.get_world_size()
    print(
        f"[r{rank}] {draw.kind} " + " ".join(f"{k}={v:.3e}" for k, v in m.items()),
        flush=True,
    )
    failures = []
    close = m["gemm2"] < 1e-3 and m["y"] < 1e-3 and m["y_cos"] >= 0.99999
    # The static recipe is also no farther from no-requant than the full-row one.
    no_worse = m["y_vs_exact"] <= 1.05 * m["full_row_vs_exact"]
    if not (close and no_worse):
        failures.append(f"r{rank} {draw.kind}: MoE {m}")
    if m["stream"] > 5e-4 or m["attn"] > 2e-3 or m["attn_cos"] < 0.99999:
        failures.append(f"r{rank} {draw.kind}: tail {m}")
    if m["qkv"] > 3e-3:
        failures.append(f"r{rank} {draw.kind}: next-layer QKV {m}")
    for key in ("y", "stream", "attn", "qkv"):
        last = _bits(runs[-1][key])
        if not all(torch.equal(_bits(run[key]), last) for run in runs):
            failures.append(f"r{rank} {draw.kind}: {key} differs across calls")
        peers = [torch.empty_like(last) for _ in range(world)]
        dist.all_gather(peers, last)
        if not all(torch.equal(p, peers[0]) for p in peers):
            failures.append(f"r{rank} {draw.kind}: ranks disagree on {key}")
    return failures


def _assert_all_ranks_pass(failures: list[str]) -> None:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, failures)
    assert not any(gathered), gathered


@pytest.mark.parametrize(
    "kinds", [("uniform",), ("shared",), ("same8", "disjoint")], ids="-".join
)
@pytest.mark.parametrize("tokens", [8, 16], ids=lambda t: f"t{t}")
@torch.inference_mode()
def test_moe_tp4_verify_matches_full_expert_reference(
    tokens: int, kinds: tuple[str, ...]
) -> None:
    group = _init_group_once()
    rank, world = group.rank(), group.size()
    case = _make_case(kinds, group, tokens)
    b = _boundary(tokens)
    rows = MODEL_HIDDEN // world
    unpool = b.unpool[rank * rows : (rank + 1) * rows].contiguous()
    runs = [[] for _ in case.draws]
    # Repeat, and alternate draws, to cycle the cross-launch state.
    for _ in range(REPEATS):
        for i, draw in enumerate(case.draws):
            runs[i].append(_run(case, draw, b, unpool))
    torch.cuda.synchronize()
    failures = []
    for draw, draw_runs in zip(case.draws, runs):
        metrics = _metrics(case, draw, draw_runs[-1], b)
        failures += _failures(draw, metrics, draw_runs)
    _assert_all_ranks_pass(failures)


@torch.inference_mode()
def test_bank_aliases_w13() -> None:
    group = _init_group_once()
    local = INTERMEDIATE // group.size()
    g = torch.Generator(device="cuda").manual_seed(group.rank())

    def noise(*shape: int) -> torch.Tensor:
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    e4m3 = torch.float8_e4m3fn
    w13 = noise(EXPERTS, 2 * local, HIDDEN // 2)
    w13_scale = noise(EXPERTS, 2 * local, HIDDEN // 16)
    w2 = noise(EXPERTS, HIDDEN, local // 2)
    w2_scale = noise(EXPERTS, HIDDEN, local // 16)
    bank = build_moe_tp4_verify_bank(
        w13=w13,
        w13_scale=w13_scale.view(e4m3),
        w2=w2,
        w2_scale=w2_scale.view(e4m3),
        group=group,
    )
    failures = []
    # The stock FI W13 views alias the bank head, byte for byte.
    head = bank.w13.data_ptr() == bank.weights2.data_ptr()
    if not (head and torch.equal(bank.w13, w13)):
        failures.append("bank.w13 does not alias the copied FI W13")
    scale_head = bank.w13_scale.view(torch.uint8)
    head = scale_head.data_ptr() == bank.scales.data_ptr()
    if not (head and torch.equal(scale_head, w13_scale.view(-1))):
        failures.append("bank.w13_scale does not alias the copied FI scales")
    _assert_all_ranks_pass(failures)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
