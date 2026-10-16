"""The verify attention-boundary megakernel (T = 8 and 16) on TP4 against the
stock sequence after attention: O-projection, all-reduce, norm/residual, pooler,
NVFP4 quant, router and top-8.

Run with ``python <this file>``; it relaunches itself under torchrun on 4 GPUs.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.kernels.ops.muse_spark.norm_fp4_quant import fused_norm_fp4_quant
from sglang.kernels.ops.muse_spark.norm_residual import fused_norm_residual
from sglang.kernels.ops.muse_spark.routing import fused_muse_spark_routing
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
    from sglang.kernels.ops.muse_spark.helion_megakernels.attention_boundary_verify_megakernel import (
        AttentionBoundaryVerifyScalars,
        MuseAttentionBoundaryVerifyMegakernel,
    )
    from sglang.kernels.ops.muse_spark.helion_megakernels.moe_tp4_verify_megakernel import (
        MOE_TOKENS,
        MoeVerifyInputs,
    )

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = [
    pytest.mark.skipif(
        not HELION_MEGAKERNELS, reason="requires Helion with cross-loop pipelining"
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="the megakernels are specialized for SM10x",
    ),
]

HIDDEN, EXPERT_HIDDEN, EXPERTS, TOPK = 8192, 4096, 256, 8
POST_EPS, NORM_EPS = 1.0e-8, 1.0e-5
MULT, ROUTE_EPS, GSI = 1.0 / math.sqrt(32.0), 1.0e-15, 1.0 / (448.0 * 6.0)


@functools.cache
def _init_group_once() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    atexit.register(dist.destroy_process_group)
    return dist.group.WORLD


@functools.cache
def _boundary_once(tokens: int) -> MuseAttentionBoundaryVerifyMegakernel:
    group = _init_group_once()
    scalars = AttentionBoundaryVerifyScalars(
        post_eps=POST_EPS,
        pre_eps=NORM_EPS,
        quant_eps=NORM_EPS,
        router_multiplier=MULT,
        route_scale=1.0,
        route_eps=ROUTE_EPS,
    )
    return MuseAttentionBoundaryVerifyMegakernel(
        torch.device("cuda", torch.cuda.current_device()), group, scalars, tokens=tokens
    )


@functools.cache
def _expert_inputs_once(tokens: int) -> MoeVerifyInputs:
    """A MoE input slot as MuseMoeTp4VerifyMegakernel allocates it, one per T."""

    def zeros(*shape: int, dtype=torch.uint8) -> torch.Tensor:
        return torch.zeros(MOE_TOKENS, *shape, dtype=dtype, device="cuda")

    return MoeVerifyInputs(
        fp4=zeros(EXPERT_HIDDEN // 2),
        scales=zeros(EXPERT_HIDDEN // 16),
        per_token_scale=zeros(dtype=torch.float32),
        packed=torch.zeros(tokens, TOPK, dtype=torch.int32, device="cuda"),
    )


def _randn(shape, seed: int, scale: float = 1.0, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, generator=g, device="cuda") * scale).to(dtype)


def _make_case(
    boundary: MuseAttentionBoundaryVerifyMegakernel, seed: int, rank: int, world: int
) -> tuple[dict, dict]:
    tokens = boundary.tokens
    """Full weights plus this rank's attention-boundary shard."""

    # add_attn gate -> (alpha, beta), as MuseSpark1xNormPreservingResidualConnection.
    gate = _randn((HIDDEN,), seed + 1, 0.1) / 0.3
    beta = torch.sigmoid(gate)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-gate) * (1.0 + beta), min=1e-3))
    w = {
        "o_proj": _randn((HIDDEN, HIDDEN), seed + 2, HIDDEN**-0.5, torch.bfloat16),
        "alpha": alpha,
        "beta": beta,
        "pre_ffn_norm": (1.0 + _randn((HIDDEN,), seed + 3, 0.02)).bfloat16(),
        "pre_expert": _randn(
            (EXPERT_HIDDEN, HIDDEN), seed + 4, HIDDEN**-0.5, torch.bfloat16
        ),
        "pre_expert_norm": (1.0 + _randn((EXPERT_HIDDEN,), seed + 5, 0.02)).bfloat16(),
        "router": _randn((EXPERTS, HIDDEN), seed + 6, 0.01),
        "router_bias": _randn((EXPERTS,), seed + 7, 0.01),
    }
    cols = slice(rank * HIDDEN // world, (rank + 1) * HIDDEN // world)
    w["o_proj_local"] = w["o_proj"][:, cols].contiguous()
    w["prepared"] = boundary.prepare_weights(
        o_proj=w["o_proj_local"],
        pre_expert=w["pre_expert"],
        router=w["router"],
        alpha=alpha,
        beta=beta,
        pre_ffn_norm=w["pre_ffn_norm"],
        pre_expert_norm=w["pre_expert_norm"],
        router_bias=w["router_bias"],
        rank=rank,
        world=world,
    )
    a = {
        "gated": _randn((tokens, HIDDEN), seed + 100, dtype=torch.bfloat16),
        "stream": _randn((tokens, HIDDEN), seed + 101),
    }
    a["gated_local"] = a["gated"][:, cols].contiguous()
    return w, a


def _run_boundary(
    boundary: MuseAttentionBoundaryVerifyMegakernel, w: dict, a: dict
) -> dict[str, torch.Tensor]:
    inputs = _expert_inputs_once(boundary.tokens)
    out = boundary(a["gated_local"], a["stream"], w["prepared"], inputs)
    return _collect(out, inputs, boundary.tokens)


def _collect(out: dict, inputs: MoeVerifyInputs, tokens: int) -> dict:
    """The outputs plus the slot's first T rows, cloned; pad rows stay zero."""

    if inputs.fp4[tokens:].any() or inputs.scales[tokens:].any():
        raise AssertionError("the boundary wrote the MoE slot's pad activations")
    slot = {
        "fp4": inputs.fp4,
        "scales": inputs.scales,
        "per_token_scale": inputs.per_token_scale,
        "packed": inputs.packed,
    }
    return {
        **{k: v.clone() for k, v in out.items()},
        **{k: v[:tokens].clone() for k, v in slot.items()},
    }


def _stock(w: dict, a: dict, world: int) -> dict[str, torch.Tensor]:
    partial = F.linear(a["gated_local"], w["o_proj_local"])
    # CustomAllReduceV2 one-shot numerics: FP32 sum of BF16 peers in rank order.
    peers = [torch.empty_like(partial) for _ in range(world)]
    dist.all_gather(peers, partial)
    branch = torch.stack(peers).float().sum(0).bfloat16()
    ffn_input, new_stream, _ = fused_norm_residual(
        branch,
        a["stream"],
        POST_EPS,
        NORM_EPS,
        pre_weight=w["pre_ffn_norm"],
        alpha=w["alpha"],
        beta=w["beta"],
    )
    expert_input = F.linear(ffn_input, w["pre_expert"])
    fp4, scales, token_scale, _ = fused_norm_fp4_quant(
        expert_input, w["pre_expert_norm"], NORM_EPS, GSI
    )
    # The served router reads the BF16 FFN input upcast, as MuseSparkMoE does.
    logits = F.linear(ffn_input.float(), w["router"])
    packed = torch.empty(len(logits), TOPK, dtype=torch.int32, device="cuda")
    topk_weights, topk_ids, _ = fused_muse_spark_routing(
        logits,
        w["router_bias"],
        TOPK,
        1.0,
        router_output_multiplier=MULT,
        router_score_division_eps=ROUTE_EPS,
        packed_topk_ids=packed,
    )
    return {
        "new_stream": new_stream,
        "ffn_in": ffn_input,
        "fp4": fp4,
        "scales": scales,
        "per_token_scale": token_scale.view(-1),
        "logits": logits,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "packed": packed,
    }


def _fp64_logits(w: dict, a: dict) -> torch.Tensor:
    x = a["gated"].double() @ w["o_proj"].double().t()
    s = a["stream"].double()
    upd = w["beta"].double() * x * torch.rsqrt((x * x).mean(-1, True) + POST_EPS)
    upd = upd + w["alpha"].double() * s
    out = upd * torch.rsqrt((upd * upd).mean(-1, True) + NORM_EPS)
    return (out * w["pre_ffn_norm"].double()) @ w["router"].double().t()


def _rel(ref: torch.Tensor, act: torch.Tensor) -> float:
    r, x = ref.double().view(-1), act.double().view(-1)
    return float((r - x).norm() / r.norm().clamp_min(1e-30))


def _max_rel(ref: torch.Tensor, act: torch.Tensor) -> float:
    r, x = ref.double(), act.double()
    return float(((r - x).abs() / r.abs().amax(-1, True).clamp_min(1e-30)).max())


def _top8_valid(logits: torch.Tensor, bias: torch.Tensor, ids: torch.Tensor) -> bool:
    """Per token, a distinct top-8 of the FP64 routing keys, up to near-ties."""

    key = torch.sigmoid(logits.double() * MULT) + bias.double()
    eighth = torch.sort(key, dim=-1, descending=True).values[:, 7:8]
    chosen = key.gather(-1, ids.long())
    distinct = torch.sort(ids, dim=-1).values.diff(dim=-1).ne(0).all()
    return bool((chosen >= eighth - 2e-4).all()) and bool(distinct)


def _route_weights(packed: torch.Tensor) -> torch.Tensor:
    bits = (packed & 0xFFFF).to(torch.int16)
    return bits.view(torch.bfloat16).float()


def _metrics(stock: dict, act: dict) -> dict[str, float]:
    ids_stock, order_stock = torch.sort(stock["topk_ids"], dim=-1)
    ids_act, order_act = torch.sort(act["topk_ids"], dim=-1)
    match = ids_stock == ids_act
    w_stock = stock["topk_weights"].gather(-1, order_stock)
    w_act = _route_weights(act["packed"]).gather(-1, order_act)
    w_err = (w_stock - w_act).abs()[match].max() if match.any() else 0.0
    return {
        "new_stream_max_rel": _max_rel(stock["new_stream"], act["new_stream"]),
        "ffn_in_rel": _rel(stock["ffn_in"], act["ffn_in"]),
        "route_id_match": float(match.float().mean()),
        "route_weight_max_abs": float(w_err),
        "packed_match": float((stock["packed"] == act["packed"]).float().mean()),
        "fp4_byte_match": float((stock["fp4"] == act["fp4"]).float().mean()),
        "scale_byte_match": float(
            (stock["scales"].view(torch.uint8) == act["scales"]).float().mean()
        ),
        "per_token_scale_rel": float(
            ((stock["per_token_scale"] - act["per_token_scale"]).abs())
            .div(stock["per_token_scale"].abs())
            .max()
        ),
    }


def _failures(
    boundary: MuseAttentionBoundaryVerifyMegakernel, w: dict, a: dict, world: int
) -> list[str]:
    stock = _stock(w, a, world)
    exact = _fp64_logits(w, a)
    torch.cuda.synchronize()
    dist.barrier()
    act = _run_boundary(boundary, w, a)
    torch.cuda.synchronize()
    dist.barrier()
    m = _metrics(stock, act)
    if dist.get_rank() == 0:
        print(" ".join(f"{k}={v:.3e}" for k, v in m.items()), flush=True)
    failures = []
    if m["new_stream_max_rel"] > 1e-3:
        failures.append(f"new_stream vs stock {m['new_stream_max_rel']:.2e}")
    if m["ffn_in_rel"] > 1e-3:
        failures.append(f"ffn_in vs stock {m['ffn_in_rel']:.2e}")
    if m["per_token_scale_rel"] > 1e-2:
        failures.append(f"per_token_scale vs stock {m['per_token_scale_rel']:.2e}")
    if m["fp4_byte_match"] < 0.95 or m["scale_byte_match"] < 0.95:
        failures.append(f"fp4/scale bytes {m['fp4_byte_match']:.3f}")
    if m["route_weight_max_abs"] > 1e-2:
        failures.append(f"route weights {m['route_weight_max_abs']:.2e}")
    if not _top8_valid(exact, w["router_bias"], act["topk_ids"]):
        failures.append(f"invalid top-8 {act['topk_ids'].tolist()}")
    if not torch.equal(act["packed"] >> 16, act["topk_ids"]):
        failures.append("packed ids != topk_ids")
    for k in ("new_stream", "fp4", "scales", "per_token_scale", "packed"):
        mine = act[k].contiguous().view(torch.uint8)
        peers = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather(peers, mine)
        if not all(torch.equal(p, peers[0]) for p in peers):
            failures.append(f"ranks disagree on {k}")
    return failures


def _assert_all_ranks_pass(failures: list[str]) -> None:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, failures)
    assert not any(gathered), gathered


@pytest.mark.parametrize("seed", [1000, 2000])
@pytest.mark.parametrize("tokens", [8, 16], ids=lambda t: f"t{t}")
@torch.inference_mode()
def test_attention_boundary_verify_matches_stock_tp4(tokens: int, seed: int) -> None:
    group = _init_group_once()
    boundary = _boundary_once(tokens)
    w, a = _make_case(boundary, seed, group.rank(), group.size())
    _assert_all_ranks_pass(_failures(boundary, w, a, group.size()))


@pytest.mark.parametrize("tokens", [8, 16], ids=lambda t: f"t{t}")
@torch.inference_mode()
def test_attention_boundary_verify_back_to_back(tokens: int) -> None:
    group = _init_group_once()
    boundary = _boundary_once(tokens)
    cases = [_make_case(boundary, s, group.rank(), group.size()) for s in (3000, 4000)]
    for _ in range(3):
        for w, a in cases:
            _run_boundary(boundary, w, a)
    _assert_all_ranks_pass(_failures(boundary, *cases[0], group.size()))


@pytest.mark.parametrize("tokens", [8, 16], ids=lambda t: f"t{t}")
@torch.inference_mode()
def test_attention_boundary_verify_cuda_graph(tokens: int) -> None:
    group = _init_group_once()
    boundary = _boundary_once(tokens)
    cases = [_make_case(boundary, s, group.rank(), group.size()) for s in (5000, 6000)]
    w, a = cases[0]
    gated, stream = a["gated_local"].clone(), a["stream"].clone()
    _run_boundary(boundary, w, {"gated_local": gated, "stream": stream})
    torch.cuda.synchronize()
    dist.barrier()
    inputs = _expert_inputs_once(tokens)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = boundary(gated, stream, w["prepared"], inputs)
    failures = []
    for _ in range(2):
        for _, case in cases:
            gated.copy_(case["gated_local"])
            stream.copy_(case["stream"])
            graph.replay()
            replayed = _collect(out, inputs, tokens)
            eager = _run_boundary(boundary, w, case)
            torch.cuda.synchronize()
            for k, v in eager.items():
                if not torch.equal(replayed[k], v):
                    failures.append(f"graph replay differs on {k}")
    _assert_all_ranks_pass(failures)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
