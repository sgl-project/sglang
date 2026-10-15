"""The verify attention-boundary megakernel (T = 8 and 16) on TP4 against the
stock sequence after the pre-FFN norm: pooler, NVFP4 quant, router and top-8.

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

from sglang.kernels.ops.muse_spark_v12.norm_fp4_quant import fused_norm_fp4_quant
from sglang.kernels.ops.muse_spark_v12.routing import fused_muse_spark_routing
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
    from sglang.kernels.ops.muse_spark_v12.helion_megakernels.moe_tp4_verify_megakernel import (
        MOE_TOKENS,
        MoeVerifyInputs,
    )
    from sglang.kernels.ops.muse_spark_v12.helion_megakernels.attention_boundary_verify_megakernel import (
        AttentionBoundaryVerifyScalars,
        MuseAttentionBoundaryVerifyMegakernel,
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
NORM_EPS = 1.0e-5
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
    """Full weights plus this rank's boundary shard, and a pre-FFN-normed input."""

    tokens = boundary.tokens
    w = {
        "pre_expert": _randn(
            (EXPERT_HIDDEN, HIDDEN), seed + 4, HIDDEN**-0.5, torch.bfloat16
        ),
        "pre_expert_norm": (1.0 + _randn((EXPERT_HIDDEN,), seed + 5, 0.02)).bfloat16(),
        "router": _randn((EXPERTS, HIDDEN), seed + 6, 0.01),
        "router_bias": _randn((EXPERTS,), seed + 7, 0.01),
    }
    w["prepared"] = boundary.prepare_weights(
        pre_expert=w["pre_expert"],
        router=w["router"],
        pre_expert_norm=w["pre_expert_norm"],
        router_bias=w["router_bias"],
        rank=rank,
        world=world,
    )
    a = {"ffn_in": _randn((tokens, HIDDEN), seed + 100, dtype=torch.bfloat16)}
    return w, a


def _run_boundary(
    boundary: MuseAttentionBoundaryVerifyMegakernel, w: dict, a: dict
) -> dict[str, torch.Tensor]:
    inputs = _expert_inputs_once(boundary.tokens)
    topk_ids = boundary(a["ffn_in"], w["prepared"], inputs)
    return _collect({"topk_ids": topk_ids}, inputs, boundary.tokens)


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


def _stock(w: dict, a: dict) -> dict[str, torch.Tensor]:
    ffn_input = a["ffn_in"]
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
        "fp4": fp4,
        "scales": scales,
        "per_token_scale": token_scale.view(-1),
        "logits": logits,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "packed": packed,
    }


def _fp64_logits(w: dict, a: dict) -> torch.Tensor:
    return a["ffn_in"].double() @ w["router"].double().t()


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
    stock = _stock(w, a)
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
    for k in ("fp4", "scales", "per_token_scale", "packed"):
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
    ffn_in = a["ffn_in"].clone()
    _run_boundary(boundary, w, {"ffn_in": ffn_in})
    torch.cuda.synchronize()
    dist.barrier()
    inputs = _expert_inputs_once(tokens)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        topk_ids = boundary(ffn_in, w["prepared"], inputs)
    failures = []
    for _ in range(2):
        for _, case in cases:
            ffn_in.copy_(case["ffn_in"])
            graph.replay()
            replayed = _collect({"topk_ids": topk_ids}, inputs, tokens)
            eager = _run_boundary(boundary, w, case)
            torch.cuda.synchronize()
            for k, v in eager.items():
                if not torch.equal(replayed[k], v):
                    failures.append(f"graph replay differs on {k}")
    _assert_all_ranks_pass(failures)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
