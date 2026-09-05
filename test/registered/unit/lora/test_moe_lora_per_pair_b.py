from __future__ import annotations

import pytest
import torch

from sglang.srt.lora.moe.execution_plan import (
    BridgeLayout,
    LoraBFamily,
    LoraBSpec,
    Site,
)
from sglang.srt.lora.moe.kernels.lora_b import (
    grouped_lora_b,
    per_pair_lora_b,
    run_lora_b,
)
from sglang.srt.lora.moe.route_view import RouteView, RouteViewKind
from sglang.srt.lora.moe.routing import build_virtual_expert_routing
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="per_pair LoRA-B needs any CUDA device"
)

_SLOTS = 2
_RANK = 16
_INTERMEDIATE = 96
_HIDDEN = 128
_POISON = 7.0

_GROUPED_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 16,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}
# The per_pair family reads only these keys; BLOCK_SIZE_K matches grouped
# so the FP32 k-tile accumulation order is identical.
_PER_PAIR_CONFIG = {
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 16,
    "num_warps": 4,
    "num_stages": 2,
}
# Same FP32 k-tile accumulation of exact BF16 products; only within-tile
# reduction order differs (tl.sum vs tl.dot), so allclose, not bitwise.
_TOLERANCE = {"atol": 5e-3, "rtol": 2e-2}


def _routing_case(num_tokens: int, top_k: int, num_experts: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), generator=generator, dtype=torch.int32
    )
    topk_ids[torch.rand((num_tokens, top_k), generator=generator) < 0.15] = -1
    topk_ids[0] = -1  # a token with zero routed pairs
    pattern = torch.tensor([0, 1, -1, 0, -1, 1], dtype=torch.int32)
    token_lora_mapping = pattern.repeat(-(-num_tokens // pattern.numel()))[:num_tokens]
    return topk_ids, token_lora_mapping


def _views(topk_ids, token_lora_mapping, num_experts, device):
    kwargs = dict(
        num_local_experts=num_experts,
        max_loras=_SLOTS,
        block_size=16,
    )
    aligned = build_virtual_expert_routing(
        topk_ids.to(device),
        token_lora_mapping.to(device),
        view=RouteViewKind.ALIGNED,
        **kwargs,
    )
    raw = build_virtual_expert_routing(
        topk_ids.to(device),
        token_lora_mapping.to(device),
        view=RouteViewKind.RAW,
        **kwargs,
    )
    return aligned, raw


def _site_tensors(site: str, num_pairs: int, num_experts: int, seed: int, device):
    generator = torch.Generator().manual_seed(seed)
    groups = _SLOTS * num_experts
    if site == "gate_up":
        num_slices, width = 2, _INTERMEDIATE
        offsets = (0, _INTERMEDIATE)
    else:
        num_slices, width = 1, _HIDDEN
        offsets = (0,)
    bridge = (
        torch.randn((num_pairs, num_slices * _RANK), generator=generator) * 0.2
    ).to(torch.bfloat16)
    weight = (
        torch.randn((groups, num_slices * width, _RANK), generator=generator) * 0.2
    ).to(torch.bfloat16)
    return bridge.to(device), weight.to(device), offsets, num_slices, width


def _destination(num_pairs: int, num_slices: int, width: int, device):
    return torch.full(
        (num_pairs, num_slices * width), _POISON, dtype=torch.bfloat16, device=device
    )


@pytest.mark.parametrize("site", ("gate_up", "down"))
@pytest.mark.parametrize(
    ("num_tokens", "top_k", "num_experts"),
    ((32, 4, 64), (13, 3, 4)),
    ids=("sparse", "partial"),
)
def test_per_pair_matches_grouped_on_identical_inputs(
    site: str, num_tokens: int, top_k: int, num_experts: int
) -> None:
    device = torch.device("cuda")
    seed = 0x1DB0 + num_tokens + num_experts
    topk_ids, token_lora_mapping = _routing_case(num_tokens, top_k, num_experts, seed)
    aligned, raw = _views(topk_ids, token_lora_mapping, num_experts, device)
    num_pairs = num_tokens * top_k
    bridge, weight, offsets, num_slices, width = _site_tensors(
        site, num_pairs, num_experts, seed, device
    )

    reference = _destination(num_pairs, num_slices, width, device)
    grouped_lora_b(
        bridge,
        weight,
        reference,
        aligned,
        destination_offsets=offsets,
        config=_GROUPED_CONFIG,
    )
    indexed = _destination(num_pairs, num_slices, width, device)
    # The raw view carries no aligned fields, so any aligned-metadata read
    # would raise instead of silently depending on the sort.
    per_pair_lora_b(
        bridge,
        weight,
        indexed,
        raw,
        destination_offsets=offsets,
        config=_PER_PAIR_CONFIG,
    )
    torch.testing.assert_close(indexed, reference, **_TOLERANCE)

    # Invalid pairs (base tokens, unrouted -1 experts) must be ZERO-stored,
    # not left at the poison: B owns every consumed destination cell.
    pair_valid = (topk_ids.view(-1) >= 0) & (
        token_lora_mapping.repeat_interleave(top_k) >= 0
    )
    invalid_rows = indexed[~pair_valid.to(device)]
    assert invalid_rows.numel() > 0
    assert torch.count_nonzero(invalid_rows) == 0
    assert torch.count_nonzero(indexed[pair_valid.to(device)]) > 0


def test_zero_pair_batches_return_without_launching() -> None:
    device = torch.device("cuda")
    topk_ids, token_lora_mapping = _routing_case(4, 2, 8, 0x1DB2)
    empty = RouteView(
        view=RouteViewKind.RAW,
        block_size=16,
        topk_ids=topk_ids[:0].to(device),
        token_lora_mapping=token_lora_mapping[:0].to(device),
        num_local_experts=8,
        is_shared_outer=False,
        max_loras=_SLOTS,
    )
    bridge, weight, offsets, num_slices, width = _site_tensors(
        "down", 0, 8, 0x1DB2, device
    )
    destination = _destination(0, num_slices, width, device)
    per_pair_lora_b(
        bridge,
        weight,
        destination,
        empty,
        destination_offsets=offsets,
        config=_PER_PAIR_CONFIG,
    )


def test_run_lora_b_dispatches_the_family_and_rejects_pdl() -> None:
    device = torch.device("cuda")
    topk_ids, token_lora_mapping = _routing_case(8, 2, 8, 0x1DB3)
    _, raw = _views(topk_ids, token_lora_mapping, 8, device)
    bridge, weight, offsets, num_slices, width = _site_tensors(
        "down", 16, 8, 0x1DB3, device
    )
    spec = LoraBSpec(Site.DOWN, LoraBFamily.PER_PAIR)
    with pytest.raises(ValueError, match="programmatic-dependent-launch"):
        run_lora_b(
            spec,
            bridge=bridge,
            weight=weight,
            destination=_destination(16, num_slices, width, device),
            routing=raw,
            destination_offsets=offsets,
            config=_PER_PAIR_CONFIG,
            consume_pdl=True,
        )
    dispatched = _destination(16, num_slices, width, device)
    run_lora_b(
        spec,
        bridge=bridge,
        weight=weight,
        destination=dispatched,
        routing=raw,
        destination_offsets=offsets,
        config=_PER_PAIR_CONFIG,
    )
    direct = _destination(16, num_slices, width, device)
    per_pair_lora_b(
        bridge,
        weight,
        direct,
        raw,
        destination_offsets=offsets,
        config=_PER_PAIR_CONFIG,
    )
    assert torch.equal(dispatched, direct)


def test_graph_replay_recomputes_from_mutated_routing_content() -> None:
    """The captured grid depends only on num_pairs; slot/expert routing is
    pure data and must be honored at replay."""
    device = torch.device("cuda")
    num_tokens, top_k, num_experts = 16, 2, 8
    topk_ids, token_lora_mapping = _routing_case(num_tokens, top_k, num_experts, 0x1DB4)
    _, raw = _views(topk_ids, token_lora_mapping, num_experts, device)
    num_pairs = num_tokens * top_k
    bridge, weight, offsets, num_slices, width = _site_tensors(
        "gate_up", num_pairs, num_experts, 0x1DB4, device
    )
    destination = torch.zeros(
        (num_pairs, num_slices * width), dtype=torch.bfloat16, device=device
    )

    def launch():
        per_pair_lora_b(
            bridge,
            weight,
            destination,
            raw,
            destination_offsets=offsets,
            config=_PER_PAIR_CONFIG,
        )

    launch()  # warm the JIT outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    new_topk_ids, new_token_lora_mapping = _routing_case(
        num_tokens, top_k, num_experts, 0x1DB5
    )
    raw.topk_ids.copy_(new_topk_ids.to(device))
    raw.token_lora_mapping.copy_(new_token_lora_mapping.to(device))
    new_bridge = (
        torch.randn(
            (num_pairs, num_slices * _RANK),
            generator=torch.Generator().manual_seed(0x1DB6),
        )
        * 0.2
    ).to(torch.bfloat16)
    bridge.copy_(new_bridge.to(device))
    graph.replay()
    torch.cuda.synchronize()
    replayed = destination.clone()

    expected = torch.zeros_like(destination)
    per_pair_lora_b(
        bridge,
        weight,
        expected,
        raw,
        destination_offsets=offsets,
        config=_PER_PAIR_CONFIG,
    )
    assert torch.equal(replayed, expected)


@pytest.mark.parametrize("top_k", (6, 1))
def test_per_pair_on_slot_planes_matches_reference(top_k: int) -> None:
    """A token-major bridge with one plane per adapter slot, as the dense
    shared A writes it, expanded through per-expert B: matches the fp32
    reference with an EP hole and a token without an adapter, invalid pairs
    zeroed. Top-k 1 must select planes too (no top-k proxy for the layout)."""

    device = torch.device("cuda")
    num_tokens, slots, num_experts, rank, inter = 9, 3, 16, 32, 128
    generator = torch.Generator().manual_seed(11)
    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, generator=generator)[:top_k]
            for _ in range(num_tokens)
        ]
    ).to(torch.int32)
    topk_ids[2, top_k - 1] = -1  # locally invalid expert (EP hole)
    mapping = torch.randint(
        0, slots, (num_tokens,), generator=generator, dtype=torch.int32
    )
    mapping[5] = -1  # no resident adapter
    topk_ids, mapping = topk_ids.to(device), mapping.to(device)
    per_expert = RouteView(
        view=RouteViewKind.RAW,
        block_size=16,
        topk_ids=topk_ids,
        token_lora_mapping=mapping,
        num_local_experts=num_experts,
        is_shared_outer=False,
        max_loras=slots,
    )
    planes = (torch.randn(slots, num_tokens, 2 * rank, generator=generator) * 0.1).to(
        device, torch.bfloat16
    )
    weight = (
        torch.randn(slots * num_experts, 2 * inter, rank, generator=generator) * 0.1
    ).to(device, torch.bfloat16)
    num_pairs = num_tokens * top_k
    reference = torch.zeros((num_pairs, 2 * inter), dtype=torch.float32, device=device)
    planes_f, weight_f = planes.float(), weight.float()
    for token in range(num_tokens):
        slot = int(mapping[token])
        for k in range(top_k):
            expert = int(topk_ids[token, k])
            if slot < 0 or expert < 0:
                continue
            block = weight_f[slot * num_experts + expert]
            pair = token * top_k + k
            reference[pair, :inter] = block[:inter] @ planes_f[slot, token, :rank]
            reference[pair, inter:] = block[inter:] @ planes_f[slot, token, rank:]

    per_pair_out = torch.full(
        (num_pairs, 2 * inter), _POISON, device=device, dtype=torch.bfloat16
    )
    # The runner hands the planes over as [S, M, 2R]; the slot stride selects a plane.
    run_lora_b(
        LoraBSpec(Site.GATE_UP, LoraBFamily.PER_PAIR, False, BridgeLayout.TOKEN_MAJOR),
        bridge=planes,
        weight=weight,
        destination=per_pair_out,
        routing=per_expert,
        destination_offsets=(0, inter),
        config=_PER_PAIR_CONFIG,
        intermediate_top_k=top_k,
    )
    torch.testing.assert_close(per_pair_out.float(), reference, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
