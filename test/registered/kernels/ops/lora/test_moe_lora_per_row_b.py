from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.lora.common.route_view import RouteView, RouteViewKind
from sglang.kernels.ops.lora.common.routing import build_route
from sglang.kernels.ops.lora.moe.lora_b import run_lora_b
from sglang.srt.lora.moe.plan import (
    BFamily,
    BridgeLayout,
    BSpec,
    Site,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="per_row LoRA-B needs any CUDA device"
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
# The per_row family reads only these keys; BLOCK_SIZE_K matches grouped
# so the FP32 k-tile accumulation order is identical.
_PER_ROW_CONFIG = {
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
        group_ids=topk_ids.to(device),
        groups_per_slot=num_experts,
        max_loras=_SLOTS,
        block_size=16,
    )
    aligned = build_route(
        token_lora_mapping.to(device), view=RouteViewKind.ALIGNED, **kwargs
    )
    raw = build_route(token_lora_mapping.to(device), view=RouteViewKind.RAW, **kwargs)
    return aligned, raw


def _site_tensors(site: str, num_pairs: int, num_experts: int, seed: int, device):
    generator = torch.Generator().manual_seed(seed)
    num_route_buckets = _SLOTS * num_experts
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
        torch.randn((num_route_buckets, num_slices * width, _RANK), generator=generator)
        * 0.2
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
def test_per_row_matches_grouped_on_identical_inputs(
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
    b_site = Site.GATE_UP if site == "gate_up" else Site.DOWN
    indexed = _destination(num_pairs, num_slices, width, device)

    def launch():
        run_lora_b(
            BSpec(b_site, BFamily.PER_ROW),
            bridge=bridge,
            weight=weight,
            destination=indexed,
            routing=raw,
            destination_offsets=offsets,
            config=_PER_ROW_CONFIG,
        )

    launch()
    if b_site is Site.GATE_UP and num_tokens == 13:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        topk_ids, token_lora_mapping = _routing_case(
            num_tokens, top_k, num_experts, seed + 1
        )
        token_lora_mapping = token_lora_mapping.roll(1)
        raw.group_ids.copy_(topk_ids.to(device))
        raw.token_slots.copy_(token_lora_mapping.to(device))
        bridge.mul_(0.5)
        indexed.fill_(_POISON)
        graph.replay()
        torch.cuda.synchronize()
        aligned = build_route(
            raw.token_slots,
            view=RouteViewKind.ALIGNED,
            group_ids=raw.group_ids,
            groups_per_slot=num_experts,
            max_loras=_SLOTS,
            block_size=16,
        )

    reference = _destination(num_pairs, num_slices, width, device)
    run_lora_b(
        BSpec(b_site, BFamily.GROUPED),
        bridge=bridge,
        weight=weight,
        destination=reference,
        routing=aligned,
        destination_offsets=offsets,
        config=_GROUPED_CONFIG,
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


@pytest.mark.parametrize("top_k", (6, 1))
def test_per_row_on_slot_planes_matches_reference(top_k: int) -> None:
    """Expand slot-plane token-major A through per-expert B against FP32.
    Include EP holes, no adapter, and top-k 1; invalid pairs must be zero.
    """

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
        group_ids=topk_ids,
        token_slots=mapping,
        groups_per_slot=num_experts,
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

    per_row_out = torch.full(
        (num_pairs, 2 * inter), _POISON, device=device, dtype=torch.bfloat16
    )
    # The runner hands the planes over as [S, M, 2R]; the slot stride selects a plane.
    run_lora_b(
        BSpec(Site.GATE_UP, BFamily.PER_ROW, False, BridgeLayout.TOKEN_MAJOR),
        bridge=planes,
        weight=weight,
        destination=per_row_out,
        routing=per_expert,
        destination_offsets=(0, inter),
        config=_PER_ROW_CONFIG,
        pair_bridge=False,
        slot_planes=True,
    )
    torch.testing.assert_close(per_row_out.float(), reference, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
