"""The grouped LoRA-A kernel's placed bridge: rows stored where a provider
row map says, on the pair's adapter-slot plane."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from sglang.kernels.ops.lora.common.lora_a import grouped_lora_a  # noqa: E402
from sglang.kernels.ops.lora.common.route_view import RouteViewKind  # noqa: E402
from sglang.kernels.ops.lora.common.routing import build_route  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the grouped LoRA-A kernel needs CUDA"
)

_CONFIG = {
    "BLOCK_SIZE_N": 16,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}


def test_grouped_lora_a_places_rows_on_slot_planes():
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(5)
    tokens, top_k, experts, slots, rank, k = 24, 2, 4, 3, 16, 128
    topk_ids = torch.stack(
        [torch.randperm(experts, generator=g)[:top_k] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[2, 0] = -1  # an EP hole
    # Slots 0, 1, 2 and the base path (-1) in turn.
    mapping = ((torch.arange(tokens, device=device) + 1) % 4 - 1).to(torch.int32)
    route = build_route(
        mapping,
        group_ids=topk_ids,
        groups_per_slot=experts,
        max_loras=slots,
        block_size=16,
        view=RouteViewKind.ALIGNED,
    )
    pairs = tokens * top_k
    x = (torch.randn(pairs, k, generator=g) * 0.5).to(
        device=device, dtype=torch.bfloat16
    )
    weight = (torch.randn(slots * experts, rank, k, generator=g) * 0.1).to(
        device=device, dtype=torch.bfloat16
    )
    # The provider row of each pair: a permutation into a padded row space.
    rows_total = pairs + 40
    perm = torch.randperm(rows_total, generator=g)[:pairs].to(
        device=device, dtype=torch.int32
    )

    pair_major = torch.zeros((pairs, rank), dtype=torch.bfloat16, device=device)
    grouped_lora_a(x, weight, pair_major, route, config=_CONFIG, pair_input=True)
    planes = torch.zeros((slots, rows_total, rank), dtype=torch.bfloat16, device=device)
    grouped_lora_a(
        x,
        weight,
        planes,
        route,
        config=_CONFIG,
        pair_input=True,
        output_rows=perm,
    )
    torch.cuda.synchronize()

    pair_slot = mapping.repeat_interleave(top_k).long()
    pair_expert = topk_ids.flatten().long()
    live = (pair_slot >= 0) & (pair_expert >= 0)
    # Both outputs store the same FP32 product, rounded to BF16 once.
    ref = torch.einsum(
        "pk,prk->pr",
        x.float(),
        weight.float()[(pair_slot.clamp(min=0) * experts + pair_expert.clamp(min=0))],
    )
    expected = torch.zeros_like(planes)
    expected[pair_slot[live], perm[live].long()] = ref.to(torch.bfloat16)[live]
    torch.testing.assert_close(planes.float(), expected.float(), rtol=2e-2, atol=2e-2)
    # A placed row is bitwise its pair-major row; sentinel pairs stay zero.
    assert torch.equal(planes[pair_slot[live], perm[live].long()], pair_major[live])
    torch.testing.assert_close(
        pair_major.float()[live],
        ref.to(torch.bfloat16).float()[live],
        rtol=2e-2,
        atol=2e-2,
    )
    assert torch.equal(pair_major[~live], torch.zeros_like(pair_major[~live]))
    # Every row off a live pair's own plane stayed zero.
    touched = torch.zeros((slots, rows_total), dtype=torch.bool, device=device)
    touched[pair_slot[live], perm[live].long()] = True
    assert torch.equal(planes[~touched], torch.zeros_like(planes[~touched]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
