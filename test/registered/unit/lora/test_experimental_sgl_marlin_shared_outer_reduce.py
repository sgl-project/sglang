"""Guards the fused shared-outer Marlin decode reduction against the three-launch
fallback it replaces; reds when down-B orientation, router weighting, the routing
scale, a ragged-tile mask, or capture-time value baking is wrong."""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


_CUDA_BF16_AVAILABLE = bool(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] >= 8
)

# Two BF16 ulps (2**-7 each), plus an absolute floor for cancelled outputs.
_RTOL = 1.6e-2
_ATOL = 1e-3


@pytest.mark.skipif(not _CUDA_BF16_AVAILABLE, reason="CUDA BF16 required")
@pytest.mark.parametrize("mapping_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_slots", [1, 3])
def test_gate_delta_activation_skips_unwritten_routes_and_adapters(
    mapping_dtype, num_slots
):
    """Private poisoned rows stay unread; adapter0 remains a real adapter slot."""
    from sglang.kernels.ops.lora.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
    from sglang.srt.lora.marlin_lora_temp.activation import silu_and_mul_add_delta

    device, dtype = "cuda", torch.bfloat16
    tokens, topk, experts, rank, hidden, width = 6, 2, 2, 8, 32, 144
    producer_ids = torch.tensor([[0, 1]] * tokens, device=device, dtype=torch.int32)
    producer_mapping = torch.tensor(
        [0, 1, 2, 0, 1, 0], device=device, dtype=torch.int32
    ).remainder(num_slots)
    a = torch.zeros((num_slots, 1, 2 * rank, hidden), device=device, dtype=dtype)
    b = torch.zeros((num_slots, experts, 2 * width, rank), device=device, dtype=dtype)
    for slot in range(min(2, num_slots)):
        a[slot, 0, 0, 0], a[slot, 0, rank, 0] = 2 + slot, 4 + slot
        b[slot, :, :width, 0], b[slot, :, width:, 0] = 3 + slot, 5 + slot
    hidden_states = torch.zeros((tokens, hidden), device=device, dtype=dtype)
    hidden_states[:, 0] = 0.125
    delta = torch.full(
        (tokens, topk, 2 * width), float("nan"), device=device, dtype=dtype
    )
    weights = torch.ones((tokens, topk), device=device)
    config = dict(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=1,
        num_warps=4,
        num_stages=2,
    )
    with override_config(config):
        merged_experts_fused_moe_lora_add(
            output=delta,
            hidden_states=hidden_states,
            lora_a=a,
            lora_b=b,
            topk_ids=producer_ids,
            topk_weights=weights,
            token_lora_mapping=producer_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=True,
            experts_shared_outer_loras_b=False,
            fuse_add_to_output=False,
            use_direct_expand_add=True,
        )
    shrunk = torch.einsum(
        "mh,mrh->mr", hidden_states.float(), a[producer_mapping.long(), 0].float()
    )
    selected_b = b[producer_mapping[:, None].long(), producer_ids.long()]
    expected_delta = (
        torch.einsum(
            "mpr,mkpir->mkpi",
            shrunk.view(tokens, 2, rank),
            selected_b.float().view(tokens, topk, 2, width, rank),
        )
        .flatten(2)
        .to(dtype)
    )
    torch.testing.assert_close(delta, expected_delta, rtol=0, atol=0)
    assert bool((delta[0] != 0).all()), "valid adapter0 must be expanded"

    base_saved, delta_saved = torch.full_like(delta, 0.5), delta.clone()
    base = base_saved.clone()
    ids = torch.full((tokens, topk * 2), -1, device=device, dtype=torch.int32)[:, ::2]
    ids.copy_(producer_ids)
    mapping = torch.full((tokens * 2,), -1, device=device, dtype=mapping_dtype)[::2]
    mapping.copy_(producer_mapping)
    enabled = torch.zeros(num_slots * 2, device=device, dtype=torch.int32)[::2]
    enabled.fill_(1)
    output = torch.empty((tokens * topk, width), device=device, dtype=dtype)

    def run():
        silu_and_mul_add_delta(
            base.view(-1, 2 * width),
            delta.view(-1, 2 * width),
            output,
            topk_ids=ids,
            token_lora_mapping=mapping,
            adapter_enabled=enabled,
        )

    run()
    default_output = torch.empty_like(output)
    silu_and_mul_add_delta(
        base.view(-1, 2 * width), delta.view(-1, 2 * width), default_output
    )
    torch.testing.assert_close(output, default_output, rtol=0, atol=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for state in ("mixed", "disabled", "all-inactive", "valid"):
        ids.copy_(torch.tensor([[0, 1]] * tokens, device=device, dtype=ids.dtype))
        mapping.copy_(producer_mapping)
        enabled.fill_(1)
        if state == "mixed":
            ids[0, 0], ids[2, 0], ids[3, 1], ids[5] = -1, -1, -1, -1
            mapping[3], mapping[4] = -1, num_slots
            if num_slots > 1:
                enabled[num_slots - 1] = 0
        elif state == "disabled":
            enabled.zero_()
        elif state == "all-inactive":
            ids.fill_(-1)
        active_route = ids >= 0
        safe_slots = mapping.clamp(0, enabled.numel() - 1).long()
        active_adapter = (
            (mapping >= 0) & (mapping < enabled.numel()) & (enabled[safe_slots] != 0)
        )
        base.copy_(base_saved).masked_fill_(~active_route.unsqueeze(-1), float("nan"))
        delta.copy_(delta_saved).masked_fill_(
            ~(active_route & active_adapter[:, None]).unsqueeze(-1), float("nan")
        )
        output.fill_(7)
        graph.replay()
        reference_delta = delta_saved.masked_fill(
            ~active_adapter[:, None, None], 0
        ).float()
        gate = base_saved[..., :width].float() + reference_delta[..., :width]
        up = base_saved[..., width:].float() + reference_delta[..., width:]
        expected = (torch.nn.functional.silu(gate) * up).to(dtype)
        expected = expected.masked_fill(~active_route.unsqueeze(-1), 7)
        torch.testing.assert_close(
            output.view_as(expected), expected, rtol=_RTOL, atol=_ATOL
        )


def _reference_reduce(
    routed_base: torch.Tensor,
    routed_rank: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_b: torch.Tensor,
    routed_scaling_factor: float,
    topk_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize the three production operations with their BF16 boundaries."""

    operand_dtype = routed_base.dtype
    if topk_ids is not None:
        inactive = topk_ids < 0
        routed_base = routed_base.masked_fill(inactive.unsqueeze(-1), 0)
        routed_rank = routed_rank.masked_fill(inactive.unsqueeze(-1), 0)
        topk_weights = topk_weights.masked_fill(inactive, 0)
    base_sum = (
        routed_base.to(torch.float32)
        .sum(dim=1)
        .mul(routed_scaling_factor)
        .to(operand_dtype)
    )
    rank_sum = (
        (routed_rank.to(torch.float32) * topk_weights.to(torch.float32).unsqueeze(-1))
        .sum(dim=1)
        .mul(routed_scaling_factor)
        .to(operand_dtype)
    )
    base_sum.addmm_(rank_sum, shared_b.T)
    return base_sum


def _mapped_reference_reduce(
    routed_base: torch.Tensor,
    routed_rank: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_b: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    routed_scaling_factor: float,
    topk_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same reduction with a per-token adapter slot; ``-1`` rows stay base-only."""

    dtype = routed_base.dtype
    if topk_ids is not None:
        inactive = topk_ids < 0
        routed_base = routed_base.masked_fill(inactive.unsqueeze(-1), 0)
        routed_rank = routed_rank.masked_fill(inactive.unsqueeze(-1), 0)
        topk_weights = topk_weights.masked_fill(inactive, 0)
    base_sum = routed_base.float().sum(dim=1).mul(routed_scaling_factor).to(dtype)
    rank_sum = (
        (routed_rank.float() * topk_weights.unsqueeze(-1))
        .sum(dim=1)
        .mul(routed_scaling_factor)
        .to(dtype)
    )
    output = base_sum.clone()
    for slot in range(shared_b.shape[0]):
        rows = token_lora_mapping == slot
        if rows.any():
            output[rows] = torch.addmm(
                base_sum[rows], rank_sum[rows], shared_b[slot, 0].T
            )
    return output


@pytest.mark.skipif(
    not _CUDA_BF16_AVAILABLE,
    reason="fused shared-outer reduction requires a CUDA GPU with BF16 tensor cores",
)
@pytest.mark.parametrize(
    ("num_tokens", "rank", "routed_scaling_factor", "hidden_width"),
    [
        # Dominant decode shape (BLOCK_M=1) with a ragged trailing column tile.
        (1, 32, 1.75, 137),
        # BLOCK_M=8 with 7 out-of-range rows -- the only param reaching the token
        # mask; scale 1.0 pairs with 1.75 so neither value can be hard-coded.
        (65, 64, 1.0, 137),
    ],
)
def test_fused_base_shared_lora_reduce_cuda_graph_parity(
    num_tokens: int,
    rank: int,
    routed_scaling_factor: float,
    hidden_width: int,
):
    """Graph replay matches sum + rounded rank reduction + shared-B addmm."""

    from sglang.srt.lora.marlin_lora_temp.shared_outer import (
        fused_base_shared_lora_reduce,
        fused_base_shared_lora_reduce_config,
        weighted_topk_rank_sum,
    )

    device = torch.device("cuda")
    topk = 6
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(
        1000 + num_tokens * 100 + rank + hidden_width
    )

    routed_base = (
        torch.randn(
            (num_tokens, topk, hidden_width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * 0.05
    )
    routed_rank = (
        torch.randn(
            (num_tokens, topk, rank),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * 0.05
    )
    topk_weights = torch.softmax(
        torch.randn(
            (num_tokens, topk),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ),
        dim=1,
    ).contiguous()
    shared_b = (
        torch.randn(
            (hidden_width, rank),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * 0.05
    )
    output = torch.empty((num_tokens, hidden_width), device=device, dtype=dtype)
    rank_output = torch.empty((num_tokens, rank), device=device, dtype=dtype)
    topk_ids = torch.zeros((num_tokens, topk), device=device, dtype=torch.int32)
    clean_base = routed_base.clone()
    clean_rank = routed_rank.clone()
    clean_weights = topk_weights.clone()

    # Production launch geometry, so the tuned tiles and their ragged tails run.
    block_m, block_k = fused_base_shared_lora_reduce_config(num_tokens)

    def invoke(with_route_ids=True) -> None:
        route_kwargs = {"topk_ids": topk_ids} if with_route_ids else {}
        fused_base_shared_lora_reduce(
            routed_base,
            routed_rank,
            topk_weights,
            shared_b,
            output,
            routed_scaling_factor,
            block_m=block_m,
            block_k=block_k,
            **route_kwargs,
        )
        weighted_topk_rank_sum(
            routed_rank,
            topk_weights,
            rank_output,
            routed_scaling_factor,
            block_m=1 if num_tokens == 1 else 2,
            **route_kwargs,
        )

    # Existing callers may omit route IDs when every payload is defined.
    invoke(with_route_ids=False)
    expected = _reference_reduce(
        routed_base, routed_rank, topk_weights, shared_b, routed_scaling_factor
    )
    torch.testing.assert_close(output, expected, rtol=_RTOL, atol=_ATOL)
    expected_rank = (
        (routed_rank.float() * topk_weights.unsqueeze(-1))
        .sum(dim=1)
        .mul(routed_scaling_factor)
        .to(dtype)
    )
    torch.testing.assert_close(rank_output, expected_rank, rtol=_RTOL, atol=_ATOL)

    # Specialize and initialize CUDA state off the capture stream.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()

    for replay, route_state in enumerate(("valid", "mixed", "idle", "valid")):
        # In-place mutation: replay must follow pointers, not capture-time values.
        routed_base.copy_(clean_base).mul_(0.75).add_(0.002 * (replay + 1))
        routed_rank.copy_(clean_rank).mul_(-0.5).add_(0.001 * (replay + 1))
        topk_weights.copy_(torch.roll(clean_weights, shifts=replay, dims=1))
        topk_ids.fill_(0)
        if route_state == "mixed":
            topk_ids[:, 0] = -1
            topk_ids[::2, 2] = -3
            if num_tokens > 1:
                topk_ids[-1].fill_(-1)
        elif route_state == "idle":
            topk_ids.fill_(-1)
        inactive = topk_ids < 0
        routed_base.masked_fill_(inactive.unsqueeze(-1), float("nan"))
        routed_rank.masked_fill_(inactive.unsqueeze(-1), float("nan"))
        topk_weights.masked_fill_(inactive, float("nan"))
        shared_b.mul_(0.875).add_(0.0005 * (replay + 1))
        output.fill_(float("nan"))
        rank_output.fill_(float("nan"))

        graph.replay()
        torch.cuda.synchronize()

        expected = _reference_reduce(
            routed_base,
            routed_rank,
            topk_weights,
            shared_b,
            routed_scaling_factor,
            topk_ids,
        )
        torch.testing.assert_close(output, expected, rtol=_RTOL, atol=_ATOL)
        expected_rank = (
            (
                routed_rank.masked_fill(inactive.unsqueeze(-1), 0).float()
                * topk_weights.masked_fill(inactive, 0).unsqueeze(-1)
            )
            .sum(dim=1)
            .mul(routed_scaling_factor)
            .to(dtype)
        )
        torch.testing.assert_close(rank_output, expected_rank, rtol=_RTOL, atol=_ATOL)
        assert bool(torch.isfinite(output).all())
        assert bool(torch.isfinite(rank_output).all())
        if route_state == "mixed":
            assert int(torch.count_nonzero(expected[0])) > 0
        elif route_state == "idle":
            assert int(torch.count_nonzero(output)) == 0
            assert int(torch.count_nonzero(rank_output)) == 0


@pytest.mark.skipif(
    not _CUDA_BF16_AVAILABLE,
    reason="mapped shared-outer reduction requires CUDA BF16 tensor cores",
)
@pytest.mark.parametrize(
    ("num_slots", "num_tokens", "hidden_width"),
    [(2, 1, 128), (3, 32, 137)],
)
def test_fused_base_mapped_shared_lora_reduce_cuda_graph_parity(
    num_slots: int, num_tokens: int, hidden_width: int
):
    """Guards the device-side per-token slot lookup; reds when the adapted-token
    predicate degrades to always-true (``-1`` rows gain a delta) or the lookup
    moves host-side (replay serves the capture-time mapping)."""

    from sglang.srt.lora.marlin_lora_temp.shared_outer import (
        fused_base_mapped_shared_lora_reduce,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(
        7000 + num_tokens * 100 + num_slots * 10 + hidden_width
    )
    routed_base = 0.05 * torch.randn(
        (num_tokens, 6, hidden_width),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    routed_rank = 0.05 * torch.randn(
        (num_tokens, 6, 32),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_weights = torch.softmax(
        torch.randn(
            (num_tokens, 6),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ),
        dim=1,
    ).contiguous()
    shared_b = 0.05 * torch.randn(
        (num_slots, 1, hidden_width, 32),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    token_lora_mapping = torch.arange(
        num_tokens, device=device, dtype=torch.int32
    ).remainder(num_slots)
    if num_tokens > 1:
        token_lora_mapping[-1] = -1
    output = torch.empty(
        (num_tokens, hidden_width), device=device, dtype=torch.bfloat16
    )
    topk_ids = torch.zeros((num_tokens, 6), device=device, dtype=torch.int32)
    clean_base = routed_base.clone()
    clean_rank = routed_rank.clone()
    clean_weights = topk_weights.clone()

    def invoke(with_route_ids=True) -> None:
        route_kwargs = {"topk_ids": topk_ids} if with_route_ids else {}
        fused_base_mapped_shared_lora_reduce(
            routed_base,
            routed_rank,
            topk_weights,
            shared_b,
            token_lora_mapping,
            output,
            1.75,
            block_k=64,
            **route_kwargs,
        )

    invoke(with_route_ids=False)
    expected = _mapped_reference_reduce(
        routed_base, routed_rank, topk_weights, shared_b, token_lora_mapping, 1.75
    )
    torch.testing.assert_close(output, expected, rtol=_RTOL, atol=_ATOL)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()

    for replay, route_state in enumerate(("valid", "mixed", "idle", "valid")):
        token_lora_mapping.copy_((token_lora_mapping + 1).remainder(num_slots))
        if num_tokens > 1:
            token_lora_mapping[0] = -1
        routed_base.copy_(clean_base).mul_(0.75 + 0.05 * replay)
        routed_rank.copy_(clean_rank).mul_(-0.5)
        topk_weights.copy_(torch.roll(clean_weights, shifts=replay, dims=1))
        topk_ids.fill_(0)
        if route_state == "mixed":
            topk_ids[:, 0] = -1
            topk_ids[::2, 2] = -3
            if num_tokens > 1:
                topk_ids[-1].fill_(-1)
        elif route_state == "idle":
            topk_ids.fill_(-1)
        inactive = topk_ids < 0
        routed_base.masked_fill_(inactive.unsqueeze(-1), float("nan"))
        routed_rank.masked_fill_(inactive.unsqueeze(-1), float("nan"))
        topk_weights.masked_fill_(inactive, float("nan"))
        shared_b.mul_(0.875)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()

        expected = _mapped_reference_reduce(
            routed_base,
            routed_rank,
            topk_weights,
            shared_b,
            token_lora_mapping,
            1.75,
            topk_ids,
        )
        torch.testing.assert_close(output, expected, rtol=_RTOL, atol=_ATOL)
        assert bool(torch.isfinite(output).all())
        if route_state == "mixed":
            assert int(torch.count_nonzero(expected[0])) > 0
        elif route_state == "idle":
            assert int(torch.count_nonzero(output)) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
