"""CUDA parity for the multi-slot shared-outer Marlin prefill factorization."""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")

# The fused MoE LoRA-add kernel's shared-memory footprint exceeds the opt-in
# ceiling of the small-GPU CI runner (~99 KiB on L4) at rank=128, so the
# generic-fallback parity case OOMs there. Skip this file on CI rather than
# shrink the production kernel to a small-GPU block config.
pytestmark = pytest.mark.skip(
    reason="fused MoE LoRA-add kernel needs more opt-in shared memory than the "
    "small-GPU CI runner provides"
)


_CUDA_BF16_AVAILABLE = bool(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] >= 8
)


def _set_mapping(mapping: torch.Tensor, num_slots: int, offset: int) -> None:
    """Select active slots while exercising both no-adapter encodings."""

    rows = torch.arange(mapping.numel(), dtype=torch.int32)
    values = (rows + offset).remainder(num_slots - 1).add_(1)
    values[(rows + 2 * offset).remainder(7) == 0] = 0
    values[(rows + 3 * offset + 1).remainder(11) == 0] = -1
    mapping.copy_(values.to(mapping.device))


def _reference_gate(
    hidden_states: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    topk_ids: torch.Tensor,
    mapping: torch.Tensor,
) -> torch.Tensor:
    """Materialize shrink/expand with the same BF16 stage boundary."""

    active = mapping >= 0
    slots = mapping.clamp_min(0).long()
    experts = topk_ids.long()
    rank = gate_b.shape[-1]
    intermediate_size = gate_b.shape[2] // 2

    selected_a = gate_a[slots, 0]
    shared_rank = torch.einsum(
        "mh,mrh->mr", hidden_states.float(), selected_a.float()
    ).to(hidden_states.dtype)
    selected_b = gate_b[slots[:, None], experts]

    gate = torch.einsum(
        "mkr,mkir->mki",
        shared_rank[:, None, :rank].expand(-1, experts.shape[1], -1).float(),
        selected_b[:, :, :intermediate_size].float(),
    )
    up = torch.einsum(
        "mkr,mkir->mki",
        shared_rank[:, None, rank:].expand(-1, experts.shape[1], -1).float(),
        selected_b[:, :, intermediate_size:].float(),
    )
    output = torch.cat((gate, up), dim=-1).to(hidden_states.dtype)
    output[~active] = 0
    return output


def _reference_down(
    activation: torch.Tensor,
    down_a: torch.Tensor,
    down_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    mapping: torch.Tensor,
    base_output: torch.Tensor,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Materialize routed shrink, weighted rank sum, and selected shared B."""

    slots = mapping.clamp_min(0).long()
    experts = topk_ids.long()
    selected_a = down_a[slots[:, None], experts]
    routed_rank = torch.einsum(
        "mki,mkri->mkr", activation.float(), selected_a.float()
    ).to(activation.dtype)
    rank_sum = (
        (routed_rank.float() * topk_weights.float().unsqueeze(-1))
        .sum(dim=1)
        .mul(routed_scaling_factor)
        .to(activation.dtype)
    )

    output = base_output.clone()
    for slot in range(down_b.shape[0]):
        rows = mapping == slot
        if rows.any():
            output[rows] = torch.addmm(
                base_output[rows], rank_sum[rows], down_b[slot, 0].T
            )
    return output


def _reference_generic_delta(
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    mapping: torch.Tensor,
    *,
    shared_a: bool,
    shared_b: bool,
    mul_routed_weight: bool,
) -> torch.Tensor:
    num_tokens, topk = topk_ids.shape
    output = hidden_states.new_zeros(num_tokens, topk, lora_b.shape[2])
    routed_inputs = hidden_states.shape[0] == topk_ids.numel()
    for token in range(num_tokens):
        slot = int(mapping[token])
        if slot < 0:
            continue
        for route in range(topk):
            expert = int(topk_ids[token, route])
            if expert < 0:
                continue
            row = token * topk + route if routed_inputs else token
            a = lora_a[slot, 0 if shared_a else expert]
            b = lora_b[slot, 0 if shared_b else expert]
            shrink = torch.mv(a.float(), hidden_states[row].float()).to(
                hidden_states.dtype
            )
            rank = b.shape[1]
            if shrink.numel() == 2 * rank:
                output_half = b.shape[0] // 2
                delta = torch.cat(
                    (
                        torch.mv(b[:output_half].float(), shrink[:rank].float()),
                        torch.mv(b[output_half:].float(), shrink[rank:].float()),
                    )
                ).to(hidden_states.dtype)
            else:
                delta = torch.mv(b.float(), shrink.float()).to(hidden_states.dtype)
            if mul_routed_weight:
                delta.mul_(topk_weights[token, route])
            output[token, route] = delta
    return output


def _run_factored_pipeline(
    *,
    hidden_states: torch.Tensor,
    activation: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    down_a: torch.Tensor,
    down_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    mapping: torch.Tensor,
    gate_rank: torch.Tensor,
    gate_output: torch.Tensor,
    down_routed_rank: torch.Tensor,
    down_rank_sum: torch.Tensor,
    down_output: torch.Tensor,
    full_routing_cache: dict,
    collapsed_routing_cache: dict,
    routed_scaling_factor: float,
) -> None:
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.lora.marlin_lora_temp.shared_outer import weighted_topk_rank_sum

    num_tokens = topk_ids.shape[0]
    num_experts = gate_b.shape[1]
    collapsed_ids = mapping.view(num_tokens, 1)
    collapsed_weights = topk_weights[:, :1]

    # Capture the same four routing domains as the production schedule.  The
    # dictionaries must remain distinct because full and collapsed top-k have
    # different flattened token domains.
    merged_experts_fused_moe_lora_add(
        output=gate_output,
        hidden_states=hidden_states,
        lora_a=gate_a,
        lora_b=gate_b,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=True,
        experts_shared_outer_loras_b=False,
        routing_cache=full_routing_cache,
        stage="routing",
        prewarm_a_routing=False,
        prewarm_b_routing=True,
        local_expert_offset=0,
        local_num_experts=num_experts,
    )

    merged_experts_fused_moe_lora_add(
        output=gate_output,
        hidden_states=hidden_states,
        lora_a=gate_a,
        lora_b=gate_b,
        topk_ids=collapsed_ids,
        topk_weights=collapsed_weights,
        token_lora_mapping=mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=True,
        experts_shared_outer_loras_b=False,
        routing_cache=collapsed_routing_cache,
        stage="routing",
        prewarm_a_routing=True,
        prewarm_b_routing=False,
        local_expert_offset=0,
        local_num_experts=num_experts,
    )
    merged_experts_fused_moe_lora_add(
        output=activation,
        hidden_states=activation,
        lora_a=down_a,
        lora_b=down_b,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=mapping,
        mul_routed_weight=True,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=True,
        routing_cache=full_routing_cache,
        stage="routing",
        prewarm_a_routing=True,
        prewarm_b_routing=False,
        local_expert_offset=0,
        local_num_experts=num_experts,
    )
    merged_experts_fused_moe_lora_add(
        output=down_output,
        hidden_states=down_rank_sum,
        lora_a=down_a,
        lora_b=down_b,
        topk_ids=collapsed_ids,
        topk_weights=collapsed_weights,
        token_lora_mapping=mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=True,
        routing_cache=collapsed_routing_cache,
        stage="routing",
        prewarm_a_routing=False,
        prewarm_b_routing=True,
        local_expert_offset=0,
        local_num_experts=num_experts,
    )

    merged_experts_fused_moe_lora_add(
        output=gate_output,
        hidden_states=hidden_states,
        lora_a=gate_a,
        lora_b=gate_b,
        topk_ids=collapsed_ids,
        topk_weights=collapsed_weights,
        token_lora_mapping=mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=True,
        experts_shared_outer_loras_b=False,
        routing_cache=collapsed_routing_cache,
        stage="shrink",
        prewarm_b_routing=False,
        intermediate_buffer=gate_rank,
        local_expert_offset=0,
        local_num_experts=num_experts,
    )
    merged_experts_fused_moe_lora_add(
        output=gate_output,
        hidden_states=hidden_states,
        lora_a=gate_a,
        lora_b=gate_b,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=True,
        experts_shared_outer_loras_b=False,
        routing_cache=full_routing_cache,
        fuse_add_to_output=False,
        use_direct_expand_add=True,
        stage="expand",
        intermediate_buffer=gate_rank,
        broadcast_intermediate=True,
        local_expert_offset=0,
        local_num_experts=num_experts,
    )

    merged_experts_fused_moe_lora_add(
        output=activation,
        hidden_states=activation,
        lora_a=down_a,
        lora_b=down_b,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=mapping,
        mul_routed_weight=True,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=True,
        routing_cache=full_routing_cache,
        stage="shrink",
        prewarm_b_routing=False,
        intermediate_buffer=down_routed_rank,
        local_expert_offset=0,
        local_num_experts=num_experts,
    )
    weighted_topk_rank_sum(
        down_routed_rank,
        topk_weights,
        down_rank_sum,
        routed_scaling_factor,
        block_m=1,
    )
    merged_experts_fused_moe_lora_add(
        output=down_output,
        hidden_states=down_rank_sum,
        lora_a=down_a,
        lora_b=down_b,
        topk_ids=collapsed_ids,
        topk_weights=collapsed_weights,
        token_lora_mapping=mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=True,
        routing_cache=collapsed_routing_cache,
        fuse_add_to_output=True,
        use_direct_expand_add=False,
        stage="expand",
        intermediate_buffer=down_rank_sum,
        local_expert_offset=0,
        local_num_experts=num_experts,
    )


@pytest.mark.skipif(
    not _CUDA_BF16_AVAILABLE,
    reason="multi-prefill parity requires a CUDA GPU with BF16 tensor cores",
)
@pytest.mark.parametrize(
    ("num_slots", "num_tokens"), [(2, 33), (3, 64), (4, 65), (5, 33), (8, 64), (16, 65)]
)
def test_multi_shared_outer_prefill_cuda_graph_parity(
    num_slots: int, num_tokens: int
) -> None:
    """Replay full+collapsed routing while adapter selections change in place."""

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_experts, router_topk = 4, 2
    hidden_size = intermediate_size = 128
    rank = 16
    scale = 1.25
    generator = torch.Generator(device=device).manual_seed(2027 + num_slots)

    def randn(*shape: int) -> torch.Tensor:
        return (
            torch.randn(*shape, device=device, dtype=dtype, generator=generator) * 0.05
        )

    hidden_states = randn(num_tokens, hidden_size)
    activation = randn(num_tokens, router_topk, intermediate_size)
    gate_a = randn(num_slots, 1, 2 * rank, hidden_size)
    gate_b = randn(num_slots, num_experts, 2 * intermediate_size, rank)
    down_a = randn(num_slots, num_experts, rank, intermediate_size)
    down_b = randn(num_slots, 1, hidden_size, rank)
    # Slot 0 is the production base/None representation: it remains a valid
    # mapping value, while every attached operand is zero in address-stable pool
    # storage.  This catches accidental stale reads that a -1-only test misses.
    gate_a[0].zero_()
    gate_b[0].zero_()
    down_a[0].zero_()
    down_b[0].zero_()

    token = torch.arange(num_tokens, device=device, dtype=torch.int32)[:, None]
    route = torch.arange(router_topk, device=device, dtype=torch.int32)[None, :]
    topk_ids = (token + 2 * route).remainder(num_experts).contiguous()
    topk_weights = (
        torch.tensor([0.25, 0.75], device=device, dtype=torch.float32)
        .expand(num_tokens, -1)
        .contiguous()
    )
    mapping = torch.empty(num_tokens, device=device, dtype=torch.int32)
    _set_mapping(mapping, num_slots, offset=0)

    gate_rank = torch.empty(num_tokens, 2 * rank, device=device, dtype=dtype)
    gate_output = torch.empty(
        num_tokens,
        router_topk,
        2 * intermediate_size,
        device=device,
        dtype=dtype,
    )
    down_routed_rank = torch.empty(
        num_tokens, router_topk, rank, device=device, dtype=dtype
    )
    down_rank_sum = torch.empty(num_tokens, rank, device=device, dtype=dtype)
    base_output = randn(num_tokens, hidden_size)
    down_output = base_output.clone()

    common = dict(
        hidden_states=hidden_states,
        activation=activation.view(num_tokens * router_topk, intermediate_size),
        gate_a=gate_a,
        gate_b=gate_b,
        down_a=down_a,
        down_b=down_b,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        mapping=mapping,
        gate_rank=gate_rank,
        gate_output=gate_output,
        down_routed_rank=down_routed_rank,
        down_rank_sum=down_rank_sum,
        down_output=down_output,
        routed_scaling_factor=scale,
    )

    # Compile JIT and initialize CUDA-library state outside capture.  These
    # throwaway caches deliberately do not enter the captured graph.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        _run_factored_pipeline(
            **common, full_routing_cache={}, collapsed_routing_cache={}
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    full_routing_cache: dict = {}
    collapsed_routing_cache: dict = {}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_factored_pipeline(
            **common,
            full_routing_cache=full_routing_cache,
            collapsed_routing_cache=collapsed_routing_cache,
        )

    assert full_routing_cache and collapsed_routing_cache
    routing_tensors = tuple(
        tensor
        for cache in (full_routing_cache, collapsed_routing_cache)
        for value in cache.values()
        for tensor in value
    )
    stable_tensors = (
        mapping,
        gate_rank,
        gate_output,
        down_routed_rank,
        down_rank_sum,
        down_output,
        *routing_tensors,
    )
    stable_addresses = tuple(tensor.data_ptr() for tensor in stable_tensors)

    for offset in (1, 3):
        _set_mapping(mapping, num_slots, offset)
        assert torch.any(mapping == -1).item()
        assert torch.any(mapping == 0).item()
        gate_rank.fill_(float("nan"))
        gate_output.fill_(float("nan"))
        down_routed_rank.fill_(float("nan"))
        down_rank_sum.fill_(float("nan"))
        down_output.copy_(base_output)

        expected_gate = _reference_gate(
            hidden_states, gate_a, gate_b, topk_ids, mapping
        )
        expected_down = _reference_down(
            activation,
            down_a,
            down_b,
            topk_ids,
            topk_weights,
            mapping,
            base_output,
            scale,
        )
        graph.replay()
        torch.cuda.synchronize()

        assert tuple(tensor.data_ptr() for tensor in stable_tensors) == stable_addresses
        torch.testing.assert_close(gate_output, expected_gate, rtol=0.025, atol=0.025)
        torch.testing.assert_close(down_output, expected_down, rtol=0.025, atol=0.025)
        base_rows = mapping <= 0
        assert torch.count_nonzero(gate_output[base_rows]).item() == 0
        torch.testing.assert_close(
            down_output[base_rows], base_output[base_rows], rtol=0, atol=0
        )
        assert torch.isfinite(gate_output).all().item()
        assert torch.isfinite(down_output).all().item()


@pytest.mark.skipif(
    not _CUDA_BF16_AVAILABLE,
    reason="generic fallback parity requires a CUDA GPU with BF16 tensor cores",
)
@pytest.mark.parametrize(
    ("num_slots", "rank", "shared_outer", "ep"),
    [(5, 32, True, False), (8, 128, True, True), (16, 128, False, True)],
)
def test_generic_fallback_cuda_graph_parity(
    num_slots: int, rank: int, shared_outer: bool, ep: bool
) -> None:
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, topk = 8, 2
    num_experts = 2 if ep else 4
    hidden_size = intermediate_size = 64
    generator = torch.Generator(device=device).manual_seed(3100 + num_slots + rank)

    def randn(*shape: int) -> torch.Tensor:
        return (
            torch.randn(*shape, device=device, dtype=dtype, generator=generator) * 0.03
        )

    gate_a = randn(num_slots, 1 if shared_outer else num_experts, 2 * rank, hidden_size)
    gate_b = randn(num_slots, num_experts, 2 * intermediate_size, rank)
    down_a = randn(num_slots, num_experts, rank, intermediate_size)
    down_b = randn(num_slots, 1 if shared_outer else num_experts, hidden_size, rank)
    hidden_states = randn(num_tokens, hidden_size)
    activation = randn(num_tokens * topk, intermediate_size)
    topk_ids = (
        torch.arange(num_tokens * topk, device=device, dtype=torch.int32)
        .remainder(num_experts)
        .view(num_tokens, topk)
    )
    if ep:
        topk_ids[::2, 1] = -1
    topk_weights = torch.tensor([0.4, 0.6], device=device, dtype=torch.float32).expand(
        num_tokens, -1
    )
    mapping = torch.arange(num_tokens, device=device, dtype=torch.int32).remainder(
        num_slots
    )
    mapping[::5] = -1
    gate_output = torch.empty(
        num_tokens, topk, 2 * intermediate_size, device=device, dtype=dtype
    )
    down_output = torch.empty(num_tokens, topk, hidden_size, device=device, dtype=dtype)

    def run(gate_cache: dict, down_cache: dict) -> None:
        gate_output.zero_()
        merged_experts_fused_moe_lora_add(
            output=gate_output,
            hidden_states=hidden_states,
            lora_a=gate_a,
            lora_b=gate_b,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=shared_outer,
            experts_shared_outer_loras_b=False,
            routing_cache=gate_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=True,
            local_num_experts=num_experts,
        )
        down_output.zero_()
        merged_experts_fused_moe_lora_add(
            output=down_output,
            hidden_states=activation,
            lora_a=down_a,
            lora_b=down_b,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=shared_outer,
            routing_cache=down_cache,
            use_direct_expand_add=False,
            local_num_experts=num_experts,
            zero_intermediate=ep and shared_outer,
        )

    def assert_expected() -> None:
        torch.testing.assert_close(
            gate_output,
            _reference_generic_delta(
                hidden_states,
                gate_a,
                gate_b,
                topk_ids,
                topk_weights,
                mapping,
                shared_a=shared_outer,
                shared_b=False,
                mul_routed_weight=False,
            ),
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            down_output,
            _reference_generic_delta(
                activation,
                down_a,
                down_b,
                topk_ids,
                topk_weights,
                mapping,
                shared_a=False,
                shared_b=shared_outer,
                mul_routed_weight=True,
            ),
            rtol=2e-2,
            atol=2e-2,
        )

    run({}, {})
    torch.cuda.synchronize()
    assert_expected()

    gate_cache: dict = {}
    down_cache: dict = {}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(gate_cache, down_cache)

    for offset in (0, 1):
        mapping.copy_(
            torch.arange(num_tokens, device=device, dtype=torch.int32)
            .add(offset)
            .remainder(num_slots)
        )
        mapping[(torch.arange(num_tokens, device=device) + offset) % 5 == 0] = -1
        graph.replay()
        assert_expected()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
