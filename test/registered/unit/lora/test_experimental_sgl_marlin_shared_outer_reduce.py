"""Guards the fused shared-outer Marlin decode reduction against the three-launch
fallback it replaces; reds when down-B orientation, router weighting, the routing
scale, a ragged-tile mask, or capture-time value baking is wrong. The direct
(no-sort) decode projections must write finite zeros on every inactive route."""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="nightly", runner_config="1-gpu-small")


_CUDA_BF16_AVAILABLE = bool(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] >= 8
)

# Two BF16 ulps (2**-7 each), plus an absolute floor for cancelled outputs.
_RTOL = 1.6e-2
_ATOL = 1e-3


@pytest.mark.skipif(not _CUDA_BF16_AVAILABLE, reason="CUDA BF16 required")
@pytest.mark.parametrize("num_slots", [1, 3])
def test_gate_delta_activation_matches_two_projection_reference(num_slots):
    """Gate/up deltas use independent rank slices, including adapter zero."""
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

    base = torch.full_like(delta, 0.5)
    output = torch.empty((tokens * topk, width), device=device, dtype=dtype)
    silu_and_mul_add_delta(base.view(-1, 2 * width), delta.view(-1, 2 * width), output)
    gate = base[..., :width].float() + expected_delta[..., :width].float()
    up = base[..., width:].float() + expected_delta[..., width:].float()
    torch.testing.assert_close(
        output.view(tokens, topk, width),
        (torch.nn.functional.silu(gate) * up).to(dtype),
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest.mark.skipif(not _CUDA_BF16_AVAILABLE, reason="CUDA BF16 required")
@pytest.mark.parametrize("shared_b", [False, True])
@pytest.mark.parametrize("slots", [1, 3])
def test_private_delta_projection_writes_inactive_routes_on_replay(shared_b, slots):
    from sglang.kernels.ops.lora.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
    from sglang.srt.lora.marlin_lora_temp.activation import silu_and_mul_add_delta

    device, dtype = "cuda", torch.bfloat16
    m, k, e, h, n, r = 6, 2, 2, 32, 32, 8
    ids = torch.tensor([[0, 1]] * m, device=device, dtype=torch.int32)
    mapping = torch.tensor(
        [0, 1, 2, 0, 1, 2], device=device, dtype=torch.int32
    ).remainder(slots)
    enabled = torch.ones(slots * 2, device=device, dtype=torch.int32)[::2]
    x = torch.zeros((m, h), device=device, dtype=dtype)
    x[:, 0] = 0.125
    a = torch.zeros((slots, 1, r, h), device=device, dtype=dtype)
    b = torch.zeros((slots, 1 if shared_b else e, 2 * n, r), device=device, dtype=dtype)
    for slot in range(slots):
        a[slot, 0, 0, 0] = slot + 2
        b[slot, :, :n, 0], b[slot, :, n:, 0] = slot + 3, slot + 5
    weights = torch.ones((m, k), device=device)
    delta = torch.empty((m, k, 2 * n), device=device, dtype=dtype)
    base = torch.full_like(delta, 0.5)
    activated = torch.empty((m * k, n), device=device, dtype=dtype)
    config = dict(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=1,
        num_warps=4,
        num_stages=2,
    )
    kwargs = dict(
        output=delta,
        hidden_states=x,
        lora_a=a,
        lora_b=b,
        topk_ids=ids,
        topk_weights=weights,
        token_lora_mapping=mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=True,
        experts_shared_outer_loras_b=shared_b,
        adapter_enabled=enabled,
        include_inactive_routes=True,
    )

    def run():
        cache = {}
        merged_experts_fused_moe_lora_add(
            **kwargs,
            routing_cache=cache,
            stage="routing",
        )
        merged_experts_fused_moe_lora_add(
            **kwargs,
            routing_cache=cache,
            fuse_add_to_output=False,
        )
        silu_and_mul_add_delta(base.view(-1, 2 * n), delta.view(-1, 2 * n), activated)

    with override_config(config):
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for state in ("valid", "mixed", "disabled", "all-inactive", "recovered"):
            ids.copy_(torch.tensor([[0, 1]] * m, device=device, dtype=ids.dtype))
            mapping.copy_(
                torch.tensor(
                    [0, 1, 2, 0, 1, 2], device=device, dtype=mapping.dtype
                ).remainder(slots)
            )
            enabled.fill_(1)
            if state == "mixed":
                ids[0, 0], mapping[1], mapping[2] = -1, -1, slots
                if slots > 1:
                    enabled[1] = 0
            elif state == "disabled":
                enabled.zero_()
            elif state == "all-inactive":
                ids.fill_(-1)
            live_route = ids >= 0
            safe_slots = mapping.clamp(0, slots - 1).long()
            live_adapter = (
                (mapping >= 0) & (mapping < slots) & (enabled[safe_slots] != 0)
            )
            live = live_route & live_adapter[:, None]
            weights.fill_(1)
            weights.masked_fill_(~live_route, float("nan"))
            # Poison both unwritten destinations and inputs that a disabled
            # producer must not read, rather than relying on zero parameters.
            x.zero_()
            x[:, 0] = 0.125
            x[~live_adapter] = float("nan")
            delta.fill_(float("nan"))
            base.fill_(0.5)
            base.masked_fill_(~live_route.unsqueeze(-1), 0)
            graph.replay()
            expected = torch.zeros_like(delta)
            for row in range(m):
                for route in range(k):
                    if bool(live[row, route]):
                        slot = int(mapping[row])
                        expert = 0 if shared_b else int(ids[row, route])
                        expected[row, route] = (
                            x[row].float() @ a[slot, 0].float().T
                        ).to(dtype).float() @ b[slot, expert].float().T
            torch.testing.assert_close(delta, expected, rtol=0, atol=0)
            gate = base[..., :n].float() + expected[..., :n].float()
            up = base[..., n:].float() + expected[..., n:].float()
            torch.testing.assert_close(
                activated.view(m, k, n),
                (torch.nn.functional.silu(gate) * up).to(dtype),
                rtol=_RTOL,
                atol=_ATOL,
            )
            if state == "mixed":
                added = torch.full_like(delta, 0.5)
                merged_experts_fused_moe_lora_add(
                    **{**kwargs, "output": added},
                    fuse_add_to_output=True,
                )
                torch.testing.assert_close(
                    added, (expected.float() + 0.5).to(dtype), rtol=0, atol=0
                )
        # Routing contains no destination identity. Each actual overwrite must
        # initialize its own omitted rows even when metadata is reused.
        cache = {}
        merged_experts_fused_moe_lora_add(
            **kwargs,
            routing_cache=cache,
            stage="routing",
        )
        second_output = torch.full_like(delta, float("nan"))
        merged_experts_fused_moe_lora_add(
            **{**kwargs, "output": second_output},
            routing_cache=cache,
            fuse_add_to_output=False,
        )
        torch.testing.assert_close(second_output, expected, rtol=0, atol=0)


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
    zero_owned_rank = torch.empty_like(routed_rank)
    zero_owned_output = torch.empty_like(rank_output)
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

    def invoke_zero_owned_rank():
        weighted_topk_rank_sum(
            zero_owned_rank,
            topk_weights,
            zero_owned_output,
            routed_scaling_factor,
            block_m=1 if num_tokens == 1 else 2,
            topk_ids=topk_ids,
            inactive_ranks_are_zero=True,
        )

    zero_owned_rank.copy_(routed_rank)
    invoke_zero_owned_rank()
    torch.cuda.synchronize()
    zero_owned_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(zero_owned_graph):
        invoke_zero_owned_rank()

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
        zero_owned_rank.copy_(routed_rank).masked_fill_(inactive.unsqueeze(-1), 0)
        zero_owned_graph.replay()
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
        torch.testing.assert_close(zero_owned_output, rank_output, rtol=0, atol=0)
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


@pytest.mark.skipif(not _CUDA_BF16_AVAILABLE, reason="CUDA BF16 required")
def test_marlin_lora_direct_inactive_routes():
    """Direct LoRA projections overwrite inactive routes with finite zeros."""
    from sglang.srt.lora.marlin_lora_temp.direct_decode import (
        direct_decode_down_shrink,
        direct_decode_gate_expand,
    )

    torch.manual_seed(51)
    rows, topk, experts, intermediate, rank = 3, 6, 8, 384, 32
    ids = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [-1, -1, -1, -1, -1, -1], [2, -1, 4, 5, -1, 7]],
        device="cuda",
        dtype=torch.int32,
    )
    slots = torch.zeros(rows, device="cuda", dtype=torch.int32)
    shared = torch.randn((rows, 2 * rank), device="cuda", dtype=torch.bfloat16)
    gate_b = (
        torch.randn(
            (1, experts, 2 * intermediate, rank), device="cuda", dtype=torch.bfloat16
        )
        * 0.05
    )
    down_a = (
        torch.randn(
            (1, experts, rank, intermediate), device="cuda", dtype=torch.bfloat16
        )
        * 0.05
    )
    activation = torch.randn(
        (rows * topk, intermediate), device="cuda", dtype=torch.bfloat16
    )
    activation.view(rows, topk, intermediate)[ids < 0] = float("nan")
    gate_output = torch.full(
        (rows, topk, 2 * intermediate),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    down_output = torch.full(
        (rows, topk, rank), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    direct_decode_gate_expand(shared, gate_b, ids, slots, gate_output)
    direct_decode_down_shrink(activation, down_a, ids, slots, down_output)
    expected_gate = torch.zeros_like(gate_output)
    expected_down = torch.zeros_like(down_output)
    for token in range(rows):
        for route in range(topk):
            expert = int(ids[token, route])
            if expert < 0:
                continue
            for half in range(2):
                start = half * intermediate
                expected_gate[token, route, start : start + intermediate] = (
                    gate_b[0, expert, start : start + intermediate].float()
                    @ shared[token, half * rank : (half + 1) * rank].float()
                ).to(gate_output.dtype)
            expected_down[token, route] = (
                down_a[0, expert].float() @ activation[token * topk + route].float()
            ).to(down_output.dtype)
    torch.testing.assert_close(gate_output, expected_gate, rtol=0.03, atol=0.01)
    torch.testing.assert_close(down_output, expected_down, rtol=0.03, atol=0.01)
    assert torch.count_nonzero(gate_output[ids < 0]) == 0
    assert torch.count_nonzero(down_output[ids < 0]) == 0

    # A dense oracle for both real adapter slots keeps the selected projection
    # independent of the kernels' inactive-address and payload masks.
    finite_shared = torch.stack((shared, shared * 2))
    finite_gate_b = torch.cat((gate_b, gate_b * 1.5))
    finite_down_a = torch.cat((down_a, down_a * -0.75))
    finite_activation = torch.randn_like(activation)
    dense_gate = (
        torch.einsum(
            "smhr,sehir->smehi",
            finite_shared.view(2, rows, 2, rank).float(),
            finite_gate_b.view(2, experts, 2, intermediate, rank).float(),
        )
        .flatten(-2)
        .to(gate_output.dtype)
    )
    dense_down = torch.einsum(
        "mti,seri->smter",
        finite_activation.view(rows, topk, intermediate).float(),
        finite_down_a.float(),
    ).to(down_output.dtype)
    slot_shared = finite_shared.clone()
    slot_gate_b = finite_gate_b.clone()
    slot_down_a = finite_down_a.clone()
    route_ids = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(rows, 1)
    route_index = torch.arange(topk, device="cuda")
    states = (
        ("valid", (0, 1, 0), (1, 1)),
        ("disabled", (0, 1, 0), (1, 0)),
        ("missing-or-out-of-range", (0, -1, 2), (1, 1)),
        ("zero", (0, 1, 0), (1, 1)),
        ("recovery", (0, 1, 0), (1, 1)),
    )
    for enabled_dtype in (torch.bool, torch.int32, torch.int64):
        # The optional metadata accepts a strided one-dimensional view.
        enabled = torch.ones(4, device="cuda", dtype=enabled_dtype)[::2]

        def project_enabled():
            direct_decode_gate_expand(
                slot_shared,
                slot_gate_b,
                ids,
                slots,
                gate_output,
                adapter_enabled=enabled,
            )
            direct_decode_down_shrink(
                activation,
                slot_down_a,
                ids,
                slots,
                down_output,
                adapter_enabled=enabled,
            )

        ids.copy_(route_ids)
        slots.copy_(torch.tensor(states[0][1], device="cuda", dtype=slots.dtype))
        activation.copy_(finite_activation)
        project_enabled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            project_enabled()
        for state, mapping, flags in states:
            ids.copy_(route_ids)
            slots.copy_(torch.tensor(mapping, device="cuda", dtype=slots.dtype))
            enabled.copy_(torch.tensor(flags, device="cuda", dtype=enabled_dtype))
            slot_shared.copy_(finite_shared)
            slot_gate_b.copy_(finite_gate_b)
            slot_down_a.copy_(finite_down_a)
            activation.copy_(finite_activation)
            if state == "disabled":
                slot_shared[1].fill_(float("nan"))
                slot_gate_b[1].fill_(float("nan"))
                slot_down_a[1].fill_(float("nan"))
                activation.view(rows, topk, intermediate)[1].fill_(float("nan"))
            elif state == "missing-or-out-of-range":
                slot_shared[:, 1:].fill_(float("nan"))
                activation.view(rows, topk, intermediate)[1:].fill_(float("nan"))
            elif state == "zero":
                ids.fill_(-1)
                slot_shared.fill_(float("nan"))
                slot_gate_b.fill_(float("nan"))
                slot_down_a.fill_(float("nan"))
                activation.fill_(float("nan"))
            gate_output.fill_(float("nan"))
            down_output.fill_(float("nan"))
            graph.replay()
            expected_gate.zero_()
            expected_down.zero_()
            active_rows = []
            if state != "zero":
                for token, slot in enumerate(mapping):
                    if 0 <= slot < len(flags) and flags[slot]:
                        active_rows.append(token)
                        expected_gate[token] = dense_gate[slot, token, route_index]
                        expected_down[token] = dense_down[
                            slot, token, route_index, route_index
                        ]
            torch.testing.assert_close(gate_output, expected_gate, rtol=0.03, atol=0.01)
            torch.testing.assert_close(down_output, expected_down, rtol=0.03, atol=0.01)
            assert torch.isfinite(gate_output).all()
            assert torch.isfinite(down_output).all()
            inactive_rows = [token for token in range(rows) if token not in active_rows]
            assert torch.count_nonzero(gate_output[inactive_rows]) == 0
            assert torch.count_nonzero(down_output[inactive_rows]) == 0
            if active_rows:
                # Slot zero is a loaded adapter and must remain a real projection.
                assert 0 in active_rows
                assert torch.count_nonzero(expected_gate[0]) > 0
                assert torch.count_nonzero(expected_down[0]) > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
