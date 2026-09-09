"""``SWAP_OUT_HALVES`` on the rank-specialized LoRA-B expand.

The gated gate_up expand emits ``[gate_delta | up_delta]``. A consumer that
stacks its base gate_up weight the other way round needs ``[up_delta | gate]``,
and ``swap_out_halves`` gives it that for free by shifting the store column by
``±N/2``. Two properties are checked:

- **off is today's kernel.** The flag defaults to ``False``, and every existing
  caller (including the Marlin overlay) relies on that, so the default must
  reproduce the torch reference exactly.
- **on is exactly the halves exchanged.** Not "close to": the swap changes the
  store address only, so the swapped result must be bit-identical to the
  unswapped one with its two ``N/2``-wide halves rolled.
- **both ways of missing the kernel are rejected.** The merged API only reaches
  this kernel when ``use_direct_expand_add`` is set AND the LoRA-B is not
  shared-outer; either condition alone falls back to the generic stock expand,
  which cannot swap, so both get a rejection test rather than a silent
  ``[gate | up]``.

Inputs are small integers held in bf16. Every product and partial sum is then
exactly representable in both fp32 (the accumulator) and bf16 (the store), so
the torch reference comparison is ``torch.equal`` rather than a tolerance.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.moe.specialized_expand import _invoke_moe_lora_expand_add
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

# (num_tokens, top_k, rank, intermediate, block_size_m, block_size_n, num_experts).
# inter=384 is Inkling at TP=8 and takes the N % 128 == 0 default tile; inter=96
# forces the launcher to halve BLOCK_SIZE_N (64 -> 32) to divide N/2, which is the
# case where "every tile sits wholly inside one half" is least obvious.
CASES = [
    (8, 2, 16, 384, 16, 128, 4),
    (5, 3, 16, 96, 32, 64, 3),
    (1, 6, 32, 128, 16, 128, 2),
]


def _config(block_size_m: int, block_size_n: int) -> dict:
    return {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
    }


def _build_routing(virtual_expert_ids: torch.Tensor, block_size_m: int):
    """Group the flattened ``(token, slot)`` slots by virtual expert, block-padded.

    Same shape contract as ``moe_align_block_size``: ``sorted_token_ids`` holds
    expanded slot indices bucketed per expert and padded up to ``block_size_m``
    with ``numel`` (an index the kernel masks off as invalid), ``expert_ids``
    holds one expert per block, and ``-1`` is the no-adapter bucket whose rows
    the kernel must zero.
    """
    flat = virtual_expert_ids.reshape(-1)
    numel = flat.numel()
    sorted_ids: list[int] = []
    expert_ids: list[int] = []
    for expert in sorted(set(flat.tolist())):
        slots = (flat == expert).nonzero(as_tuple=True)[0].tolist()
        for start in range(0, len(slots), block_size_m):
            block = slots[start : start + block_size_m]
            sorted_ids.extend(block + [numel] * (block_size_m - len(block)))
            expert_ids.append(expert)
    dev = virtual_expert_ids.device
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=dev),
        torch.tensor(expert_ids, dtype=torch.int32, device=dev),
        torch.tensor([len(sorted_ids)], dtype=torch.int32, device=dev),
    )


def _reference(
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size_m: int,
    num_valid_tokens: int,
    topk_weights: torch.Tensor,
    mul_routed_weight: bool,
    swap_out_halves: bool,
) -> torch.Tensor:
    """Block-by-block torch model of the expand, including the gate/up A split."""
    rank = weight.shape[2]
    n_total = weight.shape[1]
    half = n_total // 2
    out = torch.zeros(
        num_valid_tokens, n_total, dtype=intermediate.dtype, device=intermediate.device
    )
    weights_flat = topk_weights.reshape(-1)
    for block, expert in enumerate(expert_ids.tolist()):
        slots = sorted_token_ids[block * block_size_m : (block + 1) * block_size_m]
        slots = slots[slots < num_valid_tokens].long()
        if slots.numel() == 0:
            continue
        if expert < 0:
            # No adapter for these tokens: the kernel stores zeros over all N.
            out[slots] = 0
            continue
        # The gate half of the output contracts A[:, 0:R], the up half A[:, R:2R].
        gate = intermediate[slots, :rank].float() @ weight[expert, :half, :].float().t()
        up = (
            intermediate[slots, rank : 2 * rank].float()
            @ weight[expert, half:, :].float().t()
        )
        acc = torch.cat((gate, up), dim=-1)
        if mul_routed_weight:
            acc = acc * weights_flat[slots].unsqueeze(1)
        if swap_out_halves:
            acc = torch.cat((acc[:, half:], acc[:, :half]), dim=-1)
        out[slots] = acc.to(out.dtype)
    return out


def _make_case(num_tokens, top_k, rank, inter, num_experts, seed):
    """Small-integer bf16 operands, plus a routing table with a no-adapter bucket."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rows = num_tokens * top_k
    n_total = 2 * inter

    def _ints(*shape):
        return (
            torch.randint(-2, 3, shape, generator=gen, dtype=torch.int32)
            .to(torch.bfloat16)
            .cuda()
        )

    intermediate = _ints(rows, 2 * rank)
    weight = _ints(num_experts, n_total, rank)
    # Half-integer routing weights keep the scaled result exactly representable.
    topk_weights = (
        torch.randint(1, 3, (num_tokens, top_k), generator=gen, dtype=torch.int32)
        .to(torch.float32)
        .div(2)
        .cuda()
    )
    # -1 is the no-adapter bucket, so every case exercises the zero-fill branch
    # alongside the real experts.
    virtual_expert_ids = torch.randint(
        -1, num_experts, (num_tokens, top_k), generator=gen, dtype=torch.int32
    ).cuda()
    topk_ids = torch.zeros(
        (num_tokens, top_k), dtype=torch.int32, device="cuda"
    )  # only numel() and shape[1] are read
    return intermediate, weight, topk_weights, topk_ids, virtual_expert_ids


def _run(
    intermediate,
    weight,
    topk_weights,
    topk_ids,
    routing,
    block_size_m,
    block_size_n,
    mul_routed_weight,
    swap_out_halves,
):
    sorted_token_ids, expert_ids, num_tokens_post_padded = routing
    n_total = weight.shape[1]
    # torch.empty, like production: the kernel owes every valid row a store.
    output = torch.full(
        (topk_ids.numel(), n_total),
        float("nan"),
        dtype=intermediate.dtype,
        device=intermediate.device,
    )
    _invoke_moe_lora_expand_add(
        intermediate,
        weight,
        output,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        _config(block_size_m, block_size_n),
        mul_routed_weight,
        False,  # fuse_sum_all_reduce
        swap_out_halves=swap_out_halves,
    )
    return output


@requires_cuda
@pytest.mark.parametrize(
    "num_tokens,top_k,rank,inter,block_size_m,block_size_n,num_experts", CASES
)
@pytest.mark.parametrize("mul_routed_weight", [False, True])
def test_default_off_matches_reference(
    num_tokens,
    top_k,
    rank,
    inter,
    block_size_m,
    block_size_n,
    num_experts,
    mul_routed_weight,
):
    intermediate, weight, topk_weights, topk_ids, virtual_expert_ids = _make_case(
        num_tokens, top_k, rank, inter, num_experts, seed=inter + top_k
    )
    routing = _build_routing(virtual_expert_ids, block_size_m)

    got = _run(
        intermediate,
        weight,
        topk_weights,
        topk_ids,
        routing,
        block_size_m,
        block_size_n,
        mul_routed_weight,
        swap_out_halves=False,
    )
    ref = _reference(
        intermediate,
        weight,
        routing[0],
        routing[1],
        block_size_m,
        topk_ids.numel(),
        topk_weights,
        mul_routed_weight,
        swap_out_halves=False,
    )
    assert torch.equal(got, ref)


@requires_cuda
@pytest.mark.parametrize(
    "num_tokens,top_k,rank,inter,block_size_m,block_size_n,num_experts", CASES
)
@pytest.mark.parametrize("mul_routed_weight", [False, True])
def test_swap_exchanges_the_two_halves(
    num_tokens,
    top_k,
    rank,
    inter,
    block_size_m,
    block_size_n,
    num_experts,
    mul_routed_weight,
):
    intermediate, weight, topk_weights, topk_ids, virtual_expert_ids = _make_case(
        num_tokens, top_k, rank, inter, num_experts, seed=inter + top_k
    )
    routing = _build_routing(virtual_expert_ids, block_size_m)
    args = (
        intermediate,
        weight,
        topk_weights,
        topk_ids,
        routing,
        block_size_m,
        block_size_n,
        mul_routed_weight,
    )

    plain = _run(*args, swap_out_halves=False)
    swapped = _run(*args, swap_out_halves=True)

    # Same arithmetic, different store column -> bit-identical, not merely close.
    assert torch.equal(swapped, torch.cat((plain[:, inter:], plain[:, :inter]), dim=-1))
    # Guard against a vacuous pass on a fixture whose halves happen to agree.
    assert not torch.equal(swapped, plain)
    assert torch.equal(
        swapped,
        _reference(
            intermediate,
            weight,
            routing[0],
            routing[1],
            block_size_m,
            topk_ids.numel(),
            topk_weights,
            mul_routed_weight,
            swap_out_halves=True,
        ),
    )


@requires_cuda
def test_no_adapter_rows_stay_zero_under_swap():
    """The ``expert == -1`` branch stores at the unswapped column on purpose.

    ``off_expert`` is uniform over ``pid_n``, so every n-tile of such an m-block
    takes that branch and zeroes all N columns either way. Assert the bits.
    """
    num_tokens, top_k, rank, inter, block_size_m, block_size_n = 6, 2, 16, 384, 16, 128
    intermediate, weight, topk_weights, topk_ids, _ = _make_case(
        num_tokens, top_k, rank, inter, num_experts=3, seed=5
    )
    # Every slot of tokens 1 and 4 lands in the no-adapter bucket.
    virtual_expert_ids = torch.ones(
        (num_tokens, top_k), dtype=torch.int32, device="cuda"
    )
    virtual_expert_ids[1] = -1
    virtual_expert_ids[4] = -1
    routing = _build_routing(virtual_expert_ids, block_size_m)

    for swap in (False, True):
        out = _run(
            intermediate,
            weight,
            topk_weights,
            topk_ids,
            routing,
            block_size_m,
            block_size_n,
            False,
            swap_out_halves=swap,
        )
        zeroed = out.view(num_tokens, top_k, 2 * inter)[[1, 4]]
        assert torch.equal(zeroed, torch.zeros_like(zeroed))
        assert out.isfinite().all(), "a valid row was left unwritten"


@requires_cuda
def test_swap_survives_a_grid_wider_than_the_routing():
    """A trailing partial ``GROUP_SIZE_M`` group must not store phantom tiles.

    Production sizes the grid from ``sorted_token_ids`` but derives ``num_pid_m``
    from the device-side ``num_tokens_post_padded``, which the routing trims to a
    tighter bound -- so the grid is wider than ``num_pid_m * num_pid_n``. In the
    trailing group ``group_size_m`` is then smaller than ``GROUP_SIZE_M`` and
    ``pid_n = (pid % num_pid_in_group) // group_size_m`` runs past ``num_pid_n``.
    Those programs load ``b`` out of range (masked to zero) and hold an all-zero
    accumulator; masking the store on ``offs_n_out`` instead of ``offs_n`` pulls
    them back inside ``N`` and lets them race a real tile's store to zero.

    Repeated because the phantom store races the real one; the loser is whichever
    the scheduler runs last.
    """
    num_tokens, top_k, rank, inter = 24, 6, 16, 384
    block_size_m, block_size_n, num_experts = 16, 128, 9
    intermediate, weight, topk_weights, topk_ids, _ = _make_case(
        num_tokens, top_k, rank, inter, num_experts, seed=17
    )
    # 9 full buckets -> 9 blocks, so the second GROUP_SIZE_M=8 group holds one.
    virtual_expert_ids = (
        torch.arange(num_tokens * top_k, device="cuda", dtype=torch.int32)
        // block_size_m
    ).view(num_tokens, top_k)
    sorted_token_ids, expert_ids, num_tokens_post_padded = _build_routing(
        virtual_expert_ids, block_size_m
    )
    num_pid_m = sorted_token_ids.numel() // block_size_m
    assert num_pid_m % 8, "the trailing group must be partial"
    # Widen the grid the way the routing's tight-bound trim does, without moving
    # num_tokens_post_padded.
    extra_blocks = 3
    routing = (
        torch.cat(
            (
                sorted_token_ids,
                torch.full(
                    (extra_blocks * block_size_m,),
                    topk_ids.numel(),
                    dtype=torch.int32,
                    device="cuda",
                ),
            )
        ),
        torch.cat(
            (
                expert_ids,
                torch.full((extra_blocks,), -1, dtype=torch.int32, device="cuda"),
            )
        ),
        num_tokens_post_padded,
    )
    config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
    }

    def _launch(swap: bool) -> torch.Tensor:
        output = torch.full(
            (topk_ids.numel(), 2 * inter),
            float("nan"),
            dtype=intermediate.dtype,
            device="cuda",
        )
        _invoke_moe_lora_expand_add(
            intermediate,
            weight,
            output,
            topk_weights,
            topk_ids,
            *routing,
            config,
            False,  # mul_routed_weight
            False,  # fuse_sum_all_reduce
            swap_out_halves=swap,
        )
        return output

    plain = _launch(False)
    expected = torch.cat((plain[:, inter:], plain[:, :inter]), dim=-1)
    # The zeroed tile is 128 wide, so a fixture whose real tiles are already zero
    # would pass no matter what.
    assert (plain.view(-1, 2 * inter // block_size_n, block_size_n) != 0).any(-1).all()
    for attempt in range(8):
        assert torch.equal(_launch(True), expected), (
            f"phantom n-tile stored zeros over a real tile (attempt {attempt})"
        )


@requires_cuda
def test_swap_rejects_non_gated_layout():
    num_tokens, top_k, rank, n_total, block_size_m = 4, 2, 16, 256, 16
    rows = num_tokens * top_k
    # Non-gated: the intermediate is R wide, so there are no halves to exchange.
    intermediate = torch.zeros(rows, rank, dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros(2, n_total, rank, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.zeros((num_tokens, top_k), dtype=torch.int32, device="cuda")
    topk_weights = torch.zeros((num_tokens, top_k), dtype=torch.float32, device="cuda")
    routing = _build_routing(
        torch.zeros((num_tokens, top_k), dtype=torch.int32, device="cuda"), block_size_m
    )
    output = torch.zeros(rows, n_total, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError, match="gated gate_up layout"):
        _invoke_moe_lora_expand_add(
            intermediate,
            weight,
            output,
            topk_weights,
            topk_ids,
            routing[0],
            routing[1],
            routing[2],
            _config(block_size_m, 128),
            False,
            False,
            swap_out_halves=True,
        )


@requires_cuda
def test_kwarg_reaches_the_kernel_through_the_merged_api():
    """End-to-end through ``merged_experts_fused_moe_lora_add``.

    A dropped kwarg would leave the delta as ``[gate | up]`` and read as merely
    plausible downstream, so the threading gets its own check.
    """
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )

    max_loras, num_experts, num_tokens, top_k, rank = 2, 4, 6, 2, 16
    hidden, inter = 128, 64
    gen = torch.Generator(device="cpu").manual_seed(3)

    def _ints(*shape):
        return (
            torch.randint(-1, 2, shape, generator=gen, dtype=torch.int32)
            .to(torch.bfloat16)
            .cuda()
        )

    hidden_states = _ints(num_tokens, hidden)
    lora_a = _ints(max_loras, num_experts, 2 * rank, hidden)
    lora_b = _ints(max_loras, num_experts, 2 * inter, rank)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), generator=gen, dtype=torch.int32
    ).cuda()
    topk_weights = torch.ones((num_tokens, top_k), dtype=torch.float32, device="cuda")
    token_lora_mapping = torch.tensor(
        [0, 1, -1, 0, 1, 0], dtype=torch.int32, device="cuda"
    )

    def _delta(swap: bool) -> torch.Tensor:
        out = torch.empty(
            (num_tokens, top_k, 2 * inter), dtype=torch.bfloat16, device="cuda"
        )
        merged_experts_fused_moe_lora_add(
            output=out,
            hidden_states=hidden_states,
            lora_a=lora_a,
            lora_b=lora_b,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=False,
            fuse_add_to_output=False,
            use_direct_expand_add=True,
            # Pin the shrink buffer so the two runs cannot differ through
            # whatever an uninitialized allocation happened to hold.
            zero_intermediate=True,
            swap_out_halves=swap,
        )
        return out

    plain, swapped = _delta(False), _delta(True)
    assert torch.equal(
        swapped, torch.cat((plain[..., inter:], plain[..., :inter]), dim=-1)
    )
    assert not torch.equal(swapped, plain), "the fixture must not be half-symmetric"


@requires_cuda
def test_merged_api_rejects_swap_on_the_generic_expand():
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )

    max_loras, num_experts, num_tokens, top_k, rank = 1, 2, 4, 2, 16
    hidden, inter = 64, 32

    def zeros(*shape):
        return torch.zeros(shape, dtype=torch.bfloat16, device="cuda")

    # The guard fires before any tensor is touched, so the operands only need
    # the shapes the signature unpacks.
    with pytest.raises(ValueError, match="swap_out_halves"):
        merged_experts_fused_moe_lora_add(
            output=zeros(num_tokens, top_k, 2 * inter),
            hidden_states=zeros(num_tokens, hidden),
            lora_a=zeros(max_loras, num_experts, 2 * rank, hidden),
            lora_b=zeros(max_loras, num_experts, 2 * inter, rank),
            topk_ids=torch.zeros((num_tokens, top_k), dtype=torch.int32, device="cuda"),
            topk_weights=torch.zeros(
                (num_tokens, top_k), dtype=torch.float32, device="cuda"
            ),
            token_lora_mapping=torch.zeros(
                num_tokens, dtype=torch.int32, device="cuda"
            ),
            mul_routed_weight=False,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=False,
            fuse_add_to_output=False,
            use_direct_expand_add=False,  # generic expand: must swap in the caller
            swap_out_halves=True,
        )


@requires_cuda
def test_merged_api_rejects_swap_on_the_shared_outer_fallback():
    """``use_direct_expand_add=True`` alone is not enough to get the swap.

    ``experts_shared_outer_loras_b=True`` routes to the generic kernel inside the
    impl, which cannot swap -- so it is the second half of the guard's condition,
    and the combination that would otherwise cross the pairing without an error.
    """
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )

    max_loras, num_experts, num_tokens, top_k, rank = 1, 2, 4, 2, 16
    hidden, inter = 64, 32

    def zeros(*shape):
        return torch.zeros(shape, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError, match="swap_out_halves"):
        merged_experts_fused_moe_lora_add(
            output=zeros(num_tokens, top_k, 2 * inter),
            hidden_states=zeros(num_tokens, hidden),
            lora_a=zeros(max_loras, num_experts, 2 * rank, hidden),
            lora_b=zeros(max_loras, 1, 2 * inter, rank),
            topk_ids=torch.zeros((num_tokens, top_k), dtype=torch.int32, device="cuda"),
            topk_weights=torch.zeros(
                (num_tokens, top_k), dtype=torch.float32, device="cuda"
            ),
            token_lora_mapping=torch.zeros(
                num_tokens, dtype=torch.int32, device="cuda"
            ),
            mul_routed_weight=False,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=True,
            fuse_add_to_output=False,
            use_direct_expand_add=True,
            swap_out_halves=True,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
