"""MoE LoRA down-B fold parity and CUDA-graph replay.
The wider stage matrix lives in test_moe_lora_down_b_into_base_matrix.py.
"""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.moe_lora_down_b_kit import (
    _HIDDEN,
    _INTO_BASE_TOLERANCE,
    _RANK,
    _ROUTED_SCALING,
    _SLOTS,
    _build_runner,
    _into_base_pair,
    _make_gpu_tensors,
    _run_once,
    _standalone_output_allocation,
    _token_lora_mapping,
    _workspace_buffer_names,
    cutedsl_cuda_only,
)

# H100 only: the shipped SM100 plan table selects no fold row (overrides and
# the default table still can).
register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the scatter kernel needs any CUDA device (plain Triton)",
)


_ROW_POISON = -1  # a sentinel pair's pair_to_row entry; the finalize gates on it


def _kernel_case(num_tokens: int, top_k: int, num_experts: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), generator=generator, dtype=torch.int32
    )
    topk_ids[torch.rand((num_tokens, top_k), generator=generator) < 0.15] = -1
    topk_ids[0] = -1  # a token with no routed pair
    pattern = torch.tensor([0, 1, -1, 0, -1, 1], dtype=torch.int32)
    token_lora_mapping = pattern.repeat(-(-num_tokens // pattern.numel()))[:num_tokens]
    weights = torch.rand((num_tokens, top_k), generator=generator) + 0.1
    topk_weights = (weights / weights.sum(dim=1, keepdim=True)).float()

    pairs = num_tokens * top_k
    bridge = (torch.randn((pairs, _RANK), generator=generator) * 0.15).to(
        torch.bfloat16
    )
    b_down = (
        torch.randn((_SLOTS * num_experts, _HIDDEN, _RANK), generator=generator) * 0.15
    ).to(torch.bfloat16)
    # Slot 1 is a narrow adapter. Its rank padding is zero in the bridge and
    # in down-B. A batch with mixed ranks reaches the kernels in this form.
    b_down.view(_SLOTS, num_experts, _HIDDEN, _RANK)[1, :, :, _RANK // 2 :] = 0
    bridge_view = bridge.view(num_tokens, top_k, _RANK)
    bridge_view[token_lora_mapping == 1, :, _RANK // 2 :] = 0
    return topk_ids, token_lora_mapping, topk_weights, bridge, b_down


def _pair_to_row_rows(topk_ids: torch.Tensor, num_experts: int, style: str, seed: int):
    """Build the pair-to-row map of both row domains on the host."""
    ids = topk_ids.view(-1)
    counts = [int((ids == expert).sum()) for expert in range(num_experts)]
    if style == "masked":
        m_max = -(-max(counts + [1]) // 8) * 8 + 8
        base = [expert * m_max for expert in range(num_experts)]
        total_rows = num_experts * m_max
    else:  # contiguous rows, with each segment aligned to 8 rows
        alignment = 8
        base, offset = [], 0
        for count in counts:
            base.append(offset)
            offset += -(-count // alignment) * alignment
        total_rows = max(offset, alignment)
    pair_to_row = torch.full((ids.numel(),), _ROW_POISON, dtype=torch.int32)
    cursor = [0] * num_experts
    for pair, expert in enumerate(ids.tolist()):
        if expert >= 0:
            pair_to_row[pair] = base[expert] + cursor[expert]
            cursor[expert] += 1
    generator = torch.Generator().manual_seed(seed ^ 0xD05E)
    down_rows = (torch.randn((total_rows, _HIDDEN), generator=generator) * 0.2).to(
        torch.bfloat16
    )
    return pair_to_row, down_rows


def _fp32_finalize_oracle(
    down_rows,
    pair_to_row,
    bridge,
    b_down,
    topk_ids,
    token_lora_mapping,
    topk_weights,
    num_experts,
):
    num_tokens, top_k = topk_ids.shape
    out = torch.zeros((num_tokens, _HIDDEN), dtype=torch.float32)
    for token in range(num_tokens):
        slot = int(token_lora_mapping[token])
        for k in range(top_k):
            expert = int(topk_ids[token, k])
            if expert < 0:
                continue
            pair = token * top_k + k
            row = down_rows[int(pair_to_row[pair])].float()
            if 0 <= slot < _SLOTS:
                bucket_id = slot * num_experts + expert
                row = row + b_down[bucket_id].float() @ bridge[pair].float()
            out[token] += float(topk_weights[token, k]) * row
    return out * _ROUTED_SCALING


# Both launches use one config. The scatter must keep the same down-B tiling
# as the shipped path.
_DOWN_B_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 16,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}


def _post_reorder(
    down_rows, output, pair_to_row, topk_ids, topk_weights, lora_delta=None
):
    from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm

    num_tokens, top_k = topk_ids.shape
    post_reorder_deepgemm(
        down_rows,
        output,
        pair_to_row,
        topk_ids,
        topk_weights,
        top_k,
        num_tokens,
        _HIDDEN,
        _ROUTED_SCALING,
        lora_delta=lora_delta,
    )


@cuda_only
@pytest.mark.parametrize("row_domain", ("masked", "contiguous"))
@pytest.mark.parametrize(
    ("num_tokens", "top_k", "num_experts"),
    ((64, 2, 4), (13, 3, 4)),
    ids=("even", "partial"),
)
def test_into_base_matches_the_standalone_downb_plus_post_reorder(
    row_domain: str, num_tokens: int, top_k: int, num_experts: int
) -> None:
    from sglang.kernels.ops.lora.common.route_view import RouteViewKind
    from sglang.kernels.ops.lora.common.routing import build_route
    from sglang.kernels.ops.lora.moe.lora_b import (
        grouped_lora_b,
        invoke_down_b_into_base,
    )

    device = torch.device("cuda")
    seed = 0x5CA7 + num_tokens + num_experts
    topk_ids, token_lora_mapping, topk_weights, bridge, b_down = _kernel_case(
        num_tokens, top_k, num_experts, seed
    )
    pair_to_row, down_rows = _pair_to_row_rows(topk_ids, num_experts, row_domain, seed)
    gpu = {
        name: tensor.to(device)
        for name, tensor in {
            "topk_ids": topk_ids,
            "token_lora_mapping": token_lora_mapping,
            "topk_weights": topk_weights,
            "bridge": bridge,
            "b_down": b_down,
            "pair_to_row": pair_to_row,
            "down_rows": down_rows,
        }.items()
    }
    aligned = build_route(
        gpu["token_lora_mapping"],
        group_ids=gpu["topk_ids"],
        groups_per_slot=num_experts,
        max_loras=_SLOTS,
        block_size=16,
        view=RouteViewKind.ALIGNED,
    )

    # In the shipped path, down-B writes the LoRA delta to its own buffer.
    # post_reorder then reads that buffer and the unchanged base rows.
    lora_delta = torch.empty(
        (num_tokens * top_k, _HIDDEN), dtype=torch.bfloat16, device=device
    )
    grouped_lora_b(
        gpu["bridge"],
        gpu["b_down"],
        lora_delta,
        aligned,
        destination_offsets=(0,),
        config=_DOWN_B_CONFIG,
    )
    reference = torch.empty((num_tokens, _HIDDEN), dtype=torch.float32, device=device)
    _post_reorder(
        gpu["down_rows"],
        reference,
        gpu["pair_to_row"],
        gpu["topk_ids"],
        gpu["topk_weights"],
        lora_delta=lora_delta.view(num_tokens, top_k, _HIDDEN),
    )

    # In the scatter path, the same tiling adds the delta into a copy of the
    # base rows. post_reorder then runs with no delta buffer.
    scattered = gpu["down_rows"].clone()
    invoke_down_b_into_base(
        down_rows=scattered,
        pair_to_row=gpu["pair_to_row"],
        bridge=gpu["bridge"],
        b_down=gpu["b_down"],
        routing=aligned,
        config=_DOWN_B_CONFIG,
    )
    output = torch.empty_like(reference)
    _post_reorder(
        scattered, output, gpu["pair_to_row"], gpu["topk_ids"], gpu["topk_weights"]
    )
    torch.testing.assert_close(output, reference, **_INTO_BASE_TOLERANCE)

    # A row stays bitwise equal if no active LoRA pair targets it. A base-only
    # pair and a sentinel pair add nothing. The kernel never reads their
    # pair_to_row entries.
    lora_active = (topk_ids.view(-1) >= 0) & (
        token_lora_mapping.repeat_interleave(top_k) >= 0
    )
    touched = pair_to_row[lora_active].long()
    untouched = torch.ones(down_rows.shape[0], dtype=torch.bool)
    untouched[touched] = False
    assert torch.equal(
        scattered[untouched.to(device)], gpu["down_rows"][untouched.to(device)]
    )

    oracle = _fp32_finalize_oracle(
        down_rows,
        pair_to_row,
        bridge,
        b_down,
        topk_ids,
        token_lora_mapping,
        topk_weights,
        num_experts,
    )
    torch.testing.assert_close(output.cpu(), oracle, atol=1.8e-2, rtol=0.06)

    # With base-only traffic, the kernel must not change any base row.
    base_slots = torch.full_like(gpu["token_lora_mapping"], -1)
    aligned_base = build_route(
        base_slots,
        group_ids=gpu["topk_ids"],
        groups_per_slot=num_experts,
        max_loras=_SLOTS,
        block_size=16,
        view=RouteViewKind.ALIGNED,
    )
    scattered_base = gpu["down_rows"].clone()
    invoke_down_b_into_base(
        down_rows=scattered_base,
        pair_to_row=gpu["pair_to_row"],
        bridge=gpu["bridge"],
        b_down=gpu["b_down"],
        routing=aligned_base,
        config=_DOWN_B_CONFIG,
    )
    assert torch.equal(scattered_base, gpu["down_rows"])
    # The active routing must change the rows.
    assert (scattered - gpu["down_rows"]).abs().max().item() > 1e-3


@cutedsl_cuda_only
def test_into_base_pipeline_replays_correctly_in_a_real_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.lora.moe.runner import MoeLoraRunner

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    serial_plan, reordered_plan, launch_config = _into_base_pair()
    num_tokens, num_experts = 64, 4
    gpu = _make_gpu_tensors(num_tokens, num_experts, device)
    reference_runner = _build_runner(
        serial_plan, launch_config, "expert_major", gpu, num_experts
    )
    into_base_runner = _build_runner(
        reordered_plan, launch_config, "expert_major", gpu, num_experts
    )
    token_lora_mapping = _token_lora_mapping("active", num_tokens).to(device)

    # The workspace keys scratch by stream: warm up on the capture stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):  # warm the JIT and keep the graph buffers before capture
            _run_once(into_base_runner, gpu, token_lora_mapping, use_cuda_graph=True)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _run_once(
            into_base_runner, gpu, token_lora_mapping, use_cuda_graph=True
        )
    output = captured.hidden_states
    output_ptr = output.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)
    reference = _run_once(reference_runner, gpu, token_lora_mapping).hidden_states
    torch.testing.assert_close(output, reference, **_INTO_BASE_TOLERANCE)

    # Replay 2: every token in the batch becomes base-only.
    token_lora_mapping.fill_(-1)
    graph.replay()
    torch.cuda.synchronize(device)
    assert output.data_ptr() == output_ptr
    reference = _run_once(reference_runner, gpu, token_lora_mapping).hidden_states
    torch.testing.assert_close(output, reference, **_INTO_BASE_TOLERANCE)

    # Replay 3: new routing and new activations arrive in the same buffers.
    gpu["topk_ids"].copy_(gpu["topk_ids"].flip(dims=(1,)))
    gpu["hidden_states"].copy_((gpu["hidden_states"].float() * 1.5).to(torch.bfloat16))
    token_lora_mapping.copy_(_token_lora_mapping("mixed", num_tokens).to(device))
    graph.replay()
    torch.cuda.synchronize(device)
    assert output.data_ptr() == output_ptr
    reference = _run_once(reference_runner, gpu, token_lora_mapping).hidden_states
    torch.testing.assert_close(output, reference, **_INTO_BASE_TOLERANCE)

    assert "down_b:delta" not in _workspace_buffer_names(into_base_runner)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
