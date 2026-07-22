"""CUDA graph tests for multi-LoRA merged alignment."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

# Skipped on CI: newly-added inkling LoRA test, disabled pending stabilization.
pytestmark = pytest.mark.skip(reason="new inkling LoRA test; disabled on CI")

ALIGN_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/kernels/ops/moe/trtllm_lora_temp/virtual_experts.py"
)


def _load_align_function():
    tree = ast.parse(ALIGN_PATH.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_align_block_size_jit"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "torch": torch,
        "jit_moe_align_block_size": sys.modules[
            "sglang.kernels.ops.moe.moe_align"
        ].moe_align_block_size,
    }
    exec(compile(module, str(ALIGN_PATH), "exec"), namespace)
    return namespace["_align_block_size_jit"]


def test_experimental_alignment_geometry_and_empty_input(monkeypatch):
    calls = []

    def fake_jit_align(*args):
        (
            topk_ids,
            num_buckets,
            block_size,
            sorted_ids,
            expert_ids,
            total,
            cumsum,
            flag,
        ) = args
        calls.append((num_buckets, cumsum.numel(), flag))
        sorted_ids.fill_(topk_ids.numel())
        expert_ids[:2] = torch.tensor([-1, num_buckets - 2])
        total.fill_(2 * block_size)

    monkeypatch.setitem(
        sys.modules,
        "sglang.kernels.ops.moe.moe_align",
        types.SimpleNamespace(moe_align_block_size=fake_jit_align),
    )
    align = _load_align_function()

    topk_ids = torch.tensor([[-1, 383]], dtype=torch.int32)
    sorted_ids, expert_ids, num_tokens_post_pad = align(topk_ids, 5, 384)

    assert sorted_ids.numel() == 12  # int4-safe capacity above logical 10.
    assert expert_ids.numel() == 3
    assert calls == [(385, 386, True)]
    assert expert_ids[:2].tolist() == [-1, 383]
    assert num_tokens_post_pad.item() == 10

    outputs = align(torch.empty((0, 6), dtype=torch.int32), 16, 384)
    assert [tensor.numel() for tensor in outputs] == [0, 0, 1]
    assert outputs[2].item() == 0
    assert len(calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_experimental_alignment_cuda_sentinel_and_max_expert():
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        _align_block_size_jit,
    )

    topk_ids = torch.tensor([[-1, 383], [0, 0]], device="cuda", dtype=torch.int32)
    _, expert_ids, num_tokens_post_pad = _align_block_size_jit(topk_ids, 16, 384)
    active_experts = expert_ids[: num_tokens_post_pad.item() // 16].cpu().tolist()
    assert sorted(active_experts) == [-1, 0, 383]


def _assert_shared_outer_merged_align_semantics(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
    token_lora_mapping: torch.Tensor,
    *,
    topk: int,
    block_size: int,
    num_slots: int,
) -> None:
    """Validate routing without depending on atomic scatter order."""

    (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mask,
        virtual_num_experts,
    ) = outputs
    mapping = token_lora_mapping.cpu()
    num_routes = mapping.numel() * topk
    total_padded = int(num_tokens_post_padded.item())
    sorted_cpu = sorted_token_ids[:total_padded].cpu()
    experts_cpu = expert_ids[: total_padded // block_size].cpu()

    assert virtual_num_experts == num_slots
    assert torch.equal(token_lora_mask.cpu(), mapping >= 0)
    assert (mapping == -1).any()  # -1 is the runtime base/no-adapter sentinel.
    assert (mapping == 0).any()  # Slot 0 remains a valid adapter slot.

    expected_routes = []
    expected_total_padded = 0
    for slot in range(num_slots):
        slot_tokens = torch.nonzero(mapping == slot, as_tuple=False).flatten()
        slot_routes = (
            slot_tokens[:, None] * topk + torch.arange(topk)[None, :]
        ).flatten()
        expected_routes.extend(slot_routes.tolist())
        route_count = slot_routes.numel()
        expected_total_padded += (
            (route_count + block_size - 1) // block_size
        ) * block_size

    assert total_padded == expected_total_padded
    observed_routes = []
    for block, slot in enumerate(experts_cpu.tolist()):
        assert 0 <= slot < num_slots
        block_routes = sorted_cpu[block * block_size : (block + 1) * block_size]
        real_routes = block_routes[block_routes < num_routes].to(torch.long)
        if real_routes.numel():
            routed_tokens = torch.div(real_routes, topk, rounding_mode="floor")
            assert torch.all(mapping[routed_tokens] == slot)
            observed_routes.extend(real_routes.tolist())
        assert torch.all((block_routes >= 0) & (block_routes <= num_routes))

    assert sorted(observed_routes) == sorted(expected_routes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_slots", [2, 3, 4])
def test_multi_slot_shared_outer_merged_align_cuda_graph_parity(num_slots):
    """The fused hot path must replay with current multi-LoRA routing data."""

    from sglang.jit_kernel.trtllm_lora_temp.moe_lora_merged_align import (
        moe_lora_merged_align,
    )

    device = torch.device("cuda")
    num_tokens = 37  # Exercise the multi-LoRA prefill-size routing contract.
    topk = 6
    block_size = 16
    num_experts = 384
    generator = torch.Generator(device=device).manual_seed(9000 + num_slots)
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    token_lora_mapping = torch.arange(
        num_tokens, device=device, dtype=torch.int32
    ).remainder(num_slots)
    token_lora_mapping[0] = -1
    token_lora_mapping[-1] = -1

    def invoke(fuse_scatter: bool):
        return moe_lora_merged_align(
            topk_ids,
            token_lora_mapping,
            num_experts,
            shared_outer=True,
            max_loras=num_slots,
            block_size=block_size,
            do_skip=True,
            fuse_scatter=fuse_scatter,
        )

    # Compile both real kernel variants and initialize CUDA state off the
    # capture stream. Production selects the fused variant for this geometry.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(2):
            invoke(fuse_scatter=True)
            invoke(fuse_scatter=False)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_outputs = invoke(fuse_scatter=True)
        split_outputs = invoke(fuse_scatter=False)

    stable_outputs = (*fused_outputs[:4], *split_outputs[:4])
    stable_addresses = tuple(tensor.data_ptr() for tensor in stable_outputs)

    for replay in range(3):
        if replay:
            next_mapping = (
                torch.arange(num_tokens, device=device, dtype=torch.int32)
                .add_(replay)
                .remainder_(num_slots)
            )
            # Move the base/no-adapter rows on every replay while retaining a
            # valid adapter in slot 0.
            next_mapping[replay] = -1
            next_mapping[-replay - 1] = -1
            token_lora_mapping.copy_(next_mapping)
            topk_ids.copy_(torch.roll(topk_ids, shifts=1, dims=1))

        for outputs in (fused_outputs, split_outputs):
            outputs[0].fill_(-12345)
            outputs[1].fill_(-12345)
            outputs[2].fill_(-1)
            outputs[3].fill_(False)
        graph.replay()

        torch.cuda.synchronize()
        assert tuple(tensor.data_ptr() for tensor in stable_outputs) == stable_addresses
        for outputs in (fused_outputs, split_outputs):
            _assert_shared_outer_merged_align_semantics(
                outputs,
                token_lora_mapping,
                topk=topk,
                block_size=block_size,
                num_slots=num_slots,
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
