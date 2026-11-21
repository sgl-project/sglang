import itertools

import pytest
import torch

from sglang.kernels.ops.moe.moe_align_stable import moe_align_block_size_stable
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=480, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def reference(ids, block_size, experts):
    values = ids.cpu().flatten().tolist()
    tokens, blocks = [], []
    inactive = 0
    for expert in range(-1, experts):
        pairs = [index for index, value in enumerate(values) if value == expert]
        padded = (len(pairs) + block_size - 1) // block_size * block_size
        tokens.extend(pairs + [len(values)] * (padded - len(pairs)))
        blocks.extend([expert] * (padded // block_size))
        if expert == -1:
            inactive = padded // block_size
    return (
        torch.tensor(tokens, dtype=torch.int32),
        torch.tensor(blocks, dtype=torch.int32),
        inactive,
    )


def check(ids, block_size, experts, actual):
    tokens, blocks, inactive = reference(ids, block_size, experts)
    sorted_ids, expert_ids, counts = actual
    assert int(counts[0]) == tokens.numel()
    if counts.numel() == 2:
        assert int(counts[1]) == inactive
    torch.testing.assert_close(
        sorted_ids[: tokens.numel()].cpu(), tokens, rtol=0, atol=0
    )
    torch.testing.assert_close(
        expert_ids[: blocks.numel()].cpu(), blocks, rtol=0, atol=0
    )


CASES = list(
    itertools.product([0, 1, 6, 65, 410, 2048], [8, 16, 48], [torch.int32, torch.int64])
)


@pytest.mark.parametrize("tokens,block_size,dtype", CASES)
def test_stable_alignment(tokens, block_size, dtype):
    torch.manual_seed(1827)
    experts = 256
    ids = torch.randint(-1, experts, (tokens, 6), dtype=dtype, device="cuda")
    actual = moe_align_block_size_stable(
        ids, block_size, experts, return_inactive_prefix_count=True
    )
    check(ids, block_size, experts, actual)
    repeated = moe_align_block_size_stable(
        ids, block_size, experts, return_inactive_prefix_count=True
    )
    count = int(actual[2][0])
    assert torch.equal(actual[0][:count], repeated[0][:count])


@pytest.mark.parametrize("experts", [1, 7, 64, 1024])
def test_single_expert_and_filtered_pairs(experts):
    ids = torch.full((67, 6), experts - 1, dtype=torch.int32, device="cuda")
    ids[::3] = -1
    check(ids, 32, experts, moe_align_block_size_stable(ids, 32, experts))


@pytest.mark.parametrize("tokens", [410, 2048, 8192])
def test_cuda_graph_reads_updated_routing(tokens):
    torch.manual_seed(23)
    ids = torch.randint(-1, 256, (tokens, 6), dtype=torch.int32, device="cuda")
    for _ in range(3):
        moe_align_block_size_stable(ids, 16, 256, return_inactive_prefix_count=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = moe_align_block_size_stable(
            ids, 16, 256, return_inactive_prefix_count=True
        )
    ids.copy_(ids.flip(0))
    graph.replay()
    check(ids, 16, 256, actual)


@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] not in (9, 12),
    reason="MXFP4 Marlin requires SM90 or SM120",
)
@pytest.mark.parametrize("tokens", [1, 129])
def test_mxfp4_marlin_repeated_forward(tokens):
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        _permute_moe_fp4_scales_for_marlin,
        _repack_moe_fp4_weight_for_marlin,
        mxfp4_marlin_process_scales,
    )

    torch.manual_seed(45525)
    experts, hidden, intermediate, topk = 16, 1024, 256, 6
    packed, scales, reference_weights = [], [], []
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.float64,
    )
    for outputs, inputs in [(2 * intermediate, hidden), (hidden, intermediate)]:
        raw = torch.randint(
            0, 256, (experts, outputs, inputs // 2), device="cuda", dtype=torch.uint8
        )
        scale = torch.randint(
            119, 124, (experts, outputs, inputs // 32), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
        codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
        reference_weights.append(
            levels[codes.long()] * scale.double().repeat_interleave(32, -1)
        )
        packed.append(
            _repack_moe_fp4_weight_for_marlin(
                raw,
                num_experts=experts,
                size_n=outputs,
                size_k=inputs,
                perm=torch.empty(0, device="cuda", dtype=torch.int32),
            )
        )
        scales.append(
            _permute_moe_fp4_scales_for_marlin(
                scale.bfloat16(),
                num_experts=experts,
                size_n=outputs,
                size_k=inputs,
                group_size=32,
                process_scales=lambda value: mxfp4_marlin_process_scales(
                    value, input_dtype=torch.bfloat16
                ),
            )
        )
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    router_logits = torch.randn(tokens, experts, device="cuda")
    weights, ids = router_logits.softmax(-1).topk(topk, dim=-1)
    ids = ids.int()

    def run():
        return fused_marlin_moe(
            hidden_states=x,
            w1=packed[0],
            w2=packed[1],
            w1_scale=scales[0],
            w2_scale=scales[1],
            gating_output=router_logits,
            topk_weights=weights,
            topk_ids=ids,
            num_bits=4,
            clamp_limit=10.0,
        )

    expected_pairs = torch.zeros(
        tokens, topk, hidden, device="cuda", dtype=torch.bfloat16
    )
    for expert in range(experts):
        rows, slots = torch.where(ids == expert)
        gate_up = (x[rows].double() @ reference_weights[0][expert].T).bfloat16().float()
        gate, up = gate_up.split(intermediate, dim=-1)
        activated = torch.nn.functional.silu(gate.clamp(max=10)).bfloat16()
        activation = (activated.float() * up.clamp(-10, 10)).bfloat16()
        output = (activation.double() @ reference_weights[1][expert].T).bfloat16()
        expected_pairs[rows, slots] = (
            output.float() * weights[rows, slots, None].bfloat16().float()
        ).bfloat16()
    expected = expected_pairs.float().sum(1).bfloat16()
    first = run().clone()
    error = first.float() - expected.float()
    output_rms = expected.float().square().mean().sqrt()
    error_rms = error.square().mean().sqrt()
    # Intermediate BF16 rounding can affect outputs that cancel across
    # experts. Bound normwise accuracy and each element in BF16 units.
    epsilon = torch.finfo(torch.bfloat16).eps
    assert error_rms <= epsilon * output_rms
    torch.testing.assert_close(
        first, expected, rtol=4 * epsilon, atol=epsilon * output_rms.item()
    )
    for _ in range(20):
        assert torch.equal(run(), first)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(3):
        graph.replay()
        assert torch.equal(actual, first)
