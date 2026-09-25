"""Ratio-2 fusion preserves BF16 rounding and position-addressed pair state."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.c2_decode_pool import c2_decode_pool
from sglang.kernels.ops.layernorm.rmsnorm_fp32 import rmsnorm_fp32
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DIM = 512
EPS = 1e-6
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="The pair-pooling kernel uses CUDA libdevice arithmetic",
)


def assert_bitwise(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def pool(inputs, state, ring_size, weight=None):
    return c2_decode_pool(
        *inputs,
        state[:, :DIM],
        state[:, DIM:],
        state.shape[0] - 1,
        ring_size=ring_size,
        norm_weight=weight,
        norm_eps=EPS,
    )


@pytest.mark.parametrize("batch", [1, 3, 64])
@pytest.mark.parametrize("ring_size", [2, 8])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
def test_pool_norm_preserves_rounding_and_pending_state(
    batch, ring_size, weight_dtype, loc_dtype
):
    """Fusion must round before RMSNorm and retain raw FP32 state, including pads."""
    torch.manual_seed(0)
    kv = torch.randn(batch, DIM, device="cuda")
    score = torch.randn_like(kv)
    state = torch.randn(batch * ring_size + 1, 2 * DIM, device="cuda")
    # Equal logits, dominant current logits, and zero-valued channels share rows.
    score[:, :128] = 0
    state[:, DIM : DIM + 128] = 0
    score[:, 128:256] = 80
    state[:, DIM + 128 : DIM + 256] = -80
    kv[:, 384:] = 0
    state[:, 384:DIM] = 0
    state[-1, :DIM] = 0
    state[-1, DIM:] = -torch.inf
    weight = (torch.rand(DIM, device="cuda") + 0.5).to(weight_dtype)
    req = torch.arange(batch, device="cuda", dtype=torch.int64)
    pos = req + 2 * ring_size - 1
    raw = (513 + 2 * req).to(loc_dtype)
    out = torch.where(pos % 2 == 1, raw // 2, -1)
    if batch > 1:
        # Request 0 is live; a graph pad with the same request id must not touch it.
        req[-1] = 0
        raw[-1] = 0
        out[-1] = 0
    inputs = (kv, score, pos, raw, out, req)
    baseline_state, fused_state = state.clone(), state.clone()
    pooled, group_pos, slots = pool(inputs, baseline_state, ring_size)
    assert pooled.dtype == torch.float32
    expected = rmsnorm_fp32(pooled.bfloat16(), weight, EPS)
    actual, fused_pos, fused_slots = pool(inputs, fused_state, ring_size, weight)

    assert_bitwise(actual, expected)
    assert_bitwise(fused_pos, group_pos)
    assert_bitwise(fused_slots, slots)
    assert_bitwise(fused_pos, pos - pos % 2)
    assert_bitwise(fused_slots, out.clamp_min(0))
    live = raw != 0
    written_rows = req[live] * ring_size + pos[live] % ring_size
    state[written_rows, :DIM] = kv[live]
    state[written_rows, DIM:] = score[live]
    assert_bitwise(baseline_state, state)
    assert_bitwise(fused_state, state)


def test_pool_norm_graph_replay_updates_positions_and_padding():
    """Captured pointers must use new positions and keep padded request 0 inert."""
    torch.manual_seed(1)
    ring_size = 8
    kv = torch.randn(3, DIM, device="cuda")
    score = torch.randn_like(kv)
    initial = torch.randn(3 * ring_size + 1, 2 * DIM, device="cuda")
    initial[-1, :DIM] = 0
    initial[-1, DIM:] = -torch.inf
    weight = torch.rand(DIM, device="cuda") + 0.5
    req = torch.arange(3, device="cuda", dtype=torch.int64)
    pos = req + 31
    raw = 513 + 2 * req
    out = torch.where(pos % 2 == 1, raw // 2, -1)
    inputs = (kv, score, pos, raw, out, req)
    states = [initial.clone(), initial.clone()]
    graphs, outputs = [], []
    for state, norm_weight in zip(states, [None, weight]):
        for _ in range(3):
            warmup, _, _ = pool(inputs, state, ring_size, norm_weight)
            if norm_weight is None:
                rmsnorm_fp32(warmup.bfloat16(), weight, EPS)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            latent, group_pos, slots = pool(inputs, state, ring_size, norm_weight)
            if norm_weight is None:
                latent = rmsnorm_fp32(latent.bfloat16(), weight, EPS)
        graphs.append(graph)
        outputs.append((latent, group_pos, slots))

    for positions, requests, raw_slots in [
        ([0, 7, 8], [0, 1, 0], [513, 515, 0]),
        ([9, 10, 11], [0, 1, 2], [0, 515, 517]),
    ]:
        pos.copy_(torch.tensor(positions, device="cuda"))
        req.copy_(torch.tensor(requests, device="cuda"))
        raw.copy_(torch.tensor(raw_slots, device="cuda"))
        out.copy_(torch.where(pos % 2 == 1, raw // 2, -1))
        for graph, state in zip(graphs, states):
            state.copy_(initial)
            graph.replay()
        torch.cuda.synchronize()
        for actual, expected in zip(outputs[1], outputs[0]):
            assert_bitwise(actual, expected)
        assert_bitwise(outputs[1][1], pos - pos % 2)
        assert_bitwise(outputs[1][2], out.clamp_min(0))
        expected_state = initial.clone()
        live = raw != 0
        rows = req[live] * ring_size + pos[live] % ring_size
        expected_state[rows, :DIM] = kv[live]
        expected_state[rows, DIM:] = score[live]
        for state in states:
            assert_bitwise(state, expected_state)


@pytest.mark.parametrize("start", [6, 13])
@pytest.mark.parametrize("accepted", [0, 1, 4, 5])
def test_pool_norm_after_speculative_ring_rollback(start, accepted):
    """Rejected writes must retain the accepted predecessor across ring wraparound."""
    torch.manual_seed(2)
    ring_size = 8
    kv = torch.randn(start + 6, DIM, device="cuda")
    score = torch.randn_like(kv)
    state = torch.zeros(ring_size + 1, 2 * DIM, device="cuda")
    state[:, DIM:] = -torch.inf
    committed = state.clone()
    weight = torch.rand(DIM, device="cuda") + 0.5
    req = torch.zeros(1, device="cuda", dtype=torch.int64)
    raw = torch.full_like(req, 513)
    # Emulate verify's raw projection writes; this is not a target-verify test.
    for position in range(start + 5):
        state[position % ring_size, :DIM] = kv[position]
        state[position % ring_size, DIM:] = score[position]
        if position < start + accepted:
            committed[position % ring_size] = state[position % ring_size]
    baseline_state = state.clone()
    pos = torch.full_like(req, start + accepted)
    out = torch.where(pos % 2 == 1, raw // 2, -1)
    inputs = (kv[-1:], score[-1:], pos, raw, out, req)
    pooled, _, _ = pool(inputs, baseline_state, ring_size)
    committed_pooled, group_pos, slots = pool(inputs, committed, ring_size)
    actual, fused_pos, fused_slots = pool(inputs, state, ring_size, weight)
    expected = rmsnorm_fp32(committed_pooled.bfloat16(), weight, EPS)
    assert_bitwise(pooled, committed_pooled)
    assert_bitwise(actual, expected)
    assert_bitwise(fused_pos, group_pos)
    assert_bitwise(fused_slots, slots)
    assert_bitwise(state, baseline_state)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
