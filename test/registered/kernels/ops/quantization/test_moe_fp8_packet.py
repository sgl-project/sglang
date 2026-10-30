"""Guard the bit-exact wire contract, strided scales, padding and graph replay."""

import pytest
import torch

from sglang.kernels.ops.quantization.moe_fp8_packet import (
    pack_moe_fp8,
    unpack_moe_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.moe_fp8_packet import pack_reference, unpack_reference

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _inputs(rows, hidden, topk, column_major):
    q = torch.randint(0, 256, (rows, hidden), device="cuda", dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    # Arbitrary float bits include signed zeros, infinities and NaNs. The packet
    # must preserve bits, not approximate or canonicalize floating values.
    scale = torch.randint(
        -(2**31), 2**31 - 1, (rows, hidden // 128), device="cuda", dtype=torch.int32
    ).view(torch.float32)
    if column_major:
        scale = scale.t().contiguous().t()
    weights = torch.randn((rows, topk), device="cuda", dtype=torch.float32)
    ids = torch.randint(-1, 289, (rows, topk), device="cuda", dtype=torch.int32)
    return q, scale, weights, ids


def _bytes(tensor):
    # Explicit dense storage also fixes one-element TMA strides.
    dense = torch.empty(tensor.numel(), device=tensor.device, dtype=tensor.dtype)
    dense.copy_(tensor.reshape(-1))
    return dense.view(torch.uint8).reshape(tensor.shape[0], -1)


@pytest.mark.parametrize("rows", [1, 3, 129, 1024])
@pytest.mark.parametrize("hidden,topk", [(128, 1), (512, 9), (4096, 9)])
@pytest.mark.parametrize("column_major", [False, True])
def test_packet_exact_wire_and_roundtrip(rows, hidden, topk, column_major):
    inputs = _inputs(rows, hidden, topk, column_major)
    packet = pack_moe_fp8(*inputs)
    assert torch.equal(packet, pack_reference(*inputs))
    restored = unpack_moe_fp8(packet, hidden, topk)
    reference_restored = unpack_reference(packet, hidden, topk)
    for actual, reference, expected in zip(restored, reference_restored, inputs):
        assert torch.equal(_bytes(actual), _bytes(expected))
        assert torch.equal(_bytes(actual), _bytes(reference))
    assert restored[1].stride(0) == 1
    assert restored[1].stride(1) % 4 == 0


def test_packet_empty():
    inputs = _inputs(0, 512, 9, True)
    packet = pack_moe_fp8(*inputs)
    restored = unpack_moe_fp8(packet, 512, 9)
    assert packet.shape == (0, 608)
    for actual, expected in zip(restored, inputs):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype


def test_rank_major_gather_preserves_token_and_router_order():
    shards = [_inputs(129, 512, 9, True) for _ in range(4)]
    gathered = torch.cat([pack_moe_fp8(*shard) for shard in shards])
    actual = unpack_moe_fp8(gathered, 512, 9)
    for index, tensor in enumerate(actual):
        expected = torch.cat([shard[index] for shard in shards])
        assert torch.equal(_bytes(tensor), _bytes(expected))


def test_packet_graph_replay_reads_updated_inputs():
    inputs = _inputs(129, 512, 9, True)
    unpack_moe_fp8(pack_moe_fp8(*inputs), 512, 9)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packet = pack_moe_fp8(*inputs)
        restored = unpack_moe_fp8(packet, 512, 9)
    for old, new in zip(inputs, _inputs(129, 512, 9, True)):
        old.copy_(new)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(packet, pack_reference(*inputs))
    for actual, expected in zip(restored, inputs):
        assert torch.equal(_bytes(actual), _bytes(expected))


def test_packet_rejects_invalid_width_and_scale():
    inputs = _inputs(3, 512, 9, True)
    with pytest.raises(ValueError):
        pack_moe_fp8(inputs[0], inputs[1][:, :1], inputs[2], inputs[3])
    packet = pack_moe_fp8(*inputs)
    with pytest.raises(ValueError):
        unpack_moe_fp8(packet, 1024, 9)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
