import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.elementwise import fused_q_norm_rope
from sglang.kernels.ops.attention.dsv4.trtllm_metadata import pack_sparse_tail
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("rows,heads", [(6, 16), (9, 128), (4097, 16)])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("eps", [None, 1e-6])
def test_fp8_rope_output_padding_and_graph(rows, heads, position_dtype, eps):
    torch.manual_seed(911)
    q = torch.randn(rows, heads + 1, 512, device="cuda", dtype=torch.bfloat16)[
        :, :heads
    ]
    storage = torch.full(
        (rows, heads + 1, 512), 7.0, device="cuda", dtype=torch.float8_e4m3fn
    )
    output = storage[:, :heads]
    freqs = torch.polar(
        torch.ones(8192, 32, device="cuda"), torch.randn(8192, 32, device="cuda")
    )
    positions = torch.randint(8192, (rows,), device="cuda", dtype=position_dtype)
    expected = torch.empty_like(q)
    fused_q_norm_rope(q, output, eps, freqs, positions)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_q_norm_rope(q, output, eps, freqs, positions)
    for _ in range(3):
        q.normal_()
        original = q.clone()
        positions.random_(0, 8192)
        graph.replay()
        fused_q_norm_rope(q, expected, eps, freqs, positions)
        torch.testing.assert_close(
            output.float(), expected.to(output.dtype).float(), rtol=0, atol=0
        )
        torch.testing.assert_close(q, original, rtol=0, atol=0)
        assert (storage[:, heads:].float() == 7).all()


def test_fp8_rope_128_heads_int64_offsets():
    # Both Q and output cross 2**31 elements. Check the rows on either side
    # without allocating another full-sized reference tensor.
    rows, heads = 32769, 128
    required = rows * heads * 512 * 5
    if torch.cuda.mem_get_info()[0] < required + (1 << 30):
        pytest.skip("Requires about 11 GiB of free GPU memory")
    q = torch.ones(rows, heads, 512, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(q.shape, device="cuda", dtype=torch.float8_e4m3fn)
    freqs = torch.ones(1, 32, device="cuda", dtype=torch.complex64)
    positions = torch.zeros(rows, device="cuda", dtype=torch.int32)
    fused_q_norm_rope(q, output, None, freqs, positions)
    torch.testing.assert_close(output[-2:].float(), q[-2:].float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_sparse_tail_short_sequence_and_padding(dtype):
    # A short sequence has one valid SWA entry but still reserves 128 columns.
    table = torch.full((64, 132), -1, device="cuda", dtype=torch.int32)
    table[0, 0] = 42
    indices = torch.tensor([[100, 101, -1, -1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([2], device="cuda", dtype=dtype)
    totals = torch.zeros(64, device="cuda", dtype=torch.int32)
    pack_sparse_tail(indices, lengths, table[:1], totals[:1])
    assert totals[0].item() == 130
    assert table[0, 0].item() == 42
    assert (table[0, 1:128] == -1).all()
    torch.testing.assert_close(table[:1, 128:], indices)
    assert (table[1:] == -1).all()
    assert (totals[1:] == 0).all()


def test_sparse_tail_rejects_float_lengths():
    indices = torch.empty((0, 4), dtype=torch.int32)
    table = torch.empty((0, 132), dtype=torch.int32)
    with pytest.raises(AssertionError):
        pack_sparse_tail(
            indices, torch.empty(0), table, torch.empty(0, dtype=torch.int32)
        )


@pytest.mark.parametrize("different_device", range(4))
def test_sparse_tail_rejects_mixed_devices(different_device):
    shapes = [(0, 4), (0,), (0, 132), (0,)]
    tensors = [torch.empty(shape, dtype=torch.int32) for shape in shapes]
    tensors[different_device] = tensors[different_device].to("meta")
    with pytest.raises(AssertionError):
        pack_sparse_tail(*tensors)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
