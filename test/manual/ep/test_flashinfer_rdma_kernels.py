"""GPU contracts independent of SGLang, RDMA, and model weights."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from sglang.srt.layers.moe.token_dispatcher.flashinfer_rdma_kernels import (
    pack_expert_rows,
    unpack_expert_rows,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "counts_host", [[0, 0, 0, 0, 0], [0, 3, 0, 1, 4], [8, 0, 0, 0, 0]]
)
def test_pack_ignores_poisoned_padding_and_empty_experts(counts_host):
    experts, cap, hidden, rows = 5, 8, 4096, 16
    width, sf_width = hidden // 2, hidden // 16
    dtype = torch.uint8
    recv = torch.full((experts, cap, width), 173, dtype=dtype, device="cuda")
    sf = torch.full((experts, cap, sf_width), 219, dtype=torch.uint8, device="cuda")
    for expert, count in enumerate(counts_host):
        recv[expert, :count] = expert + 1
        sf[expert, :count] = expert + 7
    counts = torch.tensor(counts_host, dtype=torch.int32, device="cuda")
    x = torch.empty((rows, width), dtype=dtype, device="cuda")
    x_sf = torch.empty((rows, sf_width), dtype=torch.uint8, device="cuda")
    ids = torch.empty((rows, 1), dtype=torch.int32, device="cuda")
    weights = torch.empty((rows, 1), dtype=torch.float32, device="cuda")
    mapping = torch.empty(rows, dtype=torch.int32, device="cuda")
    operation = lambda: pack_expert_rows(
        recv, sf, counts, x, x_sf, ids, weights, mapping, ep_rank=2, num_experts=20
    )
    operation()
    expected_experts = [e for e, n in enumerate(counts_host) for _ in range(n)]
    valid = len(expected_experts)
    expected_ids = [e + 10 for e in expected_experts] + [20] * (rows - valid)
    assert ids.flatten().cpu().tolist() == expected_ids
    assert weights.flatten().cpu().tolist() == [1.0] * valid + [0.0] * (rows - valid)
    expected_map = [e * cap + r for e, n in enumerate(counts_host) for r in range(n)]
    assert mapping.cpu().tolist() == expected_map + [-1] * (rows - valid)
    for row, expert in enumerate(expected_experts):
        assert torch.all(x[row] == expert + 1)
        assert torch.all(x_sf[row] == expert + 7)
    assert torch.count_nonzero(x[valid:]) == 0
    # Replay must read current device counts, rather than capture-time values.
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        operation()
    counts.zero_()
    capture.replay()
    assert torch.all(ids == 20)
    assert torch.all(mapping == -1)


def test_unpack_ignores_invalid_compact_outputs_and_uses_per_row_amax():
    rows, experts, cap, hidden = 16, 5, 8, 4096
    compact = torch.full(
        (rows, hidden), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    slots = [11, 24, 25]
    mapping = torch.tensor(
        slots + [-1] * (rows - len(slots)), dtype=torch.int32, device="cuda"
    )
    compact[0] = -3
    compact[1] = 0
    compact[2] = 7
    output = torch.full((experts, cap, hidden), 91, dtype=torch.bfloat16, device="cuda")
    scales = torch.ones((experts, cap, 1), dtype=torch.float32, device="cuda")
    unpack_expert_rows(compact, mapping, output, scales)
    flat = output.reshape(-1, hidden)
    for index, value in zip(slots, [-3, 0, 7]):
        assert torch.all(flat[index] == value)
    untouched = [r for r in range(experts * cap) if r not in slots]
    assert torch.all(flat[untouched] == 91)
    torch.testing.assert_close(
        scales.flatten()[slots],
        torch.tensor([896, 1, 384], dtype=torch.float32, device="cuda"),
    )
    assert torch.all(scales.flatten()[untouched] == 1)
