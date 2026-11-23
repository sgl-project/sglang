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
    "cap, rows, counts_host",
    [
        (8, 16, [0, 0, 0, 0, 0]),
        (8, 16, [0, 3, 0, 1, 4]),
        (8, 16, [8, 0, 0, 0, 0]),
        # More received rows than the buffer holds: the excess is dropped.
        (8, 16, [8, 8, 8, 0, 0]),
        # More rows than launched programs: each program walks several rows.
        (600, 2600, [600, 0, 37, 600, 1]),
    ],
)
@pytest.mark.parametrize("codec", ["nvfp4", "bf16", "mxfp8"])
def test_pack_ignores_poisoned_padding_and_empty_experts(cap, rows, counts_host, codec):
    experts, hidden = 5, 5120
    width = hidden // 2 if codec == "nvfp4" else hidden
    sf_width = {"nvfp4": hidden // 16, "bf16": 0, "mxfp8": hidden // 32}[codec]
    dtype = torch.bfloat16 if codec == "bf16" else torch.uint8
    recv = torch.full((experts, cap, width), 173, dtype=dtype, device="cuda")
    sf = torch.full((experts, cap, sf_width), 219, dtype=torch.uint8, device="cuda")
    for expert, count in enumerate(counts_host):
        recv[expert, :count] = expert + 1
        sf[expert, :count] = expert + 7
    counts = torch.tensor(counts_host, dtype=torch.int32, device="cuda")
    # The dispatcher reuses these buffers across forwards without clearing them:
    # rows without data keep their payload and are marked by id, weight and map.
    x = torch.full((rows, width), 91, dtype=dtype, device="cuda")
    x_sf = torch.full((rows, sf_width), 92, dtype=torch.uint8, device="cuda")
    ids = torch.empty((rows, 1), dtype=torch.int32, device="cuda")
    weights = torch.empty((rows, 1), dtype=torch.float32, device="cuda")
    mapping = torch.empty(rows, dtype=torch.int32, device="cuda")
    operation = lambda: pack_expert_rows(
        recv, sf, counts, x, x_sf, ids, weights, mapping, ep_rank=2, num_experts=20
    )
    operation()
    expected_experts = [e for e, n in enumerate(counts_host) for _ in range(n)][:rows]
    valid = len(expected_experts)
    expected_ids = [e + 10 for e in expected_experts] + [20] * (rows - valid)
    assert ids.flatten().cpu().tolist() == expected_ids
    assert weights.flatten().cpu().tolist() == [1.0] * valid + [0.0] * (rows - valid)
    expected_map = [e * cap + r for e, n in enumerate(counts_host) for r in range(n)]
    expected_map = expected_map[:rows] + [-1] * (rows - valid)
    assert mapping.cpu().tolist() == expected_map
    source = torch.tensor(expected_experts, dtype=dtype, device="cuda")[:, None]
    assert torch.equal(x[:valid], (source + 1).expand(-1, width))
    assert torch.equal(x_sf[:valid], (source + 7).expand(-1, sf_width))
    assert torch.all(x[valid:] == 91)
    assert torch.all(x_sf[valid:] == 92)
    # Replay must read current device counts, rather than capture-time values.
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        operation()
    counts.zero_()
    capture.replay()
    assert torch.all(ids == 20)
    assert torch.all(mapping == -1)
    counts.copy_(torch.tensor(counts_host, dtype=torch.int32, device="cuda"))
    capture.replay()
    assert ids.flatten().cpu().tolist() == expected_ids
    assert mapping.cpu().tolist() == expected_map


def test_unpack_ignores_invalid_compact_outputs_and_uses_per_row_amax():
    rows, experts, cap, hidden = 16, 5, 8, 4096
    compact = torch.full(
        (rows, hidden), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    slots = [11, 24, 25]
    # Row 3 lies inside the received prefix without a slot. Row 4 lies beyond
    # the prefix, where a slot left by an earlier forward must not be followed.
    mapping = torch.tensor(
        slots + [-1, 30] + [-1] * (rows - len(slots) - 2),
        dtype=torch.int32,
        device="cuda",
    )
    counts = torch.tensor([0, 1, 0, 3, 0], dtype=torch.int32, device="cuda")
    compact[0] = -3
    compact[1] = 0
    compact[2] = 7
    output = torch.full((experts, cap, hidden), 91, dtype=torch.bfloat16, device="cuda")
    scales = torch.ones((experts, cap, 1), dtype=torch.float32, device="cuda")
    unpack_expert_rows(compact, mapping, counts, output, scales)
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


def test_pack_then_unpack_round_trip_over_more_rows_than_programs():
    experts, cap, hidden, rows = 5, 600, 4096, 2600
    width, sf_width = hidden // 2, hidden // 16
    counts_host = [600, 0, 37, 600, 1]
    valid = sum(counts_host)
    recv = torch.zeros((experts, cap, width), dtype=torch.uint8, device="cuda")
    sf = torch.zeros((experts, cap, sf_width), dtype=torch.uint8, device="cuda")
    counts = torch.tensor(counts_host, dtype=torch.int32, device="cuda")
    x = torch.zeros((rows, width), dtype=torch.uint8, device="cuda")
    x_sf = torch.zeros((rows, sf_width), dtype=torch.uint8, device="cuda")
    ids = torch.empty((rows, 1), dtype=torch.int32, device="cuda")
    weights = torch.empty((rows, 1), dtype=torch.float32, device="cuda")
    mapping = torch.empty(rows, dtype=torch.int32, device="cuda")
    pack_expert_rows(
        recv, sf, counts, x, x_sf, ids, weights, mapping, ep_rank=0, num_experts=5
    )
    # One distinct value per compact row; rows beyond the prefix are poisoned.
    values = (torch.arange(rows, device="cuda") % 251 + 1).to(torch.bfloat16)
    compact = values[:, None].repeat(1, hidden)
    compact[valid:] = float("nan")
    output = torch.full((experts, cap, hidden), 91, dtype=torch.bfloat16, device="cuda")
    scales = torch.ones((experts, cap, 1), dtype=torch.float32, device="cuda")
    unpack_expert_rows(compact, mapping, counts, output, scales)
    flat = output.reshape(-1, hidden)
    slots = mapping[:valid].long()
    assert torch.unique(slots).numel() == valid
    assert torch.equal(flat[slots], compact[:valid])
    untouched = torch.ones(experts * cap, dtype=torch.bool, device="cuda")
    untouched[slots] = False
    assert torch.all(flat[untouched] == 91)
    torch.testing.assert_close(
        scales.flatten()[slots], (1.0 / values[:valid].float()) * 2688.0
    )
    assert torch.all(scales.flatten()[untouched] == 1)
