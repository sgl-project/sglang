import sys

import pytest
import torch

from sglang.kernels.ops.moe.dsv4 import mega_moe_pre_dispatch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell GPU (sm_100+)",
)
def test_mxfp8_scale_output_uses_padded_row_stride() -> None:
    """Write each MXFP8 scale row without overwriting its physical padding."""
    torch.manual_seed(42)
    num_tokens, padded_max, hidden, top_k = 5, 8, 2304, 8
    num_groups = hidden // 32
    logical_scale_int32 = num_groups // 4
    scale_stride_int32 = 20
    marker = 0xA5

    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    topk_idx = (
        torch.arange(num_tokens * top_k, device="cuda", dtype=torch.int32)
        .reshape(num_tokens, top_k)
        .remainder(256)
    )
    topk_weights = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)

    buf_x = torch.empty(padded_max, hidden, device="cuda", dtype=torch.float8_e4m3fn)
    scale_bytes = torch.full(
        (padded_max, scale_stride_int32 * 4),
        marker,
        device="cuda",
        dtype=torch.uint8,
    )
    buf_x_sf = scale_bytes.view(torch.int32)[:, :logical_scale_int32]
    buf_topk_idx = torch.empty(padded_max, top_k, device="cuda", dtype=torch.int64)
    buf_topk_weights = torch.empty(
        padded_max, top_k, device="cuda", dtype=torch.float32
    )

    assert buf_x_sf.shape == (padded_max, logical_scale_int32)
    assert buf_x_sf.stride() == (scale_stride_int32, 1)
    mega_moe_pre_dispatch(
        x,
        topk_idx,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
    )
    torch.cuda.synchronize()

    logical_scale_bytes = logical_scale_int32 * 4
    assert torch.all(scale_bytes[:num_tokens, :logical_scale_bytes] != marker)
    assert torch.all(scale_bytes[:num_tokens, logical_scale_bytes:] == marker)
    assert torch.all(scale_bytes[num_tokens:] == marker)
    torch.testing.assert_close(buf_topk_idx[:num_tokens], topk_idx.to(torch.int64))
    torch.testing.assert_close(buf_topk_weights[:num_tokens], topk_weights)
    assert torch.all(buf_topk_idx[num_tokens:] == -1)
    assert torch.all(buf_topk_weights[num_tokens:] == 0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell GPU (sm_100+)",
)
def test_empty_mhc_view_clears_idle_dispatch() -> None:
    hidden, padded_max, top_k = 2304, 8, 8
    x = torch.empty(1, 4, hidden, device="cuda", dtype=torch.bfloat16)[:0, 0]
    assert x.stride(0) == 4 * hidden
    buf_x = torch.zeros(padded_max, hidden, device="cuda", dtype=torch.float8_e4m3fn)
    buf_x_sf = torch.zeros(padded_max, 20, device="cuda", dtype=torch.int32)[:, :18]
    buf_idx = torch.full((padded_max, top_k), 42, device="cuda", dtype=torch.int64)
    buf_weights = torch.ones(padded_max, top_k, device="cuda", dtype=torch.float32)
    mega_moe_pre_dispatch(
        x,
        torch.empty(0, top_k, device="cuda", dtype=torch.int32),
        torch.empty(0, top_k, device="cuda", dtype=torch.float32),
        buf_x,
        buf_x_sf,
        buf_idx,
        buf_weights,
    )
    torch.testing.assert_close(buf_idx, torch.full_like(buf_idx, -1))
    torch.testing.assert_close(buf_weights, torch.zeros_like(buf_weights))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell GPU (sm_100+)",
)
@pytest.mark.parametrize("hidden", [2304, 3584])
def test_live_rows_change_on_graph_replay(hidden: int) -> None:
    """Padded NaNs must be masked on device, including after a full replay."""
    torch.manual_seed(42)
    rows, capacity, top_k = 32, 64, 16
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    x = source.clone()
    ids = (
        torch.arange(rows * top_k, device="cuda", dtype=torch.int32).view(rows, top_k)
        % 256
    )
    weights = torch.rand(rows, top_k, device="cuda")
    count = torch.tensor(rows, device="cuda", dtype=torch.int32)

    def buffers():
        scales = torch.empty(
            capacity, ((hidden // 32 + 15) // 16) * 4, device="cuda", dtype=torch.int32
        )
        return (
            torch.empty(capacity, hidden, device="cuda", dtype=torch.float8_e4m3fn),
            scales[:, : hidden // 128],
            torch.empty(capacity, top_k, device="cuda", dtype=torch.int64),
            torch.empty(capacity, top_k, device="cuda"),
        )

    actual, expected = buffers(), buffers()
    kwargs = {"num_token_non_padded": count}
    mega_moe_pre_dispatch(x, ids, weights, *actual, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mega_moe_pre_dispatch(x, ids, weights, *actual, **kwargs)
    for live in [rows, 1, 0, rows - 1, rows]:
        x.copy_(source)
        x[live:].fill_(float("nan"))
        count.fill_(live)
        clean = source.clone()
        clean[live:] = 0
        ref_ids, ref_weights = ids.clone(), weights.clone()
        ref_ids[live:] = -1
        ref_weights[live:] = 0
        mega_moe_pre_dispatch(clean, ref_ids, ref_weights, *expected)
        graph.replay()
        torch.cuda.synchronize()
        # Input remains untouched and the producer preserves FP8 scale bytes.
        assert torch.isnan(x[live:]).all()
        torch.testing.assert_close(
            actual[0][:rows].view(torch.uint8),
            expected[0][:rows].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        for output, ref in zip(actual[1:], expected[1:]):
            n = rows if output is actual[1] else capacity
            torch.testing.assert_close(output[:n], ref[:n], rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
