"""Tracking windows preserve Torch's floor division, clamping, and padded rows."""

import pytest
import torch

from sglang.kernels.ops.mamba.sconv_tracking import fill_track_conv_indices
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _reference(query_starts, tracked, prefix, rows, width, live, chunk):
    output = torch.zeros((rows, width), dtype=torch.int64, device=query_starts.device)
    delta = tracked[:live].long()
    if prefix is not None:
        delta = delta - prefix[:live].long()
    start = query_starts[:live].long() + (delta // chunk) * chunk - width
    output[:live] = torch.minimum(
        (start[:, None] + torch.arange(width, device=output.device)).clamp_min(0),
        query_starts[-1:].long() - 1,
    )
    return output


@requires_cuda
@pytest.mark.parametrize("rows", [0, 1, 7, 32, 129, 1024])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("has_prefix", [False, True])
def test_tracking_windows_and_padding(rows, dtype, has_prefix):
    query_starts = torch.arange(rows + 1, device="cuda", dtype=dtype) * 256
    for live in sorted({0, rows // 2, rows}):
        prefix = torch.arange(live, device="cuda", dtype=dtype) * 256 + 8192
        offsets = torch.tensor([-129, -128, -1, 0, 127, 128, 257], device="cuda")
        tracked = prefix + offsets[torch.arange(live, device="cuda") % 7].to(dtype)
        prefix = prefix if has_prefix else None
        output = torch.full((rows, 3), 999, dtype=torch.int64, device="cuda")
        fill_track_conv_indices(
            query_start_loc=query_starts,
            track_seqlens=tracked,
            prefix_lens=prefix,
            output=output,
            live=live,
            chunk_size=128,
        )
        torch.testing.assert_close(
            output,
            _reference(query_starts, tracked, prefix, rows, 3, live, 128),
            rtol=0,
            atol=0,
        )


@requires_cuda
@pytest.mark.parametrize("chunk", [96, 128])
@pytest.mark.parametrize("total", [0, 1, 8192])
@pytest.mark.parametrize("width", [1, 3, 7])
def test_negative_offsets_large_prefixes_and_clamping(chunk, total, width):
    query_starts = torch.tensor([0, 0, 0, total], dtype=torch.int64, device="cuda")
    prefix = torch.tensor([2**33, 8192, 8192], dtype=torch.int64, device="cuda")
    tracked = prefix + torch.tensor([-1, -129, 257], device="cuda")
    output = torch.empty((3, width), dtype=torch.int64, device="cuda")
    fill_track_conv_indices(
        query_start_loc=query_starts,
        track_seqlens=tracked,
        prefix_lens=prefix,
        output=output,
        live=3,
        chunk_size=chunk,
    )
    torch.testing.assert_close(
        output,
        _reference(query_starts, tracked, prefix, 3, width, 3, chunk),
        rtol=0,
        atol=0,
    )


@requires_cuda
def test_graph_replay_refreshes_indices_and_zeroes_tail():
    rows, live, width = 64, 32, 3
    query_starts = torch.arange(rows + 1, dtype=torch.int32, device="cuda") * 128
    prefix = torch.full((live,), 8192, dtype=torch.int32, device="cuda")
    tracked = prefix.long() + 256
    output = torch.empty((rows, width), dtype=torch.int64, device="cuda")

    def fill():
        fill_track_conv_indices(
            query_start_loc=query_starts,
            track_seqlens=tracked,
            prefix_lens=prefix,
            output=output,
            live=live,
            chunk_size=128,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fill()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fill()
    pointer = output.data_ptr()
    for shift in (0, 128, -257):
        tracked.add_(shift)
        query_starts.add_(1)
        output.fill_(999)
        graph.replay()
        torch.testing.assert_close(
            output,
            _reference(query_starts, tracked, prefix, rows, width, live, 128),
            rtol=0,
            atol=0,
        )
        assert output.data_ptr() == pointer


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-x"]))
