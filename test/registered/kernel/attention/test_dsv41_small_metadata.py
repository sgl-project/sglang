"""Integer metadata equivalence with changing CUDA graph inputs."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv41_small_metadata import low_ratio_metadata
from sglang.kernels.ops.speculative.dspark.commit_swa import committed_swa_locations
from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def make_core(lens, loc, topk):
    rows = lens.numel()
    kw = dict(device="cuda", dtype=torch.int32)
    return DSV4AttnMetadata(
        page_size=256,
        page_table=torch.empty((rows, 1), **kw),
        raw_out_loc=loc,
        cuda_int32_kwargs=kw,
        seq_lens_casual=lens,
        positions_casual=lens - 1,
        swa_page_indices=torch.empty((rows, 128), **kw),
        swa_topk_lengths=lens.clamp_max(128),
        index_topk=topk,
        present_ratios=(1, 2),
        low_ratios=(1, 2),
    )


@pytest.mark.parametrize("rows", [1, 5, 6, 8])
@pytest.mark.parametrize("topk", [512, 1024])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_compression_graph(rows, topk, dtype):
    lens = torch.zeros(rows, device="cuda", dtype=torch.int32)
    loc = torch.zeros(rows, device="cuda", dtype=dtype)
    for _ in range(3):
        low_ratio_metadata(lens, loc, topk)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = low_ratio_metadata(lens, loc, topk)
    fields = [
        f"c{ratio}_{suffix}"
        for ratio in (1, 2)
        for suffix in (
            "out_loc",
            "topk_lengths_clamp1",
            "sparse_topk_lengths",
            "sparse_page_indices",
        )
    ]
    for replay in range(7):
        lens.copy_(torch.arange(rows, device="cuda") * 511 + replay - 2)
        # Includes padding, odd negative unused slots, and 64-bit write slots.
        values = torch.arange(rows, device="cuda", dtype=torch.int64) * 129 - 3 + replay
        if dtype == torch.int64 and replay >= 3:
            values += 1 << 33
        loc.copy_(values)
        out[3].fill_(123)
        out[7].fill_(456)
        graph.replay()
        ref = make_core(lens, loc, topk)
        ref.init_compression_metadata()
        ref.init_flashmla_related()
        candidate = make_core(lens, loc, topk)
        candidate.init_compression_metadata(low_ratio_buffers=out)
        candidate.init_flashmla_related(low_ratio_buffers=out)
        for name in fields:
            assert torch.equal(getattr(candidate, name), getattr(ref, name)), name


@pytest.mark.parametrize("bs", [1, 4, 16, 64])
@pytest.mark.parametrize("width", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_committed_locations_graph(bs, width, dtype):
    mapping = torch.randint(-1, 100000, (65536,), dtype=dtype, device="cuda")
    loc = torch.arange(bs * width, dtype=dtype, device="cuda")
    lens = torch.zeros(bs, dtype=torch.int32, device="cuda")
    for _ in range(3):
        committed_swa_locations(loc, mapping, lens, width)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = committed_swa_locations(loc, mapping, lens, width)
    for replay in range(9):
        mapping.random_(-1, 100000)
        loc.copy_(
            (torch.arange(bs * width, device="cuda") * 127 + replay - 2)
            % mapping.numel()
        )
        # Also cover valid negative torch indices, including the final element.
        if replay % 2:
            loc[::3] -= mapping.numel()
        lens.copy_((torch.arange(bs, device="cuda") + replay) % (width + 1))
        graph.replay()
        cols = torch.arange(width, device="cuda").view(1, -1)
        valid = (cols < lens.long().view(-1, 1)).reshape(-1)
        expected = torch.where(valid, mapping[loc.long()].int(), -1)
        assert torch.equal(out, expected)


def test_empty():
    empty = torch.empty(0, dtype=torch.int64, device="cuda")
    assert committed_swa_locations(empty, empty, empty, 6).numel() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
