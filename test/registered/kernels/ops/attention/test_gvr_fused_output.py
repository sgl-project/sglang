"""Opt-in GVR page mapping, optional raw output, and graph replay."""

import inspect
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.gvr_topk import (
    flashinfer_sparse_topk,
    gvr_available,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("width", [512, 1088, 8448, 16640, 32768])
@pytest.mark.parametrize("with_raw", [False, True])
def test_fused_mapping_graph(width, with_raw):
    if not torch.cuda.is_available() or not gvr_available(torch.device("cuda")):
        pytest.skip("Requires FlashInfer GVR_2")
    import flashinfer

    if (
        "backend"
        not in inspect.signature(flashinfer.top_k_page_table_transform).parameters
    ):
        pytest.skip("Requires FlashInfer fused GVR page-table API")
    torch.manual_seed(19)
    batch, k, page_size = 4, 512, 64
    scores = torch.randn(batch, width, device="cuda")
    lengths = torch.tensor(
        [0, 1, min(513, width), width], device="cuda", dtype=torch.int32
    )
    num_pages = (width + page_size - 1) // page_size
    pages = torch.stack(
        [torch.randperm(num_pages, device="cuda") for _ in range(batch)]
    ).int()
    mapping = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
    out = torch.empty(batch, k, device="cuda", dtype=torch.int32)
    raw = torch.empty_like(out) if with_raw else None

    def call():
        return flashinfer_sparse_topk(
            scores,
            lengths,
            k,
            backend="gvr_2",
            page_table=pages,
            page_size=page_size,
            row_to_batch=mapping,
            out=out,
            raw_out=raw,
        )

    def check():
        for row, length in enumerate(lengths.tolist()):
            valid = out[row][out[row] >= 0].long()
            inverse = pages[mapping[row]].argsort()
            logical = inverse[valid // page_size] * page_size + valid % page_size
            assert logical.numel() == logical.unique().numel() == min(k, length)
            assert (logical < length).all()
            torch.testing.assert_close(
                scores[row, logical].sort().values,
                scores[row, :length].topk(min(k, length)).values.sort().values,
                rtol=0,
                atol=0,
            )
            assert (out[row][out[row] < 0] == -1).all()
            if raw is not None:
                torch.testing.assert_close(raw[row][out[row] >= 0].long(), logical)
                assert (raw[row][out[row] < 0] == -1).all()

    with envs.SGLANG_DSA_GVR_FUSE_OUTPUT.override(True), patch(
        "sglang.srt.layers.attention.dsa.gvr_topk._finish_topk"
    ) as finish:
        assert call() is out
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graph.replay()
        check()
        lengths.copy_(
            torch.tensor(
                [width, 0, min(511, width), min(17, width)],
                device="cuda",
                dtype=torch.int32,
            )
        )
        pages.copy_(pages.roll(1, dims=1))
        graph.replay()
        check()
        finish.__getitem__.assert_not_called()
