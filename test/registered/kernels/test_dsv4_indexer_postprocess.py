import unittest

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import candidate_block_logits
from sglang.kernels.ops.attention.dsv4.indexer_postprocess import filter_topk_pages
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def reference_pages(scores, indices, pages, page_size):
    cols = indices.to(torch.int64)
    value = scores.gather(1, cols.clamp(0, scores.shape[1] - 1))
    valid = (cols >= 0) & (cols < scores.shape[1]) & (value > -torch.inf)
    raw = indices.masked_fill(~valid, -1)
    safe = raw.clamp_min(0).to(torch.int64)
    slots = pages.gather(1, safe // page_size) * page_size + safe % page_size
    return torch.where(raw >= 0, slots, -1).to(torch.int32), raw


class TestIndexerPostprocess(CustomTestCase):
    def test_filter_and_page_mapping(self):
        torch.manual_seed(91)
        for rows in (1, 6, 64):
            for width in (1, 65, 4096):
                for dtype in (torch.int32, torch.int64):
                    # Row-strided tensors match sliced metadata buffers.
                    scores = torch.randn(rows, width + 7, device="cuda")[:, :width]
                    indices = torch.randint(
                        -2, width + 2, (rows, 520), device="cuda", dtype=dtype
                    )[:, :512]
                    indices[:, :5] = 0
                    scores[0, 0] = -torch.inf
                    if rows > 1:
                        scores[1, 0] = torch.nan
                        scores[2, 0] = torch.inf
                        scores[3, :] = -torch.inf
                    pages = torch.randint(
                        0,
                        100000,
                        (rows, (width + 63) // 64 + 3),
                        device="cuda",
                        dtype=torch.int32,
                    )[:, :-3]
                    out = torch.empty(rows, 520, device="cuda", dtype=torch.int32)[
                        :, :512
                    ]
                    raw = torch.empty_like(indices)
                    expected, expected_raw = reference_pages(scores, indices, pages, 64)
                    for write_raw in (False, True):
                        filter_topk_pages(
                            scores, indices, pages, out, 64, raw if write_raw else None
                        )
                        torch.testing.assert_close(out, expected, rtol=0, atol=0)
                        if write_raw:
                            torch.testing.assert_close(
                                raw, expected_raw, rtol=0, atol=0
                            )

    def test_graph_replay(self):
        rows, width = 6, 4096
        scores = torch.randn(rows, width, device="cuda")
        indices = torch.randint(
            -1, width, (rows, 512), device="cuda", dtype=torch.int32
        )
        pages = torch.randint(
            0, 100000, (rows, width // 64), device="cuda", dtype=torch.int32
        )
        out, raw = torch.empty_like(indices), torch.empty_like(indices)
        filter_topk_pages(scores, indices, pages, out, 64, raw)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            filter_topk_pages(scores, indices, pages, out, 64, raw)
        for _ in range(3):
            scores.normal_()
            scores[:, :32] = -torch.inf
            indices.random_(-1, width)
            pages.random_(0, 100000)
            graph.replay()
            expected, expected_raw = reference_pages(scores, indices, pages, 64)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)

    def test_candidate_publication(self):
        torch.manual_seed(912)
        for width, group, topk in (
            (65, 32, 8),
            (4096, 128, 8),
            (32768, 2048, 8),
            (1048576, 2048, 8),
            (4096, 8, 2048),
            (1048576, 8, 2048),
        ):
            rows = 6
            x = torch.randn(rows, width + 9, device="cuda")[:, :width]
            lengths = torch.tensor(
                [0, 1, width // 2, width - 1, width, width],
                device="cuda",
                dtype=torch.int32,
            )
            # Include ties, NaN and a final partial block.
            x[3, :] = 0
            x[4, 0] = torch.nan
            x[5, :] = -torch.inf
            masked = x.masked_fill(
                torch.arange(width, device="cuda")[None, :] >= lengths[:, None],
                -torch.inf,
            )
            blocks = (width + group - 1) // group
            padded = torch.nn.functional.pad(
                masked, (0, blocks * group - width), value=-torch.inf
            )
            scores = padded.reshape(rows, blocks, group).max(-1).values
            last = (lengths - 1) // group
            scores = torch.where(
                (torch.arange(blocks, device="cuda")[None, :] == last[:, None])
                & (lengths[:, None] > 0),
                torch.inf,
                scores,
            )
            selected = scores.topk(min(topk, blocks), dim=-1)
            keep = (
                torch.zeros_like(scores, dtype=torch.bool)
                .scatter_(1, selected.indices, selected.values > -torch.inf)
                .repeat_interleave(group, dim=-1)[:, :width]
            )
            got, published = candidate_block_logits(
                x, lengths, topk_blocks=topk, block_size=group, published=None
            )
            torch.testing.assert_close(got, masked, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(published, keep, rtol=0, atol=0)
            consumer, _ = candidate_block_logits(
                x, lengths, topk_blocks=topk, block_size=group, published=published
            )
            torch.testing.assert_close(
                consumer,
                masked.masked_fill(~keep, -torch.inf),
                rtol=0,
                atol=0,
                equal_nan=True,
            )


if __name__ == "__main__":
    unittest.main()
