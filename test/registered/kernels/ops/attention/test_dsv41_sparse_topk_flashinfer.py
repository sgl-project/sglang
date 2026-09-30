"""Strict genuine selection + physical remap tests, including CUDA graphs.

These call the production utility with real stock/FlashInfer kernels. They do
not claim model or serving validation. Active-prefix NaNs are unsupported.
"""

import importlib

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    resolve_flashinfer_sparse_topk,
    topk_transform_sparse,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _operator():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    op = resolve_flashinfer_sparse_topk(
        torch.device("cuda", torch.cuda.current_device())
    )
    if op is None:
        pytest.skip(
            "FlashInfer varlen API with registered cuDNN backend is unavailable"
        )
    return op


def _inputs(rows, width, topk, seed, pattern):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randn((rows, width), device="cuda", generator=generator).bfloat16()
    lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    if pattern != "full":
        boundary = torch.tensor(
            [0, 1, topk - 1, topk, topk + 1, width - 7, width],
            dtype=torch.int32,
            device="cuda",
        )
        lengths.copy_(boundary[torch.arange(rows, device="cuda") % len(boundary)])
    if pattern == "ties":
        scores.fill_(1)
        scores[:, ::3] = -0.0
        scores[:, 1::3] = 0.0
        scores[:, 2::11] = torch.inf
        scores[:, 5::11] = -torch.inf
    columns = torch.arange(width, device="cuda")
    # The real producer leaves this suffix unspecified; NaNs must be ignored.
    scores.masked_fill_(columns[None, :] >= lengths[:, None], torch.nan)
    blocks = torch.stack(
        [
            torch.randperm(width // 8, device="cuda", generator=generator)
            + (row + 1) * (width // 8)
            for row in range(rows)
        ]
    ).int()
    storage = torch.full((rows, topk + 8), -12345, dtype=torch.int32, device="cuda")
    return scores, lengths, blocks, storage, storage[:, 4:-4]


def _check(scores, lengths, blocks, storage, output, original):
    for current, previous in zip((scores, lengths, blocks), original):
        assert torch.equal(current.view(torch.uint8), previous.view(torch.uint8))
    assert bool((storage[:, :4] == -12345).all())
    assert bool((storage[:, -4:] == -12345).all())
    x, lens, tables, got = (item.cpu() for item in (scores, lengths, blocks, output))
    topk = output.shape[1]
    for row in range(scores.shape[0]):
        n = min(topk, int(lens[row]))
        assert bool((got[row, n:] == -1).all())
        slots = got[row, :n].tolist()
        inverse = {int(block): index for index, block in enumerate(tables[row])}
        raw = [inverse[slot // 8] * 8 + slot % 8 for slot in slots]
        assert len(set(raw)) == n
        assert all(0 <= col < int(lens[row]) for col in raw)
        chosen = x[row, raw].float().sort().values
        reference = x[row, : int(lens[row])].float().topk(n).values.sort().values
        assert torch.equal(chosen, reference)


@pytest.mark.parametrize(
    "rows,topk,pattern",
    [
        (32, 512, "full"),
        (512, 512, "full"),
        (32, 512, "ragged"),
        (512, 512, "ragged"),
        (32, 512, "ties"),
        (8, 1024, "ragged"),
        (8, 2048, "ties"),
    ],
)
def test_sparse_topk_selection_and_remap(rows, topk, pattern):
    op = _operator()
    args = _inputs(rows, 16384, topk, 71237, pattern)
    scores, lengths, blocks, storage, output = args
    original = [item.clone() for item in (scores, lengths, blocks)]
    for selected in (None, op):
        output.fill_(-12345)
        topk_transform_sparse(scores, lengths, blocks, output, topk_op=selected)
        torch.cuda.synchronize()
        _check(*args, original)


def test_sparse_topk_changed_graphs_on_separate_streams(monkeypatch):
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    op = _operator()
    backend = importlib.import_module(
        "flashinfer.experimental.cudnn_topk_varlen.backend"
    )
    backend_calls = []
    original_run = backend.run

    def observed_backend(*args, **kwargs):
        backend_calls.append(torch.cuda.current_stream().cuda_stream)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(backend, "run", observed_backend)
    sets = [_inputs(512, 16384, 512, seed, "ragged") for seed in (9217, 9257)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    seen = []

    def observed(*args, **kwargs):
        seen.append(
            (
                torch.cuda.current_device(),
                torch.cuda.current_stream().cuda_stream,
                kwargs["out_indices"].data_ptr(),
            )
        )
        prior = len(backend_calls)
        result = op(*args, **kwargs)
        assert len(backend_calls) == prior + 1, "Expected actual cuDNN execution"
        assert op.suitable_auto_backends[0] == "cudnn"
        assert backend_calls[-1] == torch.cuda.current_stream().cuda_stream
        return result

    try:
        for args, stream, graph in zip(sets, streams, graphs):
            stream.wait_stream(torch.cuda.current_stream())
            scores, lengths, blocks, _, output = args
            with torch.cuda.stream(stream):
                topk_transform_sparse(scores, lengths, blocks, output, topk_op=observed)
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                topk_transform_sparse(scores, lengths, blocks, output, topk_op=observed)
        assert {entry[1] for entry in seen} == {
            stream.cuda_stream for stream in streams
        }
        assert seen[1][2] != seen[3][2], "Captured calls must own distinct raw outputs"
        for phase in range(3):
            snapshots = []
            phase_fresh_owners = []
            for index, (args, stream, graph) in enumerate(zip(sets, streams, graphs)):
                fresh = _inputs(
                    512,
                    16384,
                    512,
                    10231 + phase * 19 + index,
                    "ties" if phase == 2 else "full",
                )
                phase_fresh_owners.append(fresh)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for destination, value in zip(args[:3], fresh[:3]):
                        destination.copy_(value)
                    snapshots.append([item.clone() for item in args[:3]])
                    args[-1].fill_(-12345)
                    graph.replay()
            for stream in streams:
                stream.synchronize()
            for args, snapshot in zip(sets, snapshots):
                _check(*args, snapshot)
            # All asynchronous source copies have completed on both streams.
            phase_fresh_owners.clear()
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
