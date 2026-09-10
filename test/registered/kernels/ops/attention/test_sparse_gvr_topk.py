"""Shared DSA/DSv4 GVR_2 routing, transforms, and CUDA-graph replay."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa.gvr_topk import (
    GvrTopkState,
    flashinfer_sparse_topk,
    gvr_available,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.fixture(autouse=True)
def require_gvr():
    if not torch.cuda.is_available() or not gvr_available(torch.device("cuda")):
        pytest.skip("Requires hint-free FlashInfer GVR_2 on a supported GPU")


def assert_raw(scores, lengths, raw, k, starts=None):
    for row, length in enumerate(lengths.tolist()):
        length = min(max(length, 0), scores.shape[1])
        start = 0 if starts is None else int(starts[row])
        indices = raw[row][raw[row] >= 0].long()
        assert len(indices) == len(indices.unique()) == min(k, length)
        assert (indices < length).all()
        actual = scores[row, start + indices].sort().values
        expected = (
            scores[row, start : start + length]
            .topk(min(k, length))
            .values.sort()
            .values
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("backend", ["auto", "gvr_2"])
def test_raw_cold_and_ragged(k, backend):
    torch.manual_seed(41)
    scores = torch.randn(5, 8200, device="cuda")[:, :8192]
    lengths = torch.tensor([0, 1, k - 1, k + 1, 8192], dtype=torch.int32, device="cuda")
    raw = flashinfer_sparse_topk(scores, lengths, k, backend=backend)
    assert_raw(scores, lengths, raw, k)
    offsets = torch.arange(5, device="cuda", dtype=torch.int32) * 8192
    ragged = flashinfer_sparse_topk(
        scores, lengths, k, backend=backend, offsets=offsets
    )
    recovered = torch.where(ragged >= 0, ragged - offsets[:, None], -1)
    assert_raw(scores, lengths, recovered, k)


@pytest.mark.parametrize("width", [4096, 4093])
def test_minimally_backed_strided_view(width):
    scores = torch.empty_strided((4, width), (4100, 1), device="cuda").normal_()
    lengths = torch.full((4,), width, dtype=torch.int32, device="cuda")
    result = flashinfer_sparse_topk(scores, lengths, 512)
    assert_raw(scores, lengths, result, 512)


@pytest.mark.parametrize("width", [4096, 4093])
@pytest.mark.parametrize(
    "method", [TopkTransformMethod.PAGED, TopkTransformMethod.RAGGED]
)
def test_shared_prefill_offsets(method, width):
    # Two requests, two causal rows each, with packed KV row starts.
    k, n = 512, 4096
    scores = torch.randn(4, n, device="cuda")[:, :width]
    starts = torch.tensor([0, 0, 2048, 2048], dtype=torch.int32, device="cuda")
    lengths = torch.tensor(
        [511, 1800, 512, width - 2048], dtype=torch.int32, device="cuda"
    )
    pages = torch.stack([torch.randperm(32, device="cuda") for _ in range(2)]).int()
    metadata = SimpleNamespace(
        real_page_table=pages,
        page_table_1=None,
        cu_seqlens_k=torch.tensor([0, 2048, 4096], dtype=torch.int32, device="cuda"),
    )
    cu_q = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    with patch(
        "sglang.srt.model_executor.forward_context.get_token_to_kv_pool",
        return_value=SimpleNamespace(page_size=64),
    ):
        result = DSATopKBackend.AUTO.topk_transform(
            scores,
            lengths,
            k,
            method,
            metadata,
            cu_seqlens_q_topk=cu_q,
            row_starts=starts,
            topk_indices_offset=starts,
        )
    if method == TopkTransformMethod.RAGGED:
        raw = torch.where(result >= 0, result - starts[:, None], -1)
    else:
        inverse = pages.argsort(dim=1)
        mapping = torch.tensor([0, 0, 1, 1], device="cuda")
        raw = (
            inverse[mapping[:, None], result.clamp(min=0).long() // 64] * 64
            + result % 64
        )
        raw = raw.masked_fill(result < 0, -1)
    assert_raw(scores, lengths, raw, k, starts)


def test_graph_replay_and_slot_reset():
    k, n, rows = 512, 32768, 4
    scores = torch.randn(rows, n, device="cuda")
    lengths = torch.tensor([1024, 8000, 16000, n], dtype=torch.int32, device="cuda")
    slots = torch.tensor([3, 1, 4, 2], device="cuda")
    state = GvrTopkState(num_layers=2, num_slots=6, top_k=k, device="cuda")
    out = torch.empty(rows, k, dtype=torch.int32, device="cuda")

    def run():
        return flashinfer_sparse_topk(
            scores, lengths, k, state=state, layer_id=1, req_pool_indices=slots, out=out
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()  # Eager compilation and per-stream workspace initialization.
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for step in range(3):
        scores.normal_()
        lengths.copy_(torch.tensor([1, 8192 + step, 16000 + step, n], device="cuda"))
        slots.copy_(torch.tensor([2, 4, 1, 3], device="cuda"))
        graph.replay()
        assert_raw(scores, lengths, out, k)
        torch.testing.assert_close(state.hints[1, slots], out)
    state.reset(1, slots)
    assert (state.hints[1, slots] == -1).all()
    assert (state.hints[0] == -1).all()
    # Disaggregated decode can reuse a slot without seeing its prefill.
    state.hints.fill_(7)
    generations = torch.zeros(6, dtype=torch.int64)
    generations[3] = 1
    state.sync_generations(generations)
    assert (state.hints[:, 3] == -1).all()
    assert (state.hints[:, 1] == 7).all()


def test_selection_capability_and_overrides():
    scores = torch.randn(2, 4096, device="cuda")
    assert DSATopKBackend.AUTO.use_varlen(scores, 512)
    assert not DSATopKBackend.SGL_KERNEL.use_varlen(scores, 512)
    assert not DSATopKBackend.AUTO.use_varlen(scores, 17)
    assert not DSATopKBackend.AUTO.use_varlen(scores.half(), 512)
    with pytest.raises(RuntimeError, match="flashinfer-gvr requires"):
        DSATopKBackend.FLASHINFER_GVR.use_varlen(scores, 17)
    assert (
        DSATopKBackend.from_server_args(
            SimpleNamespace(
                dsa_topk_backend="auto",
                enable_deterministic_inference=True,
            )
        )
        == DSATopKBackend.SGL_KERNEL
    )


def test_padded_slots_and_ties():
    k, n = 512, 4096
    scores = torch.ones(4, n, device="cuda")
    lengths = torch.tensor([2048, 1, 1, 0], dtype=torch.int32, device="cuda")
    slots = torch.tensor([1, 0, 0, 0], device="cuda")
    state = GvrTopkState(num_layers=1, num_slots=2, top_k=k, device="cuda")
    raw = flashinfer_sparse_topk(
        scores, lengths, k, state=state, req_pool_indices=slots
    )
    assert_raw(scores, lengths, raw, k)
    assert (state.hints[0, 0] == -1).all()
    torch.testing.assert_close(state.hints[0, 1], raw[0])


def test_independent_graph_streams():
    runs = []
    for _ in range(2):
        scores = torch.randn(2, 65536, device="cuda")
        lengths = torch.tensor([32768, 65536], dtype=torch.int32, device="cuda")
        out = torch.empty(2, 512, dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            flashinfer_sparse_topk(scores, lengths, 512, out=out)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            flashinfer_sparse_topk(scores, lengths, 512, out=out)
        runs.append((scores, lengths, out, stream, graph))
    for scores, lengths, out, stream, graph in runs:
        with torch.cuda.stream(stream):
            graph.replay()
    torch.cuda.synchronize()
    for scores, lengths, out, _, _ in runs:
        assert_raw(scores, lengths, out, 512)
