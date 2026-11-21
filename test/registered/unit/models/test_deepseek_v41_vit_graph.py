from contextlib import contextmanager

import pytest
import torch

from sglang.srt.models.deepseek_v41_vit import assemble_image_span
from sglang.srt.multimodal.deepseek_v41_image_processing import image_token_types
from sglang.srt.multimodal.deepseek_v41_vit_cuda_graph_runner import (
    DeepseekV41ViTCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(("n_llm_h", "n_llm_w"), [(1, 1), (3, 5), (20, 19)])
def test_assemble_image_span_matches_token_types(n_llm_h, n_llm_w):
    d = 8
    features = torch.randn(n_llm_h * n_llm_w, d)
    start, newline, end = torch.randn(3, d).unbind(0)
    types = image_token_types(n_llm_h, n_llm_w)
    expected = torch.empty(len(types), d)
    expected[types == 0] = start
    expected[types == 1] = features
    expected[types == 2] = newline
    expected[types == 3] = end

    span = assemble_image_span(features, n_llm_h, n_llm_w, start, newline, end)

    assert torch.equal(span, expected)


class _FakeGraph:
    def __init__(self):
        self.replay_fn = None

    def replay(self):
        self.replay_fn()


@pytest.fixture
def fake_cuda_graph(monkeypatch):
    """CPU stand-in: encode() registers what replay() recomputes in place."""
    state = {"graph": None}

    @contextmanager
    def graph_context(graph, pool=None, capture_error_mode=None):
        state["graph"] = graph
        try:
            yield
        finally:
            state["graph"] = None

    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", graph_context)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", object)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 0)

    calls = {"eager": 0, "capture": 0}

    def encode(patches, n_h, n_w):
        out = patches * 2
        graph = state["graph"]
        if graph is None:
            calls["eager"] += 1
        else:
            calls["capture"] += 1
            graph.replay_fn = lambda: torch.mul(patches, 2, out=out)
        return out

    return encode, calls


def test_graph_runner_captures_on_second_sighting(fake_cuda_graph):
    encode, calls = fake_cuda_graph
    runner = DeepseekV41ViTCudaGraphRunner(encode, lambda *_: None, min_hits=2)
    patches = torch.randn(6, 4)

    runner.run(patches, 2, 3)
    assert calls == {"eager": 1, "capture": 0}
    assert not runner.graphs

    for _ in range(3):
        assert torch.equal(runner.run(patches, 2, 3), patches * 2)
    assert calls == {"eager": 1, "capture": 1}
    assert list(runner.graphs) == [(2, 3)]


def test_graph_runner_returned_spans_do_not_alias(fake_cuda_graph):
    # Spans are kept in the multimodal embedding cache across later replays.
    encode, _ = fake_cuda_graph
    runner = DeepseekV41ViTCudaGraphRunner(encode, lambda *_: None, min_hits=1)
    first_patches, second_patches = torch.randn(2, 6, 4).unbind(0)

    first = runner.run(first_patches, 2, 3)
    second = runner.run(second_patches, 2, 3)

    assert torch.equal(first, first_patches * 2)
    assert torch.equal(second, second_patches * 2)


def test_graph_runner_failed_capture_stays_eager(fake_cuda_graph, monkeypatch):
    encode, calls = fake_cuda_graph
    runner = DeepseekV41ViTCudaGraphRunner(encode, lambda *_: None, min_hits=1)
    attempts = []
    spans = []

    @contextmanager
    def failing_graph(graph, pool=None, capture_error_mode=None):
        attempts.append(True)
        raise RuntimeError("operation not permitted when stream is capturing")
        yield

    monkeypatch.setattr(torch.cuda, "graph", failing_graph)
    patches = torch.randn(6, 4)

    for _ in range(3):
        spans.append(runner.run(patches, 2, 3))

    assert attempts == [True]
    assert calls["eager"] == 3
    assert runner.failed_keys == {(2, 3)}
    assert not runner.graphs
    assert all(torch.equal(span, patches * 2) for span in spans)


def test_graph_runner_evicts_least_recently_used_grid(fake_cuda_graph):
    encode, _ = fake_cuda_graph
    runner = DeepseekV41ViTCudaGraphRunner(
        encode, lambda *_: None, capacity=2, min_hits=1
    )

    for n_w in (1, 2, 1, 3):
        runner.run(torch.randn(n_w, 4), 1, n_w)

    assert list(runner.graphs) == [(1, 1), (1, 3)]
