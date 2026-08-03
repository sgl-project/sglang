from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _runner(*, require_input_embeds: bool = True) -> DecodeCudaGraphRunner:
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.capture_input_embeds = True
    runner.require_decode_input_embeds = require_input_embeds
    runner.buffers = SimpleNamespace(input_embeds=torch.zeros((2, 2)))
    return runner


def test_decode_graph_embedding_transport_copies_exact_rows() -> None:
    runner = _runner()
    embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    runner._copy_decode_input_embeds(
        SimpleNamespace(input_embeds=embeddings), 2, is_ragged=False
    )

    assert torch.equal(runner.buffers.input_embeds, embeddings)


def test_decode_graph_embedding_transport_fails_without_embeddings() -> None:
    runner = _runner()

    with pytest.raises(RuntimeError, match="requires input_embeds"):
        runner._copy_decode_input_embeds(
            SimpleNamespace(input_embeds=None), 1, is_ragged=False
        )


def test_legacy_optional_embedding_transport_skips_missing_embeddings() -> None:
    runner = _runner(require_input_embeds=False)

    runner._copy_decode_input_embeds(
        SimpleNamespace(input_embeds=None), 1, is_ragged=False
    )

    assert torch.equal(runner.buffers.input_embeds, torch.zeros((2, 2)))
