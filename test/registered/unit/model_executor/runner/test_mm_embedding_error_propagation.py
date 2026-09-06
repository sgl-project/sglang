import contextlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_buffer_registry import build_eager_registry
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _batch():
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.tensor([1, 2]),
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([2]),
        out_cache_loc=torch.tensor([0, 1]),
        seq_lens_sum=2,
        positions=torch.tensor([0, 1]),
        forward_metadata_ready=True,
    )


@pytest.mark.parametrize("no_copy", [False, True])
@pytest.mark.parametrize("pdmux", [False, True])
def test_eager_private_batch_returns_mm_errors_to_worker(no_copy, pdmux):
    batch = _batch()
    errors = [(0, 576, 575)]
    seen_batches = []

    def forward(input_ids, positions, model_batch):
        seen_batches.append(model_batch)
        model_batch.mm_embedding_errors = errors
        return "logits"

    runner = object.__new__(EagerRunner)
    runner.enable_pdmux = pdmux
    runner.model_runner = SimpleNamespace(
        _extend_forward_kwargs=lambda *_: {},
        model=SimpleNamespace(forward=forward),
        device_timer=None,
        prefill_cuda_graph_runner=None,
    )
    runner._eager_registry = build_eager_registry(
        device=torch.device("cpu"),
        max_bs=1,
        max_num_token=2,
        cache_loc_dtype=torch.int64,
    )
    with (
        envs.SGLANG_EAGER_INPUT_NO_COPY.override(no_copy),
        patch(
            "sglang.srt.model_executor.runner.eager_runner.is_cp_v2_active",
            return_value=False,
        ),
    ):
        assert runner.execute(batch) == "logits"
        assert batch.mm_embedding_errors == [(0, 576, 575)]
        assert (seen_batches[0] is batch) == pdmux

        # Buffer reuse must not carry a previous request's error into a healthy one.
        errors = []
        healthy = _batch()
        assert runner.execute(healthy) == "logits"
        assert healthy.mm_embedding_errors == []


@pytest.mark.parametrize("full_graph", [False, True])
def test_captured_body_eager_wrapper_returns_mm_errors_to_worker(full_graph):
    batch = _batch()
    static_batch = replace(batch)
    seen_batches = []
    original_forward = Mock()

    def forward(input_ids, positions, model_batch):
        seen_batches.append(model_batch)
        model_batch.mm_embedding_errors = [(0, 576, 575)]
        return "logits"

    runner = SimpleNamespace(
        _is_full_backend=full_graph,
        _input_embeds_arg_idx=None,
        layer_model=SimpleNamespace(forward=original_forward),
        model_runner=SimpleNamespace(model=SimpleNamespace(forward=forward)),
        _prefill_forward_context=lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    result = PrefillCudaGraphRunner._execute_body_capture(
        runner, batch, static_batch, 2, 2, None
    )

    assert result == "logits"
    assert seen_batches[0] is not batch
    assert batch.mm_embedding_errors == [(0, 576, 575)]
    assert runner.layer_model.forward is original_forward


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
