import sys
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import torch

if find_spec("flashinfer") is None:
    flashinfer = ModuleType("flashinfer")
    flashinfer.__spec__ = ModuleSpec("flashinfer", loader=None)
    flashinfer.top_k = Mock()
    sys.modules["flashinfer"] = flashinfer

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.utils import GenerationBatchResult  # noqa: E402
from sglang.srt.speculative.uno_worker_v2 import UnoWorkerV2  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="stage-a-test-cpu-intel")


def _run_uno_prefill(mm_embedding_errors, events):
    result = GenerationBatchResult(
        next_token_ids=torch.tensor([7, 8]),
        mm_embedding_errors=mm_embedding_errors,
    )
    worker = object.__new__(UnoWorkerV2)
    worker._target_worker = SimpleNamespace(
        forward_batch_generation=Mock(return_value=result)
    )
    worker.forward_width = 2
    batch = SimpleNamespace(reqs=[object(), object()], seq_lens=torch.tensor([5, 6]))
    result = worker._forward_prefill(
        batch,
        lambda seq_lens: events.append(("worker", None, seq_lens.clone())),
    )
    return batch, result


def _publish_from_scheduler(batch, result, events):
    scheduler = object.__new__(Scheduler)
    scheduler.future_map = SimpleNamespace(
        publish=lambda indices, seq_lens: events.append(
            ("scheduler", indices.clone(), seq_lens.clone())
        )
    )
    Scheduler._publish_speculative_overlap_result(
        scheduler,
        batch,
        torch.tensor([10, 20]),
        result,
    )


def test_mixed_uno_prefill_defers_publication_until_failed_request_is_filtered():
    events = []
    errors = torch.tensor([[1, 1, 3, 2], [1, 0, 0, 0]])

    batch, result = _run_uno_prefill(errors, events)
    assert events == []

    _publish_from_scheduler(batch, result, events)
    assert len(events) == 1
    source, indices, seq_lens = events[0]
    assert source == "scheduler"
    assert torch.equal(indices, torch.tensor([20]))
    assert torch.equal(seq_lens, torch.tensor([6]))


def test_valid_uno_prefill_keeps_worker_publication_without_late_republish():
    events = []
    errors = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])

    batch, result = _run_uno_prefill(errors, events)
    _publish_from_scheduler(batch, result, events)

    assert len(events) == 1
    source, indices, seq_lens = events[0]
    assert source == "worker"
    assert indices is None
    assert torch.equal(seq_lens, torch.tensor([5, 6]))
