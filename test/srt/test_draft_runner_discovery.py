"""Unit tests for sglang/srt/speculative/draft_utils.py::iter_draft_model_runners.

The discovery helper enumerates the independent draft ``ModelRunner``s behind a
worker via ``getattr``, so most workers can be faked with ``SimpleNamespace``.
NGRAM is the one case that must be a real instance because the helper uses
``isinstance``; we build it with ``object.__new__`` and set attributes by hand,
matching the style in test_distributed_weight_update_spec_worker.py.

All cases run on CPU: the helper only inspects attributes and never touches GPU.
"""

from types import SimpleNamespace

import pytest

from sglang.srt.speculative.draft_utils import iter_draft_model_runners
from sglang.srt.speculative.ngram_worker import NGRAMWorker


def test_none_worker_returns_empty():
    assert iter_draft_model_runners(None) == []


def test_v1_eagle_style_draft_model_runner():
    # v1 EAGLE / FrozenKVMTP / Standalone v1 all expose `draft_model_runner`.
    runner = object()
    worker = SimpleNamespace(draft_model_runner=runner)
    assert iter_draft_model_runners(worker) == [("draft", runner)]


def test_v2_eagle_style_inner_draft_runner():
    # v2 EAGLE / Standalone v2 nest the runner under `draft_worker.draft_runner`.
    runner = object()
    worker = SimpleNamespace(draft_worker=SimpleNamespace(draft_runner=runner))
    assert iter_draft_model_runners(worker) == [("draft", runner)]


def test_multi_layer_v1_lists_every_step():
    r0, r1, r2 = object(), object(), object()
    worker = SimpleNamespace(model_runner_list=[r0, r1, r2])
    assert iter_draft_model_runners(worker) == [
        ("draft_step_0", r0),
        ("draft_step_1", r1),
        ("draft_step_2", r2),
    ]


def test_multi_layer_v2_lists_every_step():
    r0, r1 = object(), object()
    worker = SimpleNamespace(
        draft_worker=SimpleNamespace(draft_runner_list=[r0, r1])
    )
    assert iter_draft_model_runners(worker) == [
        ("draft_step_0", r0),
        ("draft_step_1", r1),
    ]


def test_dflash_inner_worker_falls_through_to_direct_accessor():
    # DFlash aliases `model_runner` to the target yet still owns an independent
    # `draft_model_runner`. Its inner `draft_worker` is a plain TpModelWorker with
    # NEITHER `draft_runner` NOR `draft_runner_list`, so discovery must fall
    # through the inner branch to the direct `draft_model_runner` accessor — not
    # short-circuit to [] just because an inner draft_worker exists.
    draft_runner = object()
    target_runner = object()
    worker = SimpleNamespace(
        draft_worker=SimpleNamespace(),
        draft_model_runner=draft_runner,
        model_runner=target_runner,
        target_worker=SimpleNamespace(model_runner=target_runner),
    )
    assert iter_draft_model_runners(worker) == [("draft", draft_runner)]


def test_ngram_has_no_independent_draft():
    # NGRAM's model_runner IS the target's; the isinstance branch returns [] and
    # must NOT relabel the shared target runner as a draft.
    target_runner = object()
    worker = object.__new__(NGRAMWorker)
    worker.model_runner = target_runner
    worker.target_worker = SimpleNamespace(model_runner=target_runner)

    result = iter_draft_model_runners(worker)

    assert result == []
    assert target_runner not in [r for _, r in result]


def test_aliased_target_runner_returns_empty():
    # No draft accessor, but model_runner is the target's runner -> [].
    target_runner = object()
    worker = SimpleNamespace(
        model_runner=target_runner,
        target_worker=SimpleNamespace(model_runner=target_runner),
    )
    assert iter_draft_model_runners(worker) == []


def test_unrecognized_worker_raises():
    # Nothing recognizable as a draft and no aliased-target match -> hard error,
    # rather than silently returning [].
    worker = SimpleNamespace(foo=1)
    with pytest.raises(ValueError):
        iter_draft_model_runners(worker)
