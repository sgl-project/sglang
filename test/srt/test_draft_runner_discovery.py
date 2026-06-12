"""Unit tests for the per-worker ``iter_draft_runners()`` methods.

These replace the old ``draft_utils.iter_draft_model_runners`` discovery helper:
each spec worker now reports its own independent draft ``ModelRunner``(s) as
``(role, runner)`` pairs, and the weight-check fan-out just calls the method.

Workers are built with ``object.__new__`` + manual attribute setting so no GPU /
real ``ModelRunner`` is needed — the methods only read attributes. All CPU.
"""

from types import SimpleNamespace

import pytest

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.speculative.dflash_worker import DFlashWorker
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import MultiLayerEagleWorkerV2
from sglang.srt.speculative.ngram_worker import NGRAMWorker


def _new(cls, **attrs):
    # Build a worker without running __init__, then set the attributes the method
    # reads — mirrors the object.__new__ trick the old test used for NGRAM.
    w = object.__new__(cls)
    for k, v in attrs.items():
        setattr(w, k, v)
    return w


# --- TpModelWorker default (covers all four v1 workers by inheritance) ---


def test_tp_worker_target_has_no_draft():
    # The target worker (is_draft_worker=False) owns no independent draft.
    w = _new(TpModelWorker, is_draft_worker=False)
    assert w.iter_draft_runners() == []


def test_tp_worker_single_layer_draft_is_own_model_runner():
    # v1 EAGLE / Standalone / FrozenKVMTP: the draft model IS this worker's own
    # model_runner; model_runner_list stays empty.
    runner = object()
    w = _new(
        TpModelWorker, is_draft_worker=True, model_runner_list=[], _model_runner=runner
    )
    assert w.iter_draft_runners() == [("draft", runner)]


def test_tp_worker_multi_layer_lists_every_step():
    # v1 MultiLayerEagle: model_runner_list holds one runner per step (index 0 is
    # model_runner itself).
    r0, r1, r2 = object(), object(), object()
    w = _new(
        TpModelWorker, is_draft_worker=True, model_runner_list=[r0, r1, r2], _model_runner=r0
    )
    assert w.iter_draft_runners() == [
        ("draft_step_0", r0),
        ("draft_step_1", r1),
        ("draft_step_2", r2),
    ]


def test_v1_subclass_inherits_default():
    # EAGLEWorker is a TpModelWorker subclass and inherits the default unchanged
    # (StandaloneWorker/FrozenKVMTPWorker/MultiLayerEagleWorker likewise).
    runner = object()
    w = _new(
        EAGLEWorker, is_draft_worker=True, model_runner_list=[], _model_runner=runner
    )
    assert w.iter_draft_runners() == [("draft", runner)]


# --- v2 family (override; the runner lives on the inner draft worker) ---


def test_eagle_v2_uses_inner_draft_runner():
    # EAGLEWorkerV2.draft_worker is a property -> inner EagleDraftWorker.draft_runner.
    runner = object()
    w = _new(EAGLEWorkerV2, _draft_worker=SimpleNamespace(draft_runner=runner))
    assert w.iter_draft_runners() == [("draft", runner)]


def test_multi_layer_v2_lists_every_step():
    r0, r1 = object(), object()
    w = _new(
        MultiLayerEagleWorkerV2,
        _draft_worker=SimpleNamespace(draft_runner_list=[r0, r1]),
    )
    assert w.iter_draft_runners() == [("draft_step_0", r0), ("draft_step_1", r1)]


# --- bare workers (no shared spec base) ---


def test_dflash_returns_independent_draft():
    # model_runner aliases the target, but draft_model_runner is independent.
    runner = object()
    w = _new(DFlashWorker, draft_model_runner=runner)
    assert w.iter_draft_runners() == [("draft", runner)]


def test_ngram_has_no_independent_draft():
    # NGRAM's model_runner IS the target's; no independent draft to check.
    w = _new(NGRAMWorker)
    assert w.iter_draft_runners() == []


def test_worker_without_method_fails_loudly():
    # A worker that neither inherits the TpModelWorker default nor defines the
    # method raises AttributeError at call time — the polymorphic analog of the
    # old helper's ValueError on an unrecognized worker.
    with pytest.raises(AttributeError):
        SimpleNamespace().iter_draft_runners()
