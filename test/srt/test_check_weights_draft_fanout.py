"""Unit tests for Scheduler.check_weights draft fan-out / merge + reset coverage.

Covers ``SchedulerUpdateWeightsMixin.check_weights`` selector routing
(``target`` / ``draft`` / ``all``), checksum key merging + collision detection,
role-labeled error wrapping, and the complete-coverage reset rule of
``WeightChecker._reset_tensors`` (selecting a runner resets everything it covers,
including storage shared with another runner).

A fake scheduler is built exactly like test_distributed_weight_update_spec_worker.py:
``class _TestScheduler(SchedulerUpdateWeightsMixin): pass`` instantiated directly,
with ``tp_worker`` / ``draft_worker`` set as attributes. Each runner's
``check_weights`` is mocked to return a checksum payload. Everything runs on CPU.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.managers.io_struct import CheckWeightsReqInput
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.utils.weight_checker import _RESET_SENTINEL, WeightChecker


class _TestScheduler(SchedulerUpdateWeightsMixin):
    pass


def _checksum_runner(checksums, tp_rank=0):
    """A model_runner mock whose check_weights returns a checksum payload."""
    runner = Mock()
    runner.check_weights.return_value = {
        "checksums": dict(checksums),
        "parallelism_info": {"tp_rank": tp_rank},
    }
    return runner


def _draft_worker(*pairs):
    # Fake draft worker exposing iter_draft_runners() for the fan-out tests;
    # discovery itself is covered by test_draft_runner_discovery.py.
    return SimpleNamespace(iter_draft_runners=lambda: list(pairs))


def _scheduler(tp_worker=None, draft_worker=None):
    scheduler = _TestScheduler()
    scheduler.tp_worker = tp_worker
    scheduler.draft_worker = draft_worker
    return scheduler


def _call(scheduler, action, selector=None):
    return SchedulerUpdateWeightsMixin.check_weights(
        scheduler, CheckWeightsReqInput(action=action, selector=selector)
    )


# ---------------------------------------------------------------------------
# selector=None / "target": byte-identical backward-compat fast path
# ---------------------------------------------------------------------------


def test_default_selector_returns_target_payload_verbatim():
    target_payload = {
        "checksums": {"w": "a"},
        "parallelism_info": {"tp_rank": 0},
    }
    target_runner = Mock()
    target_runner.check_weights.return_value = target_payload
    draft_worker = Mock()  # must NOT be consulted on the default path
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=draft_worker,
    )

    out = _call(scheduler, action="checksum", selector=None)

    assert out.success is True
    # Exact same object handed back: no "runners" key, keys unprefixed.
    assert out.payload is target_payload
    assert "runners" not in out.payload
    target_runner.check_weights.assert_called_once_with(action="checksum")
    draft_worker.assert_not_called()


# ---------------------------------------------------------------------------
# selector="all": merge target + draft checksums
# ---------------------------------------------------------------------------


def test_all_selector_merges_target_and_single_draft():
    target_runner = _checksum_runner({"w": "a"})
    draft_runner = _checksum_runner({"w": "b"})
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=_draft_worker(("draft", draft_runner)),
    )

    out = _call(scheduler, action="checksum", selector="all")

    assert out.success is True
    # Target keys stay unprefixed; draft keys get the "draft." role prefix.
    assert out.payload["checksums"] == {"w": "a", "draft.w": "b"}
    roles = [info["role"] for info in out.payload["runners"]]
    assert roles == ["target", "draft"]


def test_all_selector_multi_step_draft_prefixes_each_step():
    target_runner = _checksum_runner({"w": "a"})
    r0 = _checksum_runner({"w": "b0"})
    r1 = _checksum_runner({"w": "b1"})
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=_draft_worker(("draft_step_0", r0), ("draft_step_1", r1)),
    )

    out = _call(scheduler, action="checksum", selector="all")

    assert out.success is True
    assert out.payload["checksums"] == {
        "w": "a",
        "draft_step_0.w": "b0",
        "draft_step_1.w": "b1",
    }


def test_all_selector_checksum_key_collision_fails():
    # Target already owns an (unprefixed) "draft.w" key; the draft runner's "w"
    # prefixes to the same "draft.w" -> collision the outer handler reports.
    target_runner = _checksum_runner({"draft.w": "a"})
    draft_runner = _checksum_runner({"w": "b"})
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=_draft_worker(("draft", draft_runner)),
    )

    out = _call(scheduler, action="checksum", selector="all")

    assert out.success is False
    assert "collision" in out.message


# ---------------------------------------------------------------------------
# selector="draft": only draft weights, no target relabeling
# ---------------------------------------------------------------------------


def test_draft_selector_without_draft_worker_succeeds_empty():
    target_runner = _checksum_runner({"w": "a"})
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=None,
    )

    out = _call(scheduler, action="checksum", selector="draft")

    assert out.success is True
    assert "no separate draft weights" in out.message
    assert out.payload["checksums"] == {}


def test_draft_selector_ngram_returns_empty_without_target_relabel():
    # NGRAM's model_runner IS the target's. selector="draft" must succeed with
    # empty checksums and must NOT return/relabel the target's checksums.
    target_runner = _checksum_runner({"w": "a"})
    ngram = object.__new__(NGRAMWorker)
    ngram.model_runner = target_runner
    ngram.target_worker = SimpleNamespace(model_runner=target_runner)
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=ngram,
    )

    out = _call(scheduler, action="checksum", selector="draft")

    assert out.success is True
    assert out.payload["checksums"] == {}
    # The shared target runner was never asked for a checksum on the draft path.
    target_runner.check_weights.assert_not_called()


# ---------------------------------------------------------------------------
# Validation precedes mutation; role-labeled errors
# ---------------------------------------------------------------------------


def test_invalid_selector_fails_before_touching_any_runner():
    target_runner = _checksum_runner({"w": "a"})
    draft_runner = _checksum_runner({"w": "b"})
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=_draft_worker(("draft", draft_runner)),
    )

    out = _call(scheduler, action="reset_tensors", selector="bogus")

    assert out.success is False
    assert "invalid selector" in out.message
    # Validation runs first, so no runner is mutated.
    target_runner.check_weights.assert_not_called()
    draft_runner.check_weights.assert_not_called()


def test_empty_string_selector_is_rejected_not_coerced_to_target():
    # selector="" must be rejected (not coerced to "target" by `selector or
    # "target"`). It is distinct from None, which still defaults to "target".
    target_runner = _checksum_runner({"w": "a"})
    draft_runner = _checksum_runner({"w": "b"})
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=_draft_worker(("draft", draft_runner)),
    )

    out = _call(scheduler, action="reset_tensors", selector="")

    assert out.success is False
    assert "invalid selector" in out.message
    # Rejected before any mutation.
    target_runner.check_weights.assert_not_called()
    draft_runner.check_weights.assert_not_called()


def test_compare_error_message_carries_role_label():
    # Target's compare is a no-op (Mock default); only the draft raises. The
    # wrapped message must carry the failing runner's role label.
    target_runner = Mock()
    target_runner.check_weights.return_value = None
    draft_runner = Mock()
    draft_runner.check_weights.side_effect = AssertionError("no snapshot")
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=_draft_worker(("draft", draft_runner)),
    )

    out = _call(scheduler, action="compare", selector="all")

    assert out.success is False
    assert "[draft]" in out.message


# ---------------------------------------------------------------------------
# reset_tensors complete coverage: a runner's selection resets shared storage
# ---------------------------------------------------------------------------


class _NamedParamModel:
    """Model surface for _model_state: yields a caller-supplied set of named
    Parameters, so two models can share a Parameter object (shared storage) while
    each also holds private ones."""

    def __init__(self, named_params):
        self._named_params = list(named_params)

    def named_parameters(self):
        return list(self._named_params)

    def named_buffers(self):
        return []


class _RealCheckerRunner:
    """Minimal model_runner whose check_weights drives a real WeightChecker, so
    reset_tensors / snapshot / compare run their actual logic."""

    def __init__(self, named_params):
        self.model = _NamedParamModel(named_params)
        self._checker = WeightChecker(self)

    def check_weights(self, action):
        return self._checker.handle(action)


def test_draft_reset_covers_target_owned_shared_storage():
    # Complete coverage: under selector="draft" + reset_tensors the scheduler
    # resets everything the draft runner covers, INCLUDING embed/head shared with
    # the target via set_embed_and_head. The shared storage is set to the reset
    # sentinel (no protection / no skip), and draft-private weights too.
    shared = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    t_priv = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    d_priv = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

    target_runner = _RealCheckerRunner([("embed", shared), ("t_priv", t_priv)])
    draft_runner = _RealCheckerRunner([("embed", shared), ("d_priv", d_priv)])
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        # the draft worker yields [("draft", draft_runner)] via iter_draft_runners.
        draft_worker=_draft_worker(("draft", draft_runner)),
    )

    t_priv_before = t_priv.clone()

    out = _call(scheduler, action="reset_tensors", selector="draft")

    assert out.success is True
    # Shared storage (target-owned, tied to the draft) IS reset to the sentinel.
    assert torch.equal(shared, torch.full_like(shared, _RESET_SENTINEL))
    # The draft's private weight is reset to the sentinel too.
    assert torch.equal(d_priv, torch.full_like(d_priv, _RESET_SENTINEL))
    # The target-private weight, NOT covered by the draft runner, is untouched.
    assert torch.equal(t_priv, t_priv_before)


def test_all_selector_reset_shared_storage_independent_of_order():
    # selector="all" writes the shared storage from both target and draft runners.
    # Because the sentinel is idempotent, the final value is the sentinel
    # regardless of which runner wrote last.
    shared = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    t_priv = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    d_priv = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

    target_runner = _RealCheckerRunner([("embed", shared), ("t_priv", t_priv)])
    draft_runner = _RealCheckerRunner([("embed", shared), ("d_priv", d_priv)])
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=_draft_worker(("draft", draft_runner)),
    )

    out = _call(scheduler, action="reset_tensors", selector="all")

    assert out.success is True
    sentinel = torch.full_like(shared, _RESET_SENTINEL)
    assert torch.equal(shared, sentinel)
    assert torch.equal(t_priv, sentinel)
    assert torch.equal(d_priv, sentinel)


# ---------------------------------------------------------------------------
# Action validation precedes the empty-draft-runner fast path
# ---------------------------------------------------------------------------


def test_draft_selector_rejects_unsupported_action_when_no_draft_runner():
    # The empty-draft-runner fast path must NOT swallow an unsupported or deleted
    # action (e.g. the removed "mark_reset_storage") as a success. Action is
    # validated before the empty-runner return, so these fail and no runner runs.
    target_runner = Mock()
    target_runner.check_weights.side_effect = AssertionError("target must not be touched")
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=None,
    )

    for action in ("mark_reset_storage", "nonsense_action"):
        out = _call(scheduler, action=action, selector="draft")
        assert out.success is False, action
        assert "Unsupported" in out.message, action

    target_runner.check_weights.assert_not_called()


# ---------------------------------------------------------------------------
# selector="draft" compare fans out only to draft runners (target excluded)
# ---------------------------------------------------------------------------


def test_draft_compare_scope_excludes_target_and_labels_draft():
    # Scheduler fan-out scope: selector="draft" compare must touch ONLY the draft
    # runner(s); the target runner is never consulted, and a draft-side failure
    # carries the [draft] role label.
    target_runner = Mock()
    target_runner.check_weights.side_effect = AssertionError(
        "target must not be compared on a draft selection"
    )
    draft_runner = Mock()
    draft_runner.check_weights.side_effect = AssertionError("draft-private stale")
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=_draft_worker(("draft", draft_runner)),
    )

    out = _call(scheduler, action="compare", selector="draft")

    assert out.success is False
    assert "[draft]" in out.message
    draft_runner.check_weights.assert_called_once_with(action="compare")
    target_runner.check_weights.assert_not_called()


# ---------------------------------------------------------------------------
# explicit selector="target": target-only scope + verbatim payload shape
# ---------------------------------------------------------------------------


def test_target_selector_checksum_returns_target_payload_only():
    # Explicit selector="target" takes the byte-identical fast path: only the
    # target runner is consulted, the draft worker is never touched, and the
    # checksum payload is the target's verbatim (no "runners" key, keys unprefixed).
    target_payload = {"checksums": {"w": "a"}, "parallelism_info": {"tp_rank": 0}}
    target_runner = Mock()
    target_runner.check_weights.return_value = target_payload
    draft_runner = Mock()
    draft_runner.check_weights.side_effect = AssertionError(
        "draft must not be touched on a target selection"
    )
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=_draft_worker(("draft", draft_runner)),
    )

    out = _call(scheduler, action="checksum", selector="target")

    assert out.success is True
    assert out.payload is target_payload
    assert "runners" not in out.payload
    target_runner.check_weights.assert_called_once_with(action="checksum")
    draft_runner.check_weights.assert_not_called()


def test_target_selector_reset_and_compare_touch_target_only():
    # selector="target" routes reset_tensors / compare to the target runner only;
    # the draft runner is never consulted.
    for action in ("reset_tensors", "compare"):
        target_runner = Mock()
        target_runner.check_weights.return_value = None
        draft_runner = Mock()
        draft_runner.check_weights.side_effect = AssertionError(
            "draft must not be touched on a target selection"
        )
        scheduler = _scheduler(
            tp_worker=SimpleNamespace(model_runner=target_runner),
            draft_worker=_draft_worker(("draft", draft_runner)),
        )

        out = _call(scheduler, action=action, selector="target")

        assert out.success is True, action
        target_runner.check_weights.assert_called_once_with(action=action)
        draft_runner.check_weights.assert_not_called()
