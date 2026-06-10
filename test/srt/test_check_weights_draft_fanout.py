"""Unit tests for Scheduler.check_weights draft fan-out / merge + reset dedup.

Covers ``SchedulerUpdateWeightsMixin.check_weights`` selector routing
(``target`` / ``draft`` / ``all``), checksum key merging + collision detection,
role-labeled error wrapping, and the storage-dedup invariant of
``WeightChecker._reset_tensors`` (a storage shared across runners is randomized
once).

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
from sglang.srt.utils.weight_checker import WeightChecker


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
        draft_worker=SimpleNamespace(draft_model_runner=draft_runner),
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
        draft_worker=SimpleNamespace(model_runner_list=[r0, r1]),
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
        draft_worker=SimpleNamespace(draft_model_runner=draft_runner),
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
        draft_worker=SimpleNamespace(draft_model_runner=draft_runner),
    )

    out = _call(scheduler, action="reset_tensors", selector="bogus")

    assert out.success is False
    assert "invalid selector" in out.message
    # Validation runs first, so no runner is mutated.
    target_runner.check_weights.assert_not_called()
    draft_runner.check_weights.assert_not_called()


def test_empty_string_selector_is_rejected_not_coerced_to_target():
    # P1 regression: selector="" must be rejected (it used to be coerced to
    # "target" by `selector or "target"`). It is distinct from None, which still
    # defaults to "target".
    target_runner = _checksum_runner({"w": "a"})
    draft_runner = _checksum_runner({"w": "b"})
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=SimpleNamespace(draft_model_runner=draft_runner),
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
        draft_worker=SimpleNamespace(draft_model_runner=draft_runner),
    )

    out = _call(scheduler, action="compare", selector="all")

    assert out.success is False
    assert "[draft]" in out.message


# ---------------------------------------------------------------------------
# reset_tensors storage dedup (AC-4) — exercised at the WeightChecker level
# ---------------------------------------------------------------------------


class _SharedParamModel:
    """Exposes the named_parameters/named_buffers surface _model_state iterates,
    with a caller-supplied Parameter so two runners can share one storage."""

    def __init__(self, param):
        self._param = param

    def named_parameters(self):
        return [("w", self._param)]

    def named_buffers(self):
        return []


def test_reset_dedups_shared_storage_across_runners():
    # Two runners whose models expose the SAME Parameter object (shared storage,
    # as embed/head tied to the target would be).
    shared = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    checker_a = WeightChecker(SimpleNamespace(model=_SharedParamModel(shared)))
    checker_b = WeightChecker(SimpleNamespace(model=_SharedParamModel(shared)))

    visited = set()
    checker_a.handle("reset_tensors", visited_storage=visited)
    after_a = shared.clone()
    # A randomized it away from zeros.
    assert not torch.equal(after_a, torch.zeros(4))

    checker_b.handle("reset_tensors", visited_storage=visited)
    # B saw the storage already in `visited` and skipped it.
    assert torch.equal(shared, after_a)


def test_reset_without_shared_visited_set_randomizes_twice():
    # Contrast case proving the dedup matters: with independent visited sets,
    # B re-randomizes the same storage and the value changes again.
    shared = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    checker_a = WeightChecker(SimpleNamespace(model=_SharedParamModel(shared)))
    checker_b = WeightChecker(SimpleNamespace(model=_SharedParamModel(shared)))

    checker_a.handle("reset_tensors", visited_storage=set())
    after_a = shared.clone()
    checker_b.handle("reset_tensors", visited_storage=set())

    assert not torch.equal(shared, after_a)


# ---------------------------------------------------------------------------
# selector="draft" + reset_tensors: target-owned storage is protected (P0)
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
    reset_tensors / mark_reset_storage run their actual storage-dedup logic."""

    def __init__(self, named_params):
        self.model = _NamedParamModel(named_params)
        self._checker = WeightChecker(self)

    def check_weights(self, action, visited_storage=None):
        return self._checker.handle(action, visited_storage=visited_storage)


def test_draft_only_reset_protects_target_owned_shared_storage():
    # P0 regression: under selector="draft" + reset_tensors the scheduler must
    # pre-seed visited_storage with target-owned storage (via mark_reset_storage)
    # so embed/head shared with the draft (set_embed_and_head) is NOT scrambled;
    # only draft-PRIVATE weights get randomized.
    shared = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    t_priv = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    d_priv = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

    target_runner = _RealCheckerRunner(
        [("embed", shared), ("t_priv", t_priv)]
    )
    draft_runner = _RealCheckerRunner(
        [("embed", shared), ("d_priv", d_priv)]
    )
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        # iter_draft_model_runners yields [("draft", draft_runner)].
        draft_worker=SimpleNamespace(draft_model_runner=draft_runner),
    )

    shared_before = shared.clone()
    d_priv_before = d_priv.clone()

    out = _call(scheduler, action="reset_tensors", selector="draft")

    assert out.success is True
    # Shared/target-owned storage left untouched...
    assert torch.equal(shared, shared_before)
    # ...while the draft's private weight was randomized.
    assert not torch.equal(d_priv, d_priv_before)
