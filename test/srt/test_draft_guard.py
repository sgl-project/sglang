"""Tests for the draft-worker capture opt-out pass and fail-closed guard.

The opt-out pass (`disable_routed_experts_capture_for_draft`) walks the
constructed draft model and flips every `TopK` to
`allow_routed_experts_capture=False`; the guard
(`check_draft_capture_optout`) re-walks and asserts that invariant. Both
are architecture-agnostic. These tests build minimal `nn.Module`
stand-ins holding fake `TopK` instances and exercise:

  - the opt-out pass flips every TopK to False (MoE), no-ops on a dense
    model, is idempotent, and leaves non-TopK siblings untouched;
  - guard is a no-op when routed-experts capture is disabled;
  - a properly opted-out MoE draft (all TopK flags False) -> success;
  - a dense draft (zero TopK) -> success;
  - any TopK with the flag left True -> RuntimeError.

A source-level tripwire also pins the order of the two passes inside
`ModelRunner.initialize()`: the opt-out must run after `_prepare_moe_topk`
and before the guard, on every draft worker.
"""

import inspect
import unittest

import torch.nn as nn

from sglang.srt.layers.moe.topk import TopK
from sglang.srt.state_capturer.draft_guard import (
    check_draft_capture_optout,
    disable_routed_experts_capture_for_draft,
)


class _FakeDraftModel(nn.Module):
    """Wraps a list of `TopK` modules so `.modules()` yields them."""

    def __init__(self, *topks: TopK) -> None:
        super().__init__()
        self._topks = nn.ModuleList(topks)


def _make_topk(flag: bool) -> TopK:
    return TopK(top_k=4, allow_routed_experts_capture=flag)


class _ModelWithSibling(nn.Module):
    """Holds `TopK` modules plus a plain non-TopK sibling carrying its own
    `allow_routed_experts_capture` attribute the pass must not touch."""

    def __init__(self, *topks: TopK) -> None:
        super().__init__()
        self._topks = nn.ModuleList(topks)
        self.sibling = nn.Linear(2, 2)
        # A bystander attribute with the same name the pass flips on TopK.
        self.sibling.allow_routed_experts_capture = True


class DisableRoutedExpertsCaptureForDraftTest(unittest.TestCase):
    def _flags(self, model: nn.Module):
        return [
            m.topk_config.allow_routed_experts_capture
            for m in model.modules()
            if isinstance(m, TopK)
        ]

    def test_all_default_true_topk_flipped_false(self):
        """Several default-True TopK -> every one is False after the pass."""
        model = _FakeDraftModel(_make_topk(True), _make_topk(True), _make_topk(True))
        disable_routed_experts_capture_for_draft(model)
        self.assertEqual(self._flags(model), [False, False, False])

    def test_dense_model_zero_topk_is_noop(self):
        """A dense draft (zero TopK) -> no-op, no exception."""
        model = _FakeDraftModel()
        disable_routed_experts_capture_for_draft(model)
        self.assertEqual(self._flags(model), [])

    def test_idempotent(self):
        """Calling twice keeps every TopK False with no error."""
        model = _FakeDraftModel(_make_topk(True), _make_topk(False))
        disable_routed_experts_capture_for_draft(model)
        disable_routed_experts_capture_for_draft(model)
        self.assertEqual(self._flags(model), [False, False])

    def test_non_topk_sibling_untouched(self):
        """A non-TopK module carrying the same attribute name must keep its
        value; the pass only rewrites `TopK.topk_config`."""
        model = _ModelWithSibling(_make_topk(True))
        disable_routed_experts_capture_for_draft(model)
        self.assertEqual(self._flags(model), [False])
        self.assertTrue(model.sibling.allow_routed_experts_capture)


class InitializeOrderingTripwireTest(unittest.TestCase):
    """Source-level tripwire on `ModelRunner.initialize()`: the draft branch
    must import and call `disable_routed_experts_capture_for_draft` after
    `_prepare_moe_topk` and before `check_draft_capture_optout`. A full
    `initialize()` needs GPU + weights, so this asserts against source text
    instead of executing it."""

    def _initialize_source(self) -> str:
        from sglang.srt.model_executor import model_runner as mr

        return inspect.getsource(mr.ModelRunner.initialize)

    def _require_index(self, source: str, needle: str) -> int:
        idx = source.find(needle)
        self.assertNotEqual(
            idx,
            -1,
            f"ModelRunner.initialize() must contain {needle!r}; the draft "
            "opt-out wiring is missing or was renamed",
        )
        return idx

    def test_optout_runs_between_prepare_topk_and_guard(self):
        source = self._initialize_source()
        prepare_idx = self._require_index(source, "_prepare_moe_topk(")
        disable_idx = self._require_index(
            source, "disable_routed_experts_capture_for_draft(self.model)"
        )
        guard_idx = self._require_index(source, "check_draft_capture_optout(")
        self.assertLess(
            prepare_idx,
            disable_idx,
            "disable_routed_experts_capture_for_draft must run AFTER "
            "_prepare_moe_topk so the draft TopK modules already exist",
        )
        self.assertLess(
            disable_idx,
            guard_idx,
            "disable_routed_experts_capture_for_draft must run BEFORE "
            "check_draft_capture_optout so the guard sees the opted-out flags",
        )

    def test_disable_helper_imported_in_draft_branch(self):
        source = self._initialize_source()
        self._require_index(source, "disable_routed_experts_capture_for_draft,")
        # The import must sit ahead of the call site.
        import_idx = self._require_index(
            source, "from sglang.srt.state_capturer.draft_guard import"
        )
        call_idx = self._require_index(
            source, "disable_routed_experts_capture_for_draft(self.model)"
        )
        self.assertLess(
            import_idx,
            call_idx,
            "the draft branch must import disable_routed_experts_capture_for_draft "
            "before calling it",
        )


class GuardSuccessTest(unittest.TestCase):
    def test_capture_disabled_is_noop(self):
        """When routed-experts capture is not enabled, the guard must
        return without inspecting the model. Even a polluting state
        (default-True TopK) is allowed because capture cannot happen."""
        model = _FakeDraftModel(_make_topk(True))  # would otherwise be a violation
        check_draft_capture_optout(model, routed_experts_capture_enabled=False)

    def test_moe_optout_satisfied(self):
        """A properly opted-out MoE draft must pass: every TopK has
        allow_routed_experts_capture=False."""
        model = _FakeDraftModel(_make_topk(False), _make_topk(False))
        check_draft_capture_optout(model, routed_experts_capture_enabled=True)

    def test_dense_draft_with_zero_topk(self):
        model = _FakeDraftModel()  # no TopK modules
        check_draft_capture_optout(model, routed_experts_capture_enabled=True)


class GuardFailureTest(unittest.TestCase):
    def test_any_topk_with_true_flag_fails(self):
        """If any TopK on the draft model has the flag left at True, the
        guard must fail closed rather than let it pollute the target's R3
        buffer."""
        model = _FakeDraftModel(_make_topk(False), _make_topk(True))  # one offender
        with self.assertRaisesRegex(RuntimeError, "TopK module"):
            check_draft_capture_optout(model, routed_experts_capture_enabled=True)

    def test_all_topk_true_reports_count(self):
        model = _FakeDraftModel(_make_topk(True), _make_topk(True))
        with self.assertRaisesRegex(RuntimeError, "2 MoE TopK module"):
            check_draft_capture_optout(model, routed_experts_capture_enabled=True)


if __name__ == "__main__":
    unittest.main()
