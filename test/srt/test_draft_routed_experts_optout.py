"""Tests for the draft-worker routed-experts (R3) capture opt-out.

R3 routed-experts capture is target-only. A speculative draft worker shares
the process with its target, so three things keep a draft from polluting the
target's capture:

  - `disable_routed_experts_capture_for_draft(model)` walks the constructed
    draft model and flips every `TopK` to `allow_routed_experts_capture=False`,
    so a draft `TopK` never writes the capture buffer;
  - `ModelRunner.init_routed_experts_capturer()` early-returns for a draft
    worker, so a draft never replaces the target's process-global capturer;
  - `ModelRunner.forward()` gates the routed-experts `on_forward_end()` on
    `not self.is_draft_worker`, so a draft forward never finalizes the
    target's capturer with the draft's batch.

The functional tests exercise the walker on minimal `nn.Module` stand-ins.
The source-level tripwires pin the two `ModelRunner` guards and the
`initialize()` ordering in place (a full run needs GPU + weights).
"""

import inspect
import unittest

import torch.nn as nn

from sglang.srt.layers.moe.topk import TopK
from sglang.srt.state_capturer.routed_experts import (
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
        self.sibling.allow_routed_experts_capture = True


class DisableRoutedExpertsCaptureForDraftTest(unittest.TestCase):
    def _flags(self, model: nn.Module):
        return [
            m.topk_config.allow_routed_experts_capture
            for m in model.modules()
            if isinstance(m, TopK)
        ]

    def test_all_default_true_topk_flipped_false(self):
        model = _FakeDraftModel(_make_topk(True), _make_topk(True), _make_topk(True))
        disable_routed_experts_capture_for_draft(model)
        self.assertEqual(self._flags(model), [False, False, False])

    def test_dense_model_zero_topk_is_noop(self):
        model = _FakeDraftModel()
        disable_routed_experts_capture_for_draft(model)
        self.assertEqual(self._flags(model), [])

    def test_idempotent(self):
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


class CapturerGuardTripwireTest(unittest.TestCase):
    """Source-level tripwire: `init_routed_experts_capturer()` must early-return
    for a draft worker before `set_global_experts_capturer(`. Asserted on source
    text because a real call needs GPU + weights.
    """

    def _capturer_source(self) -> str:
        from sglang.srt.model_executor import model_runner as mr

        return inspect.getsource(mr.ModelRunner.init_routed_experts_capturer)

    def test_draft_guard_returns_before_setting_global_capturer(self):
        source = self._capturer_source()

        guard_idx = source.find("if self.is_draft_worker:")
        self.assertNotEqual(
            guard_idx,
            -1,
            "init_routed_experts_capturer() must contain the draft-worker "
            "guard 'if self.is_draft_worker:'; the target-only opt-out was "
            "removed or renamed",
        )

        return_idx = source.find("return", guard_idx)
        self.assertNotEqual(
            return_idx,
            -1,
            "the draft-worker guard must early-return; no 'return' follows "
            "'if self.is_draft_worker:'",
        )

        set_idx = source.find("set_global_experts_capturer(")
        self.assertNotEqual(
            set_idx,
            -1,
            "init_routed_experts_capturer() must call "
            "set_global_experts_capturer(); the capture install site was "
            "removed or renamed",
        )

        self.assertLess(
            guard_idx,
            set_idx,
            "the draft-worker guard 'if self.is_draft_worker:' must appear "
            "BEFORE set_global_experts_capturer(); otherwise a draft worker "
            "would overwrite the target's process-global R3 capturer",
        )
        self.assertLess(
            return_idx,
            set_idx,
            "the draft-worker guard's 'return' must appear BEFORE "
            "set_global_experts_capturer(); otherwise a draft worker would "
            "fall through and replace the target's process-global R3 capturer",
        )


class ForwardFinalizeGateTripwireTest(unittest.TestCase):
    """Source-level tripwire: `forward()` must gate the routed-experts
    `on_forward_end(` finalization on `not self.is_draft_worker`, in the same
    `if` as `get_global_experts_capturer()`. Asserted on source text because a
    real forward needs GPU + weights. The sibling `indexer_capturer` is
    intentionally not gated this way and is out of scope here.
    """

    def _forward_source(self) -> str:
        from sglang.srt.model_executor import model_runner as mr

        return inspect.getsource(mr.ModelRunner.forward)

    def test_routed_experts_finalize_gated_on_not_draft_worker(self):
        source = self._forward_source()

        getter_idx = source.find("get_global_experts_capturer()")
        self.assertNotEqual(
            getter_idx,
            -1,
            "forward() must call get_global_experts_capturer() to finalize "
            "routed-experts capture; the finalization site was removed or "
            "renamed",
        )

        finalize_idx = source.find("experts_capturer.on_forward_end(")
        self.assertNotEqual(
            finalize_idx,
            -1,
            "forward() must finalize via experts_capturer.on_forward_end(); "
            "the routed-experts finalization call was removed or renamed",
        )

        # The draft guard must sit in the SAME `if` condition that guards the
        # get_global_experts_capturer() walrus, i.e. shortly before the getter
        # call. rfind locates the guard token immediately preceding the getter.
        guard_idx = source.rfind("not self.is_draft_worker", 0, getter_idx)
        self.assertNotEqual(
            guard_idx,
            -1,
            "the routed-experts finalization in forward() must be gated by "
            "'not self.is_draft_worker' in the same `if` as "
            "get_global_experts_capturer(); the draft guard is missing, so a "
            "draft forward would finalize the target's process-global R3 "
            "capturer",
        )

        # Offsets must be: draft-guard < getter < on_forward_end. This proves
        # the draft guard precedes (gates) the routed-experts finalization
        # rather than appearing somewhere unrelated below it.
        self.assertLess(
            guard_idx,
            getter_idx,
            "'not self.is_draft_worker' must appear BEFORE "
            "get_global_experts_capturer() in the gating `if`",
        )
        self.assertLess(
            getter_idx,
            finalize_idx,
            "get_global_experts_capturer() must appear BEFORE "
            "experts_capturer.on_forward_end(); the gate must short-circuit "
            "the finalization for a draft worker",
        )

        # Guard, getter, and finalization must form one contiguous gated block,
        # not a match from an earlier unrelated statement. The gate spans well
        # under 400 chars, so a far-away match means it isn't on this `if`.
        self.assertLess(
            finalize_idx - guard_idx,
            400,
            "the 'not self.is_draft_worker' guard must be on the SAME `if` "
            "that immediately guards get_global_experts_capturer() and "
            "on_forward_end(); a distant match means the routed-experts "
            "finalization is not actually gated on the draft-worker check",
        )


class InitializeOrderingTripwireTest(unittest.TestCase):
    """Source-level tripwire on `ModelRunner.initialize()`: the draft branch
    must import and call `disable_routed_experts_capture_for_draft` after
    `_prepare_moe_topk` and inside the `if self.is_draft_worker:` branch, and
    before any backend/graph/memory init. A full `initialize()` needs GPU +
    weights, so this asserts against source text instead of executing it."""

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

    def test_optout_runs_after_prepare_topk_in_draft_branch(self):
        source = self._initialize_source()
        prepare_idx = self._require_index(source, "_prepare_moe_topk(")
        disable_idx = self._require_index(
            source, "disable_routed_experts_capture_for_draft(self.model)"
        )
        self.assertLess(
            prepare_idx,
            disable_idx,
            "disable_routed_experts_capture_for_draft must run AFTER "
            "_prepare_moe_topk so the draft TopK modules already exist",
        )
        # The disable pass must be gated inside the draft-worker branch, not run
        # unconditionally on every worker: an `if self.is_draft_worker:` line
        # must sit between _prepare_moe_topk and the disable call.
        branch_idx = self._require_index(source, "if self.is_draft_worker:")
        self.assertLess(
            prepare_idx,
            branch_idx,
            "the `if self.is_draft_worker:` branch guarding the opt-out must "
            "appear AFTER _prepare_moe_topk",
        )
        self.assertLess(
            branch_idx,
            disable_idx,
            "disable_routed_experts_capture_for_draft must sit INSIDE the "
            "`if self.is_draft_worker:` branch, not run unconditionally; the "
            "branch guard must precede the disable call",
        )

    def test_no_backend_or_graph_init_before_disable(self):
        """The disable pass must run before any backend/graph/memory init, so no
        captured graph records the default-True capture decision before the draft
        TopK modules are opted out. Checks that no init-call token appears between
        `_prepare_moe_topk(` and the disable call.
        """
        source = self._initialize_source()
        prepare_idx = self._require_index(source, "_prepare_moe_topk(")
        disable_idx = self._require_index(
            source, "disable_routed_experts_capture_for_draft(self.model)"
        )
        between = source[prepare_idx + len("_prepare_moe_topk(") : disable_idx]

        # Candidate backend/graph/memory init-call tokens. Whichever of these
        # the codebase actually uses must NOT appear between prepare_topk and
        # the disable pass.
        init_call_tokens = [
            "init_attention_backend",
            "init_cuda_graph",
            "init_device_graph",
            "init_piecewise",
            "init_backends",
            "init_memory_pool",
            "init_routed_experts_capturer",
            "capture_cuda_graph",
        ]
        offenders = [tok for tok in init_call_tokens if tok in between]
        self.assertEqual(
            offenders,
            [],
            "no backend/graph/memory init call may appear between "
            "_prepare_moe_topk() and disable_routed_experts_capture_for_draft(); "
            f"found {offenders!r} in the between-region. The draft opt-out must "
            "run BEFORE any graph/backend/memory init records the (default-True) "
            "capture decision",
        )

    def test_disable_helper_imported_in_draft_branch(self):
        source = self._initialize_source()
        self._require_index(source, "disable_routed_experts_capture_for_draft,")
        import_idx = self._require_index(
            source, "from sglang.srt.state_capturer.routed_experts import"
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


if __name__ == "__main__":
    unittest.main()
