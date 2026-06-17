"""Tests for the fail-closed draft-worker capture guard.

The guard's contract is architecture-agnostic: it walks the constructed
draft model and asserts every `TopK` carries
`allow_routed_experts_capture=False`. These tests build minimal
`nn.Module` stand-ins holding fake `TopK` instances and exercise:

  - guard is a no-op when routed-experts capture is disabled;
  - a properly opted-out MoE draft (all TopK flags False) -> success;
  - a dense draft (zero TopK) -> success;
  - any TopK with the flag left True -> RuntimeError.
"""

import unittest

import torch.nn as nn

from sglang.srt.layers.moe.topk import TopK
from sglang.srt.state_capturer.draft_guard import check_draft_capture_optout


class _FakeDraftModel(nn.Module):
    """Wraps a list of `TopK` modules so `.modules()` yields them."""

    def __init__(self, *topks: TopK) -> None:
        super().__init__()
        self._topks = nn.ModuleList(topks)


def _make_topk(flag: bool) -> TopK:
    return TopK(top_k=4, allow_routed_experts_capture=flag)


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
