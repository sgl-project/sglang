"""Tests for the fail-closed draft-worker capture guard.

These tests construct minimal `nn.Module` stand-ins that look like draft
models (a couple of fake `TopK` instances) and drive
`check_draft_capture_optout` against each inventory category to prove the
guard's contract:

  - unknown architecture -> RuntimeError;
  - MoE entry with opted_out=False -> RuntimeError;
  - MoE entry with a TopK whose flag is True -> RuntimeError;
  - dense entry that grew a TopK -> RuntimeError;
  - properly opted-out MoE entry -> success;
  - dense entry with zero TopK -> success;
  - guard is a no-op when routed-experts capture is disabled.
"""

import unittest
from dataclasses import dataclass

import torch.nn as nn

from sglang.srt.layers.moe.topk import TopK
from sglang.srt.state_capturer.draft_guard import check_draft_capture_optout


@dataclass
class _FakeHfConfig:
    architectures: list


class _FakeDraftModel(nn.Module):
    """Wraps a list of `TopK` modules so `.modules()` yields them.

    The architecture identity is exposed via the wrapping class name
    (one fake subclass per inventory architecture we want to simulate).
    """

    def __init__(self, *topks: TopK) -> None:
        super().__init__()
        self._topks = nn.ModuleList(topks)


def _make_class(name: str) -> type:
    """Construct a one-off subclass of `_FakeDraftModel` whose `__name__`
    matches an inventory architecture, so `_resolved_architecture` picks
    it up via `type(model).__name__`."""
    return type(name, (_FakeDraftModel,), {})


def _make_topk(flag: bool) -> TopK:
    return TopK(top_k=4, allow_routed_experts_capture=flag)


def _make_hf_config(arch: str) -> _FakeHfConfig:
    return _FakeHfConfig(architectures=[arch])


class GuardSuccessTest(unittest.TestCase):
    def test_capture_disabled_is_noop(self):
        """When routed-experts capture is not enabled, the guard must
        return without inspecting the model. Even a polluting state
        (default-True TopK on a known dense-allowlist architecture) is
        allowed because capture cannot happen."""
        cls = _make_class("Eagle3DeepseekV2ForCausalLM")
        model = cls(_make_topk(True))  # would otherwise be a violation
        check_draft_capture_optout(
            model,
            _make_hf_config("Eagle3DeepseekV2ForCausalLM"),
            routed_experts_capture_enabled=False,
        )

    def test_moe_optout_satisfied(self):
        """A properly opted-out MoE draft must pass: every TopK has
        allow_routed_experts_capture=False."""
        cls = _make_class("DeepseekV3ForCausalLMNextN")
        model = cls(_make_topk(False), _make_topk(False))
        check_draft_capture_optout(
            model,
            _make_hf_config("DeepseekV3ForCausalLMNextN"),
            routed_experts_capture_enabled=True,
        )

    def test_dense_allowlist_with_zero_topk(self):
        cls = _make_class("LlamaForCausalLMEagle")
        model = cls()  # no TopK modules
        check_draft_capture_optout(
            model,
            _make_hf_config("LlamaForCausalLMEagle"),
            routed_experts_capture_enabled=True,
        )


class GuardFailureTest(unittest.TestCase):
    def test_unknown_architecture_fails(self):
        cls = _make_class("SomeArchitectureNotInInventory")
        model = cls()
        with self.assertRaisesRegex(RuntimeError, "not registered in draft_inventory"):
            check_draft_capture_optout(
                model,
                _make_hf_config("SomeArchitectureNotInInventory"),
                routed_experts_capture_enabled=True,
            )

    def test_moe_entry_pending_optout_fails(self):
        """An inventory entry that's MoE-bearing but `opted_out=False`
        (the plumbing isn't done yet) must fail closed even if the
        current `TopK` flags happen to look correct.

        After round 2 every MoE-bearing family in the inventory has
        `opted_out=True`, so to exercise this guard branch we inject a
        synthetic pending entry via mock instead of relying on a real
        pending family."""
        from unittest.mock import patch

        from sglang.srt.state_capturer.draft_inventory import DraftInventoryEntry

        pending_entry = DraftInventoryEntry(
            source_architectures=("FakeBaseForCausalLM",),
            draft_architecture="FakePendingForCausalLMMTP",
            moe_bearing=True,
            draft_signal="is_mtp",
            opt_out_injection_point="python/sglang/srt/models/fake_pending.py:FakePendingMoE",
            rationale="synthetic pending entry for guard test",
            opted_out=False,
        )
        cls = _make_class("FakePendingForCausalLMMTP")
        model = cls(_make_topk(False))
        with patch(
            "sglang.srt.state_capturer.draft_guard.lookup_draft_arch",
            return_value=pending_entry,
        ):
            with self.assertRaisesRegex(RuntimeError, "opted_out=False"):
                check_draft_capture_optout(
                    model,
                    _make_hf_config("FakePendingForCausalLMMTP"),
                    routed_experts_capture_enabled=True,
                )

    def test_moe_entry_with_true_flag_fails(self):
        """A properly registered, opted-out MoE family must still fail
        if any TopK on the draft model has the flag back to True (a
        regression introduced by a future model edit)."""
        cls = _make_class("DeepseekV3ForCausalLMNextN")
        model = cls(_make_topk(False), _make_topk(True))  # one offender
        with self.assertRaisesRegex(RuntimeError, "TopK module"):
            check_draft_capture_optout(
                model,
                _make_hf_config("DeepseekV3ForCausalLMNextN"),
                routed_experts_capture_enabled=True,
            )

    def test_dense_entry_with_topk_fails(self):
        """A dense allowlist entry must contain zero TopK modules; if
        someone adds an MoE block to a previously-dense draft wrapper,
        the guard catches it."""
        cls = _make_class("Eagle3DeepseekV2ForCausalLM")
        model = cls(_make_topk(False))  # any TopK violates the dense contract
        with self.assertRaisesRegex(RuntimeError, "allowlisted as dense"):
            check_draft_capture_optout(
                model,
                _make_hf_config("Eagle3DeepseekV2ForCausalLM"),
                routed_experts_capture_enabled=True,
            )


class GuardArchitectureResolutionTest(unittest.TestCase):
    def test_prefers_model_class_name_over_hf_config(self):
        """`type(self.model).__name__` is the authoritative architecture
        identity. If hf_config disagrees, the model class wins."""
        # Build a model whose class name is registered + opted out,
        # while passing an hf_config that names an unknown architecture.
        cls = _make_class("DeepseekV3ForCausalLMNextN")
        model = cls(_make_topk(False))
        check_draft_capture_optout(
            model,
            _make_hf_config("SomeOtherArchitecture"),
            routed_experts_capture_enabled=True,
        )


if __name__ == "__main__":
    unittest.main()
