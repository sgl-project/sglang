"""Tests for the per-config `allow_routed_experts_capture` opt-out.

Two layers of coverage:

1. **Behavior tests** (primary): patch the global capturer and exercise the
   gate decision through `capture_routed_experts_if_allowed` and the default-True
   target case. These prove the runtime contract.
2. **Structural tests** (supplementary tripwires): source-level assertions
   that the helper is wired into both CUDA and NPU sites and that the
   BYPASS custom-op signature carries the flag. These guard against a
   future refactor that inlines a capture call and bypasses the helper.

End-to-end runtime correctness against a real MTP target (Frozen-KV MTP
+ overlap + cuda-graph + BYPASS variant) is covered by
`test/registered/rl/test_return_routed_experts_mtp.py`.
"""

import inspect
import unittest
from typing import List

import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.fused_moe_triton import layer as fmt_layer
from sglang.srt.layers.moe.topk import (
    TopK,
    TopKConfig,
    _post_process_topk_ids,
    capture_routed_experts_if_allowed,
)


class _FakeCapturer:
    """Records each `capture(...)` invocation. Stateless beyond a call log."""

    def __init__(self) -> None:
        self.calls: List[dict] = []

    def capture(self, *, layer_id, topk_indices) -> None:
        self.calls.append({"layer_id": layer_id, "topk_indices": topk_indices})


class _InstallGlobalCapturer:
    """Context manager: swap the global capturer for a fake and restore."""

    def __init__(self, capturer) -> None:
        self._new = capturer
        self._old = None

    def __enter__(self):
        # Patch the module-level getter so callers see our fake without
        # touching the real `_global_expert_capturer` singleton state.
        self._old = topk_module.get_global_experts_capturer
        topk_module.get_global_experts_capturer = lambda: self._new
        return self._new

    def __exit__(self, *exc):
        topk_module.get_global_experts_capturer = self._old


# -------------------------------------------------------------------------
# Behavior tests
# -------------------------------------------------------------------------


class CaptureRoutedExpertsIfAllowedBehaviorTest(unittest.TestCase):
    """AC-1 (CUDA + NPU share this helper) + AC-7 (default True still captures)."""

    def _topk_ids(self) -> torch.Tensor:
        return torch.zeros(2, 4, dtype=torch.int32)

    def test_default_true_invokes_capture(self):
        """AC-7: default behavior must still record routed experts on the
        target path. Proves the dataclass default doesn't accidentally
        suppress capture."""
        cfg = TopKConfig(top_k=8)
        self.assertTrue(cfg.allow_routed_experts_capture)
        fake = _FakeCapturer()
        with _InstallGlobalCapturer(fake):
            capture_routed_experts_if_allowed(
                cfg, layer_id=3, topk_ids=self._topk_ids()
            )
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["layer_id"], 3)

    def test_explicit_false_skips_capture(self):
        """AC-1: draft-side MoE layers (`allow_routed_experts_capture=False`)
        must not call into the capturer."""
        cfg = TopKConfig(top_k=8, allow_routed_experts_capture=False)
        fake = _FakeCapturer()
        with _InstallGlobalCapturer(fake):
            capture_routed_experts_if_allowed(
                cfg, layer_id=0, topk_ids=self._topk_ids()
            )
        self.assertEqual(fake.calls, [])

    def test_no_global_capturer_is_safe(self):
        """When no capturer is installed (e.g. when
        `--enable-return-routed-experts` is off), the gate must short-
        circuit cleanly without raising."""
        cfg = TopKConfig(top_k=8)
        # No global capturer installed.
        old = topk_module.get_global_experts_capturer
        topk_module.get_global_experts_capturer = lambda: None
        try:
            capture_routed_experts_if_allowed(
                cfg, layer_id=0, topk_ids=self._topk_ids()
            )
        finally:
            topk_module.get_global_experts_capturer = old


class PostProcessTopkIdsGateTest(unittest.TestCase):
    """AC-1 (CUDA path): the production capture site
    `_post_process_topk_ids` must consult the helper."""

    def _drive(self, allow_flag: bool):
        topk_ids = torch.zeros(2, 4, dtype=torch.int32)
        topk_weights = torch.zeros(2, 4, dtype=torch.float32)
        router_logits = torch.zeros(2, 8, dtype=torch.float32)
        cfg = TopKConfig(top_k=4, allow_routed_experts_capture=allow_flag)
        fake = _FakeCapturer()
        with _InstallGlobalCapturer(fake):
            try:
                _post_process_topk_ids(
                    topk_ids,
                    topk_weights,
                    cfg,
                    router_logits,
                    layer_id=0,
                )
            except Exception:
                # The gate decision happens before any downstream CUDA
                # branch could fail on a CPU-only host; ignoring the
                # downstream failure does not invalidate the gate check.
                pass
        return fake

    def test_post_process_true_calls_capture(self):
        fake = self._drive(allow_flag=True)
        self.assertEqual(
            len(fake.calls),
            1,
            "with allow_routed_experts_capture=True, the CUDA path's gate must "
            "invoke the capturer exactly once at the post-process site",
        )

    def test_post_process_false_skips_capture(self):
        fake = self._drive(allow_flag=False)
        self.assertEqual(
            fake.calls,
            [],
            "with allow_routed_experts_capture=False, the CUDA path's gate must "
            "never invoke the capturer at the post-process site",
        )


# -------------------------------------------------------------------------
# Structural tests (supplementary regression tripwires)
# -------------------------------------------------------------------------


class TopKConfigStructuralTest(unittest.TestCase):
    """AC-1: dataclass + TopK construction propagation."""

    def test_dataclass_default_is_true(self):
        cfg = TopKConfig(top_k=8)
        self.assertTrue(cfg.allow_routed_experts_capture)

    def test_dataclass_accepts_explicit_false(self):
        cfg = TopKConfig(top_k=8, allow_routed_experts_capture=False)
        self.assertFalse(cfg.allow_routed_experts_capture)

    def test_topk_init_propagates_default(self):
        topk = TopK(top_k=8)
        self.assertTrue(topk.topk_config.allow_routed_experts_capture)

    def test_topk_init_propagates_false(self):
        topk = TopK(top_k=8, allow_routed_experts_capture=False)
        self.assertFalse(topk.topk_config.allow_routed_experts_capture)


class CaptureSiteRoutingStructuralTest(unittest.TestCase):
    """Tripwires that catch a future refactor inlining a capture call
    instead of routing through `capture_routed_experts_if_allowed`."""

    def test_cuda_post_process_calls_helper(self):
        source = inspect.getsource(_post_process_topk_ids)
        self.assertIn(
            "capture_routed_experts_if_allowed(",
            source,
            "the CUDA capture site must route through the shared helper",
        )

    def test_npu_fused_topk_calls_helper(self):
        # The NPU MoE topk file lives behind an `sgl_kernel_npu` import,
        # so we cannot import the module on a CPU host. Read its source
        # directly off disk and assert the helper is used.
        from pathlib import Path

        npu_topk_path = (
            Path(topk_module.__file__).resolve().parent.parent.parent
            / "hardware_backend"
            / "npu"
            / "moe"
            / "topk.py"
        )
        npu_source = npu_topk_path.read_text()
        self.assertIn(
            "capture_routed_experts_if_allowed(",
            npu_source,
            "the NPU capture site must route through the shared helper "
            "to inherit the per-TopKConfig opt-out",
        )
        self.assertNotIn(
            "cap := get_global_experts_capturer()",
            npu_source,
            "the NPU path must not retain an inline capture gate that "
            "could drift from the helper",
        )


class BypassPathStructuralTest(unittest.TestCase):
    """AC-2 structural tripwires."""

    def test_caller_decomposes_allow_flag(self):
        source = inspect.getsource(fmt_layer)
        self.assertIn("topk_output.topk_config.allow_routed_experts_capture", source)

    def test_custom_op_signature_declares_flag(self):
        source = inspect.getsource(fmt_layer)
        self.assertRegex(
            source,
            r"def fused_moe_bypassed_piecewise_cuda_graph_impl\([^)]*"
            r"allow_routed_experts_capture:\s*bool",
        )

    def test_rebuilt_topkconfig_carries_flag(self):
        source = inspect.getsource(fmt_layer)
        self.assertIn(
            "allow_routed_experts_capture=allow_routed_experts_capture", source
        )


if __name__ == "__main__":
    unittest.main()
