"""Tests for the per-config `allow_routed_experts_capture` opt-out.

Behavior tests patch the global capturer and check the gate decision in
`capture_routed_experts_if_allowed`. Structural tests assert at source level
that the CUDA/NPU sites and the BYPASS custom-op all route through the helper,
so a later refactor can't inline a capture call and skip the opt-out.

End-to-end correctness against a real MTP target lives in
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
    """Records each `capture(...)` invocation."""

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
        # Patch the getter, not the singleton, so the real capturer state is
        # left untouched.
        self._old = topk_module.get_global_experts_capturer
        topk_module.get_global_experts_capturer = lambda: self._new
        return self._new

    def __exit__(self, *exc):
        topk_module.get_global_experts_capturer = self._old


# -------------------------------------------------------------------------
# Behavior tests
# -------------------------------------------------------------------------


class CaptureRoutedExpertsIfAllowedBehaviorTest(unittest.TestCase):
    def _topk_ids(self) -> torch.Tensor:
        return torch.zeros(2, 4, dtype=torch.int32)

    def test_default_true_invokes_capture(self):
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
        cfg = TopKConfig(top_k=8, allow_routed_experts_capture=False)
        fake = _FakeCapturer()
        with _InstallGlobalCapturer(fake):
            capture_routed_experts_if_allowed(
                cfg, layer_id=0, topk_ids=self._topk_ids()
            )
        self.assertEqual(fake.calls, [])

    def test_no_global_capturer_is_safe(self):
        """With no capturer installed (capture disabled), the gate must not raise."""
        cfg = TopKConfig(top_k=8)
        old = topk_module.get_global_experts_capturer
        topk_module.get_global_experts_capturer = lambda: None
        try:
            capture_routed_experts_if_allowed(
                cfg, layer_id=0, topk_ids=self._topk_ids()
            )
        finally:
            topk_module.get_global_experts_capturer = old


class PostProcessTopkIdsGateTest(unittest.TestCase):
    """The CUDA capture site `_post_process_topk_ids` must consult the helper."""

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
                # The gate runs before any downstream CUDA branch that would
                # fail on a CPU-only host, so the gate check still holds.
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
    """The BYPASS piecewise-cuda-graph path must thread the flag end to end."""

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
