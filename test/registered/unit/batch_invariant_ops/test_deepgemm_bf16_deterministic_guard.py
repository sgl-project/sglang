"""Hermetic CPU unit tests for DeepGEMM BF16 deterministic algorithm toggling.

Upstream ``deep_gemm`` wheels (e.g. 2.8.x) default ``bf16_gemm_nn`` to cuBLASLt
(whose split-K reduction varies with batch size M) and expose
``use_deterministic_algorithms(bool)`` without ``get_deterministic_algorithms()``.
Previously ``_matmul_persistent_deepgemm`` treated a missing getter as already
deterministic (``else True``) and never called ``use_deterministic_algorithms(True)``.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepGemmBf16DeterministicGuard(CustomTestCase):
    def _inputs(self):
        a = torch.zeros((4, 32), dtype=torch.bfloat16)
        b = torch.zeros((32, 32), dtype=torch.bfloat16)
        return a, b

    def test_setter_without_getter_enables_deterministic_during_gemm(self):
        """Regression for upstream deep_gemm 2.8.x: setter exists, getter is absent."""
        events = []

        def fake_gemm(a, b, out):
            events.append("gemm")
            out.zero_()

        fake_dg = SimpleNamespace(
            use_deterministic_algorithms=lambda flag: events.append(("set", flag)),
            bf16_gemm_nn=fake_gemm,
        )
        a, b = self._inputs()

        # Outside batch-invariant mode: enables True before gemm, restores False after.
        with (
            mock.patch.object(batch_invariant_ops, "deep_gemm", fake_dg, create=True),
            mock.patch.object(batch_invariant_ops, "_batch_invariant_MODE", False),
        ):
            batch_invariant_ops._matmul_persistent_deepgemm(a, b, torch.bfloat16)
        self.assertEqual(events, [("set", True), "gemm", ("set", False)])

        # Inside batch-invariant mode: enables True and keeps it True.
        events.clear()
        with (
            mock.patch.object(batch_invariant_ops, "deep_gemm", fake_dg, create=True),
            mock.patch.object(batch_invariant_ops, "_batch_invariant_MODE", True),
        ):
            batch_invariant_ops._matmul_persistent_deepgemm(a, b, torch.bfloat16)
        self.assertEqual(events, [("set", True), "gemm"])

    def test_setter_and_getter_preserves_caller_state(self):
        """sgl-deep-gemm wheel: getter + setter both present."""
        state = {"det": False}
        events = []

        def set_det(flag: bool):
            state["det"] = flag
            events.append(("set", flag))

        def fake_gemm(a, b, out):
            events.append(("gemm", state["det"]))
            out.zero_()

        fake_dg = SimpleNamespace(
            get_deterministic_algorithms=lambda: state["det"],
            use_deterministic_algorithms=set_det,
            bf16_gemm_nn=fake_gemm,
        )
        a, b = self._inputs()

        with mock.patch.object(batch_invariant_ops, "deep_gemm", fake_dg, create=True):
            batch_invariant_ops._matmul_persistent_deepgemm(a, b, torch.bfloat16)
        self.assertEqual(events, [("set", True), ("gemm", True), ("set", False)])
        self.assertFalse(state["det"])

        # Already deterministic: no redundant toggle, stays True.
        state["det"] = True
        events.clear()
        with mock.patch.object(batch_invariant_ops, "deep_gemm", fake_dg, create=True):
            batch_invariant_ops._matmul_persistent_deepgemm(a, b, torch.bfloat16)
        self.assertEqual(events, [("gemm", True)])
        self.assertTrue(state["det"])

    def test_legacy_wheel_without_setter_runs_cleanly(self):
        """Older deep_gemm (<0.2) without use_deterministic_algorithms."""
        called = []
        fake_dg = SimpleNamespace(
            bf16_gemm_nn=lambda a, b, out: called.append(True) or out.zero_()
        )
        a, b = self._inputs()
        with mock.patch.object(batch_invariant_ops, "deep_gemm", fake_dg, create=True):
            batch_invariant_ops._matmul_persistent_deepgemm(a, b, torch.bfloat16)
        self.assertEqual(called, [True])

    def test_mode_lifecycle_syncs_deepgemm_deterministic_flag(self):
        """enable/disable/set_batch_invariant_mode synchronize deep_gemm state."""
        toggles = []
        fake_dg = SimpleNamespace(
            use_deterministic_algorithms=lambda flag: toggles.append(flag)
        )
        with (
            mock.patch.object(batch_invariant_ops, "ENABLE_JIT_DEEPGEMM", True),
            mock.patch.object(batch_invariant_ops, "deep_gemm", fake_dg, create=True),
            # CPU CI has no accelerator; register on CPU like test_batch_invariant_context.
            mock.patch.object(
                batch_invariant_ops, "get_dispatch_device_backend", lambda: "CPU"
            ),
            mock.patch.object(batch_invariant_ops, "_is_npu", False),
        ):
            batch_invariant_ops.disable_batch_invariant_mode()
            toggles.clear()
            with batch_invariant_ops.set_batch_invariant_mode(True):
                self.assertEqual(toggles, [True])
            self.assertEqual(toggles, [True, False])


if __name__ == "__main__":
    unittest.main()
