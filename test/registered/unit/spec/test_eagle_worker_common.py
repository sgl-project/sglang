"""Regression test for the idle-rank verify capture mode.

``can_run_graph`` derives capture mode from ``requested = max(fb, spec)`` where
``fb = forward_batch.capture_hidden_mode`` and ``spec = spec_info.capture_hidden_mode``.

Invariant: an idle rank's ``spec`` must not raise ``requested`` above ``fb``.

Why NULL: ``NULL`` is the enum floor, so ``max(fb, NULL) = fb``. Idle and active
share the same ``fb`` (set by ``eagle_prepare_for_verify``); the active path
leaves ``spec_info=None`` (treated as ``NULL``), so idle's explicit ``NULL``
collapses ``requested`` to ``fb``, keeping both ranks aligned.

The idle path returns before any tree-mask or kernel work, so this is CPU-only.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.eagle_worker_common import build_eagle_verify_input

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestIdleVerifyInputCaptureMode(CustomTestCase):
    def test_idle_verify_input_abstains_with_null(self):
        # Pre-fix idle hardcoded ``FULL``, which lifted standalone's ``requested``
        # above active and desynced DP graph-vs-eager.
        idle_verify_input = build_eagle_verify_input(
            SimpleNamespace(forward_mode=ForwardMode.IDLE),
            # Draft input and tree tensors are unused on the idle path.
            None,
            None,
            None,
            None,
            None,
            target_worker=None,  # idle path never reads it
            topk=1,
            num_steps=3,
            num_draft_tokens=4,
            tree_mask_mode=None,
            device="cpu",
        )
        self.assertEqual(idle_verify_input.capture_hidden_mode, CaptureHiddenMode.NULL)


if __name__ == "__main__":
    unittest.main()
