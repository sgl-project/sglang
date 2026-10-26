"""Unit tests for the NPU draft-step geometry (KV window / query split).

Deliberately small: these are the two formulas that were hard-coded to the
topk == 1 shape, and topk == 1 must keep reproducing the old values.

Pure ints, no torch/npu import: runs on any host.
"""

import importlib.util
import os
import unittest
from pathlib import Path

try:  # normal path (Linux CI, sglang installed / PYTHONPATH=python)
    from sglang.srt.hardware_backend.npu.attention.spec_step_geometry import (
        draft_step_kv_offset,
        draft_step_query_lens,
    )
except Exception:  # Windows dev boxes cannot import sglang (stdlib resource)
    _HERE = Path(__file__).resolve()
    _MODULE = None
    for _parent in _HERE.parents:
        _candidate = (
            _parent
            / "python"
            / "sglang"
            / "srt"
            / "hardware_backend"
            / "npu"
            / "attention"
            / "spec_step_geometry.py"
        )
        if os.path.isfile(_candidate):
            _MODULE = _candidate
            break
    if _MODULE is None:
        raise
    _spec = importlib.util.spec_from_file_location("spec_step_geometry", _MODULE)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    draft_step_kv_offset = _mod.draft_step_kv_offset
    draft_step_query_lens = _mod.draft_step_query_lens

try:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=5, suite="base-a-test-cpu")
except Exception:
    pass


class TestDraftStepGeometry(unittest.TestCase):
    def test_topk1_reproduces_the_chain_values(self):
        """Regression guard: the production chain path must not move."""
        self.assertEqual(
            [draft_step_kv_offset(s, 1) for s in range(4)], [1, 2, 3, 4]
        )
        self.assertEqual(draft_step_query_lens(3, 1), [1, 2, 3])

    def test_a_wider_tree_scales_both_quantities(self):
        # One step writes topk slots, so the window ends at (step + 1) * topk.
        self.assertEqual([draft_step_kv_offset(s, 2) for s in range(3)], [2, 4, 6])
        self.assertEqual(draft_step_query_lens(2, 5), [5, 10])

    def test_invalid_inputs_raise(self):
        for bad in ((-1, 2), (0, 0)):
            with self.assertRaises(ValueError):
                draft_step_kv_offset(*bad)
        for bad in ((-1, 2), (2, 0)):
            with self.assertRaises(ValueError):
                draft_step_query_lens(*bad)


if __name__ == "__main__":
    unittest.main()
