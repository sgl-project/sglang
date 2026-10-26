"""Unit tests for the NPU draft-step geometry (KV window / query split).

Two facts have to hold for every draft step, and both were hard-coded to the
``topk == 1`` shape before:

* by draft step ``i`` the request has written ``topk`` KV slots per step, so the
  live window is ``prefix + (i + 1) * topk`` (the self token plus every earlier
  level);
* the step runs ``topk`` queries per request, so the TND query split is
  ``[topk, 2*topk, ..., bs*topk]`` and not ``[1, 2, ..., bs]``.

``topk == 1`` must reproduce the old values exactly: that path is in production.

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
except Exception:  # Windows dev boxes cannot import sglang (stdlib `resource`)
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


class TestDraftStepKvOffset(unittest.TestCase):
    def test_topk1_keeps_the_production_values(self):
        """Regression guard: topk=1 must stay `step + 1`."""
        self.assertEqual(
            [draft_step_kv_offset(s, 1) for s in range(6)], [1, 2, 3, 4, 5, 6]
        )

    def test_tree_width_scales_the_window(self):
        # Step i writes `topk` slots, so level i ends at (i + 1) * topk.
        self.assertEqual(
            [draft_step_kv_offset(s, 2) for s in range(4)], [2, 4, 6, 8]
        )
        self.assertEqual(draft_step_kv_offset(0, 5), 5)
        self.assertEqual(draft_step_kv_offset(2, 5), 15)

    def test_zero_step_always_covers_the_whole_first_level(self):
        for topk in (1, 2, 4, 8):
            self.assertEqual(draft_step_kv_offset(0, topk), topk)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            draft_step_kv_offset(-1, 2)
        with self.assertRaises(ValueError):
            draft_step_kv_offset(0, 0)


class TestDraftStepQueryLens(unittest.TestCase):
    def test_topk1_keeps_the_production_values(self):
        """Regression guard: topk=1 must stay `[1, 2, ..., bs]`."""
        self.assertEqual(draft_step_query_lens(3, 1), [1, 2, 3])

    def test_each_request_contributes_topk_queries(self):
        self.assertEqual(draft_step_query_lens(3, 2), [2, 4, 6])
        self.assertEqual(draft_step_query_lens(2, 5), [5, 10])

    def test_length_is_request_count_and_total_is_rows(self):
        for bs, topk in ((1, 1), (1, 4), (7, 2), (64, 3)):
            lens = draft_step_query_lens(bs, topk)
            self.assertEqual(len(lens), bs, f"bs={bs} topk={topk}")
            self.assertEqual(lens[-1], bs * topk, f"bs={bs} topk={topk}")

    def test_empty_batch(self):
        self.assertEqual(draft_step_query_lens(0, 2), [])

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            draft_step_query_lens(-1, 2)
        with self.assertRaises(ValueError):
            draft_step_query_lens(2, 0)


class TestConsistency(unittest.TestCase):
    def test_window_covers_every_query_of_the_step(self):
        """The KV window of step i must reach the last query of that level."""
        prefix = 7
        for topk in (1, 2, 3, 4):
            for step in range(4):
                first_slot = prefix + step * topk
                last_query = first_slot + topk - 1
                # Inclusive causal length, i.e. positions [0, last_query].
                self.assertGreaterEqual(
                    prefix + draft_step_kv_offset(step, topk),
                    last_query + 1,
                    f"topk={topk} step={step}",
                )


if __name__ == "__main__":
    unittest.main()
