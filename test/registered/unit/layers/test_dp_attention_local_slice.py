"""Unit tests for the issue #30712 fix.

All tests run on CPU -- no GPU required.

Issue #30712: on the prefill / piecewise CUDA-graph path with DP attention and
routed-expert return enabled, ``ModelRunner`` fed the DP-attention state
capturers ``getattr(self.decode_cuda_graph_runner, "bs", None)`` regardless of
which runner actually ran. When the decode runner had not run (or was absent),
that resolved to ``None`` and ``get_dp_local_slice_cpu`` computed
``dp_rank * None`` -> ``TypeError`` for ``dp_rank > 0``.

Two layers are pinned here:

1. ``TestGetDpLocalSliceCpu`` -- the rank-padded slicing invariant the capturer
   relies on. When a forward pass runs through a CUDA graph each DP rank's data
   lives in a rank-padded buffer whose per-rank stride is the graph's padded
   extent, so rank ``r`` starts at ``r * padded_extent`` -- NOT the eager
   prefix-sum offset. (This function was already correct; these cases guard
   against a "None-guard" fallback that would silently read the wrong slice.)

2. ``TestModelRunnerOutputCudaGraphExtent`` -- the actual fix wiring. The bug
   was NOT in ``get_dp_local_slice_cpu`` (unchanged by the fix) but in the
   ``ModelRunner`` call sites. The fix carries the real padded extent on
   ``ModelRunnerOutput.cuda_graph_padded_extent`` and reads it at the capturer
   call sites. These tests FAIL on the stock (unpatched) tree, where the field
   does not exist.
"""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.dp_attention import get_dp_local_slice_cpu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GLOBAL_NUM_TOKENS = [11, 13, 17, 19]


class TestGetDpLocalSliceCpu(CustomTestCase):
    def _slice(self, dp_rank, can_run_graph, cuda_graph_batch):
        forward_batch = SimpleNamespace(global_num_tokens_cpu=GLOBAL_NUM_TOKENS)
        with patch(
            "sglang.srt.layers.dp_attention.get_attention_dp_rank",
            return_value=dp_rank,
        ):
            return get_dp_local_slice_cpu(
                forward_batch, can_run_graph, cuda_graph_batch
            )

    def test_cuda_graph_uses_rank_padded_stride(self):
        # Graph path: start = dp_rank * padded_extent; length = local tokens.
        self.assertEqual(self._slice(2, True, 32), (64, 17))
        self.assertEqual(self._slice(0, True, 32), (0, 11))

    def test_eager_uses_prefix_sum(self):
        # Eager path ignores the graph extent and packs ranks contiguously.
        self.assertEqual(self._slice(2, False, None), (24, 17))

    def test_graph_and_eager_layouts_diverge_for_nonzero_rank(self):
        # The two layouts must not coincide for dp_rank > 0 -- this is why the
        # capturer must receive the actual padded extent (issue #30712), not a
        # prefix-sum fallback that happens to avoid the crash.
        graph_start, _ = self._slice(3, True, 32)
        eager_start, _ = self._slice(3, False, None)
        self.assertNotEqual(graph_start, eager_start)

    def test_none_extent_on_graph_path_crashes_for_nonzero_rank(self):
        # The exact #30712 symptom: had the capturer received None on the graph
        # path (as the stock getattr(self.decode_cuda_graph_runner, "bs", None)
        # did), the slice math raises for dp_rank > 0. The fix removes the None
        # at the source so this branch is never taken with a real graph run.
        with self.assertRaises(TypeError):
            self._slice(2, True, None)


class TestModelRunnerOutputCudaGraphExtent(CustomTestCase):
    """Regression tests for the fix wiring itself (fail on the stock tree)."""

    def test_output_exposes_cuda_graph_padded_extent(self):
        from sglang.srt.model_executor.model_runner import ModelRunnerOutput

        field_names = {f.name for f in dataclasses.fields(ModelRunnerOutput)}
        # Absent on stock -> the capturer could only fall back to the decode
        # runner's bs (or None). Present on the fix.
        self.assertIn("cuda_graph_padded_extent", field_names)

    def test_output_defaults_extent_to_none_for_eager(self):
        from sglang.srt.model_executor.model_runner import ModelRunnerOutput

        # Eager path leaves the extent None (prefix-sum layout is used).
        out = ModelRunnerOutput(logits_output=None, can_run_graph=False)
        self.assertIsNone(out.cuda_graph_padded_extent)

    def test_graph_output_extent_feeds_a_valid_slice(self):
        from sglang.srt.model_executor.model_runner import ModelRunnerOutput

        # End-to-end contract: the value the capturer receives on the graph
        # path is ModelRunnerOutput.cuda_graph_padded_extent -- a real int, so
        # get_dp_local_slice_cpu returns the rank-padded slice instead of doing
        # dp_rank * None.
        out = ModelRunnerOutput(
            logits_output=None,
            can_run_graph=True,
            cuda_graph_padded_extent=32,
        )
        self.assertIsNotNone(out.cuda_graph_padded_extent)
        forward_batch = SimpleNamespace(global_num_tokens_cpu=GLOBAL_NUM_TOKENS)
        with patch(
            "sglang.srt.layers.dp_attention.get_attention_dp_rank",
            return_value=2,
        ):
            start, length = get_dp_local_slice_cpu(
                forward_batch, out.can_run_graph, out.cuda_graph_padded_extent
            )
        self.assertEqual((start, length), (64, 17))


if __name__ == "__main__":
    unittest.main()
