"""Draft forwards must not occupy the target expert-distribution window (#48603)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import ast
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.eplb.expert_distribution import (
    _ExpertDistributionRecorderNoop,
    _ExpertDistributionRecorderReal,
    _SelectExpertsSinglePassGatherer,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.utils import Withable
from sglang.test.test_utils import CustomTestCase


def _batch(mode: ForwardMode = ForwardMode.DECODE):
    return SimpleNamespace(
        forward_mode=mode,
        extend_num_tokens=0,
        is_extend_in_batch=False,
    )


def _model_runner_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/model_executor/model_runner.py"
    )


def _model_runner_forward_node(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    return item
    raise AssertionError("ModelRunner.forward not found")


class TestDisabledForwardPass(CustomTestCase):
    def _make_recorder(self, *, gatherer=None):
        recorder = _ExpertDistributionRecorderReal.__new__(
            _ExpertDistributionRecorderReal
        )
        if gatherer is None:
            gatherer = Mock()
            gatherer.collect.return_value = {"global_physical_count": torch.zeros(1)}
        accumulator = Mock()
        accumulator.get_single_pass_gatherer_key.return_value = "primary"
        recorder._recording = True
        recorder._disable_all = False
        recorder._record_current_pass = False
        recorder._single_pass_gatherers = {"primary": gatherer}
        recorder._accumulator = accumulator
        recorder._current_debug_name = Withable()
        recorder._current_forward_pass_id = Withable()
        recorder._current_layer_idx = SimpleNamespace(value=0)
        return recorder

    def _run_pass(self, recorder, pass_id, *, is_draft, batch=None):
        batch = _batch() if batch is None else batch
        record_ctx = (
            recorder.disable_this_region() if is_draft else contextlib.nullcontext()
        )
        with record_ctx, recorder.with_forward_pass(pass_id, batch):
            pass

    def test_draft_passes_do_not_append_target_records(self):
        """A draft forward that also uses TARGET_VERIFY must not consume a window slot."""
        recorder = self._make_recorder()
        for i in range(6):
            self._run_pass(
                recorder,
                i,
                is_draft=(i % 2 == 1),
                batch=_batch(
                    ForwardMode.TARGET_VERIFY if i % 2 == 1 else ForwardMode.DECODE
                ),
            )
        self.assertEqual(recorder._accumulator.append.call_count, 3)
        appended_ids = [
            call.args[0] for call in recorder._accumulator.append.call_args_list
        ]
        self.assertEqual(appended_ids, [0, 2, 4])
        recorder._single_pass_gatherers["primary"].reset.assert_called()
        self.assertEqual(recorder._single_pass_gatherers["primary"].reset.call_count, 3)

    def test_target_without_routes_still_appends(self):
        """A quiet target pass is a valid zero observation, not a skipped draft."""
        recorder = self._make_recorder()
        self._run_pass(recorder, 7, is_draft=False)
        recorder._accumulator.append.assert_called_once()
        self.assertEqual(recorder._accumulator.append.call_args.args[0], 7)

    def test_nested_disabled_forward_preserves_outer_pass(self):
        recorder = self._make_recorder()
        gatherer = recorder._single_pass_gatherers["primary"]
        with recorder.with_forward_pass(10, _batch()):
            self.assertEqual(recorder._current_forward_pass_id.value, 10)
            self.assertTrue(recorder._record_current_pass)
            gatherer.reset.reset_mock()
            gatherer.collect.reset_mock()
            with recorder.disable_this_region():
                with recorder.with_forward_pass(99, _batch()):
                    self.assertEqual(recorder._current_forward_pass_id.value, 10)
                    self.assertTrue(recorder._record_current_pass)
                    self.assertTrue(recorder._disable_all)
            gatherer.reset.assert_not_called()
            gatherer.collect.assert_not_called()
            self.assertEqual(recorder._current_forward_pass_id.value, 10)
            self.assertTrue(recorder._record_current_pass)
        recorder._accumulator.append.assert_called_once()
        self.assertEqual(recorder._accumulator.append.call_args.args[0], 10)

    def test_disable_restored_after_exception(self):
        recorder = self._make_recorder()
        with self.assertRaises(RuntimeError):
            with recorder.disable_this_region():
                with recorder.with_forward_pass(2, _batch()):
                    raise RuntimeError("draft failed")
        self.assertFalse(recorder._disable_all)
        self._run_pass(recorder, 3, is_draft=False)
        recorder._accumulator.append.assert_called_once()

    def test_disabled_region_hides_count_buffer_left_by_target(self):
        """Fused kernels must not write the target buffer during draft work."""
        gatherer = _SelectExpertsSinglePassGatherer.__new__(
            _SelectExpertsSinglePassGatherer
        )
        gatherer._data = torch.zeros((2, 4), dtype=torch.int32)
        recorder = self._make_recorder(gatherer=gatherer)
        recorder._record_current_pass = True

        enabled = recorder.get_current_pass_count_buffer()
        self.assertIsNotNone(enabled)
        self.assertEqual(tuple(enabled.shape), (4,))
        with recorder.disable_this_region():
            self.assertIsNone(recorder.get_current_pass_count_buffer())
            self.assertTrue(recorder._record_current_pass)
        self.assertIsNotNone(recorder.get_current_pass_count_buffer())

    def test_noop_recorder_still_runs_forward_body(self):
        recorder = _ExpertDistributionRecorderNoop()
        ran = []
        with recorder.disable_this_region():
            with recorder.with_forward_pass(1, _batch()):
                ran.append(1)
        self.assertEqual(ran, [1])

    def test_model_runner_disables_draft_before_entering_forward_pass(self):
        source = _model_runner_path().read_text()
        forward = _model_runner_forward_node(source)
        recording_with = None
        for stmt in forward.body:
            if not isinstance(stmt, ast.With):
                continue
            names = [ast.unparse(item.context_expr) for item in stmt.items]
            if any("with_forward_pass" in name for name in names):
                recording_with = names
                break
        self.assertIsNotNone(recording_with)
        self.assertIn("record_ctx", recording_with)
        self.assertLess(
            recording_with.index("record_ctx"),
            next(
                i
                for i, name in enumerate(recording_with)
                if "with_forward_pass" in name
            ),
        )
        forward_src = ast.get_source_segment(source, forward)
        self.assertIsNotNone(forward_src)
        self.assertIn("disable_this_region()", forward_src)
        self.assertIn("if self.is_draft_worker", forward_src)


if __name__ == "__main__":
    unittest.main()
