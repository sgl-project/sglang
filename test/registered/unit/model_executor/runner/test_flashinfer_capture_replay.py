"""Decode graph capture is served entirely by the managed store.

Each shape is captured under strict replay, where a lookup without a stored winner
raises instead of baking FlashInfer's default into the graph. On such a miss the
shape's forward is tuned under the store and the capture runs once more; a miss
that survives the retry is logged as a warning and captured as it would be without
the check. The lifecycle cases use FlashInfer's real AutoTuner and store, with GPU
timing replaced by a table.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

import sys
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_lora,
    get_parallel,
    override_platform,
)
from sglang.test.test_utils import CustomTestCase

OP = "sglang_test::capture_replay"
OTHER_OP = "sglang_test::capture_replay_other"
# Tactic 1 wins; -1 is FlashInfer's default, which races in every tuning pass.
TIMES = {-1: 4.0, 0: 3.0, 1: 1.0, 2: 2.0}


def _dummy_runner():
    from flashinfer.autotuner import TunableRunner

    class DummyRunner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return [0, 1, 2]

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            return inputs[0]

    return DummyRunner()


def _model_runner(is_draft_worker=False):
    return SimpleNamespace(
        device="cuda",
        gpu_id=torch.cuda.current_device() if torch.cuda.is_available() else 0,
        forward_stream=torch.cuda.Stream() if torch.cuda.is_available() else None,
        is_draft_worker=is_draft_worker,
    )


def _decode_runner(**fields):
    runner = dict(
        backend=FullCudaGraphBackend.__new__(FullCudaGraphBackend),
        capture_forward_mode=ForwardMode.DECODE,
        enable_torch_compile=False,
        model_runner=_model_runner(),
    )
    runner.update(fields)
    return SimpleNamespace(**runner)


class _PublishedConfig(CustomTestCase):
    """A published single-rank configuration with cache reuse on."""

    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.enter(autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(True))

    def enter(self, context):
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)


@unittest.skipUnless(torch.cuda.is_available(), "FlashInfer requires CUDA")
class TestCaptureReplayLifecycle(_PublishedConfig):
    def setUp(self):
        super().setUp()
        from flashinfer import autotune_v2_reload
        from flashinfer.autotuner import AutoTuner, TuningConfig

        self.tuner = AutoTuner.get()
        self.config = TuningConfig()
        self.reload = autotune_v2_reload
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        for context in (
            get_parallel().override(
                tp_group=SimpleNamespace(world_size=1), tp_rank=0, pp_rank=0, dp_rank=0
            ),
            autotune.envs.SGLANG_CACHE_DIR.override(directory.name),
            patch.object(
                autotune,
                "_MANAGED_AUTOTUNE_CACHE_SM_MAJORS",
                (torch.cuda.get_device_capability()[0],),
            ),
            patch.object(autotune, "should_run_flashinfer_autotune", return_value=True),
            # Eligibility has its own cases below.
            patch.object(autotune, "uses_flashinfer_capture_replay", return_value=True),
            patch.object(AutoTuner, "_profile_single_kernel", self._profile),
        ):
            self.enter(context)
        self.profiled = []
        self.captures = 0
        self.hook = Mock()
        self.target = _model_runner()
        self.runner = _decode_runner(model_runner=self.target)
        self.tuner.clear_cache()
        self.addCleanup(self._detach)

    def _profile(self, runner, inputs, tactic, tuning_config, **kwargs):
        self.profiled.append(tactic)
        return float(TIMES[tactic])

    def _detach(self):
        self.tuner.clear_cache()
        self.tuner._managed_cache = None
        self.tuner._managed_stores.clear()

    def _choose(self, op):
        inputs = [torch.zeros(8, 16)]
        return self.tuner.choose_one(op, [_dummy_runner()], self.config, inputs)[1]

    def _startup(self, *ops):
        autotune.attach_flashinfer_managed_autotune_cache(self.target)
        with autotune.flashinfer_autotune_context(self.target, run_lm_head=False):
            for op in ops:
                self._choose(op)

    def _capture(self, *ops, fill_ops=None):
        """One shape; returns the tactics its capture served."""
        served = []

        def capture():
            self.captures += 1
            served[:] = [self._choose(op) for op in ops]

        def forward():
            for op in ops if fill_ops is None else fill_ops:
                self._choose(op)

        autotune.capture_with_flashinfer_replay(
            self.runner, capture, forward, self.hook
        )
        return served

    def _restart(self):
        self.tuner.clear_cache()
        self.reload()
        self.profiled.clear()
        self.captures = 0
        self.hook.reset_mock()

    def test_a_shape_the_store_covers_captures_once_without_tuning(self):
        self._startup(OP)
        self.profiled.clear()
        self.assertEqual(self._capture(OP), [1])
        self.assertEqual((self.captures, self.profiled), (1, []))
        self.hook.assert_not_called()

    def test_a_miss_is_tuned_and_the_shape_captured_again(self):
        self._startup(OP)
        self.profiled.clear()
        self.assertEqual(self._capture(OP, OTHER_OP), [1, 1])
        # Only the profile startup never looked up was measured.
        self.assertEqual(sorted(self.profiled), [-1, 0, 1, 2])
        self.assertEqual(self.captures, 2)
        self.hook.assert_called_once_with()

    def test_a_warm_restart_captures_once_without_tuning(self):
        self._startup(OP)
        self._capture(OP, OTHER_OP)
        self._restart()
        self._startup(OP)
        self.assertEqual(self._capture(OP, OTHER_OP), [1, 1])
        self.assertEqual((self.captures, self.profiled), (1, []))

    def test_a_miss_the_retry_cannot_fix_warns_and_captures_the_default(self):
        # Never worse than without the check: the unfixable lookup gets the default.
        self._startup(OP)
        self.profiled.clear()
        with self.assertLogs(autotune.logger, "WARNING") as logs:
            served = self._capture(OP, OTHER_OP, fill_ops=(OP,))
        self.assertEqual(served, [1, -1])
        self.assertIn("strict replay miss", logs.output[0])
        self.assertEqual((self.captures, self.profiled), (3, []))

    def test_a_skipped_op_is_neither_tuned_nor_a_miss(self):
        with get_exec().kernel.override(flashinfer_autotune_skip_ops=[OTHER_OP]):
            self._startup(OP)
            self.profiled.clear()
            self.assertEqual(self._capture(OP, OTHER_OP), [1, -1])
        self.assertEqual((self.captures, self.profiled), (1, []))

    def test_other_capture_errors_are_not_retried(self):
        def capture():
            self.captures += 1
            raise RuntimeError("capture failed")

        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            autotune.capture_with_flashinfer_replay(
                self.runner, capture, Mock(), self.hook
            )
        self.assertEqual(self.captures, 1)
        self.hook.assert_not_called()

    def test_the_hook_runs_after_a_failed_fill(self):
        self._startup(OP)

        def forward():
            raise ValueError("fill failed")

        def capture():
            self.captures += 1
            self._choose(OTHER_OP)

        with self.assertRaisesRegex(ValueError, "fill failed"):
            autotune.capture_with_flashinfer_replay(
                self.runner, capture, forward, self.hook
            )
        self.assertEqual(self.captures, 1)
        self.hook.assert_called_once_with()


class TestCaptureReplayEligibility(_PublishedConfig):
    """Only single-rank full-graph decode capture of a tuning target qualifies."""

    def setUp(self):
        super().setUp()
        self.enter(override_platform(has_flashinfer=True, device_capability=(10, 0)))
        self.enter(
            patch.object(autotune, "should_run_flashinfer_autotune", return_value=True)
        )

    def assert_eligible(self, runner, expected):
        self.assertEqual(autotune.uses_flashinfer_capture_replay(runner), expected)

    def test_single_rank_decode_capture_on_the_full_backend(self):
        self.assert_eligible(_decode_runner(), True)

    def test_runner_exclusions(self):
        for fields in (
            dict(backend=SimpleNamespace()),
            dict(capture_forward_mode=ForwardMode.TARGET_VERIFY),
            dict(enable_torch_compile=True),
            dict(model_runner=_model_runner(is_draft_worker=True)),
        ):
            with self.subTest(fields=fields):
                self.assert_eligible(_decode_runner(**fields), False)

    def test_a_draft_runner_is_excluded_before_its_fields_are_read(self):
        # The EAGLE draft runners subclass the decode runner but set only some of
        # its fields; the capture-time check must not reach a missing one.
        from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
            EAGLEDraftCudaGraphRunner,
        )

        runner = EAGLEDraftCudaGraphRunner.__new__(EAGLEDraftCudaGraphRunner)
        runner.model_runner = _model_runner(is_draft_worker=True)
        self.assert_eligible(runner, False)

    def test_config_exclusions(self):
        for context in (
            get_lora().override(enable_lora=True),
            get_exec().overlap.override(enable_two_batch_overlap=True),
        ):
            with self.subTest(context=context), context:
                self.assert_eligible(_decode_runner(), False)

    def test_foundry_graph_persistence(self):
        # Foundry captures or restores each shape itself.
        foundry = SimpleNamespace(enabled=True)
        with patch.object(autotune, "get_foundry_adapter", return_value=foundry):
            self.assert_eligible(_decode_runner(), False)

    def test_processes_without_the_managed_store(self):
        with patch.object(
            autotune, "uses_flashinfer_managed_autotune_cache", return_value=False
        ):
            self.assert_eligible(_decode_runner(), False)

    def test_multi_rank_processes_even_with_the_managed_store(self):
        # Strict replay is single-rank, whatever processes the store covers.
        with (
            patch.object(
                autotune, "uses_flashinfer_managed_autotune_cache", return_value=True
            ),
            get_parallel().override(max_world_size=2),
        ):
            self.assert_eligible(_decode_runner(), False)

    def test_runners_that_decline_tuning(self):
        with patch.object(
            autotune, "should_run_flashinfer_autotune", return_value=False
        ):
            self.assert_eligible(_decode_runner(), False)


class TestCaptureReplayContexts(_PublishedConfig):
    """Every context targets P1's store, with the skip ops."""

    def setUp(self):
        super().setUp()
        self.events = []
        self.misses = []  # consumed by successive strict captures

        @contextmanager
        def autotune_v2(**kw):
            self.events.append(kw["mode"])
            strict = kw.get("strict_replay", False)
            yield SimpleNamespace(
                misses=self.misses.pop(0) if strict and self.misses else 0
            )

        self.autotune_v2 = Mock(side_effect=autotune_v2)
        self.enter(
            patch.dict(
                sys.modules,
                {"flashinfer": SimpleNamespace(autotune_v2=self.autotune_v2)},
            )
        )
        self.forward = Mock(side_effect=lambda: self.events.append("forward"))
        self.hook = Mock(side_effect=lambda: self.events.append("hook"))

    def capture(self, eligible, misses=0):
        """``misses``: how many strict captures in a row miss."""
        attempts = []

        def capture():
            attempts.append(1)
            self.events.append("capture")
            if len(attempts) <= misses:
                raise RuntimeError("autotune strict replay miss for op")

        self.misses = [1] * misses
        with patch.object(
            autotune, "uses_flashinfer_capture_replay", return_value=eligible
        ):
            autotune.capture_with_flashinfer_replay(
                _decode_runner(), capture, self.forward, self.hook
            )

    def store(self):
        return dict(
            cache_root=autotune.flashinfer_managed_autotune_cache_root(),
            cache_scope="context",
            skip_ops=autotune.get_flashinfer_autotune_skip_ops(),
        )

    def test_a_covered_shape_captures_once_under_strict_replay(self):
        with get_exec().kernel.override(flashinfer_autotune_skip_ops=[OP]):
            self.capture(eligible=True)
            store = self.store()
        self.assertEqual(self.events, ["replay", "capture"])
        self.assertEqual(
            self.autotune_v2.call_args_list,
            [call(mode="replay", strict_replay=True, **store)],
        )
        self.assertIn(OP, store["skip_ops"])

    def test_a_miss_tunes_then_captures_again(self):
        self.capture(eligible=True, misses=1)
        store = self.store()
        self.assertEqual(
            self.events,
            ["replay", "capture", "tune", "forward", "hook", "replay", "capture"],
        )
        self.assertEqual(
            self.autotune_v2.call_args_list,
            [
                call(mode="replay", strict_replay=True, **store),
                call(mode="tune", **store),
                call(mode="replay", strict_replay=True, **store),
            ],
        )

    def test_a_second_miss_captures_outside_any_context(self):
        with self.assertLogs(autotune.logger, "WARNING"):
            self.capture(eligible=True, misses=2)
        self.assertEqual(
            self.events,
            ["replay", "capture", "tune", "forward", "hook", "replay", "capture"]
            + ["capture"],
        )

    def test_ineligible_capture_enters_no_context(self):
        self.capture(eligible=False)
        self.assertEqual(self.events, ["capture"])
        self.autotune_v2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
