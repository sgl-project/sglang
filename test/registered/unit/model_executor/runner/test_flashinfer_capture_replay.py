"""Decode graph capture is served entirely by the managed store.

Each captured shape first tunes, under the store, whatever its forward looks up that
startup did not, then captures under strict replay, where a lookup without a stored
winner raises instead of baking FlashInfer's default into the graph. The lifecycle
cases use FlashInfer's real AutoTuner and store with GPU timing replaced by a table.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

import sys
import tempfile
import unittest
from contextlib import nullcontext
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
        enable_two_batch_overlap=False,
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

    def _capture(self, *ops):
        """One shape: the fill runs the forward, the body stands in for capture."""

        def forward():
            for op in ops:
                self._choose(op)

        with autotune.flashinfer_capture_replay(self.runner, forward, self.hook):
            return [self._choose(op) for op in ops]

    def _restart(self):
        self.tuner.clear_cache()
        self.reload()
        self.profiled.clear()

    def test_fill_tunes_what_startup_missed_and_capture_replays_it(self):
        self._startup(OP)
        self.profiled.clear()
        self.assertEqual(self._capture(OP, OTHER_OP), [1, 1])
        # Only the profile startup never looked up was measured.
        self.assertEqual(sorted(self.profiled), [-1, 0, 1, 2])
        self.hook.assert_called_once_with()

    def test_warm_restart_fills_nothing(self):
        self._startup(OP)
        self._capture(OP, OTHER_OP)
        self._restart()
        self._startup(OP)
        self.assertEqual(self._capture(OP, OTHER_OP), [1, 1])
        self.assertEqual(self.profiled, [])

    def test_a_capture_lookup_the_fill_did_not_make_raises(self):
        self._startup(OP)

        def forward():
            self._choose(OP)

        self.profiled.clear()
        with self.assertRaisesRegex(RuntimeError, "strict replay miss"):
            with autotune.flashinfer_capture_replay(self.runner, forward, self.hook):
                self._choose(OTHER_OP)
        self.assertEqual(self.profiled, [])

    def test_a_skipped_op_is_neither_tuned_nor_a_miss(self):
        with get_exec().kernel.override(flashinfer_autotune_skip_ops=[OTHER_OP]):
            self._startup(OP)
            self.profiled.clear()
            self.assertEqual(self._capture(OP, OTHER_OP), [1, -1])
        self.assertEqual(self.profiled, [])

    def test_the_hook_runs_after_a_failed_fill(self):
        self._startup(OP)

        def forward():
            raise ValueError("fill failed")

        body = Mock()
        with self.assertRaisesRegex(ValueError, "fill failed"):
            with autotune.flashinfer_capture_replay(self.runner, forward, self.hook):
                body()
        self.hook.assert_called_once_with()
        body.assert_not_called()


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
            dict(enable_two_batch_overlap=True),
            dict(model_runner=_model_runner(is_draft_worker=True)),
        ):
            with self.subTest(fields=fields):
                self.assert_eligible(_decode_runner(**fields), False)

    def test_lora(self):
        with get_lora().override(enable_lora=True):
            self.assert_eligible(_decode_runner(), False)

    def test_processes_without_the_managed_store(self):
        # Strict replay is single-rank; so is the store here.
        with get_parallel().override(max_world_size=2):
            self.assert_eligible(_decode_runner(), False)

    def test_runners_that_decline_tuning(self):
        with patch.object(
            autotune, "should_run_flashinfer_autotune", return_value=False
        ):
            self.assert_eligible(_decode_runner(), False)


class TestCaptureReplayContexts(_PublishedConfig):
    """The fill and the strict replay both target P1's store with the skip ops."""

    def setUp(self):
        super().setUp()
        self.events = []
        self.autotune_v2 = Mock(
            side_effect=lambda **kw: self.events.append(kw["mode"]) or nullcontext()
        )
        self.enter(
            patch.dict(
                sys.modules,
                {"flashinfer": SimpleNamespace(autotune_v2=self.autotune_v2)},
            )
        )
        self.forward = Mock(side_effect=lambda: self.events.append("forward"))
        self.hook = Mock(side_effect=lambda: self.events.append("hook"))

    def capture(self, eligible):
        with (
            patch.object(
                autotune, "uses_flashinfer_capture_replay", return_value=eligible
            ),
            autotune.flashinfer_capture_replay(
                _decode_runner(), self.forward, self.hook
            ),
        ):
            self.events.append("capture")

    def test_tune_then_strict_replay_on_the_managed_root(self):
        with get_exec().kernel.override(flashinfer_autotune_skip_ops=[OP]):
            self.capture(eligible=True)
            store = dict(
                cache_root=autotune.flashinfer_managed_autotune_cache_root(),
                cache_scope="context",
                skip_ops=autotune.get_flashinfer_autotune_skip_ops(),
            )
        self.assertEqual(self.events, ["tune", "forward", "hook", "replay", "capture"])
        self.assertEqual(
            self.autotune_v2.call_args_list,
            [
                call(mode="tune", **store),
                call(mode="replay", strict_replay=True, **store),
            ],
        )
        self.assertIn(OP, store["skip_ops"])

    def test_ineligible_capture_enters_no_context(self):
        self.capture(eligible=False)
        self.assertEqual(self.events, ["capture"])
        self.autotune_v2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
