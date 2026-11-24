"""Single-rank startups keep tuned tactics in FlashInfer's managed store.

The lifecycle cases drive SGLang's autotune context against FlashInfer's real
AutoTuner and store. Only the GPU timing is replaced, by a table, so each case
knows which tactic must win and can tell a profiled startup from one served by
the store. A restart drops every in-memory winner and re-reads the store, which
is what a new process attaching the same root sees.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_parallel,
    override_platform,
)
from sglang.test.test_utils import CustomTestCase

OP = "sglang_test::managed_cache"
OTHER_OP = "sglang_test::managed_cache_other"
# Tactic 1 wins; -1 is FlashInfer's default, which races in every tuning pass.
TIMES = {-1: 4.0, 0: 3.0, 1: 1.0, 2: 2.0}
# A startup that profiled under these would pick 2, one served by the store 1.
SHIFTED_TIMES = {-1: 4.0, 0: 3.0, 1: 5.0, 2: 0.5}


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
        # Read by the JSON cache path only.
        dtype=torch.bfloat16,
        model_config=SimpleNamespace(hf_config=None, quantization=None),
    )


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
class TestManagedAutotuneCache(_PublishedConfig):
    def setUp(self):
        super().setUp()
        from flashinfer import autotune_v2_reload
        from flashinfer.autotuner import AutoTuner, TuningConfig

        self.tuner = AutoTuner.get()
        self.config = TuningConfig()
        self.reload = autotune_v2_reload
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache_dir = Path(directory.name)
        for context in (
            # One rank; the JSON cache path names its file after these.
            get_parallel().override(
                tp_group=SimpleNamespace(world_size=1), tp_rank=0, pp_rank=0, dp_rank=0
            ),
            autotune.envs.SGLANG_CACHE_DIR.override(str(self.cache_dir)),
            # The store is validated per architecture; state that this GPU is.
            patch.object(
                autotune,
                "_MANAGED_AUTOTUNE_CACHE_SM_MAJORS",
                (torch.cuda.get_device_capability()[0],),
            ),
            # These runners stand in for ones whose warmup tunes.
            patch.object(autotune, "should_run_flashinfer_autotune", return_value=True),
            patch.object(AutoTuner, "_profile_single_kernel", self._profile),
        ):
            self.enter(context)

        self.times = dict(TIMES)
        self.profiled = []
        self.target = _model_runner()
        self.draft = _model_runner(is_draft_worker=True)
        self.tuner.clear_cache()
        self.addCleanup(self._detach)

    def _profile(self, runner, inputs, tactic, tuning_config, **kwargs):
        self.profiled.append(tactic)
        return float(self.times[tactic])

    def _detach(self):
        # A process attaches one store for its lifetime; tests share a process.
        self.tuner.clear_cache()
        self.tuner._managed_cache = None
        self.tuner._managed_stores.clear()

    def _choose(self, op=OP):
        inputs = [torch.zeros(8, 16)]
        return self.tuner.choose_one(op, [_dummy_runner()], self.config, inputs)[1]

    def _tune(self, *ops, runner=None):
        with autotune.flashinfer_autotune_context(
            runner or self.target, run_lm_head=False
        ):
            return [self._choose(op) for op in ops or (OP,)]

    def _restart(self):
        self.tuner.clear_cache()
        self.reload()
        self.profiled.clear()
        autotune.attach_flashinfer_managed_autotune_cache(self.target)

    def _root(self):
        return autotune.flashinfer_managed_autotune_cache_root()

    def _entries(self):
        return sorted(self._root().glob("v2/*/entries/*.json"))

    def _entry_for(self, op):
        (path,) = [p for p in self._entries() if op in json.loads(p.read_text())["key"]]
        return path

    def test_cold_start_publishes_and_serving_reads_the_store(self):
        autotune.attach_flashinfer_managed_autotune_cache(self.target)
        self.assertEqual(self._tune(), [1])
        self.assertEqual(sorted(self.profiled), [-1, 0, 1, 2])
        (entry,) = self._entries()
        self.assertEqual(json.loads(entry.read_text())["tactic"], 1)
        # Serving runs outside any autotune context.
        self.profiled.clear()
        self.assertEqual(self._choose(), 1)
        self.assertEqual(self.profiled, [])

    def test_warm_restart_serves_the_published_winner_without_profiling(self):
        self._tune()
        self.times = dict(SHIFTED_TIMES)
        for _ in range(2):
            self._restart()
            self.assertEqual(self._tune(), [1])
            self.assertEqual(self._choose(), 1)
            self.assertEqual(self.profiled, [])

    def test_partial_store_tunes_only_the_missing_entry(self):
        self._tune(OP, OTHER_OP)
        self._entry_for(OTHER_OP).unlink()
        self.times = dict(SHIFTED_TIMES)
        self._restart()
        self.assertEqual(self._tune(OP, OTHER_OP), [1, 2])
        self.assertEqual(sorted(self.profiled), [-1, 0, 1, 2])
        self.assertEqual(json.loads(self._entry_for(OTHER_OP).read_text())["tactic"], 2)

    def test_corrupt_entry_is_retuned_and_republished(self):
        self._tune()
        entry = self._entry_for(OP)
        entry.write_text(entry.read_text()[:-7])
        self.times = dict(SHIFTED_TIMES)
        self._restart()
        self.assertEqual(self._tune(), [2])
        self.assertEqual(json.loads(self._entry_for(OP).read_text())["tactic"], 2)

    def test_unwritable_root_still_serves_what_it_tuned(self):
        blocker = self.cache_dir / "not_a_directory"
        blocker.write_text("")
        with autotune.envs.SGLANG_CACHE_DIR.override(str(blocker)):
            self.assertEqual(self._tune(), [1])
            self.profiled.clear()
            self.assertEqual(self._choose(), 1)
            self.assertEqual(self.profiled, [])

    def test_skipped_op_is_neither_profiled_nor_published(self):
        with get_exec().kernel.override(flashinfer_autotune_skip_ops=[OP]):
            self._tune(OP)
            self.assertEqual(self.profiled, [])
            self.assertEqual(self._entries(), [])

    def test_draft_tuning_keeps_the_targets_winners(self):
        autotune.attach_flashinfer_managed_autotune_cache(self.target)
        self.assertEqual(self._tune(OP), [1])
        autotune.attach_flashinfer_managed_autotune_cache(self.draft)
        self.times = {**TIMES, 1: 5.0, 2: 0.5}
        self.assertEqual(self._tune(OTHER_OP, runner=self.draft), [2])
        self.profiled.clear()
        self.assertEqual([self._choose(OP), self._choose(OTHER_OP)], [1, 2])
        self._restart()
        self.assertEqual([self._choose(OP), self._choose(OTHER_OP)], [1, 2])
        self.assertEqual(self.profiled, [])

    def test_json_and_managed_caches_select_the_same_winners(self):
        tables = [
            TIMES,
            SHIFTED_TIMES,
            {-1: 0.5, 0: 1.0, 1: 2.0, 2: 3.0},  # the default wins
            {-1: 2.0, 0: 1.0, 1: 1.0, 2: 1.0},  # a three-way tie
        ]
        winners = {}
        for managed in (False, True):
            with patch.object(
                autotune,
                "uses_flashinfer_managed_autotune_cache",
                return_value=managed,
            ):
                for i, table in enumerate(tables):
                    self._detach()
                    self.times = dict(table)
                    self.profiled.clear()
                    op = f"{OP}_{i}"
                    winners[managed, i] = (self._tune(op), sorted(self.profiled))
        for i in range(len(tables)):
            self.assertEqual(winners[False, i], winners[True, i], tables[i])


class TestManagedAutotuneCacheEligibility(_PublishedConfig):
    """Which processes use the managed store; the rest keep the JSON cache."""

    def setUp(self):
        super().setUp()
        self.enter(override_platform(has_flashinfer=True, device_capability=(10, 0)))

    def assert_managed(self, expected):
        self.assertEqual(autotune.uses_flashinfer_managed_autotune_cache(), expected)

    def test_single_rank_on_a_validated_gpu(self):
        for capability in ((9, 0), (10, 0), (10, 3)):
            with override_platform(device_capability=capability):
                self.assert_managed(True)

    def test_disabled_reuse_keeps_its_fresh_json_file(self):
        with autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(False):
            self.assert_managed(False)

    def test_megamoe_keeps_its_per_gpu_records(self):
        with get_exec().moe.override(moe_runner_backend="flashinfer_megamoe"):
            self.assert_managed(False)

    def test_any_capacity_beyond_one_rank_keeps_the_per_rank_cache(self):
        # Multi-rank launches, and elastic EP that may grow past one rank.
        with get_parallel().override(max_world_size=2):
            self.assert_managed(False)

    def test_platforms_without_flashinfer(self):
        # ROCm reports gfx942 as capability (9, 4) but has no FlashInfer.
        with override_platform(has_flashinfer=False, device_capability=(9, 4)):
            self.assert_managed(False)

    def test_unvalidated_architecture(self):
        for capability in ((8, 0), (11, 0), (12, 0), (None, None)):
            with (
                self.subTest(capability=capability),
                override_platform(device_capability=capability),
            ):
                self.assert_managed(False)


class TestManagedAutotuneCacheAttach(_PublishedConfig):
    """Warmup attaches the store only for an eligible runner that will tune."""

    def setUp(self):
        super().setUp()
        self.autotune_v2 = Mock(side_effect=lambda **_: nullcontext())
        self.enter(override_platform(has_flashinfer=True, device_capability=(10, 0)))
        self.enter(
            patch.dict(
                sys.modules,
                {"flashinfer": SimpleNamespace(autotune_v2=self.autotune_v2)},
            )
        )
        self.runner = SimpleNamespace(device="cuda", is_draft_worker=False)

    def attach(self):
        autotune.attach_flashinfer_managed_autotune_cache(self.runner)

    def test_attaches_the_root_for_a_runner_that_tunes(self):
        with patch.object(
            autotune, "should_run_flashinfer_autotune", return_value=True
        ):
            self.attach()
        self.autotune_v2.assert_called_once_with(
            mode="replay",
            cache_root=autotune.flashinfer_managed_autotune_cache_root(),
        )

    def test_runners_that_decline_tuning_do_not_attach(self):
        # A store would otherwise serve tuned tactics to these processes.
        for context in (
            get_exec().deterministic.override(enable_deterministic_inference=True),
            get_exec().kernel.override(disable_flashinfer_autotune=True),
        ):
            with self.subTest(context=context), context:
                self.attach()
        self.autotune_v2.assert_not_called()

    def test_ineligible_processes_do_not_attach(self):
        with (
            autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(False),
            patch.object(autotune, "should_run_flashinfer_autotune", return_value=True),
        ):
            self.attach()
        self.autotune_v2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
