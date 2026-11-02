"""Unit tests for the release/resume memory-occupation state machine.

RL Test Consolidation (sgl-project/sglang#47624) — sleep/wake area, CPU tier.
File ownership per RFC section 5 (sleep/wake, CPU/unit).

Coverage:
- release asserts a fully idle engine; an engine_paused release passes with
  ignore_waiting=True.
- Default tags pause all memory types in the fixed order kv -> weights ->
  cuda_graph; the request-side tag order is normalized.
- The device is synchronized during release. NOTE: on main the synchronize
  currently runs AFTER the pool pauses — the Ascend SDMA-race fix (synchronize
  BEFORE the first pause) is pending as a separate defect-fix PR; a strict
  ordering assertion is kept (commented) in
  test_release_synchronizes_device_during_release and must be enabled when
  that fix lands.
- kv_cache release flushes the cache exactly once; weights release does not.
- release is idempotent (offload_tags is a set); unknown tags are silently
  tracked today — the RFC proposes strict rejection, so this test pins the
  current behavior to make any future tightening a deliberate change.
- resume is NOT idempotent: an unknown tag raises KeyError; a weights resume
  without the release-time static-state stash raises AttributeError.
- a proper weights resume imports the stashed static state and clears it.

NPU risk probed (indirectly, via call ordering): the synchronize/pause race
that aborts Ascend devices with SDMA channel errors.

BF16 precision scope (RFC section 4): this unit tier exercises the
release/resume state machine against mocked devices and is deliberately
dtype-agnostic; the BF16 contract (explicit --dtype bfloat16 + /server_info
dtype assertion) is enforced by the 1-NPU smoke sibling
test/registered/npu/rl/test_npu_release_resume_memory.py (S0).

Sibling CUDA test:
  test/registered/rl/test_multi_instance_release_memory_occupation.py
Empirical background: npu188 TMS probes (region content zeroing, SDMA error
without prior synchronize) — see sleep-mode consolidation report.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

WU_MODULE = "sglang.srt.managers.scheduler_components.weight_updater"


class TestSchedulerMemoryRelease(CustomTestCase):
    def setUp(self):
        self.updater = SchedulerWeightUpdaterManager.__new__(
            SchedulerWeightUpdaterManager
        )
        self.updater.scheduler = MagicMock()
        self.updater.scheduler._engine_paused = False
        self.updater.scheduler.disaggregation_mode = DisaggregationMode.NULL
        self.updater.is_fully_idle = lambda **kwargs: True
        self.updater.offload_tags = set()
        self.updater.memory_saver_adapter = MagicMock()
        self.updater.tp_worker = MagicMock()
        self.updater.tp_cpu_group = None
        self.updater.flush_cache = MagicMock()
        # _assert_weight_cache_inactive is a class method (slots dataclass):
        # patch it at class level instead of assigning an instance attribute.
        patcher_wci = patch.object(
            SchedulerWeightUpdaterManager, "_assert_weight_cache_inactive"
        )
        self.mock_wci = patcher_wci.start()
        self.addCleanup(patcher_wci.stop)
        # Shared ledger for device-sync and pool-pause call ordering.
        self.events = []

        patcher_barrier = patch("torch.distributed.barrier")
        patcher_device = patch("torch.get_device_module")
        patcher_export = patch(
            f"{WU_MODULE}._export_static_state", return_value="STASH"
        )
        patcher_import = patch(f"{WU_MODULE}._import_static_state")
        self.mock_barrier = patcher_barrier.start()
        self.mock_device_module = patcher_device.start()
        self.mock_export = patcher_export.start()
        self.mock_import = patcher_import.start()
        self.mock_device_module.return_value.synchronize.side_effect = lambda: (
            self.events.append("sync")
        )
        self.updater.memory_saver_adapter.pause.side_effect = lambda tag: (
            self.events.append(f"pause:{tag}")
        )
        self.addCleanup(patcher_barrier.stop)
        self.addCleanup(patcher_device.stop)
        self.addCleanup(patcher_export.stop)
        self.addCleanup(patcher_import.stop)

    def _release(self, tags=None):
        return self.updater.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=tags)
        )

    def _resume(self, tags=None):
        return self.updater.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=tags)
        )

    def _pause_calls(self):
        return [e for e in self.events if e.startswith("pause:")]

    # --- idle prerequisite -------------------------------------------------

    def test_release_rejected_while_requests_in_flight(self):
        """release with an unfinished engine must assert before touching pools."""
        self.updater.is_fully_idle = lambda **kwargs: False

        with self.assertRaises(AssertionError):
            self._release(tags=["kv_cache"])

        self.updater.memory_saver_adapter.pause.assert_not_called()
        self.assertEqual(self._pause_calls(), [])

    def test_release_while_engine_paused_passes_ignore_waiting(self):
        """A paused engine holds queued requests: release must consult
        is_fully_idle with ignore_waiting=True instead of failing."""
        seen = {}
        self.updater.is_fully_idle = lambda **kwargs: seen.update(kwargs) or True
        self.updater.scheduler._engine_paused = True

        self._release(tags=["kv_cache"])

        self.assertEqual(seen, {"ignore_waiting": True})

    # --- execution order contracts ----------------------------------------

    def test_release_default_tags_pause_all_types_in_fixed_order(self):
        """tags=None must pause kv, weights, cuda_graph — in that order."""
        self._release(tags=None)

        self.assertEqual(
            self._pause_calls(),
            [
                f"pause:{GPU_MEMORY_TYPE_KV_CACHE}",
                f"pause:{GPU_MEMORY_TYPE_WEIGHTS}",
                f"pause:{GPU_MEMORY_TYPE_CUDA_GRAPH}",
            ],
        )

    def test_release_request_tag_order_is_normalized(self):
        """The caller's tag order must not change execution order: kv still
        pauses before weights (single-request normalization contract)."""
        self._release(tags=["weights", "kv_cache"])

        self.assertEqual(
            self._pause_calls(),
            [
                f"pause:{GPU_MEMORY_TYPE_KV_CACHE}",
                f"pause:{GPU_MEMORY_TYPE_WEIGHTS}",
            ],
        )

    def test_release_synchronizes_device_during_release(self):
        """release must synchronize the device at least once.

        On main the synchronize currently runs AFTER the pool pauses. The
        Ascend SDMA-race fix (synchronize BEFORE the first pause and after the
        static-state export) is pending as a separate defect-fix PR; enable
        the strict ordering below together with that fix:
        """
        self._release(tags=None)

        self.assertIn("sync", self.events)

        # Strict ordering (enable once the SDMA sync-before-pause fix lands):
        # first_pause = next(
        #     i for i, e in enumerate(self.events) if e.startswith("pause:")
        # )
        # self.assertLess(self.events.index("sync"), first_pause)

    def test_release_kv_cache_flushes_exactly_once(self):
        """kv_cache release flushes the cache once; weights release must not."""
        self._release(tags=["kv_cache"])
        self.assertEqual(self.updater.flush_cache.call_count, 1)

        self.updater.flush_cache.reset_mock()
        self._release(tags=["weights"])
        self.updater.flush_cache.assert_not_called()

    # --- idempotency and tag contracts -------------------------------------

    def test_double_release_is_idempotent(self):
        """Re-releasing the same tags must not raise nor duplicate state."""
        self._release(tags=["weights"])
        self._release(tags=["weights"])

        self.assertEqual(self.updater.offload_tags, {GPU_MEMORY_TYPE_WEIGHTS})
        self.assertEqual(len(self._pause_calls()), 2)

    def test_release_unknown_tag_is_silently_tracked(self):
        """Current contract: unknown tags are added to offload_tags without
        validation and never reach any pool pause. The consolidation RFC
        proposes rejecting them; this test pins today's behavior so a future
        strict-validation change updates this test deliberately."""
        self._release(tags=["weights", "bogus-tag"])

        self.assertIn("bogus-tag", self.updater.offload_tags)
        self.assertEqual(self._pause_calls(), [f"pause:{GPU_MEMORY_TYPE_WEIGHTS}"])

    def test_resume_unknown_tag_raises_keyerror(self):
        """resume removes tags from offload_tags: resuming a tag that was never
        released must raise KeyError (resume is not idempotent today)."""
        with self.assertRaises(KeyError):
            self._resume(tags=["kv_cache"])

    def test_resume_weights_without_prior_release_raises_attribute_error(self):
        """A weights resume requires the static-state stash exported at release
        time. With the tag manually present but no stash, resume must fail
        loudly (AttributeError) rather than resume into undefined state."""
        self.updater.offload_tags = {
            GPU_MEMORY_TYPE_KV_CACHE,
            GPU_MEMORY_TYPE_WEIGHTS,
        }

        with self.assertRaises(AttributeError):
            self._resume(tags=["weights"])

    def test_resume_weights_imports_stash_and_clears_it(self):
        """The happy path: weights resume restores the exported static state
        and drops the stash reference."""
        self._release(tags=["weights"])
        self.assertEqual(self.updater.stashed_model_static_state, "STASH")

        self._resume(tags=["weights"])

        self.mock_import.assert_called_once_with(
            self.updater.tp_worker.model_runner.model, "STASH"
        )
        self.assertFalse(hasattr(self.updater, "stashed_model_static_state"))


if __name__ == "__main__":
    unittest.main()
