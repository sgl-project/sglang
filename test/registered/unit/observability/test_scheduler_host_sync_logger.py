"""SchedulerHostSyncLogger counts torch sync-debug reports by scheduler stage,
forward mode and call site, and dumps them with the launched batches."""

import unittest
import warnings
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_PROCESS_BATCH_RESULT,
    SCHEDULER_STAGE_RUN_BATCH,
    SchedulerStageMetricsRecorder,
)
from sglang.srt.utils import scheduler_host_sync_logger as host_sync_module
from sglang.srt.utils.scheduler_host_sync_logger import SchedulerHostSyncLogger
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

_SYNC_MESSAGE = "called a synchronizing CUDA operation"


def _batch(mode: str):
    return SimpleNamespace(forward_mode=SimpleNamespace(name=mode))


def _issue_sync():
    warnings.warn(_SYNC_MESSAGE, UserWarning)


class TestSchedulerHostSyncLogger(CustomTestCase):
    def setUp(self):
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        self._debug_mode = patch.object(torch.cuda, "set_sync_debug_mode")
        self.set_sync_debug_mode = self._debug_mode.start()
        self.recorder = SchedulerStageMetricsRecorder(enabled=True)
        self.recorder.start(wall_ns=0)
        self.records = []
        self._log_json = patch.object(
            host_sync_module,
            "log_json",
            side_effect=lambda loggers, event, data: self.records.append((event, data)),
        )
        self._log_json.start()
        self.logger = SchedulerHostSyncLogger(
            targets=["stdout"], dump_interval=0.0, stage_recorder=self.recorder
        )

    def tearDown(self):
        self._log_json.stop()
        self._debug_mode.stop()
        self._warnings.__exit__(None, None, None)

    def test_maybe_create(self):
        self.assertIsNone(
            SchedulerHostSyncLogger.maybe_create(
                enable_metrics=True, stage_recorder=self.recorder
            )
        )
        with envs.SGLANG_LOG_SCHEDULER_HOST_SYNCS_TARGET.override("stdout"):
            with self.assertRaises(ValueError):
                SchedulerHostSyncLogger.maybe_create(
                    enable_metrics=False, stage_recorder=self.recorder
                )
            logger = SchedulerHostSyncLogger.maybe_create(
                enable_metrics=True, stage_recorder=self.recorder
            )
        self.assertIsInstance(logger, SchedulerHostSyncLogger)
        self.set_sync_debug_mode.assert_not_called()

    def test_counts_by_stage_mode_and_call_site(self):
        with self.recorder.record(SCHEDULER_STAGE_RUN_BATCH):
            self.logger.on_step_launch(_batch("DECODE"))
            _issue_sync()
            _issue_sync()
        self.set_sync_debug_mode.assert_called_once_with("warn")
        with self.recorder.record(SCHEDULER_STAGE_GET_NEXT_BATCH):
            _issue_sync()
        with self.recorder.record(SCHEDULER_STAGE_PROCESS_BATCH_RESULT):
            self.logger.on_step_result(_batch("EXTEND"))
            _issue_sync()
        self.logger.on_step_launch(_batch("DECODE"))

        self.assertEqual(len(self.records), 1)
        event, data = self.records[0]
        self.assertEqual(event, "scheduler.host_syncs")
        self.assertEqual(data["launched_batches"], {"decode": 1})
        self.assertEqual(data["total_syncs"], 4)
        counts = Counter()
        for sync in data["syncs"]:
            counts[sync["stage"], sync["forward_mode"]] += sync["count"]
        self.assertEqual(
            counts,
            {
                (SCHEDULER_STAGE_RUN_BATCH, "decode"): 2,
                (SCHEDULER_STAGE_GET_NEXT_BATCH, ""): 1,
                (SCHEDULER_STAGE_PROCESS_BATCH_RESULT, "extend"): 1,
            },
        )
        for sync in data["syncs"]:
            innermost = sync["call_site"].split(" <- ")[0]
            self.assertTrue(innermost.endswith(" _issue_sync"), innermost)

    def test_window_is_cleared_after_dump(self):
        with self.recorder.record(SCHEDULER_STAGE_RUN_BATCH):
            self.logger.on_step_launch(_batch("EXTEND"))
            _issue_sync()
        self.logger.on_step_launch(_batch("DECODE"))
        self.logger.on_step_launch(_batch("DECODE"))
        self.assertEqual([data["total_syncs"] for _, data in self.records], [1, 0])
        self.assertEqual(self.records[1][1]["launched_batches"], {"decode": 1})

    def test_other_warnings_pass_through(self):
        shown = []
        warnings.showwarning = lambda message, *args, **kwargs: shown.append(
            str(message)
        )
        self.logger.on_step_launch(_batch("DECODE"))
        warnings.warn("unrelated", UserWarning)
        self.assertEqual(shown, ["unrelated"])
        self.logger.on_step_launch(_batch("DECODE"))
        self.assertEqual(self.records[0][1]["total_syncs"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_counts_real_device_read(self):
        self._debug_mode.stop()
        previous_mode = torch.cuda.get_sync_debug_mode()
        try:
            x = torch.ones(4, device="cuda")
            with self.recorder.record(SCHEDULER_STAGE_RUN_BATCH):
                self.logger.on_step_launch(_batch("DECODE"))
                x.sum().item()
            self.logger.on_step_launch(_batch("DECODE"))
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)
            self._debug_mode.start()
        (sync,) = self.records[0][1]["syncs"]
        self.assertEqual(sync["stage"], SCHEDULER_STAGE_RUN_BATCH)
        self.assertIn("test_counts_real_device_read", sync["call_site"])


if __name__ == "__main__":
    unittest.main()
