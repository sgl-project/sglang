"""Unit tests for server warmup progress reporting."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.server_warmup import SchedulerWarmupMixin


class TestServerWarmupProgress(unittest.TestCase):
    def test_ci_progress_uses_scheduler_counter_when_tqdm_is_disabled(self):
        scheduler = SchedulerWarmupMixin()
        scheduler._show_warmup_progress = True
        scheduler._warmup_total = 1
        scheduler._warmup_processed = 1
        progress_bar = MagicMock(total=1, n=0)
        scheduler._warmup_progress_bar = progress_bar

        with (
            patch(
                "sglang.multimodal_gen.runtime.server_warmup._is_ci_log_env",
                return_value=True,
            ),
            patch("sglang.multimodal_gen.runtime.server_warmup.logger") as logger,
        ):
            scheduler._advance_warmup_progress_bar(object(), OutputBatch())

        logger.info.assert_called_once_with(
            "Warmup requests: %s/%s %s", 1, 1, "warmup req"
        )
        progress_bar.close.assert_called_once_with()
        self.assertIsNone(scheduler._warmup_progress_bar)


if __name__ == "__main__":
    unittest.main()
