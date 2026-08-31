"""Buffer-mode transfers must drain before destructive cache administration."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerBufferModeIdle(unittest.TestCase):
    def setUp(self):
        self.pipeline = BufferModePipeline.__new__(BufferModePipeline)
        self.pipeline.reset()

        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.running_batch = SimpleNamespace(is_empty=lambda: True, reqs=[])
        self.scheduler.chunked_req = None
        self.scheduler.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        self.scheduler.last_batch = None
        self.scheduler.enable_overlap = True
        self.scheduler.result_queue = []
        self.scheduler.ps = SimpleNamespace(pp_size=1)
        self.scheduler.waiting_queue = []
        self.scheduler._engine_paused = False
        self.scheduler.disaggregation_mode = DisaggregationMode.NULL
        self.scheduler.grammar_manager = MagicMock(grammar_queue=[])
        self.scheduler.enable_hisparse = False
        self.scheduler.enable_hierarchical_cache = True
        # These cache-mode maps intentionally stay empty: buffer-mode transfers
        # are owned by the pipeline, not by the similarly named cache maps.
        self.scheduler.tree_cache = SimpleNamespace(
            ongoing_write_through={},
            ongoing_load_back={},
            ongoing_prefetch={},
            ongoing_backup={},
            enable_storage=True,
            buffer_pipeline=self.pipeline,
            reset=MagicMock(),
        )
        self.scheduler.req_to_token_pool = MagicMock()
        self.scheduler.token_to_kv_pool_allocator = MagicMock()
        self.scheduler.metrics_reporter = MagicMock(is_stats_logging_rank=False)
        self.scheduler.draft_worker = None

        memory = patch(
            "sglang.srt.managers.scheduler.get_memory",
            return_value=SimpleNamespace(hicache_host_memory_mode="buffer_only"),
        )
        memory.start()
        self.addCleanup(memory.stop)

    def test_empty_pipeline_is_idle(self):
        self.assertTrue(self.pipeline.is_idle())
        self.assertTrue(self.scheduler.is_fully_idle())

    def test_each_buffer_stage_blocks_flush_until_drained(self):
        stages = (
            self.pipeline.pending_write_queue,
            self.pipeline.staged_prefetches,
            self.pipeline.ongoing_write_through,
            self.pipeline.ongoing_backup,
            self.pipeline.ongoing_buffer_load_back,
        )
        for name, stage in zip(("queued", "staged", "D2H", "storage", "H2D"), stages):
            with self.subTest(stage=name):
                if stage is self.pipeline.pending_write_queue:
                    stage.append(object())
                else:
                    stage[1] = object()

                try:
                    self.assertFalse(self.scheduler.flush_cache(empty_cache=False))
                    self.assertFalse(self.pipeline.is_idle())
                    self.assertFalse(self.scheduler.is_fully_idle())
                    self.scheduler.tree_cache.reset.assert_not_called()
                    self.scheduler.token_to_kv_pool_allocator.clear.assert_not_called()
                finally:
                    stage.clear()
                    self.scheduler.tree_cache.reset.reset_mock()
                    self.scheduler.token_to_kv_pool_allocator.clear.reset_mock()

                self.assertTrue(self.pipeline.is_idle())
                self.assertTrue(self.scheduler.flush_cache(empty_cache=False))
                self.scheduler.tree_cache.reset.assert_called_once()
                self.scheduler.token_to_kv_pool_allocator.clear.assert_called_once()
                self.scheduler.tree_cache.reset.reset_mock()
                self.scheduler.token_to_kv_pool_allocator.clear.reset_mock()

    def test_inflight_transfers_do_not_change_health_check(self):
        self.pipeline.ongoing_write_through[1] = object()
        self.pipeline.ongoing_buffer_load_back[-1] = object()

        self.assertFalse(self.scheduler.is_fully_idle())
        self.assertTrue(self.scheduler.is_fully_idle(for_health_check=True))


if __name__ == "__main__":
    unittest.main()
