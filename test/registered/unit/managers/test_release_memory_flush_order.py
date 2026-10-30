import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestReleaseMemoryFlushOrder(CustomTestCase):
    """Releasing the KV region must flush the pools exactly once, before the region is tagged as offloaded."""

    def _harness(self):
        scheduler = SimpleNamespace(
            _engine_paused=False,
            is_fully_idle=lambda ignore_waiting=False: True,
            cur_batch_for_debug=None,
            last_batch=None,
            tree_cache=Mock(),
            req_to_token_pool=Mock(),
            token_to_kv_pool_allocator=Mock(),
            grammar_manager=Mock(),
            metrics_reporter=SimpleNamespace(
                reset_metrics=lambda: None, is_stats_logging_rank=False
            ),
            draft_worker=None,
            waiting_queue=[],
            running_batch=SimpleNamespace(reqs=[]),
            disaggregation_mode=DisaggregationMode.NULL,
        )
        with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
            manager = SchedulerWeightUpdaterManager(
                tp_worker=None,
                draft_worker=None,
                memory_saver_adapter=Mock(),
                flush_cache=lambda empty_cache=False: Scheduler.flush_cache(
                    scheduler, empty_cache=False
                ),
                is_fully_idle=scheduler.is_fully_idle,
                scheduler=scheduler,
            )
        scheduler.weight_updater = manager
        return scheduler, manager

    def test_release_flushes_once_before_the_pause_and_later_flushes_are_skipped(self):
        scheduler, manager = self._harness()
        calls = Mock()
        manager.flush_cache = Mock(wraps=manager.flush_cache)
        calls.attach_mock(manager.flush_cache, "flush")
        calls.attach_mock(manager.memory_saver_adapter.pause, "pause")
        with patch(
            "sglang.srt.managers.scheduler_components.weight_updater.torch.get_device_module",
            return_value=SimpleNamespace(synchronize=calls.synchronize),
        ):
            manager.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            )

        self.assertEqual(
            calls.mock_calls,
            [
                call.flush(),
                call.synchronize(),
                call.pause(GPU_MEMORY_TYPE_KV_CACHE),
                call.synchronize(),
            ],
        )
        self.assertEqual(scheduler.tree_cache.reset.call_count, 1)
        self.assertEqual(manager.memory_saver_adapter.pause.call_count, 1)
        self.assertIn(GPU_MEMORY_TYPE_KV_CACHE, manager.offload_tags)

        # the trainer's /flush_cache while the KV region is unmapped must not touch the pools
        self.assertTrue(Scheduler.flush_cache(scheduler, empty_cache=False))
        self.assertEqual(scheduler.tree_cache.reset.call_count, 1)

        manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        )
        self.assertNotIn(GPU_MEMORY_TYPE_KV_CACHE, manager.offload_tags)
        self.assertTrue(Scheduler.flush_cache(scheduler, empty_cache=False))
        self.assertEqual(scheduler.tree_cache.reset.call_count, 2)

    def test_flush_and_release_drop_cached_embeddings(self):
        """/flush_cache and a KV release must drop ViT embeddings with the other pools."""
        mm_schedule.init_mm_embedding_cache(1 << 20)
        self.addCleanup(mm_schedule.init_mm_embedding_cache, 0)
        scheduler, manager = self._harness()
        entry = EmbeddingResult(embedding=torch.ones(2, 4))

        mm_schedule.embedding_cache.set(7, entry)
        self.assertTrue(Scheduler.flush_cache(scheduler, empty_cache=False))
        self.assertFalse(mm_schedule.embedding_cache.has(7))

        mm_schedule.embedding_cache.set(7, entry)
        with patch(
            "sglang.srt.managers.scheduler_components.weight_updater.torch.get_device_module"
        ):
            manager.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            )
        self.assertFalse(mm_schedule.embedding_cache.has(7))


if __name__ == "__main__":
    unittest.main()
