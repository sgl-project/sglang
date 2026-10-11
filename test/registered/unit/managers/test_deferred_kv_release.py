"""Deferred KV releases of finished requests: a capped number per decode step,
early for a waiting turn that continues one."""

import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import batch_result_processor
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_processor() -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=Mock(),
        tree_cache=Mock(),
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=Mock(),
        draft_worker=None,
        model_worker=Mock(),
        logprob_result_processor=None,
        output_streamer=Mock(),
        beam_coordinator=Mock(),
        abort_request=lambda *args, **kwargs: None,
    )


def _finished(rid, ids):
    return SimpleNamespace(rid=rid, full_untruncated_fill_ids=array("q", ids))


def _waiting(ids):
    return SimpleNamespace(origin_input_ids=array("q", ids))


class TestDeferredKvRelease(CustomTestCase):
    def setUp(self):
        self.processor = _make_processor()
        self.released = []
        recorder = patch.object(
            batch_result_processor,
            "release_kv_cache",
            side_effect=lambda req, _cache, checkpoint: self.released.append(req.rid),
        )
        recorder.start()
        self.addCleanup(recorder.stop)

    def _defer(self, *reqs):
        self.processor.deferred_kv_releases.extend((req, True) for req in reqs)

    def _deferred_rids(self):
        return [req.rid for req, _ in self.processor.deferred_kv_releases]

    def test_a_capped_step_releases_the_oldest(self):
        self._defer(*(_finished(i, [i]) for i in range(20)))

        self.processor.release_deferred_kv_caches(limit=8)

        self.assertEqual(self.released, list(range(8)))
        self.assertEqual(self._deferred_rids(), list(range(8, 20)))

    def test_an_uncapped_release_drains_everything(self):
        self._defer(*(_finished(i, [i]) for i in range(3)))

        self.processor.release_deferred_kv_caches()
        self.processor.release_deferred_kv_caches()

        self.assertEqual(self.released, [0, 1, 2])
        self.assertEqual(self._deferred_rids(), [])

    def test_only_a_continued_turn_is_released_early(self):
        self._defer(
            _finished("a", [1, 2, 3, 4]),
            _finished("b", [1, 5, 6]),
            _finished("c", [9, 9]),
        )

        self.processor.release_deferred_kv_caches_continued_by(
            [_waiting([1, 2, 3, 4, 7, 8])]
        )

        self.assertEqual(self.released, ["a"])
        self.assertEqual(self._deferred_rids(), ["b", "c"])

    def test_a_shorter_or_diverging_waiting_request_releases_nothing(self):
        self._defer(_finished("a", [1, 2, 3, 4]))

        self.processor.release_deferred_kv_caches_continued_by(
            [_waiting([1, 2, 3]), _waiting([2, 2, 3, 4, 5]), _waiting([1, 2, 3, 5, 6])]
        )
        self.processor.release_deferred_kv_caches_continued_by([])

        self.assertEqual(self.released, [])
        self.assertEqual(self._deferred_rids(), ["a"])

    def test_deferred_queue_is_per_processor(self):
        self._defer(_finished("a", [1]))

        self.assertEqual(len(_make_processor().deferred_kv_releases), 0)
        self.assertIsInstance(self.processor.deferred_kv_releases, deque)


class TestSchedulerDrainsDeferredReleases(CustomTestCase):
    def _scheduler(self):
        scheduler = object.__new__(Scheduler)
        processor = Mock(deferred_kv_releases=deque([(object(), True)]))
        processor.release_deferred_kv_caches.side_effect = (
            processor.deferred_kv_releases.clear
        )
        scheduler.batch_result_processor = processor
        return scheduler

    def test_kv_pressure_drains_deferred_releases_before_retracting(self):
        scheduler = self._scheduler()
        scheduler.enable_skip_finishing_decode = False
        scheduler.result_queue = deque()
        scheduler.new_token_ratio_tracker = Mock()
        batch = Mock()
        batch.batch_size.return_value = 2
        batch.is_empty.return_value = False
        # Full until the deferred KV goes back, then enough for the step.
        batch.check_decode_mem.side_effect = [False, True]

        self.assertIs(scheduler.update_running_batch(batch), batch)

        scheduler.batch_result_processor.release_deferred_kv_caches.assert_called_once_with()
        batch.retract_decode.assert_not_called()
        batch.prepare_for_decode.assert_called_once_with()

    def test_deferred_releases_keep_the_scheduler_busy(self):
        scheduler = self._scheduler()
        scheduler.running_batch = Mock(is_empty=Mock(return_value=True))
        scheduler.chunked_req = None
        scheduler.dllm_manager = Mock(any_staging_reqs=Mock(return_value=False))
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_lmcache = False
        scheduler._pp_microbatches_drained = lambda: True

        self.assertFalse(scheduler.is_fully_idle())
        scheduler.batch_result_processor.deferred_kv_releases.clear()
        self.assertTrue(scheduler.is_fully_idle())


if __name__ == "__main__":
    unittest.main()
