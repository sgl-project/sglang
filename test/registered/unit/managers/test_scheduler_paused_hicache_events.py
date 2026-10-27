"""A paused scheduler keeps reaping HiCache events, so a flush can succeed."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    PauseGenerationReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import weight_updater
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TreeCache:
    """The HiCache state is_fully_idle reads. A write-through is queued at insert
    and an L3 prefetch at prefetch_from_storage; check_hicache_events retires
    both, as writing_check and _drain_ack_prefetch do once the copies are acked."""

    def __init__(self, enable_storage):
        self.enable_storage = enable_storage
        self.ongoing_write_through, self.ongoing_load_back = {}, {}
        self.ongoing_prefetch, self.ongoing_backup = {}, {}
        self.hicache_storage_pass_prefix_keys = False
        self.prefetch_threshold = 256

    def insert(self, key):
        self.ongoing_write_through[key] = object()

    def check_hicache_events(self):
        self.ongoing_write_through.clear()
        self.ongoing_prefetch.clear()

    def is_backuped(self, node):
        return False

    def is_root(self, node):
        return True

    def get_last_hash_value(self, node):
        return None

    def prefetch_from_storage(self, handle, last_host_node, new_input_tokens, *a, **k):
        # Page-aligned length against the threshold, as the unified tree does.
        if len(new_input_tokens) // 64 * 64 >= self.prefetch_threshold:
            self.ongoing_prefetch[handle] = object()

    def reset(self):
        pass


def _match_nothing(self, tree_cache, cow_mamba=False):
    # The stub tree holds nothing, so the requeued request matches no prefix.
    self.refresh_fill_ids()
    self.prefix_len, self.host_hit_length = 0, 0
    self.last_host_node = self.last_node = "root"


class TestPausedHiCacheReap(unittest.TestCase):
    def setUp(self):
        enter_scope(self, published_topology())
        self.calls = Mock()
        self.calls.idle.return_value = True

    def _stub_scheduler(self, mode, hicache=True, decode_hicache=False):
        s = Scheduler.__new__(Scheduler)
        s.disaggregation_mode = mode
        s.enable_hierarchical_cache = hicache
        s.enable_decode_hicache = decode_hicache
        s.enable_unified_cache_external_linker = False
        s.enable_lmcache = False
        s.enable_hicache_storage = True
        s.tree_cache = SimpleNamespace(check_hicache_events=self.calls.drain)
        s._process_storage_prefetch_retries = self.calls.retry
        s.is_fully_idle = self.calls.idle
        s.metrics_reporter = Mock()
        s.kv_events_publisher = Mock()
        return s

    def test_paused_iteration_reaps_once_before_recording_state(self):
        """Each paused iteration reaps HiCache events once, before it records
        idle or active, and retries no L3 prefetch."""
        for mode, decode_hicache in (
            (DisaggregationMode.NULL, False),
            (DisaggregationMode.PREFILL, False),
            (DisaggregationMode.DECODE, True),
        ):
            with self.subTest(mode=mode.name):
                self.calls.reset_mock()
                s = self._stub_scheduler(mode, decode_hicache=decode_hicache)
                s._record_scheduler_state_for_paused_engine()
                self.assertEqual(self.calls.mock_calls, [call.drain(), call.idle()])

    def test_no_reap_without_hicache_or_decode_radix_cache(self):
        """No reap without HiCache, and none on a PD decode worker without the
        decode radix cache, whose running loop does not reap either."""
        for mode, hicache in (
            (DisaggregationMode.NULL, False),
            (DisaggregationMode.DECODE, True),
        ):
            with self.subTest(mode=mode.name, hicache=hicache):
                self.calls.reset_mock()
                s = self._stub_scheduler(mode, hicache=hicache, decode_hicache=False)
                s._record_scheduler_state_for_paused_engine()
                self.assertEqual(self.calls.mock_calls, [call.idle()])


class TestPausedFlushAfterRetract(unittest.TestCase):
    """The real event_loop_normal, pause_generation(retract), flush_cache and
    update_weights_from_disk; the forward, retract_all and the tree are stubbed."""

    def setUp(self):
        enter_scope(self, published_topology())

    def _scheduler(self, storage, n_output, insert):
        tc = _TreeCache(enable_storage=storage)
        req = Req(
            rid="r",
            origin_input_text="",
            origin_input_ids=array("q", range(1, 33)),
            sampling_params=SamplingParams(max_new_tokens=4096),
        )
        req.output_ids = array("q", range(100, 100 + n_output))
        s = Scheduler.__new__(Scheduler)
        s.__dict__.update(
            gracefully_exit=False,
            _engine_paused=False,
            _pause_started_at=None,
            enable_overlap=False,
            enable_overlap_mlx=False,
            running_batch=SimpleNamespace(
                reqs=[req],
                batch_is_full=False,
                is_empty=lambda: not s.running_batch.reqs,
            ),
            last_batch=None,
            chunked_req=None,
            waiting_queue=[],
            disaggregation_mode=DisaggregationMode.NULL,
            decode_offload_manager=None,
            hisparse_coordinator=None,
            enable_hierarchical_cache=True,
            enable_hicache_storage=storage,
            enable_lmcache=False,
            enable_hisparse=False,
            enable_priority_scheduling=False,
            max_queued_requests=None,
            processed_tokens_counter=0,
            tree_cache=tc,
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            draft_worker=None,
            dllm_manager=SimpleNamespace(any_staging_reqs=lambda: False),
            grammar_manager=MagicMock(grammar_queue=[]),
            metrics_reporter=MagicMock(
                current_scheduler_metrics_enabled=False, is_stats_logging_rank=False
            ),
            kv_events_publisher=MagicMock(),
        )
        s._process_storage_prefetch_retries = lambda: None

        def get_next_batch_to_run(running_batch, last_batch):
            s._process_hicache_events()  # the real getter's first step
            batch = SimpleNamespace(
                reqs=running_batch.reqs,
                is_empty=lambda: False,
                forward_mode=SimpleNamespace(is_extend=lambda: False),
            )
            return SimpleNamespace(running_batch=running_batch, batch_to_run=batch)

        s.get_next_batch_to_run = get_next_batch_to_run
        s.run_batch = lambda batch: None

        def process_batch_result(batch, result):
            # cache_unfinished_req inserts the decoded tokens: one write-through.
            if insert:
                tc.insert("r")

        s.process_batch_result = process_batch_result

        with patch.object(
            weight_updater,
            "get_parallel",
            return_value=SimpleNamespace(tp_group=SimpleNamespace(cpu_group=None)),
        ):
            updater = weight_updater.SchedulerWeightUpdaterManager(
                tp_worker=None,
                draft_worker=None,
                memory_saver_adapter=None,
                flush_cache=s.flush_cache,
                is_fully_idle=s.is_fully_idle,
                scheduler=s,
            )
        return s, tc, updater

    def _run(self, storage=False, n_output=0, insert=True):
        """One running iteration, a retract pause, one paused iteration, then
        update_weights_from_disk(flush_cache=True) and flush_cache()."""
        s, tc, updater = self._scheduler(storage, n_output, insert)
        seen = {}
        runner = SimpleNamespace(
            weight_updater=SimpleNamespace(
                update_weights_from_disk=lambda *a, **k: (
                    True,
                    "Succeeded to update model weights.",
                )
            )
        )

        def ingest_requests():
            seen["step"] = seen.get("step", 0) + 1
            if seen["step"] == 2:
                s.pause_generation(PauseGenerationReqInput(mode="retract"))
                seen["after_pause"] = (
                    len(s.waiting_queue),
                    len(tc.ongoing_write_through),
                    len(tc.ongoing_prefetch),
                )
            elif seen["step"] == 3:
                out = updater.update_weights_from_disk(
                    UpdateWeightFromDiskReqInput(model_path="dummy", flush_cache=True)
                )
                seen["update"] = (out.success, out.message)
                seen["flush_cache"] = s.flush_cache()
                s.gracefully_exit = True

        s.ingest_requests = ingest_requests
        with (
            patch("sglang.srt.managers.scheduler.retract_all"),
            patch.object(Req, "init_next_round_input", _match_nothing),
            patch.object(
                weight_updater.SchedulerWeightUpdaterManager,
                "_select_runners",
                lambda self, selector="all": [("target", runner)],
            ),
        ):
            Scheduler.event_loop_normal(s)
        return seen

    def test_write_through_queued_before_pause_does_not_block_flush(self):
        """After a retract pause and one paused iteration, a write-through queued
        before the pause does not block the weight update's flush or flush_cache."""
        seen = self._run()
        self.assertEqual(seen["after_pause"], (1, 1, 0))
        self.assertEqual(seen["update"], (True, "Succeeded to update model weights."))
        self.assertTrue(seen["flush_cache"])

    def test_prefetch_issued_by_retract_requeue_is_retired_while_paused(self):
        """With L3 storage, requeuing a retracted request with 300 generated
        tokens starts one prefetch; a paused iteration retires it."""
        seen = self._run(storage=True, n_output=300, insert=False)
        self.assertEqual(seen["after_pause"], (1, 0, 1))
        self.assertEqual(seen["update"], (True, "Succeeded to update model weights."))
        self.assertTrue(seen["flush_cache"])


if __name__ == "__main__":
    unittest.main()
