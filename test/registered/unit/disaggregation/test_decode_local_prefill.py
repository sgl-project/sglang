import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.pd_disaggregation_hook import (
    validate_decode_local_prefill,
)
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq, PauseGenerationReqInput
from sglang.srt.managers.schedule_batch import Req, release_req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _req(rid: str, local: bool, num_tokens: int = 4) -> Req:
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", range(num_tokens)),
        sampling_params=SamplingParams(max_new_tokens=8),
        disagg_local_prefill=local,
    )


class TestDecodeLocalPrefillValidation(CustomTestCase):
    def _args(self, **kwargs) -> ServerArgs:
        fields = dict(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_local_prefill=True,
            chunked_prefill_size=1024,
        )
        fields.update(kwargs)
        return ServerArgs(**fields)

    def test_accepts_supported_decode_config(self):
        validate_decode_local_prefill(self._args())

    def test_requires_decode_mode(self):
        with self.assertRaisesRegex(ValueError, "requires --disaggregation-mode"):
            validate_decode_local_prefill(self._args(disaggregation_mode="prefill"))

    def test_rejects_unsupported_configs(self):
        for kwargs, reason in (
            ({"pp_size": 2}, "pipeline parallelism"),
            ({"moe_a2a_backend": "deepep"}, "--moe-a2a-backend deepep"),
            ({"chunked_prefill_size": -1}, "disabled chunked prefill"),
        ):
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, reason):
                    validate_decode_local_prefill(self._args(**kwargs))


class TestDecodeLocalPrefillScheduler(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")

    def _scheduler(self, local_prefill: bool = True) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.enable_priority_scheduling = False
        scheduler.abort_on_priority_when_disabled = False
        scheduler.max_queued_requests = None
        scheduler.processed_tokens_counter = 0
        scheduler.enable_hicache_storage = False
        scheduler.enable_lmcache = False
        scheduler.enable_hisparse = False
        scheduler.waiting_queue = []
        scheduler.disagg_decode_prebuilt_queue = [] if local_prefill else None
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        scheduler.disagg_decode_transfer_queue = MagicMock()
        scheduler.tree_cache = MagicMock()
        scheduler.ipc_channels = MagicMock()
        scheduler.beam_coordinator = MagicMock()
        scheduler.mm_receiver = None
        scheduler.dllm_config = None
        scheduler.grammar_manager = MagicMock()
        scheduler.chunked_req = None
        scheduler.collect_inflight_reqs = MagicMock(return_value=set())
        scheduler.scheduler_stage_metrics = None
        scheduler.decode_offload_manager = None
        return scheduler

    def test_local_request_waits_for_prefill_on_this_server(self):
        scheduler = self._scheduler()
        local, remote = _req("local", True), _req("remote", False)

        scheduler._add_request_to_queue(local)
        scheduler._add_request_to_queue(remote)

        self.assertEqual(scheduler.waiting_queue, [local])
        scheduler.disagg_decode_prealloc_queue.add.assert_called_once_with(
            remote, is_retracted=False
        )

    def test_retracted_local_request_is_prefilled_again(self):
        scheduler = self._scheduler()
        local = _req("local", True)

        scheduler._add_request_to_queue(local, is_retracted=True)

        self.assertEqual(scheduler.waiting_queue, [local])
        scheduler.disagg_decode_prealloc_queue.add.assert_not_called()

    def test_transferred_requests_skip_the_local_waiting_queue(self):
        scheduler = self._scheduler()
        local = _req("local", True)
        scheduler.waiting_queue = [local]
        transferred = _req("transferred", False)
        scheduler.enable_decode_hicache = False
        scheduler.polling_count = 0
        scheduler.polling_interval = 1
        scheduler.disagg_decode_prealloc_queue.retracted_queue = []
        scheduler.disagg_decode_prealloc_queue.resume_retracted_reqs.return_value = []
        scheduler.disagg_decode_prealloc_queue.pop_preallocated.return_value = ([], [])
        scheduler.disagg_decode_transfer_queue.pop_transferred.return_value = [
            transferred
        ]

        scheduler.process_decode_queue()

        self.assertEqual(scheduler.waiting_queue, [local])
        self.assertEqual(scheduler.disagg_decode_prebuilt_queue, [transferred])

    def test_without_local_prefill_transferred_requests_use_waiting_queue(self):
        scheduler = self._scheduler(local_prefill=False)

        self.assertIs(
            scheduler.get_disagg_decode_prebuilt_reqs(), scheduler.waiting_queue
        )

    def test_next_batch_runs_through_the_aggregated_scheduler(self):
        scheduler = self._scheduler()
        running_batch, last_batch = MagicMock(), MagicMock()
        scheduler.get_new_prebuilt_batch = MagicMock(return_value=None)
        scheduler.get_next_batch_to_run = MagicMock(return_value="plan")

        plan = scheduler.get_next_disagg_decode_batch_to_run(
            running_batch=running_batch, last_batch=last_batch
        )

        self.assertEqual(plan, "plan")
        scheduler.get_next_batch_to_run.assert_called_once_with(
            running_batch, last_batch
        )

    def test_prebuilt_admission_counts_local_requests_joining(self):
        scheduler = self._scheduler()
        scheduler.max_running_requests = 4
        scheduler.req_to_token_pool = SimpleNamespace(size=8)
        scheduler.token_to_kv_pool_allocator = None
        scheduler.enable_overlap = False
        scheduler.model_config = None
        scheduler.spec_algorithm = None
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.chunked_req = _req("chunked", True)
        last_batch = SimpleNamespace(
            reqs=[_req("prefilled", True), scheduler.chunked_req],
            forward_mode=ForwardMode.EXTEND,
        )
        transferred = [MagicMock(kv=SimpleNamespace(kv_committed_len=None))] * 3
        scheduler.disagg_decode_prebuilt_queue = list(transferred)
        running_batch = SimpleNamespace(batch_size=lambda: 1)

        joining = scheduler._num_local_prefill_reqs_joining(last_batch)
        with patch("sglang.srt.disaggregation.decode.ScheduleBatch.init_new"):
            scheduler._get_new_prebuilt_batch(running_batch, joining)

        # 4 slots: 1 running, 2 local requests joining, so 1 transferred fits.
        self.assertEqual(joining, 2)
        self.assertEqual(len(scheduler.disagg_decode_prebuilt_queue), 2)

    def test_abort_releases_kv_only_for_transferred_requests(self):
        scheduler = self._scheduler()
        local, transferred = _req("a-local", True), _req("a-transferred", False)
        scheduler.waiting_queue = [local]
        scheduler.disagg_decode_prebuilt_queue = [transferred]
        scheduler.disagg_decode_prealloc_queue.queue = []
        scheduler.disagg_decode_prealloc_queue.retracted_queue = []
        scheduler.disagg_decode_transfer_queue.queue = []

        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
            scheduler.abort_request(AbortReq(rid="a-"))

        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(scheduler.disagg_decode_prebuilt_queue, [])
        self.assertEqual([c.args[0] for c in release.call_args_list], [transferred])
        self.assertEqual(
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_count, 2
        )

    def test_prebuilt_requests_keep_the_scheduler_busy(self):
        scheduler = self._scheduler()
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler._pp_microbatches_drained = MagicMock(return_value=True)
        scheduler.grammar_manager.grammar_queue = []
        scheduler.disagg_decode_prealloc_queue.queue = []
        scheduler.disagg_decode_prealloc_queue.retracted_queue = []
        scheduler.disagg_decode_transfer_queue.queue = []
        scheduler.disagg_decode_transfer_queue.has_pending_deferred_releases.return_value = False
        scheduler.disagg_prefill_bootstrap_queue = None
        scheduler.disagg_prefill_inflight_queue = None
        scheduler.enable_hierarchical_cache = False
        scheduler._engine_paused = False

        scheduler.disagg_decode_prebuilt_queue = [_req("transferred", False)]
        self.assertFalse(scheduler.is_fully_idle(ignore_waiting=True))
        scheduler.disagg_decode_prebuilt_queue = []
        self.assertTrue(scheduler.is_fully_idle(ignore_waiting=True))

    def test_pause_requeues_local_requests_for_local_prefill(self):
        scheduler = self._scheduler()
        scheduler._engine_paused = False
        scheduler.enable_overlap = False
        scheduler.enable_overlap_mlx = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.req_to_token_pool = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.hisparse_coordinator = None
        scheduler.decode_offload_manager = None
        scheduler.result_queue = deque()
        scheduler.metrics_reporter = MagicMock()
        scheduler.metrics_reporter.current_scheduler_metrics_enabled = False
        scheduler.kv_events_publisher = MagicMock()
        local, remote = _req("local", True), _req("remote", False)
        scheduler.running_batch = SimpleNamespace(
            reqs=[local, remote], batch_is_full=True
        )

        with patch("sglang.srt.managers.scheduler.retract_all"):
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(scheduler.waiting_queue, [local])
        scheduler.disagg_decode_prealloc_queue.hold_rebootstrap.assert_called_once_with(
            remote
        )


class TestDecodeLocalPrefillMemory(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", disaggregation_mode="decode"), role="test"
        )

    def test_retraction_skips_host_backup_for_local_request(self):
        for local, backed_up in ((True, False), (False, True)):
            with self.subTest(local=local):
                req = _req("req", local)
                with (
                    patch(
                        "sglang.srt.managers.schedule_batch.backup_kv_cache",
                        return_value=True,
                    ) as backup,
                    patch("sglang.srt.managers.schedule_batch.release_kv_cache"),
                    patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
                ):
                    release_req(
                        req=req,
                        remaing_req_count=0,
                        req_to_token_pool=MagicMock(),
                        token_to_kv_pool_allocator=MagicMock(),
                        tree_cache=MagicMock(),
                        hisparse_coordinator=None,
                    )
                self.assertEqual(backup.called, backed_up)

    def test_prealloc_budget_reserves_remaining_local_chunks(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.num_reserved_decode_tokens = 0
        queue.retracted_queue = []
        queue.transfer_queue = SimpleNamespace(queue=[])
        queue.token_to_kv_pool_allocator = MagicMock()
        queue.token_to_kv_pool_allocator.available_size.return_value = 1000
        queue._uses_swa_reservation = MagicMock(return_value=False)
        chunked = _req("chunked", True, num_tokens=600)
        chunked.refresh_fill_ids()
        chunked.extend_end = 200
        queue.scheduler = SimpleNamespace(
            enable_hisparse=False,
            running_batch=SimpleNamespace(reqs=[]),
            waiting_queue=[],
            disagg_decode_prebuilt_queue=[],
            last_batch=None,
            chunked_req=None,
        )

        self.assertEqual(queue._allocatable_token_budgets(), 1000)
        queue.scheduler.chunked_req = chunked
        self.assertEqual(queue._allocatable_token_budgets(), 600)

    def test_load_counts_local_requests_as_pending_prefill(self):
        local = SimpleNamespace(seqlen=300, num_matched_prefix_tokens=200)
        transferred = SimpleNamespace(seqlen=50)
        prealloc = SimpleNamespace(queue=[], retracted_queue=[])
        stats = MagicMock(gen_throughput=0.0, cache_hit_rate=0.0, utilization=0.0)
        inquirer = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            get_stats=lambda: stats,
            get_running_batch=lambda: SimpleNamespace(reqs=[]),
            get_waiting_queue=lambda: [local],
            waiting_queue_prefix_matched=lambda: True,
            get_recent_cache_hit_rate=lambda: 0.0,
            get_chunked_req=lambda: None,
            get_disagg_decode_prebuilt_queue=lambda: [transferred],
            get_disagg_decode_prealloc_queue=lambda: prealloc,
            get_disagg_decode_transfer_queue=lambda: SimpleNamespace(queue=[]),
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(
                    get_kv_token_stats=lambda: (1000, 0.5)
                )
            ),
            tp_worker=None,
            token_to_kv_pool_allocator=None,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            max_total_num_tokens=2000,
            max_running_requests=8,
            get_decode_moment_totals=lambda: (0,),
            get_total_prefill_uncached_tokens=lambda: 0,
            get_total_prefill_busy_us=lambda: 0,
        )
        inquirer.get_num_waiting_uncached_tokens = lambda: (
            SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer)
        )

        with get_parallel().override(dp_rank=0):
            snapshot = SchedulerLoadInquirer.get_loads(inquirer)

        # Both queues wait; only the local request still needs KV.
        self.assertEqual(snapshot.num_waiting_reqs, 2)
        self.assertEqual(snapshot.num_total_tokens, 1300)
        self.assertEqual(snapshot.num_waiting_uncached_tokens, 100)


if __name__ == "__main__":
    unittest.main()
