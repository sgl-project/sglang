import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.pd_disaggregation_hook import validate_decode_local_prefill
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    SchedulerDisaggregationDecodeMixin,
    _num_local_prefill_reqs_joining,
)
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.model_executor.forward_mode import ForwardMode
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
    """The flag is refused where the decode server cannot run a prefill chunk."""

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

    def test_rejects_unsupported_configs(self):
        for kwargs, reason in (
            ({"disaggregation_mode": "prefill"}, "requires --disaggregation-mode"),
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
        scheduler.enable_unified_cache_external_linker = False
        scheduler.enable_hisparse = False
        scheduler.uocr_runtime = None
        scheduler.invariant_checker = MagicMock()
        scheduler.waiting_queue = IndexedReqList(ListKind.WAITING)
        scheduler.running_reqs = KeyedReqList(ListKind.DECODING)
        scheduler.disagg_decode_prebuilt_queue = (
            IndexedReqList(ListKind.PD_PREBUILT, abort=scheduler._abort_prebuilt)
            if local_prefill
            else None
        )
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        scheduler.disagg_decode_transfer_queue = MagicMock()
        scheduler.tree_cache = MagicMock()
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        scheduler.scheduler_stage_metrics = None
        scheduler.decode_offload_manager = None
        scheduler.hisparse_coordinator = None
        scheduler.dllm_config = None
        scheduler.request_retirer = MagicMock()
        return scheduler

    def test_local_request_waits_for_prefill_on_this_server(self):
        """A marked request must never enter the preallocation path: it has no
        prefill peer to receive KV from."""
        scheduler = self._scheduler()
        local, remote = _req("local", True), _req("remote", False)

        scheduler._add_request_to_queue(local)
        scheduler._add_request_to_queue(remote)

        self.assertEqual(list(scheduler.waiting_queue), [local])
        scheduler.disagg_decode_prealloc_queue.add.assert_called_once_with(
            remote, is_retracted=False
        )

    def test_retracted_local_request_is_prefilled_again(self):
        scheduler = self._scheduler()
        local = _req("local", True)

        scheduler._add_request_to_queue(local, is_retracted=True)

        self.assertEqual(list(scheduler.waiting_queue), [local])
        scheduler.disagg_decode_prealloc_queue.add.assert_not_called()

    def test_transferred_requests_skip_the_local_waiting_queue(self):
        """With local prefill on, waiting_queue is the aggregated prefill queue,
        so a transferred request admitted there would be prefilled twice."""
        scheduler = self._scheduler()
        local = _req("local", True)
        move([local], scheduler.waiting_queue)
        transferred = _req("transferred", False)
        scheduler.enable_decode_hicache = False
        scheduler.polling_count = 0
        scheduler.polling_interval = 1
        scheduler.tree_cache.linker = None
        prealloc = scheduler.disagg_decode_prealloc_queue
        prealloc.retracted_queue = []
        prealloc.move_resumed = lambda dst: []
        transfer = scheduler.disagg_decode_transfer_queue
        transfer.move_transferred = lambda dst: (
            move([transferred], dst),
            [transferred],
        )[1]
        scheduler._preallocate_new_requests = MagicMock()

        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        self.assertEqual(list(scheduler.waiting_queue), [local])
        self.assertEqual(list(scheduler.disagg_decode_prebuilt_queue), [transferred])

    def test_without_local_prefill_transferred_requests_use_waiting_queue(self):
        scheduler = self._scheduler(local_prefill=False)
        transferred = _req("transferred", False)
        scheduler.enable_decode_hicache = False
        scheduler.polling_count = 0
        scheduler.polling_interval = 1
        scheduler.tree_cache.linker = None
        prealloc = scheduler.disagg_decode_prealloc_queue
        prealloc.retracted_queue = []
        prealloc.move_resumed = lambda dst: []
        transfer = scheduler.disagg_decode_transfer_queue
        transfer.move_transferred = lambda dst: (
            move([transferred], dst),
            [transferred],
        )[1]
        scheduler._preallocate_new_requests = MagicMock()

        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        self.assertEqual(list(scheduler.waiting_queue), [transferred])

    def test_without_local_prefill_the_decode_path_is_unchanged(self):
        scheduler = self._scheduler(local_prefill=False)
        running_batch = MagicMock()
        running_batch.is_empty.return_value = True
        scheduler.get_new_prebuilt_batch = MagicMock(return_value=None)
        scheduler.get_next_batch_to_run = MagicMock()
        scheduler.dp_attn_adapter = MagicMock()
        scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.return_value = None
        scheduler.ngram_embedding_manager = MagicMock()
        scheduler.ngram_embedding_manager.prepare_for_forward.return_value = None

        plan = scheduler.get_next_disagg_decode_batch_to_run(
            running_batch=running_batch, last_batch=MagicMock()
        )

        self.assertIsNone(plan.batch_to_run)
        scheduler.get_next_batch_to_run.assert_not_called()

    def test_prebuilt_admission_counts_local_requests_joining(self):
        """Local prefill requests from the last extend batch and the chunked
        request join the running batch after prebuilt admission in the same
        step; counting them keeps the batch within --max-running-requests."""
        scheduler = self._scheduler()
        scheduler.max_running_requests = 4
        scheduler.req_to_token_pool = SimpleNamespace(size=8)
        scheduler.token_to_kv_pool_allocator = None
        scheduler.enable_overlap = False
        scheduler.model_config = None
        scheduler.spec_algorithm = None
        chunked = _req("chunked", True)
        move([chunked], scheduler.chunked_reqs)
        last_batch = SimpleNamespace(
            reqs=[_req("prefilled", True), chunked], forward_mode=ForwardMode.EXTEND
        )
        transferred = [_req(f"t{i}", False) for i in range(3)]
        for req in transferred:
            req.kv.req_pool_idx = 0
        move(transferred, scheduler.disagg_decode_prebuilt_queue)
        running_batch = SimpleNamespace(batch_size=lambda: 1)

        with (
            patch("sglang.srt.disaggregation.decode.ScheduleBatch.init_new"),
            patch("sglang.srt.managers.schedule_batch.Req.init_next_round_input"),
        ):
            scheduler.get_next_disagg_decode_batch_to_run = MagicMock()
            joining = _num_local_prefill_reqs_joining(
                last_batch=last_batch, chunked_req=chunked
            )
            scheduler._get_new_prebuilt_batch(
                running_batch=running_batch, num_joining_reqs=joining
            )

        # 4 slots: 1 running, 2 local requests joining, so 1 transferred fits.
        self.assertEqual(joining, 2)
        self.assertEqual(len(scheduler.disagg_decode_prebuilt_queue), 2)
        self.assertEqual(len(scheduler.running_reqs), 1)

    def test_prebuilt_admission_counts_mixed_decode_rows_once(self):
        """A mixed chunk step empties the running batch and carries its decode
        rows in the MIXED last batch, so every request of that batch joins at
        the merge; excluding the rows already listed as running would admit
        transferred requests past --max-running-requests."""
        scheduler = self._scheduler()
        decoding = [_req(f"d{i}", False) for i in range(2)]
        move(decoding, scheduler.running_reqs)
        last_batch = SimpleNamespace(
            reqs=[_req("prefilled", True), *decoding], forward_mode=ForwardMode.MIXED
        )

        joining = _num_local_prefill_reqs_joining(
            last_batch=last_batch, chunked_req=None
        )

        self.assertEqual(joining, 3)

    def test_abort_releases_kv_only_for_transferred_requests(self):
        """A queued local request owns no KV; a transferred one owns its row and
        pages and keeps them in the cache."""
        scheduler = self._scheduler()
        scheduler.waiting_queue = IndexedReqList(
            ListKind.WAITING, abort=scheduler._abort_waiting
        )
        local, transferred = _req("a-local", True), _req("a-transferred", False)
        move([local], scheduler.waiting_queue)
        move([transferred], scheduler.disagg_decode_prebuilt_queue)
        transferred.kv.req_pool_idx = 0
        send_abort_output = MagicMock()
        scheduler.request_retirer = RequestRetirer(
            tree_cache=scheduler.tree_cache,
            beam_coordinator=MagicMock(),
            hisparse_coordinator=None,
            uocr_runtime=None,
            send_abort_output=send_abort_output,
            stream_output=MagicMock(),
        )
        recv_req = AbortReq(rid="a-")

        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache"
        ) as release:
            for request_list in (
                scheduler.waiting_queue,
                scheduler.disagg_decode_prebuilt_queue,
            ):
                request_list.abort(recv_req)

        self.assertEqual(len(scheduler.waiting_queue), 0)
        self.assertEqual(len(scheduler.disagg_decode_prebuilt_queue), 0)
        self.assertEqual([c.args[0] for c in release.call_args_list], [transferred])
        self.assertTrue(release.call_args.kwargs["checkpoint"])
        self.assertEqual(send_abort_output.call_count, 2)

    def test_pause_requeues_local_requests_for_local_prefill(self):
        """A retracted local request has no prefill peer to rebootstrap with; a
        transferred one waiting for a slot is retracted with the running ones."""
        scheduler = self._scheduler()
        scheduler.request_retirer = MagicMock()
        scheduler._add_request_to_queue = MagicMock()
        local, remote, waiting = (
            _req("local", True),
            _req("remote", False),
            _req("waiting", False),
        )
        move([waiting], scheduler.disagg_decode_prebuilt_queue)
        running_batch = SimpleNamespace(
            reqs=[local, remote], batch_is_full=True, filter_batch=MagicMock()
        )

        with patch(
            "sglang.srt.managers.scheduler.retract_all", return_value=[]
        ) as retract_all:
            scheduler._retract_batches([(running_batch, None)])

        self.assertEqual(
            set(retract_all.call_args.kwargs["reqs"]), {local, remote, waiting}
        )
        scheduler._add_request_to_queue.assert_called_once_with(local)
        self.assertEqual(
            [
                c.args[0]
                for c in scheduler.disagg_decode_prealloc_queue.hold_rebootstrap.call_args_list
            ],
            [remote, waiting],
        )

    def test_prebuilt_requests_keep_the_scheduler_busy(self):
        scheduler = self._scheduler()
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler._pp_microbatches_drained = MagicMock(return_value=True)
        scheduler.pause_state = SimpleNamespace(paused=False)
        scheduler.request_lists = lambda: {
            ListKind.WAITING: scheduler.waiting_queue,
            ListKind.PD_PREBUILT: scheduler.disagg_decode_prebuilt_queue,
        }

        move([_req("transferred", False)], scheduler.disagg_decode_prebuilt_queue)
        self.assertFalse(
            scheduler.is_fully_idle(for_health_check=True, ignore_waiting=True)
        )
        # Admitted, the request leaves the prebuilt list for a running one.
        move(list(scheduler.disagg_decode_prebuilt_queue), scheduler.running_reqs)
        self.assertTrue(
            scheduler.is_fully_idle(for_health_check=True, ignore_waiting=True)
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
        """A local request is prefilled again on this server; a host backup of
        its KV would be taken and never restored."""
        retirer = RequestRetirer(
            tree_cache=MagicMock(),
            beam_coordinator=MagicMock(),
            hisparse_coordinator=None,
            uocr_runtime=None,
            send_abort_output=MagicMock(),
            stream_output=MagicMock(),
        )
        for local, backed_up in ((True, False), (False, True)):
            with self.subTest(local=local):
                req = _req("req", local)
                with (
                    patch(
                        "sglang.srt.managers.scheduler_components.request_retirer.backup_kv_cache",
                        return_value=True,
                    ) as backup,
                    patch(
                        "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache"
                    ),
                ):
                    self.assertTrue(retirer.retract(req, offload_kv=True))
                self.assertEqual(backup.called, backed_up)

    def _prealloc_queue(self, local_prefill: bool) -> DecodePreallocQueue:
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.local_prefill = local_prefill
        queue.num_reserved_decode_tokens = 0
        queue.retracted_queue = []
        queue.transfer_queue = SimpleNamespace(queue=SimpleNamespace(items=lambda: []))
        queue.token_to_kv_pool_allocator = MagicMock()
        queue.token_to_kv_pool_allocator.available_size.return_value = 1000
        queue._uses_swa_reservation = lambda: False
        queue._budgets_count_from_zero = lambda: False
        queue._need_space_for_single_req = lambda *_: 0
        queue._host_staged_admission_stats = None
        queue.scheduler = SimpleNamespace(
            enable_hisparse=False,
            running_batch=SimpleNamespace(reqs=[]),
            waiting_queue=[],
            disagg_decode_prebuilt_queue=[_req("transferred", False)],
            last_batch=None,
            chunked_req=None,
        )
        return queue

    def test_prealloc_budget_reserves_remaining_local_chunks(self):
        """Transferred requests admitted while a local prefill is mid-chunk must
        leave room for the chunks it has not computed yet."""
        queue = self._prealloc_queue(local_prefill=True)
        chunked = _req("chunked", True, num_tokens=600)
        chunked.refresh_fill_ids()
        chunked.extend_end = 200

        self.assertEqual(queue._allocatable_token_budgets(), 1000)
        queue.scheduler.chunked_req = chunked
        self.assertEqual(queue._allocatable_token_budgets(), 600)

    def test_prealloc_counts_prebuilt_requests_only_with_local_prefill(self):
        for local_prefill, active in ((True, 1), (False, 0)):
            with self.subTest(local_prefill=local_prefill):
                queue = self._prealloc_queue(local_prefill)
                self.assertEqual(queue._active_req_count(), active)

    def test_load_counts_local_requests_as_pending_prefill(self):
        """Transferred requests already hold their KV; only the local request
        still needs tokens, and only its uncached part is prefill work."""
        local = SimpleNamespace(seqlen=300, num_matched_prefix_tokens=200)
        transferred = SimpleNamespace(seqlen=50)
        prealloc = SimpleNamespace(
            queue=SimpleNamespace(items=lambda: []),
            host_pool_blocked=False,
            transfer_queue=SimpleNamespace(num_host_staged_landed=0),
        )
        stats = MagicMock(
            gen_throughput=0.0,
            cache_hit_rate=0.0,
            utilization=0.0,
            num_grammar_queue_reqs=0,
            num_paused_reqs=0,
            num_retracted_reqs=0,
        )
        lists = {
            ListKind.PD_PREALLOC: [],
            ListKind.PD_RECEIVING: [],
            ListKind.PD_RETRACTED: [],
            ListKind.PD_PREBUILT: [transferred],
        }
        inquirer = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            get_decode_local_prefill=lambda: True,
            get_stats=lambda: stats,
            get_running_batch=lambda: SimpleNamespace(reqs=[]),
            get_waiting_queue=lambda: [local],
            get_request_lists=lambda: lists,
            waiting_queue_prefix_matched=lambda: True,
            get_recent_cache_hit_rate=lambda: 0.0,
            get_chunked_req=lambda: None,
            get_waiting_storage_span=None,
            get_disagg_decode_prealloc_queue=lambda: prealloc,
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

        self.assertEqual(snapshot.num_waiting_reqs, 2)
        self.assertEqual(snapshot.num_total_tokens, 1300)
        self.assertEqual(snapshot.num_waiting_uncached_tokens, 100)


if __name__ == "__main__":
    unittest.main()
