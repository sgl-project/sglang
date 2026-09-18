import types
import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    AbortReq,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    release_req,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import RetractionBackup, release_kv_cache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.session.session_controller import Session
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE = 1
KV_SIZE = 512
MAMBA_SIZE = 8
VOCAB_SIZE = 32000


def _build():
    server_args = ServerArgs(model_path="dummy", page_size=PAGE)
    server_args._mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, PAGE)
    server_args.max_mamba_cache_size = MAMBA_SIZE
    server_args.mamba_max_states_per_path = 2
    set_global_server_args_for_scheduler(server_args)

    full_attention_layer_ids = [3, 7]
    mamba_layer_ids = [i for i in range(8) if i not in full_attention_layer_ids]
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=64,
        n_groups=2,
        num_heads=4,
        head_dim=16,
        state_size=16,
        conv_kernel=4,
    )
    cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layer_ids)
    with torch.device("cpu"):
        req_to_token_pool = HybridReqToTokenPool(
            size=8,
            mamba_size=MAMBA_SIZE,
            mamba_spec_state_size=8,
            max_context_len=512,
            device="cpu",
            enable_memory_saver=False,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=True,
            speculative_num_draft_tokens=None,
        )
    kv_pool = HybridLinearKVPool(
        size=KV_SIZE,
        dtype=torch.bfloat16,
        page_size=PAGE,
        head_num=2,
        head_dim=16,
        full_attention_layer_ids=full_attention_layer_ids,
        device="cpu",
        enable_memory_saver=False,
        mamba_pool=req_to_token_pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=KV_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    cache = UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE,
            disable=False,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=True,
            eviction_policy="lru",
        )
    )
    sessions = SimpleNamespace(sessions={})
    observer = SchedulerPoolStatsObserver(
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
        session_controller=sessions,
        hisparse_coordinator=None,
        is_hybrid_swa=False,
        is_hybrid_ssm=True,
        enable_hisparse=False,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=KV_SIZE,
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
    )
    checker = SchedulerInvariantChecker(
        is_hybrid_swa=False,
        is_hybrid_ssm=True,
        disaggregation_mode=DisaggregationMode.NULL,
        page_size=PAGE,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=KV_SIZE,
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
        pool_stats_observer=observer,
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
        scheduler_stage_metrics=None,
    )
    return server_args, cache, allocator, req_to_token_pool, observer, checker


def _recv(rid, input_ids):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", input_ids),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(temperature=0, max_new_tokens=4),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=True,
        session_params=SessionParams(
            id="session-a",
            rid=None,
            offset=None,
            replace=False,
            drop_previous_output=False,
        ),
        lora_id=None,
        custom_logit_processor=None,
        return_sampling_mask=False,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        priority=None,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


def _prefill(req, cache, allocator, req_to_token_pool):
    key = RadixKey(array("q", req.origin_input_ids))
    match = cache.match_prefix(MatchPrefixParams(key=key, req=req, cow_mamba=True))
    prefix_len = len(match.device_indices)
    req.lock_receipt = cache.inc_lock_ref(match.last_device_node).to_dec_params()
    if req.kv.req_pool_idx is None:
        req_to_token_pool.alloc([req])
    total = len(req.origin_input_ids)
    if prefix_len:
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, prefix_len)), match.device_indices
        )
    if total > prefix_len:
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(prefix_len, total)),
            allocator.alloc(total - prefix_len),
        )
    req.output_ids = array("q")
    req.prefix_indices = match.device_indices
    req.last_node = match.last_device_node
    req.kv.cache_protected_len = (
        match.cache_protected_len
        if match.cache_protected_len is not None
        else prefix_len
    )
    req.kv.kv_committed_len = total
    req.kv.kv_allocated_len = total
    req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
    req.set_extend_range(prefix_len, total)
    req.kv.mamba_last_track_seqlen = 0
    if req.kv.mamba_next_track_idx is None:
        req.kv.mamba_next_track_idx = 0
    cache.cache_unfinished_req(req)


def _finish_first_turn(req, cache):
    req.finished_reason = FINISH_LENGTH(length=0)
    req.finished_len = 0
    release_kv_cache(req, cache)


def _scheduler_stub(cache):
    output = Mock()
    stub = SimpleNamespace(
        chunked_req=None,
        _pending_chunked_abort_req=None,
        mm_receiver=None,
        waiting_queue=[],
        tree_cache=cache,
        ipc_channels=SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=output)
        ),
        beam_coordinator=SimpleNamespace(retire_group=Mock()),
        disaggregation_mode=DisaggregationMode.NULL,
        disagg_prefill_pending_chunk_rids=set(),
        enable_hicache_storage=False,
        dllm_config=None,
        enable_overlap=False,
        result_queue=deque(),
        grammar_manager=SimpleNamespace(abort_requests=Mock()),
        ps=SimpleNamespace(pp_size=1),
        running_batch=SimpleNamespace(reqs=[]),
        last_batch=SimpleNamespace(reqs=[]),
    )
    stub._release_aborted_request = types.MethodType(
        Scheduler._release_aborted_request, stub
    )
    stub._release_dropped_waiting_req_mm_inputs = types.MethodType(
        Scheduler._release_dropped_waiting_req_mm_inputs, stub
    )
    stub.collect_inflight_reqs = types.MethodType(Scheduler.collect_inflight_reqs, stub)
    stub.clear_pending_chunk_send = types.MethodType(
        SchedulerDisaggregationPrefillMixin.clear_pending_chunk_send, stub
    )
    return stub


class TestSessionQueueAbort(CustomTestCase):
    def _setup_first_turn(self):
        (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
        ) = _build()
        session = Session(capacity_of_str_len=0, session_id="session-a", streaming=True)
        req1 = session.create_req(
            _recv("turn-1", list(range(16))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        _prefill(req1, cache, allocator, req_to_token_pool)
        _finish_first_turn(req1, cache)
        self.assertTrue(cache.session.has_slot(session.session_id))
        return (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        )

    def _assert_idle(self, observer, checker):
        pool_stats = observer.get_pool_stats()
        mamba_leak, mamba_msg = checker._check_mamba_pool(pool_stats)
        self.assertFalse(mamba_leak, mamba_msg)
        all_leak, all_messages = checker._check_all_pools(pool_stats)
        self.assertFalse(all_leak, all_messages)

    def test_queue_abort_of_restored_turn_finishes_and_releases_slot(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertIsNotNone(req.kv.req_pool_idx)

        scheduler = _scheduler_stub(cache)
        scheduler.waiting_queue.append(req)
        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        self.assertNotIn(session.session_id, cache.session.slots)
        self.assertEqual(cache.session.session_held_mamba_slots(), 0)
        self._assert_idle(observer, checker)

        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_queue_abort_of_retracted_turn_finishes(self):
        (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertIsNotNone(req.kv.req_pool_idx)
        release_req(
            req=req,
            remaing_req_count=1,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            hisparse_coordinator=None,
            offload_kv=False,
        )
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.req_pool_idx)

        scheduler = _scheduler_stub(cache)
        scheduler.waiting_queue.append(req)
        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertTrue(req.finished())
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_retracted_queue_abort_in_decode_mode_finishes(self):
        """PD decode: a retracted req (KV already nuked by release_req) sits in
        disagg_decode_prealloc_queue.retracted_queue with a retraction backup.
        Aborting it must stamp FINISH_ABORT and clear the session's inflight
        turn."""
        (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        release_req(
            req=req,
            remaing_req_count=1,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            hisparse_coordinator=None,
            offload_kv=False,
        )
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)

        req.kv.retraction_backup = RetractionBackup(cpu_tensors=torch.empty(0))
        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[req], held_rebootstrap_reqs=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.retracted_queue, [])
        self.assertIsNone(req.kv.retraction_backup)
        send_output.assert_called_once()
        self.assertIsInstance(send_output.call_args[0][0], AbortReq)
        self.assertEqual(send_output.call_args[0][0].rid, req.rid)
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_prealloc_queue_abort_in_decode_mode_finishes_session_turn(self):
        """PD decode: a prealloc-queue req has no KV yet, so release_kv_cache
        never runs and nothing clears the session inflight turn. The abort
        path must stamp FINISH_ABORT and the queue finish must clear the
        turn."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)

        decode_req = SimpleNamespace(req=req, kv_receiver=Mock())
        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[decode_req], retracted_queue=[], held_rebootstrap_reqs=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        decode_req.kv_receiver.abort.assert_called_once()
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.message, "Aborted")
        # The session stays inflight until the queue actually finishes the
        # request — a follow-up must not be admitted before that.
        self.assertTrue(session.has_unfinished_request())
        # Removal + output streaming are pop_preallocated's job, not abort's.
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.queue, [decode_req])
        send_output.assert_not_called()

        # Drive the prealloc queue's abort-scan: it finishes the req, clears
        # the receiver, and unblocks the session.
        prealloc_queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        prealloc_queue.queue = [decode_req]
        prealloc_queue.pending_reqs = []
        prealloc_queue.retracted_queue = []
        prealloc_queue.pp_size = 1
        prealloc_queue._resolve_pending_reqs = MagicMock()
        prealloc_queue._update_handshake_waiters = MagicMock()
        prealloc_queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        prealloc_queue._uses_swa_reservation = MagicMock(return_value=False)
        prealloc_queue._allocatable_token_budgets = MagicMock(return_value=0)
        prealloc_queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        prealloc_queue.scheduler = scheduler

        preallocated, failed = prealloc_queue.pop_preallocated()

        self.assertEqual(failed, [decode_req])
        self.assertEqual(prealloc_queue.queue, [])
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_prealloc_handshake_failure_of_user_aborted_req_still_cleans_up(self):
        """PD decode: a user-aborted prealloc-queue req that polls Failed must
        still run failure_exception() -- the receiver's transfer-record cleanup
        -- while keeping the FINISH_ABORT stamp (no 500 restamp, no failure
        metric)."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )

        decode_req = SimpleNamespace(req=req, kv_receiver=Mock())
        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[decode_req], retracted_queue=[], held_rebootstrap_reqs=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        decode_req.kv_receiver.failure_exception.side_effect = RuntimeError("aborted")

        prealloc_queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        prealloc_queue.queue = [decode_req]
        prealloc_queue.pp_size = 1
        prealloc_queue.gloo_group = MagicMock()
        prealloc_queue.tp_rank = 0
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = Mock()
        prealloc_queue.scheduler = scheduler

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ):
            prealloc_queue._update_handshake_waiters()

        decode_req.kv_receiver.failure_exception.assert_called_once_with()
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertIsNone(req.finished_reason.status_code)
        scheduler.metrics_collector.increment_bootstrap_failed_reqs.assert_not_called()

    def test_transfer_queue_abort_in_decode_mode_finishes_session_turn(self):
        """PD decode: a transfer-queue req holds real KV; abort must stamp
        FINISH_ABORT and clear the session turn while leaving the KV for
        pop_transferred's Failed branch to release."""
        (
            _server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)

        decode_req = SimpleNamespace(req=req, kv_receiver=Mock())
        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[], held_rebootstrap_reqs=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[decode_req])
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        decode_req.kv_receiver.abort.assert_called_once()
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.message, "Aborted")
        # The session stays inflight until pop_transferred finishes the req
        # and releases its KV.
        self.assertTrue(session.has_unfinished_request())
        self.assertEqual(scheduler.disagg_decode_transfer_queue.queue, [decode_req])
        send_output.assert_not_called()
        self.assertIsNotNone(req.kv.req_pool_idx)

        # Drive the transfer queue's Failed branch (the aborted receiver
        # polls Failed): releases KV, unblocks the session, and must NOT
        # count a user abort as a transfer failure.
        # failure_exception() is also the receiver's transfer-record cleanup,
        # so it must still run for a user abort, with its exception swallowed.
        # (pop_transferred clears decode_req.kv_receiver, so keep a reference.)
        receiver = decode_req.kv_receiver
        receiver.failure_exception.side_effect = RuntimeError("aborted")
        decode_req.metadata_buffer_index = 3
        decode_req.hicache_restore_status = HiCacheRestoreResult.READY
        transfer_queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer_queue.queue = [decode_req]
        transfer_queue.enable_staging = False
        transfer_queue.enable_deferred_kv_release = False
        transfer_queue.gloo_group = MagicMock()
        transfer_queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        transfer_queue.tp_rank = 0
        transfer_queue.tree_cache = cache
        transfer_queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        transfer_queue.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        transfer_queue._clean_hicache_prefetch_resources = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = Mock()
        transfer_queue.scheduler = scheduler

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ):
            transferred = transfer_queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(transfer_queue.queue, [])
        receiver.failure_exception.assert_called_once_with()
        self.assertIsNone(req.finished_reason.status_code)
        scheduler.metrics_collector.increment_transfer_failed_reqs.assert_not_called()
        self.assertFalse(session.has_unfinished_request())
        self.assertIsNone(req.kv.req_pool_idx)
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_transfer_failure_of_statusless_internal_abort_counts_as_failure(self):
        """A statusless internal FINISH_ABORT (e.g. a grammar accept error)
        is not a client cancel either: abort_request records the client
        origin explicitly, so a bare stamp must still log, restamp, and
        count the transfer failure."""
        (
            _server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)
        # An internal failure already stamped a statusless abort.
        req.finished_reason = FINISH_ABORT()

        decode_req = SimpleNamespace(req=req, kv_receiver=Mock())
        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        receiver = decode_req.kv_receiver
        receiver.failure_exception.side_effect = RuntimeError("boom")
        decode_req.metadata_buffer_index = 3
        decode_req.hicache_restore_status = HiCacheRestoreResult.READY
        transfer_queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer_queue.queue = [decode_req]
        transfer_queue.enable_staging = False
        transfer_queue.enable_deferred_kv_release = False
        transfer_queue.gloo_group = MagicMock()
        transfer_queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        transfer_queue.tp_rank = 0
        transfer_queue.tree_cache = cache
        transfer_queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        transfer_queue.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        transfer_queue._clean_hicache_prefetch_resources = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = Mock()
        transfer_queue.scheduler = scheduler

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ):
            transferred = transfer_queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(transfer_queue.queue, [])
        receiver.failure_exception.assert_called_once_with()
        self.assertEqual(req.finished_reason.status_code, 500)
        self.assertIn("Decode transfer failed", req.finished_reason.message)
        scheduler.metrics_collector.increment_transfer_failed_reqs.assert_called_once()
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)

    def test_transfer_failure_of_internally_aborted_req_counts_as_failure(self):
        """An internal 500 FINISH_ABORT (e.g. corruption) is not a user
        cancel: when the receiver then polls Failed, the transfer error log,
        the restamp with the transfer error, and the failure metric must
        still fire -- treating it as a user abort would underreport real
        transfer failures."""
        (
            _server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)
        # An internal failure already stamped a 500 abort.
        req.finished_reason = FINISH_ABORT("corruption detected", 500)

        decode_req = SimpleNamespace(req=req, kv_receiver=Mock())
        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        receiver = decode_req.kv_receiver
        receiver.failure_exception.side_effect = RuntimeError("boom")
        decode_req.metadata_buffer_index = 3
        decode_req.hicache_restore_status = HiCacheRestoreResult.READY
        transfer_queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer_queue.queue = [decode_req]
        transfer_queue.enable_staging = False
        transfer_queue.enable_deferred_kv_release = False
        transfer_queue.gloo_group = MagicMock()
        transfer_queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        transfer_queue.tp_rank = 0
        transfer_queue.tree_cache = cache
        transfer_queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        transfer_queue.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        transfer_queue._clean_hicache_prefetch_resources = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = Mock()
        transfer_queue.scheduler = scheduler

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ):
            transferred = transfer_queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(transfer_queue.queue, [])
        receiver.failure_exception.assert_called_once_with()
        self.assertEqual(req.finished_reason.status_code, 500)
        self.assertIn("Decode transfer failed", req.finished_reason.message)
        scheduler.metrics_collector.increment_transfer_failed_reqs.assert_called_once()
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)

    def test_queue_full_reject_of_incoming_turn_clears_session(self):
        """A queue-full reject of the incoming request (not a priority
        eviction) must run the dropped-request cleanup for it: the session's
        inflight marker clears and its req-owned early mamba alloc returns,
        or the session bricks for later turns."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-full", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        assert session.has_unfinished_request()

        scheduler = _scheduler_stub(cache)
        scheduler.max_queued_requests = 0
        scheduler.enable_priority_scheduling = False
        scheduler._release_dropped_waiting_req_mamba_slot = types.MethodType(
            Scheduler._release_dropped_waiting_req_mamba_slot, scheduler
        )

        rejected = Scheduler._abort_on_queued_limit(scheduler, req)

        assert rejected is True
        self.assertFalse(session.has_unfinished_request())
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()

    def test_queue_full_reject_of_non_streaming_session_turn_is_terminal(self):
        """A non-streaming session's dropped turn must be stamped terminal:
        req_nodes only unblocks appends and session close once the req is
        finished, and abort_req alone clears just the streaming inflight
        marker, so an unstamped drop bricks the session."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
        ) = _build()
        session = Session(
            capacity_of_str_len=0, session_id="session-a", streaming=False
        )
        req = session.create_req(
            _recv("turn-1", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        assert session.req_nodes
        self.assertTrue(session.has_unfinished_request())

        scheduler = _scheduler_stub(cache)
        scheduler.max_queued_requests = 0
        scheduler.enable_priority_scheduling = False
        scheduler._release_dropped_waiting_req_mamba_slot = types.MethodType(
            Scheduler._release_dropped_waiting_req_mamba_slot, scheduler
        )

        rejected = Scheduler._abort_on_queued_limit(scheduler, req)

        assert rejected is True
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, 503)
        self.assertFalse(session.has_unfinished_request())
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()

    def test_abort_all_over_multiple_retracted_and_queued_decode_reqs(self):
        """abort_all must hit retracted, prealloc, and transfer reqs in one
        pass, tolerating plain Reqs with no session attached."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            _session,
        ) = self._setup_first_turn()
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)

        def _plain_req(rid):
            return Req(
                rid=rid,
                origin_input_text="",
                origin_input_ids=array("q", [1, 2, 3]),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=4),
                vocab_size=VOCAB_SIZE,
            )

        retracted_req = _plain_req("retracted-1")
        retracted_req.kv.retraction_backup = RetractionBackup(
            cpu_tensors=torch.empty(0)
        )
        prealloc_req = _plain_req("prealloc-1")
        transfer_req = _plain_req("transfer-1")
        prealloc_decode_req = SimpleNamespace(req=prealloc_req, kv_receiver=Mock())
        transfer_decode_req = SimpleNamespace(req=transfer_req, kv_receiver=Mock())

        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[prealloc_decode_req],
            retracted_queue=[retracted_req],
            held_rebootstrap_reqs=[],
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[transfer_decode_req]
        )
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        Scheduler.abort_request(scheduler, AbortReq(rid="", abort_all=True))

        for req in (retracted_req, prealloc_req, transfer_req):
            self.assertTrue(req.finished())
            self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.retracted_queue, [])
        self.assertIsNone(retracted_req.kv.retraction_backup)
        send_output.assert_called_once()
        self.assertIsInstance(send_output.call_args[0][0], AbortReq)
        self.assertEqual(send_output.call_args[0][0].rid, "retracted-1")
        prealloc_decode_req.kv_receiver.abort.assert_called_once()
        transfer_decode_req.kv_receiver.abort.assert_called_once()

    def test_retracted_queue_abort_retry_after_send_failure(self):
        """If send_output raises, the req must stay fully intact in
        retracted_queue so a retry re-runs idempotent steps: the backup
        discard happens only after a successful send."""
        (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        release_req(
            req=req,
            remaing_req_count=1,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            hisparse_coordinator=None,
            offload_kv=False,
        )
        req.kv.retraction_backup = RetractionBackup(cpu_tensors=torch.empty(0))

        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[req], held_rebootstrap_reqs=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        send_output.side_effect = [RuntimeError("ipc"), None]

        with self.assertRaises(RuntimeError):
            Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))
        self.assertIsNotNone(req.kv.retraction_backup)
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.retracted_queue, [req])

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.retracted_queue, [])
        self.assertIsNone(req.kv.retraction_backup)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)

    def test_abort_all_partial_retracted_failure_keeps_failed_entry_retryable(self):
        """A multi-entry retracted abort must commit each removal in place:
        if send_output raises for entry B after A succeeded, the queue must
        already hold only [B] so a retry does not re-run A's cleanup."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            _session,
        ) = self._setup_first_turn()
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)

        def _plain_req(rid):
            req = Req(
                rid=rid,
                origin_input_text="",
                origin_input_ids=array("q", [1, 2, 3]),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=4),
                vocab_size=VOCAB_SIZE,
            )
            req.kv.retraction_backup = RetractionBackup(cpu_tensors=torch.empty(0))
            return req

        req_a = _plain_req("retracted-a")
        req_b = _plain_req("retracted-b")

        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[req_a, req_b], held_rebootstrap_reqs=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        send_output.side_effect = [None, RuntimeError("ipc down")]

        with self.assertRaises(RuntimeError):
            Scheduler.abort_request(scheduler, AbortReq(rid="", abort_all=True))
        self.assertEqual(
            scheduler.disagg_decode_prealloc_queue.retracted_queue, [req_b]
        )
        self.assertIsNone(req_a.kv.retraction_backup)
        self.assertIsNotNone(req_b.kv.retraction_backup)

        send_output.side_effect = None
        Scheduler.abort_request(scheduler, AbortReq(rid="", abort_all=True))
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.retracted_queue, [])
        self.assertIsNone(req_b.kv.retraction_backup)
        self.assertTrue(req_a.finished())
        self.assertTrue(req_b.finished())
        self.assertEqual(send_output.call_count, 3)

    def test_prefill_bootstrap_queue_abort_finishes_session_turn(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(req.kv.req_pool_idx)
        req.disagg_kv_sender = Mock()
        req.bootstrap_room = 0
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda **kwargs: None)
        )

        scheduler = _scheduler_stub(cache)
        scheduler.ps.tp_rank = 0
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[req])
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = Mock()
        scheduler.req_to_metadata_buffer_idx_allocator = None

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertIsNone(req.finished_reason.status_code)
        self.assertFalse(session.has_unfinished_request())
        req.disagg_kv_sender.abort.assert_called_once()
        self.assertEqual(scheduler.disagg_prefill_bootstrap_queue.queue, [req])

        SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure(scheduler, req)

        self.assertIsNone(req.finished_reason.status_code)
        scheduler.metrics_collector.increment_bootstrap_failed_reqs.assert_not_called()
        scheduler.output_streamer.stream_output.assert_called_once()
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)

    def test_prefill_inflight_queue_abort_stamps_user_abort(self):
        (
            _server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)
        req.disagg_kv_sender = Mock()
        req.bootstrap_room = 0
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda **kwargs: None)
        )

        scheduler = _scheduler_stub(cache)
        scheduler.ps.tp_rank = 0
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler.disagg_prefill_inflight_queue = [req]
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = Mock()

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertIsNone(req.finished_reason.status_code)
        req.disagg_kv_sender.abort.assert_called_once()
        # The sender's failure_exception() is also its cleanup; it must still
        # run for a user abort, with its exception swallowed.
        req.disagg_kv_sender.failure_exception.side_effect = RuntimeError("aborted")

        exc = SchedulerDisaggregationPrefillMixin.handle_inflight_transfer_failure(
            scheduler, req
        )

        req.disagg_kv_sender.failure_exception.assert_called_once_with()
        self.assertIsNone(exc)
        self.assertIsNone(req.finished_reason.status_code)
        scheduler.metrics_collector.increment_transfer_failed_reqs.assert_not_called()
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)

    def test_natural_prefill_transfer_failure_still_reports_500(self):
        (
            _server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)
        req.disagg_kv_sender = Mock()
        req.bootstrap_room = 0
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda **kwargs: None)
        )

        scheduler = _scheduler_stub(cache)
        scheduler.ps.tp_rank = 0
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler.disagg_prefill_inflight_queue = [req]
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = Mock()

        exc = SchedulerDisaggregationPrefillMixin.handle_inflight_transfer_failure(
            scheduler, req
        )

        self.assertIsNone(exc)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, 500)
        scheduler.metrics_collector.increment_transfer_failed_reqs.assert_called_once()
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)

    def test_prefill_inflight_failure_of_internally_aborted_req_counts_as_failure(self):
        """An internal 500 FINISH_ABORT on the prefill side is not a user
        cancel either: the sender's exception must propagate to the caller,
        the req is restamped with the transfer error, and the failure metric
        fires."""
        (
            _server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)
        req.disagg_kv_sender = Mock()
        req.bootstrap_room = 0
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda **kwargs: None)
        )
        # An internal failure already stamped a 500 abort.
        req.finished_reason = FINISH_ABORT("corruption detected", 500)

        scheduler = _scheduler_stub(cache)
        scheduler.ps.tp_rank = 0
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler.disagg_prefill_inflight_queue = [req]
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = Mock()
        boom = RuntimeError("boom")
        req.disagg_kv_sender.failure_exception.side_effect = boom

        exc = SchedulerDisaggregationPrefillMixin.handle_inflight_transfer_failure(
            scheduler, req
        )

        self.assertIs(exc, boom)
        req.disagg_kv_sender.failure_exception.assert_called_once_with()
        self.assertEqual(req.finished_reason.status_code, 500)
        self.assertIn("Prefill transfer failed", req.finished_reason.message)
        scheduler.metrics_collector.increment_transfer_failed_reqs.assert_called_once()
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)

    def test_held_rebootstrap_abort_targeted_and_abort_all(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req_a = SimpleNamespace(
            rid="held-a",
            session=None,
            multimodal_inputs=None,
            finished_reason=None,
            return_logprob=False,
            weight_version_events=[],
            output_ids=array("q"),
        )
        req_b = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertTrue(session.has_unfinished_request())

        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[],
            retracted_queue=[],
            held_rebootstrap_reqs=[req_a, req_b],
            add=Mock(),
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output

        Scheduler.abort_request(scheduler, AbortReq(rid=req_a.rid))
        held = scheduler.disagg_decode_prealloc_queue.held_rebootstrap_reqs
        self.assertEqual(held, [req_b])
        self.assertIsInstance(req_a.finished_reason, FINISH_ABORT)
        send_output.assert_called_once()
        self.assertEqual(send_output.call_args[0][0].rid, req_a.rid)
        self.assertTrue(session.has_unfinished_request())

        Scheduler.abort_request(scheduler, AbortReq(rid="", abort_all=True))
        self.assertEqual(held, [])
        self.assertIsInstance(req_b.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(send_output.call_count, 2)

        DecodePreallocQueue.enqueue_held_rebootstrap(
            scheduler.disagg_decode_prealloc_queue
        )
        scheduler.disagg_decode_prealloc_queue.add.assert_not_called()
        self._assert_idle(observer, checker)

    def test_dllm_queue_abort_stamps_finish_abort_and_clears_session(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertTrue(session.has_unfinished_request())

        scheduler = _scheduler_stub(cache)
        scheduler.dllm_config = object()
        scheduler.dllm_manager = SimpleNamespace(
            pop_aborted_reqs=lambda abort_all, rid: [req]
        )
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        send_output.assert_called_once()
        self.assertIsInstance(send_output.call_args[0][0], AbortReq)
        self.assertEqual(send_output.call_args[0][0].rid, req.rid)
        self._assert_idle(observer, checker)

    def test_dllm_queue_abort_skips_already_finished_req(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        # Finished in the last forward but not yet dropped by
        # filter_finished_reqs(); the abort must not restamp it.
        req.finished_reason = FINISH_LENGTH(length=1)
        req.multimodal_inputs = Mock()

        scheduler = _scheduler_stub(cache)
        scheduler.dllm_config = object()
        scheduler.dllm_manager = SimpleNamespace(
            pop_aborted_reqs=lambda abort_all, rid: [req]
        )
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertIsNotNone(req.multimodal_inputs)
        send_output.assert_not_called()

    def test_dllm_abort_defers_req_with_pending_overlap_result(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        self.assertTrue(session.has_unfinished_request())

        from sglang.srt.dllm.mixin.scheduler import DllmManager

        scheduler = _scheduler_stub(cache)
        scheduler.enable_overlap = True
        scheduler.result_queue = deque([(SimpleNamespace(reqs=[req]), None)])
        scheduler.dllm_config = object()
        scheduler.dllm_manager = DllmManager()
        scheduler.dllm_manager.staging_queue = [req]
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertFalse(req.finished())
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertEqual(scheduler.dllm_manager.staging_queue, [req])
        self.assertTrue(session.has_unfinished_request())
        send_output.assert_not_called()
        self.assertIsNotNone(req.kv.req_pool_idx)

        # The same req with no pending result takes the direct abort path.
        req.to_finish = None
        scheduler.enable_overlap = False
        scheduler.result_queue = deque()

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        send_output.assert_called_once()

    def test_dllm_deferred_abort_keeps_timeout_reason(self):
        """A running-timeout abort deferred for a pending overlap result must
        keep its message + 503 on to_finish; a bare FINISH_ABORT would stream
        a generic statusless abort to the client."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)

        from sglang.srt.dllm.mixin.scheduler import DllmManager

        scheduler = _scheduler_stub(cache)
        scheduler.enable_overlap = True
        scheduler.result_queue = deque([(SimpleNamespace(reqs=[req]), None)])
        scheduler.dllm_config = object()
        scheduler.dllm_manager = DllmManager()
        scheduler.dllm_manager.staging_queue = [req]

        Scheduler.abort_request(
            scheduler,
            AbortReq(rid=req.rid, abort_message="Request running timeout reached."),
        )

        self.assertFalse(req.finished())
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertEqual(req.to_finish.message, "Request running timeout reached.")
        self.assertEqual(req.to_finish.status_code, 503)
        self.assertEqual(scheduler.dllm_manager.staging_queue, [req])

    def test_dllm_direct_abort_keeps_timeout_reason(self):
        """A running-timeout abort on a dLLM queued req with no pending
        overlap result takes the direct branch; the streamed abort must keep
        the timeout message + 503 rather than a generic statusless stamp."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)

        from sglang.srt.dllm.mixin.scheduler import DllmManager

        scheduler = _scheduler_stub(cache)
        scheduler.dllm_config = object()
        scheduler.dllm_manager = DllmManager()
        scheduler.dllm_manager.staging_queue = [req]

        Scheduler.abort_request(
            scheduler,
            AbortReq(rid=req.rid, abort_message="Request running timeout reached."),
        )

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(
            req.finished_reason.message, "Request running timeout reached."
        )
        self.assertEqual(req.finished_reason.status_code, 503)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()

    def _dllm_result_stub(self, fdfo, empty=True):
        return SimpleNamespace(
            copy_done=None,
            next_token_ids=(
                [torch.zeros(4, dtype=torch.long)]
                if fdfo
                else [torch.tensor([], dtype=torch.long)]
            ),
            accept_length_per_req_cpu=[0] if fdfo else None,
            dllm_algo_state=None,
            can_run_cuda_graph=False,
        )

    def _run_process_batch_result_dllm(self, session, cache, req, fdfo):
        from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin

        scheduler = _scheduler_stub(cache)
        scheduler.dllm_config = SimpleNamespace(
            first_done_first_out_mode=fdfo, block_size=4
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=Mock(), free_group_end=Mock()
        )
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        scheduler.metrics_reporter = SimpleNamespace(
            num_generated_tokens=0, report_prefill_stats=Mock()
        )
        batch = SimpleNamespace(
            batch_size=lambda: 1,
            reqs=[req],
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        with patch("sglang.srt.dllm.mixin.scheduler.release_kv_cache") as release_mock:
            SchedulerDllmMixin.process_batch_result_dllm(
                scheduler, batch, self._dllm_result_stub(fdfo)
            )
        return scheduler, release_mock

    def test_dllm_process_result_finalizes_to_finish_on_empty_result(self):
        for fdfo in (False, True):
            with self.subTest(fdfo=fdfo):
                (
                    _server_args,
                    cache,
                    _allocator,
                    _req_to_token_pool,
                    _observer,
                    _checker,
                    session,
                ) = self._setup_first_turn()
                req = session.create_req(
                    _recv("turn-2", list(range(32, 48))),
                    tokenizer=None,
                    vocab_size=VOCAB_SIZE,
                )
                req.to_finish = FINISH_ABORT()
                req.time_stats = SimpleNamespace(
                    set_completion_time=Mock(), set_first_token_time=Mock()
                )

                scheduler, release_mock = self._run_process_batch_result_dllm(
                    session, cache, req, fdfo
                )

                self.assertTrue(req.finished())
                self.assertIsInstance(req.finished_reason, FINISH_ABORT)
                self.assertIsNone(req.to_finish)
                release_mock.assert_called_once()
                scheduler.output_streamer.stream_output.assert_called_once_with(
                    [req], False
                )

    def test_dllm_process_result_finalizes_deferred_abort_on_empty_token_list(self):
        """A non-FDFO algorithm can return a globally empty token list; a
        deferred (overlap) abort must still finalize and release KV instead of
        stranding with its session and KV held."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.to_finish = FINISH_ABORT()
        req.time_stats = SimpleNamespace(
            set_completion_time=Mock(), set_first_token_time=Mock()
        )

        from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin

        scheduler = _scheduler_stub(cache)
        scheduler.dllm_config = SimpleNamespace(
            first_done_first_out_mode=False, block_size=4
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=Mock(), free_group_end=Mock()
        )
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        scheduler.metrics_reporter = SimpleNamespace(
            num_generated_tokens=0, report_prefill_stats=Mock()
        )
        batch = SimpleNamespace(
            batch_size=lambda: 1,
            reqs=[req],
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            copy_done=None,
            next_token_ids=[],
            accept_length_per_req_cpu=None,
            dllm_algo_state=None,
            can_run_cuda_graph=False,
        )

        with patch("sglang.srt.dllm.mixin.scheduler.release_kv_cache") as release_mock:
            SchedulerDllmMixin.process_batch_result_dllm(scheduler, batch, result)

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        release_mock.assert_called_once_with(req, cache)
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)

    def test_first_turn_drop_releases_own_mm_inputs(self):
        """A first turn dropped before ever committing owns its multimodal
        inputs; the dropped-req cleanup must release their features, since
        session close only scans committed req_nodes."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            session,
        ) = self._setup_first_turn()
        fresh = Session(capacity_of_str_len=0, session_id="fresh-first", streaming=True)
        req = fresh.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        mm = Mock()
        req.multimodal_inputs = mm
        scheduler = _scheduler_stub(cache)

        scheduler._release_dropped_waiting_req_mm_inputs(req)

        mm.release_features.assert_called_once_with()
        self.assertIsNone(req.multimodal_inputs)
        self.assertFalse(fresh.has_unfinished_request())
        self._assert_idle(_observer, _checker)

    def test_restored_turn_drop_keeps_session_owned_mm_inputs(self):
        """A restored turn inherits multimodal inputs from the committed turn;
        the dropped-req cleanup must NOT release them (the session owns and
        releases them at close)."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            session,
        ) = self._setup_first_turn()
        # The session already has a committed first turn from setup; a new
        # turn inherits its multimodal inputs from the committed turn.
        shared_mm = Mock()
        req2 = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req2.multimodal_inputs = shared_mm
        scheduler = _scheduler_stub(cache)

        scheduler._release_dropped_waiting_req_mm_inputs(req2)

        shared_mm.release_features.assert_not_called()
        self.assertIs(req2.multimodal_inputs, shared_mm)
        self.assertFalse(session.has_unfinished_request())


if __name__ == "__main__":
    unittest.main()
