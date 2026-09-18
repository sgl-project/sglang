import dataclasses
import types
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    AbortReq,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    MultimodalInputs,
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
        dllm_config=None,
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
            queue=[], retracted_queue=[req]
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
        # The committed first turn holds the session's shared multimodal
        # inputs; a new turn inherits that exact object (see create_req).
        shared_mm = Mock()
        [committed] = session.req_nodes.values()
        committed.req.multimodal_inputs = shared_mm
        req2 = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIs(req2.multimodal_inputs, shared_mm)
        scheduler = _scheduler_stub(cache)

        scheduler._release_dropped_waiting_req_mm_inputs(req2)

        shared_mm.release_features.assert_not_called()
        self.assertIs(req2.multimodal_inputs, shared_mm)
        self.assertFalse(session.has_unfinished_request())

    def test_restored_turn_drop_releases_only_turn_added_mm_items(self):
        """A restored turn that appended its own media owns those additions:
        the dropped-req cleanup releases them while preserving the shared
        inherited history, and the session stays usable for the next turn."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            session,
        ) = self._setup_first_turn()
        inherited_item = Mock()
        inherited_item.feature = Mock()
        own_item = Mock()
        own_item.feature = Mock()
        [committed] = session.req_nodes.values()
        committed.req.multimodal_inputs = MultimodalInputs(mm_items=[inherited_item])
        req2 = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        # The turn appends its own media on a private copy, leaving the
        # session's shared object untouched (see _extend_session_image_inputs).
        req2.multimodal_inputs = dataclasses.replace(req2.multimodal_inputs)
        req2.multimodal_inputs.mm_items = req2.multimodal_inputs.mm_items + [own_item]
        scheduler = _scheduler_stub(cache)

        scheduler._release_dropped_waiting_req_mm_inputs(req2)

        # The turn's own addition is released...
        own_item.release_transport_proxies.assert_called_once_with()
        self.assertIsNone(own_item.feature)
        # ...while the shared inherited item and history are preserved.
        inherited_item.release_transport_proxies.assert_not_called()
        self.assertIsNotNone(inherited_item.feature)
        self.assertEqual(committed.req.multimodal_inputs.mm_items, [inherited_item])
        self.assertFalse(session.has_unfinished_request())
        # The session stays usable: a follow-up turn inherits the intact
        # shared history and marks the session inflight again.
        req3 = session.create_req(
            _recv("turn-3", list(range(48, 64))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIs(req3.multimodal_inputs, committed.req.multimodal_inputs)
        self.assertTrue(session.has_unfinished_request())


if __name__ == "__main__":
    unittest.main()
