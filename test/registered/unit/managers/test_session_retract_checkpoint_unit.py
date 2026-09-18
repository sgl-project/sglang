import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    ScheduleBatch,
    release_req,
)
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
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


def _recv(rid, input_ids, parent_rid=None, session_id="session-a"):
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
            id=session_id,
            rid=parent_rid,
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
    key = RadixKey(array("q", req.origin_input_ids + req.output_ids))
    match = cache.match_prefix(MatchPrefixParams(key=key, req=req, cow_mamba=True))
    prefix_len = len(match.device_indices)
    req.lock_receipt = cache.inc_lock_ref(match.last_device_node).to_dec_params()
    if req.kv.req_pool_idx is None:
        req_to_token_pool.alloc([req])
    total = len(req.origin_input_ids) + len(req.output_ids)
    if prefix_len:
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, prefix_len)), match.device_indices
        )
    if total > prefix_len:
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(prefix_len, total)),
            allocator.alloc(total - prefix_len),
        )
    req.prefix_indices = match.device_indices
    req.last_node = match.last_device_node
    req.kv.cache_protected_len = (
        match.cache_protected_len
        if match.cache_protected_len is not None
        else prefix_len
    )
    req.kv.kv_committed_len = total
    req.kv.kv_allocated_len = total
    req.full_untruncated_fill_ids = array("q", req.origin_input_ids + req.output_ids)
    req.set_extend_range(prefix_len, total)
    req.kv.mamba_last_track_seqlen = 0
    if req.kv.mamba_next_track_idx is None:
        req.kv.mamba_next_track_idx = 0
    cache.cache_unfinished_req(req)


def _decode_step(req, allocator, req_to_token_pool, token):
    pos = req.kv.kv_allocated_len
    idx = allocator.alloc(1)
    req_to_token_pool.write((req.kv.req_pool_idx, slice(pos, pos + 1)), idx)
    req.output_ids.append(token)
    req._refresh_fill_ids()
    req.kv.kv_committed_len += 1
    req.kv.kv_allocated_len += 1


def _finish_turn(req, cache, finished_len):
    req.finished_reason = FINISH_LENGTH(length=finished_len)
    req.finished_len = finished_len
    release_kv_cache(req, cache)


class TestSessionRetractCheckpoint(CustomTestCase):
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
        _finish_turn(req1, cache, finished_len=0)
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

    def test_oom_retract_of_last_streaming_turn_aborts_session_turn(self):
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
        _prefill(req, cache, allocator, req_to_token_pool)

        batch = ScheduleBatch(reqs=[req])
        batch.req_to_token_pool = req_to_token_pool
        batch.token_to_kv_pool_allocator = allocator
        batch.tree_cache = cache
        batch.hisparse_coordinator = None
        batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)

        with patch.object(batch, "check_decode_mem", return_value=False):
            with patch.object(batch, "filter_batch"):
                with patch.object(
                    NewTokenRatioTracker,
                    "estimate_new_token_ratio_after_retract",
                    return_value=0.0,
                ):
                    retracted, _ratio, reqs_to_abort = batch.retract_decode()

        self.assertEqual(reqs_to_abort, [req])
        self.assertEqual(retracted, [])
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertFalse(session.has_unfinished_request())
        self.assertFalse(cache.session.has_slot(session.session_id))
        self.assertEqual(set(session.req_nodes), {"turn-1"})
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_retract_backup_failure_aborts_session_turn(self):
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

        # A bystander decode request the retraction order prefers to keep
        # (more output tokens), so the loop retracts the session turn.
        bystander_session = Session(
            capacity_of_str_len=0, session_id="session-b", streaming=True
        )
        bystander = bystander_session.create_req(
            _recv("bystander", list(range(48, 64)), session_id="session-b"),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        _prefill(bystander, cache, allocator, req_to_token_pool)
        _decode_step(bystander, allocator, req_to_token_pool, 100)

        batch = ScheduleBatch(reqs=[req, bystander])
        batch.req_to_token_pool = req_to_token_pool
        batch.token_to_kv_pool_allocator = allocator
        batch.tree_cache = cache
        batch.hisparse_coordinator = None
        batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)

        fake_disagg = SimpleNamespace(
            disaggregation_mode="decode",
            disaggregation_decode_retraction_backup=None,
        )
        # check_decode_mem True throughout: the first_iter pass retracts
        # exactly one request and the last-request OOM branch never fires.
        with patch.object(batch, "check_decode_mem", return_value=True):
            with patch.object(batch, "filter_batch"):
                with patch.object(
                    NewTokenRatioTracker,
                    "estimate_new_token_ratio_after_retract",
                    return_value=0.0,
                ):
                    with patch(
                        "sglang.srt.managers.schedule_batch.get_disagg",
                        return_value=fake_disagg,
                    ):
                        with patch(
                            "sglang.srt.managers.schedule_batch.retraction_backup",
                            return_value=False,
                        ) as backup:
                            retracted, _ratio, reqs_to_abort = batch.retract_decode()

        backup.assert_called_once()
        self.assertEqual(retracted, [])
        self.assertEqual(reqs_to_abort, [req])
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertIn("Retraction host KV pool exhausted", req.to_finish.message)
        # The retract nuke ran (turn KV + slot dropped) and the terminal
        # abort cleared the session inflight marker.
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertFalse(session.has_unfinished_request())
        self.assertFalse(cache.session.has_slot(session.session_id))
        self.assertEqual(set(session.req_nodes), {"turn-1"})
        # The bystander was never retracted and stays decode-active.
        self.assertIsNone(bystander.to_finish)
        self.assertTrue(bystander_session.has_unfinished_request())

        _finish_turn(bystander, cache, finished_len=1)
        cache.session.release_session(bystander_session.session_id)
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_bootstrap_failure_before_allocation_aborts_session_turn(self):
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
        req.disagg_kv_sender = SimpleNamespace(failure_exception=lambda: None)
        req.bootstrap_room = 0
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda **kwargs: None)
        )
        scheduler = SimpleNamespace(
            tree_cache=cache,
            disagg_prefill_pending_chunk_rids=set(),
            req_to_metadata_buffer_idx_allocator=None,
            output_streamer=SimpleNamespace(stream_output=lambda reqs, rl: None),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
        )
        scheduler.clear_pending_chunk_send = (
            SchedulerDisaggregationPrefillMixin.clear_pending_chunk_send.__get__(
                scheduler
            )
        )

        with patch(
            "sglang.srt.disaggregation.prefill.get_parallel",
            return_value=SimpleNamespace(tp_rank=0),
        ):
            SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure(scheduler, req)

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(set(session.req_nodes), {"turn-1"})
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_retract_readmit_finish_matches_no_retract(self):
        worlds = []
        for retract in (False, True):
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
            _decode_step(req, allocator, req_to_token_pool, 100)
            _decode_step(req, allocator, req_to_token_pool, 101)
            if retract:
                release_req(
                    req=req,
                    remaing_req_count=1,
                    req_to_token_pool=req_to_token_pool,
                    token_to_kv_pool_allocator=allocator,
                    tree_cache=cache,
                    hisparse_coordinator=None,
                    offload_kv=False,
                )
                self.assertFalse(cache.session.has_slot(session.session_id))
                self.assertEqual(set(session.req_nodes), {"turn-1"})
                self.assertTrue(session.has_unfinished_request())
                self.assertEqual(cache.session.session_held_mamba_slots(), 0)
                self._assert_idle(observer, checker)
                req.init_next_round_input(cache)
                _prefill(req, cache, allocator, req_to_token_pool)
                _decode_step(req, allocator, req_to_token_pool, 102)
                _decode_step(req, allocator, req_to_token_pool, 103)
            else:
                _decode_step(req, allocator, req_to_token_pool, 102)
                _decode_step(req, allocator, req_to_token_pool, 103)
            _finish_turn(req, cache, finished_len=4)
            worlds.append((cache, observer, checker, session, req))

        cache_a, observer_a, checker_a, session_a, req_a = worlds[0]
        cache_b, observer_b, checker_b, session_b, req_b = worlds[1]
        self.assertEqual(set(session_a.req_nodes), set(session_b.req_nodes))
        self.assertEqual(list(req_a.output_ids), list(req_b.output_ids))
        self.assertEqual(session_a.committed_origin_len, session_b.committed_origin_len)
        self.assertEqual(
            session_a.committed_unpadded_len, session_b.committed_unpadded_len
        )
        self.assertEqual(session_a.committed_fill_len, session_b.committed_fill_len)
        slot_a = cache_a.session.slots[session_a.session_id]
        slot_b = cache_b.session.slots[session_b.session_id]
        self.assertEqual(slot_a.kv.kv_committed_len, slot_b.kv.kv_committed_len)
        self.assertEqual(slot_a.kv.kv_allocated_len, slot_b.kv.kv_allocated_len)
        self.assertFalse(session_a.has_unfinished_request())
        self.assertFalse(session_b.has_unfinished_request())
        self.assertEqual(
            cache_a.session.session_held_mamba_slots(),
            cache_b.session.session_held_mamba_slots(),
        )
        self._assert_idle(observer_a, checker_a)
        self._assert_idle(observer_b, checker_b)
        next_a = session_a.create_req(
            _recv("turn-3", [99], parent_rid="turn-2"),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        next_b = session_b.create_req(
            _recv("turn-3", [99], parent_rid="turn-2"),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertEqual(list(next_a.origin_input_ids), list(next_b.origin_input_ids))

    def test_unflagged_release_of_unfinished_req_is_a_normal_finish(self):
        # Disaggregated prefill releases the KV of a successful transfer
        # *before* stamping FINISH_LENGTH; without explicit retract intent
        # that must still commit the turn (slot saved, finish_req ran).
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
        self.assertIsNone(req.finished_reason)

        release_kv_cache(req, cache)
        req.finished_reason = FINISH_LENGTH(length=0)
        req.finished_len = 0

        self.assertTrue(cache.session.has_slot(session.session_id))
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(set(session.req_nodes), {"turn-2"})
        self.assertIs(session.req_nodes["turn-2"].req, req)
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertGreater(cache.session.session_held_mamba_slots(), 0)
        committed = list(req.origin_input_ids) + list(req.output_ids)
        follow_up = session.create_req(
            _recv("turn-3", [99], parent_rid="turn-2"),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)
        self.assertEqual(list(follow_up.origin_input_ids), committed + [99])
        cache.session.release_session(session.session_id)
        self._assert_idle(observer, checker)


if __name__ == "__main__":
    unittest.main()
