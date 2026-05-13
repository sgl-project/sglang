# SPDX-License-Identifier: Apache-2.0

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from types import SimpleNamespace

import pytest
import torch

from sglang.omni.core.protocol import TemporaryForwardPrepared
from sglang.omni.runtime.srt_scheduler_state import OmniSchedulerState
from sglang.srt.omni_session.srt_executor import (
    OmniSRTSchedulerExecutor,
    OmniSRTSchedulerExecutorError,
)
from sglang.srt.managers.schedule_batch import (
    _normalize_custom_position_ids_for_batch,
)


def test_idle_cleanup_drains_active_batch_when_no_pending_requests():
    scheduler = _FakeScheduler()
    executor = OmniSRTSchedulerExecutor(scheduler)

    executor.run_idle_cleanup()

    assert scheduler.cleanup_steps == 1
    assert scheduler.is_fully_idle()


def test_idle_cleanup_clears_finished_batch_references():
    scheduler = _FakeScheduler()
    finished = _FakeReq(finished=True)
    scheduler.last_batch = _FakeBatch([finished])
    scheduler.running_batch = _FakeBatch([finished])
    scheduler.cur_batch = _FakeBatch([finished])
    executor = OmniSRTSchedulerExecutor(scheduler)

    executor.run_idle_cleanup()

    assert scheduler.cleanup_steps == 0
    assert scheduler.is_fully_idle()


def test_sync_execute_budget_scales_with_request_decode_length():
    scheduler = _FakeSyncScheduler(finish_after_steps=12)
    executor = OmniSRTSchedulerExecutor(scheduler, max_sync_steps=8)
    req = _FakeReq(finished=False, max_new_tokens=16)

    executor.execute_omni_request(
        record=SimpleNamespace(session_id="s0"),
        req=req,
        state=None,
    )

    assert req.finished()
    assert scheduler.run_steps == 12
    assert scheduler.sample_launches == 12


def test_sync_execute_restores_outer_scheduler_batch_state():
    scheduler = _FakeSyncScheduler(finish_after_steps=1)
    executor = OmniSRTSchedulerExecutor(scheduler)
    req = _FakeReq(finished=False)
    outer_last_batch = _FakeBatch([])
    scheduler.last_batch = outer_last_batch

    executor.execute_omni_request(
        record=SimpleNamespace(session_id="s0"),
        req=req,
        state=None,
    )

    assert req.finished()
    assert scheduler.last_batch is outer_last_batch
    assert scheduler.cur_batch is None


def test_explicit_omni_position_count_overrides_raw_custom_positions():
    req = _FakeReq(finished=False)
    req.custom_position_ids = list(range(1024)) + [37]
    req.omni_srt_position_count = 38

    assert OmniSRTSchedulerExecutor._request_position_count(req) == 38


def test_request_token_indices_for_active_req_reads_tree_cache_session_slot():
    scheduler = _FakeScheduler()
    scheduler.tree_cache.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(20, dtype=torch.int64).view(2, 10)
    )
    scheduler.tree_cache.session_slots["s0"] = SimpleNamespace(
        req_pool_idx=1,
        kv_committed_len=3,
    )
    executor = OmniSRTSchedulerExecutor(scheduler)
    req = _FakeReq(finished=False)
    req.session = SimpleNamespace(session_id="s0")

    token_indices = executor._request_token_indices_for_active_req(req)

    assert token_indices.tolist() == [10, 11, 12]


def test_custom_position_ids_promote_text_positions_for_mrope_batching():
    positions = _normalize_custom_position_ids_for_batch(
        [0, [1, 0, 0], 2, [3, 4, 5]]
    )

    assert positions == [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 4, 5]]


def test_async_execute_submits_native_req_to_scheduler_state():
    scheduler = _FakeSyncScheduler(finish_after_steps=1)
    scheduler.omni_scheduler_state = OmniSchedulerState()
    scheduler.omni_scheduler_state.bind_scheduler_thread()
    executor = OmniSRTSchedulerExecutor(scheduler)
    req = _FakeReq(finished=False)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            executor.execute_omni_request,
            record=SimpleNamespace(session_id="s0"),
            req=req,
            state=None,
        )
        _wait_until(
            lambda: not scheduler.omni_scheduler_state.pending_srt_requests.empty()
        )

        scheduler.omni_scheduler_state.admit_srt_requests(scheduler)
        assert scheduler.waiting_queue == [req]

        batch = _FakeBatch([scheduler.waiting_queue.pop(0)])
        observation = scheduler.omni_scheduler_state.observe_srt_batch_before_process(
            batch
        )
        req._finished = True
        scheduler.omni_scheduler_state.finalize_srt_batch_after_process(observation)
        scheduler.omni_scheduler_state.retire_finished_srt_requests(scheduler)

        future.result(timeout=1)

    assert req.finished()
    assert not scheduler.omni_scheduler_state.running_srt_requests
    assert not scheduler.omni_scheduler_state.finished_srt_requests


def test_temporary_context_idle_check_skips_cleanup_when_already_idle():
    scheduler = _FakeScheduler()
    scheduler.cur_batch = _FakeBatch([])
    executor = OmniSRTSchedulerExecutor(scheduler)

    executor._check_scheduler_idle_for_temporary_context()

    assert scheduler.cleanup_steps == 0


def test_temporary_context_idle_check_drains_waiting_queue():
    scheduler = _FakeSyncScheduler(finish_after_steps=1)
    req = _FakeReq(finished=False)
    scheduler.waiting_queue.append(req)
    executor = OmniSRTSchedulerExecutor(scheduler)

    executor._check_scheduler_idle_for_temporary_context()

    assert req.finished()
    assert scheduler.is_fully_idle()


def test_temporary_context_forward_runs_on_scheduler_thread_and_releases():
    scheduler = _FakeScheduler()
    scheduler.cur_batch = _FakeBatch([])
    scheduler.omni_scheduler_state = OmniSchedulerState()
    scheduler.omni_scheduler_state.bind_scheduler_thread()
    executor = _FakeTemporaryContextExecutor(scheduler)
    prepared = TemporaryForwardPrepared(generation_input={}, srt_session_id="s0")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            executor.run_temporary_context_forward,
            prepared=prepared,
            forward=lambda forward_batch: forward_batch,
        )
        _wait_until(
            lambda: not scheduler.omni_scheduler_state
            .pending_scheduler_thread_calls.empty()
        )

        scheduler.omni_scheduler_state.drain_scheduler_thread_calls()

        assert future.result(timeout=1) == "temporary-batch"

    assert (
        executor.build_thread_id
        == scheduler.omni_scheduler_state.scheduler_thread_id
    )
    assert executor.temporary_batch.released


def test_temporary_context_cache_allocates_scratch_kv_without_evicting_tree_cache():
    scheduler = _FakeScheduler()
    scheduler.tree_cache = _FakeTreeCache(page_size=16)
    executor = OmniSRTSchedulerExecutor(scheduler)
    allocator = _FakeKVAllocator(size=64)

    out_cache_loc, owned_cache_loc = executor._alloc_temporary_context_cache(
        token_to_kv_pool_allocator=allocator,
        extend_num_tokens=33,
        device=torch.device("cpu"),
    )

    assert out_cache_loc.tolist() == list(range(33))
    assert owned_cache_loc.tolist() == list(range(48))
    assert scheduler.tree_cache.evict_calls == []


def test_temporary_context_cache_reports_pressure_without_evicting_tree_cache():
    scheduler = _FakeScheduler()
    scheduler.tree_cache = _FakeTreeCache(page_size=16)
    executor = OmniSRTSchedulerExecutor(scheduler)
    allocator = _FakeKVAllocator(size=32)

    with pytest.raises(
        OmniSRTSchedulerExecutorError,
        match="without evicting committed SRT context",
    ):
        executor._alloc_temporary_context_cache(
            token_to_kv_pool_allocator=allocator,
            extend_num_tokens=33,
            device=torch.device("cpu"),
        )

    assert scheduler.tree_cache.evict_calls == []


def _wait_until(predicate):
    import time

    deadline = time.time() + 1
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met")


class _FakeScheduler:
    def __init__(self):
        self.session_controller = object()
        self.omni_scheduler_state = None
        self.tree_cache = _FakeTreeCache()
        self.last_batch = None
        self.running_batch = _FakeBatch([])
        self.cur_batch = _FakeBatch([_FakeReq(finished=False)])
        self.waiting_queue = []
        self.grammar_manager = _FakeGrammarManager()
        self.result_queue = deque()
        self.cleanup_steps = 0

    def init_req_max_new_tokens(self, req):
        pass

    def is_fully_idle(self):
        return (
            self.running_batch.is_empty()
            and self.last_batch is None
            and (self.cur_batch is None or self.cur_batch.is_empty())
            and len(self.waiting_queue) == 0
            and len(self.grammar_manager.grammar_queue) == 0
        )

    def get_next_batch_to_run(self):
        self.cleanup_steps += 1
        self.running_batch = _FakeBatch([])
        self.cur_batch = _FakeBatch([])
        return None

    def on_idle(self):
        pass

    def process_batch_result(self, batch, result):
        pass


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.batch_is_full = False

    def is_empty(self):
        return not self.reqs

    def filter_batch(self):
        self.reqs = [req for req in self.reqs if not req.finished()]


class _FakeReq:
    def __init__(self, *, finished, max_new_tokens=0):
        self._finished = finished
        self.rid = "r0"
        self.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)
        self.session = None
        self.req_pool_idx = None
        self.kv_committed_len = 0
        self.custom_position_ids = None
        self.custom_decode_position_id = None
        self.omni_srt_position_count = None

    def finished(self):
        return self._finished


class _FakeSyncScheduler(_FakeScheduler):
    def __init__(self, *, finish_after_steps):
        super().__init__()
        self.cur_batch = None
        self.finish_after_steps = finish_after_steps
        self.run_steps = 0
        self.sample_launches = 0

    def _add_request_to_queue(self, req):
        self.waiting_queue.append(req)

    def get_next_batch_to_run(self):
        if self.waiting_queue:
            self.running_batch = _FakeBatch([self.waiting_queue.pop(0)])
            return self.running_batch
        if self.running_batch.is_empty():
            return None
        return self.running_batch

    def run_batch(self, batch):
        self.run_steps += 1
        return object()

    def launch_batch_sample_if_needed(self, result):
        self.sample_launches += 1

    def process_batch_result(self, batch, result):
        if self.run_steps >= self.finish_after_steps:
            for req in batch.reqs:
                req._finished = True
            self.running_batch = _FakeBatch([])


class _TruthyEmptyQueue:
    def __len__(self):
        return 0

    def __bool__(self):
        return True


class _FakeGrammarManager:
    grammar_queue = _TruthyEmptyQueue()


class _FakeTemporaryContextBatch:
    def __init__(self):
        self.forward_batch = "temporary-batch"
        self.released = False

    def release(self):
        self.released = True


class _FakeTemporaryContextExecutor(OmniSRTSchedulerExecutor):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        self.build_thread_id = None
        self.temporary_batch = _FakeTemporaryContextBatch()

    def build_temporary_context_forward_batch(self, *, prepared):
        self.build_thread_id = get_ident()
        return self.temporary_batch


class _FakeTreeCache:
    def __init__(self, page_size=1):
        self.page_size = page_size
        self.evict_calls = []
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.empty((0, 0), dtype=torch.int64)
        )
        self.session_slots = {}

    def evict(self, params):
        self.evict_calls.append(params)

    def get_session_slot(self, session_id):
        return self.session_slots.get(session_id)


class _FakeKVAllocator:
    def __init__(self, size):
        self.free_pages = torch.arange(size, dtype=torch.int64)

    def available_size(self):
        return int(self.free_pages.numel())

    def alloc(self, need_size):
        if need_size > self.available_size():
            return None
        selected = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return selected
