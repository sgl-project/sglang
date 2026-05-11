# SPDX-License-Identifier: Apache-2.0

from collections import deque
from types import SimpleNamespace

from sglang.srt.omni_session.srt_executor import OmniSRTSchedulerExecutor


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


def test_temporary_context_idle_check_skips_cleanup_when_already_idle():
    scheduler = _FakeScheduler()
    scheduler.cur_batch = _FakeBatch([])
    executor = OmniSRTSchedulerExecutor(scheduler)

    executor._check_scheduler_idle_for_temporary_context()

    assert scheduler.cleanup_steps == 0


class _FakeScheduler:
    def __init__(self):
        self.session_controller = object()
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
