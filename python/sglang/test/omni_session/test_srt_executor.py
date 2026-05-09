# SPDX-License-Identifier: Apache-2.0

from sglang.srt.omni_session.srt_executor import UGSRTSchedulerExecutor


def test_idle_cleanup_drains_active_batch_when_no_pending_requests():
    scheduler = _FakeScheduler()
    executor = UGSRTSchedulerExecutor(scheduler)

    executor.run_idle_cleanup()

    assert scheduler.cleanup_steps == 1
    assert scheduler.is_fully_idle()


def test_idle_cleanup_clears_finished_batch_references():
    scheduler = _FakeScheduler()
    finished = _FakeReq(finished=True)
    scheduler.last_batch = _FakeBatch([finished])
    scheduler.running_batch = _FakeBatch([finished])
    scheduler.cur_batch = _FakeBatch([finished])
    executor = UGSRTSchedulerExecutor(scheduler)

    executor.run_idle_cleanup()

    assert scheduler.cleanup_steps == 0
    assert scheduler.is_fully_idle()


class _FakeScheduler:
    def __init__(self):
        self.session_controller = object()
        self.last_batch = None
        self.running_batch = _FakeBatch([])
        self.cur_batch = _FakeBatch([_FakeReq(finished=False)])
        self.waiting_queue = []
        self.grammar_manager = _FakeGrammarManager()
        self.cleanup_steps = 0

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


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs

    def is_empty(self):
        return not self.reqs

    def filter_batch(self):
        self.reqs = [req for req in self.reqs if not req.finished()]


class _FakeReq:
    def __init__(self, *, finished):
        self._finished = finished

    def finished(self):
        return self._finished


class _TruthyEmptyQueue:
    def __len__(self):
        return 0

    def __bool__(self):
        return True


class _FakeGrammarManager:
    grammar_queue = _TruthyEmptyQueue()
