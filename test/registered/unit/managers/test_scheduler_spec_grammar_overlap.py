import unittest
from collections import deque
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeForwardMode:
    def __init__(self, *, decode: bool):
        self._decode = decode

    def is_decode(self):
        return self._decode

    def is_extend(self):
        return False


class _FakeSpecAlgorithm:
    def __init__(self, *, is_none: bool):
        self._is_none = is_none

    def is_none(self):
        return self._is_none

    def is_some(self):
        return not self._is_none


class _FakeBatch:
    def __init__(
        self,
        *,
        has_grammar: bool = True,
        is_spec: bool = True,
        is_decode: bool = True,
    ):
        self.has_grammar = has_grammar
        self.spec_algorithm = _FakeSpecAlgorithm(is_none=not is_spec)
        self.forward_mode = _FakeForwardMode(decode=is_decode)
        self.is_extend_in_batch = False
        self.copied_batch = object()

    def __bool__(self):
        return True

    def copy(self):
        return self.copied_batch


class TestSchedulerSpecGrammarOverlap(unittest.TestCase):
    def _new_scheduler_for_guard(self, batch, *, has_result: bool):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.last_batch = batch
        scheduler.result_queue = deque([(object(), object())] if has_result else [])
        return scheduler

    def test_pending_spec_grammar_guard(self):
        cases = [
            (
                "spec grammar decode with a pending result",
                _FakeBatch(),
                True,
                True,
            ),
            (
                "empty result queue",
                _FakeBatch(),
                False,
                False,
            ),
            (
                "non grammar",
                _FakeBatch(has_grammar=False),
                True,
                False,
            ),
            (
                "non spec",
                _FakeBatch(is_spec=False),
                True,
                False,
            ),
            (
                "non decode",
                _FakeBatch(is_decode=False),
                True,
                False,
            ),
        ]

        for name, batch, has_result, expected in cases:
            with self.subTest(name=name):
                scheduler = self._new_scheduler_for_guard(
                    batch,
                    has_result=has_result,
                )

                self.assertEqual(
                    scheduler._has_pending_spec_grammar_result(),
                    expected,
                )

    def test_processes_pending_spec_grammar_result_before_next_batch(self):
        batch = _FakeBatch()

        calls, scheduler = self._run_two_overlap_iterations(batch)

        self.assertLess(
            calls.index("process_batch_result"),
            calls.index("get_next_batch_to_run_2"),
        )
        self.assertIsNone(scheduler.last_batch)

    def test_keeps_original_order_when_guard_does_not_match(self):
        cases = [
            ("non grammar", _FakeBatch(has_grammar=False)),
            ("non spec", _FakeBatch(is_spec=False)),
            ("non decode", _FakeBatch(is_decode=False)),
        ]

        for name, batch in cases:
            with self.subTest(name=name):
                calls, _ = self._run_two_overlap_iterations(batch)

                self.assertLess(
                    calls.index("get_next_batch_to_run_2"),
                    calls.index("process_batch_result"),
                )

    def _run_two_overlap_iterations(self, first_batch):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler.require_mlp_sync = False
        scheduler.is_generation = False
        scheduler.last_batch = None
        scheduler.request_receiver = SimpleNamespace(recv_requests=lambda: [])
        scheduler.process_input_requests = lambda recv_reqs: None
        scheduler._apply_war_barrier = lambda: None
        scheduler.on_idle = lambda: None

        calls = []
        batch_result = object()
        get_next_call_count = 0

        def get_next_batch_to_run():
            nonlocal get_next_call_count
            get_next_call_count += 1
            calls.append(f"get_next_batch_to_run_{get_next_call_count}")
            if get_next_call_count == 1:
                return first_batch
            scheduler.gracefully_exit = True
            return None

        def run_batch(batch):
            calls.append("run_batch")
            self.assertIs(batch, first_batch)
            return batch_result

        def process_batch_result(batch, result):
            calls.append("process_batch_result")
            self.assertIs(batch, first_batch.copied_batch)
            self.assertIs(result, batch_result)

        scheduler.get_next_batch_to_run = get_next_batch_to_run
        scheduler.run_batch = run_batch
        scheduler.process_batch_result = process_batch_result

        Scheduler.event_loop_overlap(scheduler)

        return calls, scheduler


if __name__ == "__main__":
    unittest.main()
