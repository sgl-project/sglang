import unittest
from types import SimpleNamespace

from sglang.srt.managers.shortest_prefill_wait_guard import ShortestPrefillWaitGuard
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def make_schedule(**overrides):
    config = dict(
        schedule_policy="shortest-prefill-first",
        shortest_prefill_wait_guard_seconds=60.0,
        shortest_prefill_wait_guard_age_gap_seconds=15.0,
        shortest_prefill_wait_guard_cooldown_seconds=30.0,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def make_parallel(**overrides):
    config = dict(attn_tp_size=1, attn_cp_size=1, pp_size=1, attn_dp_size=1)
    config.update(overrides)
    return SimpleNamespace(**config)


def make_request(received, *, bootstrap=0.0, queued=0.0, priority=0, finished=False):
    return SimpleNamespace(
        time_stats=SimpleNamespace(
            scheduler_recv_time=received,
            prefill_bootstrap_queue_entry_time=bootstrap,
            wait_queue_entry_time=queued,
        ),
        priority=priority,
        finished=lambda: finished,
    )


class TestShortestPrefillWaitGuardConfig(unittest.TestCase):
    def test_disabled_with_other_policy_or_topology(self):
        guard = ShortestPrefillWaitGuard.from_config(
            make_schedule(
                schedule_policy="fcfs", shortest_prefill_wait_guard_seconds=0
            ),
            make_parallel(attn_tp_size=2),
            is_pd_prefill=False,
        )
        self.assertIsNone(guard)

    def test_enabled_only_for_pd_prefill_with_spf(self):
        with self.assertRaises(ValueError):
            ShortestPrefillWaitGuard.from_config(
                make_schedule(schedule_policy="fcfs"),
                make_parallel(),
                is_pd_prefill=True,
            )
        with self.assertRaises(ValueError):
            ShortestPrefillWaitGuard.from_config(
                make_schedule(), make_parallel(), is_pd_prefill=False
            )

    def test_rejects_ranks_that_share_prefill_admission(self):
        for field in ("attn_tp_size", "attn_cp_size", "pp_size"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ShortestPrefillWaitGuard.from_config(
                    make_schedule(), make_parallel(**{field: 2}), is_pd_prefill=True
                )

    def test_independent_dp_replicas_are_supported(self):
        guards = [
            ShortestPrefillWaitGuard.from_config(
                make_schedule(),
                make_parallel(attn_dp_size=2, attn_dp_rank=rank, tp_size=2),
                is_pd_prefill=True,
            )
            for rank in (0, 1)
        ]
        self.assertTrue(all(isinstance(g, ShortestPrefillWaitGuard) for g in guards))
        guards[0].record_admission(100.0)
        old = make_request(15.0)
        queue = [make_request(90.0), old]
        self.assertIsNone(guards[0].pick(queue, now=101.0, has_active_chunk=False))
        self.assertIs(guards[1].pick(queue, now=101.0, has_active_chunk=False), old)

    def test_enabled_thresholds_must_be_finite_and_positive(self):
        fields_and_values = (
            ("shortest_prefill_wait_guard_seconds", -1.0),
            ("shortest_prefill_wait_guard_age_gap_seconds", 0.0),
            ("shortest_prefill_wait_guard_age_gap_seconds", float("nan")),
            ("shortest_prefill_wait_guard_cooldown_seconds", 0.0),
            ("shortest_prefill_wait_guard_cooldown_seconds", float("inf")),
        )
        for field, value in fields_and_values:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                ShortestPrefillWaitGuard.from_config(
                    make_schedule(**{field: value}), make_parallel(), is_pd_prefill=True
                )


class TestShortestPrefillWaitGuard(unittest.TestCase):
    def setUp(self):
        self.guard = ShortestPrefillWaitGuard(60.0, 15.0, 30.0)

    def pick(
        self, queue, *, now=100.0, has_active_chunk=False, priority_scheduling=False
    ):
        return self.guard.pick(
            queue,
            now=now,
            has_active_chunk=has_active_chunk,
            priority_scheduling=priority_scheduling,
        )

    def test_wait_and_age_gap_boundaries(self):
        old = make_request(40.0)
        queue = [make_request(55.0), old]
        self.assertIsNone(self.pick(queue, now=99.9))
        self.assertIsNone(self.pick([make_request(54.9), old]))
        self.assertIs(self.pick(queue), old)

    def test_simultaneous_cold_arrivals_keep_spf_order(self):
        queue = [make_request(10.0), make_request(10.0), make_request(11.0)]
        self.assertIsNone(self.pick(queue))

    def test_empty_or_single_request_queue_needs_no_promotion(self):
        self.assertIsNone(self.pick([]))
        self.assertIsNone(self.pick([make_request(10.0)]))

    def test_promotes_oldest_live_request_without_reordering_queue(self):
        head = make_request(90.0)
        younger = make_request(30.0)
        old = make_request(20.0)
        tied = make_request(20.0)
        aborted = make_request(5.0, finished=True)
        queue = [head, younger, old, tied, aborted]
        order = [id(req) for req in queue]
        self.assertIs(self.pick(queue), old)
        self.assertEqual([id(req) for req in queue], order)

    def test_uses_scheduler_receive_time_before_admission_queue_fallbacks(self):
        head = make_request(90.0)
        waiting_before_bootstrap = make_request(20.0, bootstrap=80.0, queued=90.0)
        self.assertIs(
            self.pick([head, waiting_before_bootstrap]), waiting_before_bootstrap
        )

        for old in (
            make_request(0.0, bootstrap=20.0, queued=90.0),
            make_request(0.0, queued=20.0),
        ):
            self.assertIs(self.pick([head, old]), old)
        self.assertIsNone(self.pick([head, make_request(0.0)]))

    def test_does_not_interrupt_active_chunk(self):
        queue = [make_request(90.0), make_request(10.0)]
        self.assertIsNone(self.pick(queue, has_active_chunk=True))
        self.assertIs(self.pick(queue), queue[1])

    def test_cooldown_starts_only_after_successful_admission(self):
        head = make_request(90.0)
        old = make_request(10.0)
        queue = [head, old]
        self.assertIs(self.pick(queue), old)
        self.assertIs(self.pick(queue, now=101.0), old)
        self.guard.record_admission(101.0)
        self.assertIsNone(self.pick(queue, now=130.9))
        self.assertIs(self.pick(queue, now=131.0), old)

    def test_only_promotes_within_queue_head_priority_when_enabled(self):
        head = make_request(90.0, priority=10)
        older_other_priority = make_request(10.0, priority=5)
        old_same_priority = make_request(25.0, priority=10)
        self.assertIsNone(
            self.pick([head, older_other_priority], priority_scheduling=True)
        )
        queue = [head, old_same_priority, older_other_priority]
        self.assertIs(self.pick(queue, priority_scheduling=True), old_same_priority)
        self.assertIs(self.pick(queue), older_other_priority)


if __name__ == "__main__":
    unittest.main()
