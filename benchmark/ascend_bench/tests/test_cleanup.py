from asc_bench.cleanup import STRAGGLER_PATTERNS, kill_stragglers, wait_hbm_freed


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_passes_when_usage_returns_to_baseline():
    clock = Clock()
    samples = iter([{0: 5000}, {0: 120}])
    ok = wait_hbm_freed(
        lambda: next(samples),
        {0: 100},
        budget_mb=500,
        timeout_s=100,
        poll_s=1,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert ok is True
    assert clock.now == 1.0  # one dirty probe, one clean probe


def test_extra_kill_runs_once_then_success():
    clock = Clock()
    samples = iter([{0: 5000}, {0: 5000}, {0: 100}])
    kills = []

    def extra_kill():
        kills.append(1)

    ok = wait_hbm_freed(
        lambda: next(samples),
        {0: 100},
        budget_mb=100,
        timeout_s=100,
        poll_s=1,
        extra_kill=extra_kill,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert ok is True
    assert len(kills) == 1


def test_timeout_returns_false_when_still_dirty():
    clock = Clock()
    kills = []

    def extra_kill():
        kills.append(1)

    ok = wait_hbm_freed(
        lambda: {0: 9000},
        {0: 100},
        budget_mb=100,
        timeout_s=10,
        poll_s=2,
        extra_kill=extra_kill,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert ok is False
    assert len(kills) == 1
    assert clock.now >= 10


def test_no_probe_degrades_to_grace_wait():
    clock = Clock()
    ok = wait_hbm_freed(
        None,
        None,
        timeout_s=30,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert ok is True
    assert clock.now == 10.0


def test_kill_stragglers_invokes_pkill_per_pattern():
    calls = []

    class FakeResult:
        returncode = 0

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeResult()

    kill_stragglers(run=fake_run)
    assert calls == [["pkill", "-9", "-f", p] for p in STRAGGLER_PATTERNS]


def test_kill_stragglers_ignores_missing_pkill():
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("no pkill")

    kill_stragglers(run=fake_run)  # must not raise
