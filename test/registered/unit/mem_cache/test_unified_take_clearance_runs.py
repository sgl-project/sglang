"""`PageUsage.take_clearance_runs` gives the same waits, and leaves the same
fences, as asking `take_clearance` run by run, whatever the runs, the fences
and the state of their signals.
"""

import random
import unittest

from sglang.srt.mem_cache.kv.alloc import page_usage as page_usage_module
from sglang.srt.mem_cache.kv.alloc.page_usage import Consumer, PageUsage
from sglang.srt.mem_cache.kv.alloc.signals import StreamTransfer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CONSUMERS = (
    (Consumer.FORWARD,),
    (Consumer.SCHEDULE_COPY,),
    (Consumer.FORWARD, Consumer.ZERO),
    (Consumer.HOST_LOAD,),
)


class _Event:
    def __init__(self, fired):
        self.fired = fired

    def query(self):
        return self.fired


def _ledger(rng, page_bytes):
    """A ledger with random fences: forward events (some retired, some
    fired), host copies, and pairs of both."""
    usage = PageUsage()
    forwards = [_Event(rng.random() < 0.5) for _ in range(4)]
    for event in forwards:
        usage.register_forward(event)
    usage.retired_seq = rng.randrange(0, 5)
    copies = [StreamTransfer(_Event(rng.random() < 0.5)) for _ in range(3)]
    for _ in range(rng.randrange(1, 12)):
        kind = rng.randrange(3)
        if kind == 0:
            key = rng.choice(forwards)
        elif kind == 1:
            key = rng.choice(copies)
        else:
            key = (rng.choice(forwards), rng.choice(copies))
        first = rng.randrange(0, 60)
        end = first + rng.randrange(1, 12)
        usage.vacate(
            key,
            [(first, end)],
            page_bytes=page_bytes,
            writes=rng.random() < 0.3,
        )
    return usage


def _runs(rng):
    pages = sorted(rng.sample(range(70), rng.randrange(1, 30)))
    runs, start, prev = [], pages[0], pages[0]
    for page in pages[1:]:
        if page != prev + 1:
            runs.append((start, prev + 1))
            start = page
        prev = page
    runs.append((start, prev + 1))
    return runs


def _shape(usage):
    return sorted(
        (use.lo, use.hi, id(use.signal), use.writes)
        for use in usage.uses
        if use.pages is None
    )


def _keys(clearance):
    return (
        {id(k) for k in clearance.stream},
        {id(k) for k in clearance.host},
        {id(k) for k in clearance.after},
        clearance.excluded,
    )


class TestTakeClearanceRuns(CustomTestCase):
    def test_one_walk_equals_run_by_run(self):
        page_bytes = 16
        for seed in range(400):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                consumers = rng.choice(CONSUMERS)
                runs = _runs(rng)
                spans = [(a * page_bytes, b * page_bytes) for a, b in runs]
                one = _ledger(random.Random(seed), page_bytes)
                each = _ledger(random.Random(seed), page_bytes)
                # The two ledgers hold equal fences on shared signals.
                each.uses = [
                    page_usage_module.PageUse(
                        u.kind, v.signal, writes=u.writes, lo=u.lo, hi=u.hi
                    )
                    for u, v in zip(each.uses, one.uses)
                ]
                each.event_seq, each.retired_seq = one.event_seq, one.retired_seq
                expected = page_usage_module.Clearance()
                for lo, hi in spans:
                    expected.merge(each.take_clearance(lo, hi, consumers))
                got = one.take_clearance_runs(spans, consumers)
                self.assertEqual(_keys(got), _keys(expected))
                self.assertEqual(_shape(one), _shape(each))


if __name__ == "__main__":
    unittest.main()
