import unittest

from sglang.srt.managers.prefill_delayer import (
    CoalesceParams,
    CoalesceState,
    decide_coalesce,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

P = CoalesceParams(dp_size=2, chunk=16384, interval=10, stall_ticks=3, ttft_ticks=8)


def tick(
    state,
    armed=False,
    prefillable=(1, 1),
    running=(30, 30),
    pending=(16384, 16384),
    queue=(5, 5),
    partial=(0, 0),
    kv_high=(0, 0),
    params=P,
):
    return decide_coalesce(
        state,
        params,
        armed=armed,
        prefillable=prefillable,
        running=running,
        pending=pending,
        queue=queue,
        partial=partial,
        kv_high=kv_high,
    )


class TestPrefillDelayerCoalesce(CustomTestCase):
    def test_full_aligned_group_fires(self):
        _, allow, reason = tick(CoalesceState())
        self.assertEqual((allow, reason), (True, "fill"))

    def test_interval_holds_for_n_passes_then_fires(self):
        state, allow, reason = tick(CoalesceState(), armed=True)
        self.assertEqual((allow, reason), (False, "interval"))
        for _ in range(P.interval - 1):
            state, allow, reason = tick(state)
            self.assertEqual((allow, reason), (False, "interval"))
        state, allow, reason = tick(state)
        self.assertEqual((allow, reason), (True, "fill"))
        self.assertEqual(state.hold_ticks, 0)

    def test_interval_skipped_without_decode(self):
        _, allow, reason = tick(CoalesceState(), armed=True, running=(0, 0))
        self.assertEqual((allow, reason), (True, "no_decode"))

    def test_nothing_prefillable_is_vacuous(self):
        _, allow, reason = tick(CoalesceState(), prefillable=(0, 0), pending=(0, 0))
        self.assertEqual((allow, reason), (True, ""))

    def test_unaligned_holds_until_ttft_when_peer_decodes(self):
        state = CoalesceState()
        for _ in range(P.ttft_ticks):
            state, allow, reason = tick(state, prefillable=(1, 0), pending=(16384, 0))
            self.assertEqual((allow, reason), (False, "align"))
        _, allow, reason = tick(state, prefillable=(1, 0), pending=(16384, 0))
        self.assertEqual((allow, reason), (True, "ttft"))

    def test_unaligned_with_idle_peer_fires_after_stall(self):
        state = CoalesceState()
        out = []
        for _ in range(P.stall_ticks + 1):
            state, allow, reason = tick(
                state, prefillable=(1, 0), running=(30, 0), pending=(4000, 0)
            )
            out.append((allow, reason))
        self.assertEqual(out[:-1], [(False, "align")] * P.stall_ticks)
        self.assertEqual(out[-1], (True, "stall_unaligned"))

    def test_underfilled_waits_while_work_grows_then_stalls(self):
        state, allow, reason = tick(CoalesceState(), pending=(2000, 2000))
        self.assertEqual((allow, reason), (False, "fill_wait"))
        state, allow, reason = tick(state, pending=(3000, 2000), queue=(6, 5))
        self.assertEqual((allow, reason), (False, "fill_wait"))
        self.assertEqual(state.stall, 0)
        for i in range(P.stall_ticks):
            state, allow, reason = tick(state, pending=(3000, 2000), queue=(6, 5))
            expect = (True, "stall") if i == P.stall_ticks - 1 else (False, "fill_wait")
            self.assertEqual((allow, reason), expect)

    def test_kv_high_fires_unaligned(self):
        _, allow, reason = tick(
            CoalesceState(), prefillable=(1, 0), pending=(100, 0), kv_high=(1, 0)
        )
        self.assertEqual((allow, reason), (True, "kv_high"))

    def test_kv_high_on_non_prefillable_rank_ignored(self):
        _, allow, reason = tick(
            CoalesceState(), prefillable=(1, 0), pending=(100, 0), kv_high=(0, 1)
        )
        self.assertEqual((allow, reason), (False, "align"))

    def test_partial_bound(self):
        state = CoalesceState(hold_ticks=P.partial_ticks)
        _, allow, reason = tick(
            state, prefillable=(1, 0), pending=(16384, 0), partial=(1, 0)
        )
        self.assertEqual((allow, reason), (True, "partial"))

    def test_pending_capped_at_chunk(self):
        _, allow, reason = tick(CoalesceState(), pending=(100000, 0), queue=(9, 9))
        self.assertEqual((allow, reason), (False, "fill_wait"))


if __name__ == "__main__":
    unittest.main()
