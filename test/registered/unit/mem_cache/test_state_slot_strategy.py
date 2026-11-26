"""Per-request state-slot demand: the sizing count must match what the runtime
actually allocates.

`StateSlotStrategy.slots_per_request()` is what the memory pools are sized
from. Every required allocation at runtime takes `initial_tracking_slots`
(see `fresh_request_slots`, `pending_request_slots`, `held_request_slots` and
`acquire` in `sglang.srt.mem_cache.kv.state_storage`), which honours the lazy
extra buffer; the second tracking slot is taken later and optionally by
`take_tracking_slot`, and a shortfall there only skips a snapshot. These cases
pin the counts so the two cannot drift apart again.

CPU only: no device, no engine.
"""

import unittest

from sglang.srt.mem_cache.kv.state_policy import StateSlotStrategy


def strategy(**kwargs):
    base = dict(
        radix_cache=True,
        decode_only=False,
        extra_buffer=True,
        overlap=True,
        lazy=False,
        skip_decode_lock=False,
    )
    base.update(kwargs)
    return StateSlotStrategy(**base)


class TestStateSlotStrategy(unittest.TestCase):
    def test_lazy_with_decode_lock_skip_takes_three_slots(self):
        # working slot + one radix checkpoint + one lazy tracking slot
        self.assertEqual(
            strategy(lazy=True, skip_decode_lock=True).slots_per_request(), 3
        )

    def test_lazy_without_decode_lock_skip_takes_four_slots(self):
        # two checkpoints without the skip, still one lazy tracking slot
        self.assertEqual(
            strategy(lazy=True, skip_decode_lock=False).slots_per_request(), 4
        )

    def test_non_lazy_counts_both_tracking_slots(self):
        for skip, expected in ((True, 4), (False, 5)):
            with self.subTest(skip_decode_lock=skip):
                self.assertEqual(
                    strategy(lazy=False, skip_decode_lock=skip).slots_per_request(),
                    expected,
                )

    def test_lazy_does_not_change_the_count_without_the_extra_buffer(self):
        # No tracking slots exist to defer, so the count is the working slot
        # plus two radix checkpoints. The decode-lock skip drops a checkpoint
        # only together with the extra buffer, so it does not apply here.
        for lazy in (True, False):
            for skip in (True, False):
                with self.subTest(lazy=lazy, skip_decode_lock=skip):
                    self.assertEqual(
                        strategy(
                            extra_buffer=False, lazy=lazy, skip_decode_lock=skip
                        ).slots_per_request(),
                        3,
                    )

    def test_decode_only_ignores_lazy(self):
        # PD decode builds its count with lazy off, so the two must agree here
        for lazy in (True, False):
            with self.subTest(lazy=lazy):
                self.assertEqual(
                    strategy(decode_only=True, lazy=lazy).slots_per_request(), 3
                )

    def test_radix_cache_off_ignores_lazy(self):
        for lazy in (True, False):
            with self.subTest(lazy=lazy):
                self.assertEqual(
                    strategy(radix_cache=False, lazy=lazy).slots_per_request(), 3
                )

    def test_tracking_slot_properties(self):
        lazy = strategy(lazy=True)
        self.assertEqual(lazy.initial_tracking_slots, 1)
        self.assertEqual(lazy.tracking_slots, 2)
        eager = strategy(lazy=False)
        self.assertEqual(eager.initial_tracking_slots, eager.tracking_slots)

    def test_only_lazy_configurations_differ_from_the_eager_count(self):
        """Exhaustive over the axes that reach slots_per_request()."""
        import itertools

        moved = []
        for radix, dec, xbuf, overlap, skip in itertools.product(
            (True, False), repeat=5
        ):
            kw = dict(
                radix_cache=radix,
                decode_only=dec,
                extra_buffer=xbuf,
                overlap=overlap,
                skip_decode_lock=skip,
            )
            a = strategy(lazy=True, **kw).slots_per_request()
            b = strategy(lazy=False, **kw).slots_per_request()
            if a != b:
                moved.append((kw, b, a))
        for kw, b, a in moved:
            self.assertEqual(b - a, 1, f"{kw}: expected one fewer slot, got {b}->{a}")
            self.assertTrue(kw["radix_cache"] and not kw["decode_only"])
            self.assertTrue(kw["extra_buffer"] and kw["overlap"])
        self.assertEqual(len(moved), 2, f"unexpected set of changed cells: {moved}")


if __name__ == "__main__":
    unittest.main()
