"""Window retirement must keep the SWA window behind the replay-trimmed match.

Matching drops the decoder-replay tail before page alignment, so a match can
end a page below the window the default retention keeps. Freeing that window
makes the first repeat fall back to an older checkpoint despite a Full-KV hit.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.mem_cache.kv.advance import retire_window_rows
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def retire(
    length,
    window=128,
    page=256,
    replay=128,
    sharing=True,
    retain_floor=None,
    previous=0,
):
    """Retire a request at ``length - 1`` and return its new window cursor."""
    kv = SimpleNamespace(
        holds_kv=True,
        cache_protected_len=0,
        req_pool_idx=0,
        evicted=previous,
        window_dead_lo=lambda _: 0,
        get_evicted_seqlen=lambda _: previous,
    )
    kv.set_evicted_seqlen = lambda _, value: setattr(kv, "evicted", value)
    table = Mock()
    table.__getitem__ = Mock(return_value=object())
    retire_window_rows(
        SimpleNamespace(kv=kv),
        length - 1,
        window_sub_pool="swa",
        sliding_window_size=window,
        page_size=page,
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        token_to_kv_pool_allocator=SimpleNamespace(),
        supports_prefix_sharing=sharing,
        retain_floor=retain_floor,
        free_segment=Mock(),
        reprefill_tail_tokens=replay,
    )
    return kv.evicted


class TestReplayRetention(unittest.TestCase):
    def test_keeps_window_before_trimmed_match(self):
        for length in (32576, 32448, 269913, 1029986):
            with self.subTest(length=length):
                boundary = (length - 128) // 256 * 256
                self.assertLessEqual(retire(length), boundary - 128)

    def test_all_page_alignments(self):
        for page, window, replay in ((256, 128, 128), (64, 128, 128), (16, 8, 8)):
            for remainder in range(page):
                length = 64 * page + 2 * window + remainder
                with self.subTest(page=page, window=window, remainder=remainder):
                    boundary = (length - replay) // page * page
                    self.assertLessEqual(
                        retire(length, window, page, replay), boundary - window
                    )

    def test_no_replay_keeps_existing_policy(self):
        for length in range(32000, 32512):
            self.assertEqual(retire(length, replay=0), (length - 1 - 256) // 256 * 256)

    def test_retains_at_most_one_extra_page(self):
        for length in range(32000, 32512):
            extra = retire(length, replay=0) - retire(length)
            self.assertGreaterEqual(extra, 0)
            self.assertLessEqual(extra, 256)

    def test_bad_alignment(self):
        self.assertEqual(retire(32576, replay=0), 32256)
        self.assertEqual(retire(32576), 32000)
        self.assertEqual(retire(32448), 32000)

    def test_no_prefix_sharing_ignores_replay(self):
        for replay in (0, 128):
            self.assertEqual(retire(32576, replay=replay, sharing=False), 32256)

    def test_floor_and_cursor_still_apply(self):
        self.assertEqual(retire(32576, retain_floor=30000), 29952)
        # Freed rows stay freed.
        self.assertEqual(retire(32576, previous=32256), 32256)


if __name__ == "__main__":
    unittest.main()
