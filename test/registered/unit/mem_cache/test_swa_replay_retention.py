"""Exercise SWA eviction frontiers without accelerator execution or model weights."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.mem_cache import common as cache_common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ReplayRetentionTest(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(cache_common.__file__).parent
        cls.free_window = staticmethod(cache_common.free_swa_out_of_window_slots)

    def frontier(
        self,
        length,
        window=128,
        page=256,
        replay=128,
        sharing=True,
        retain_floor=None,
        previous=0,
    ):
        kv = SimpleNamespace(
            holds_kv=True,
            cache_protected_len=0,
            req_pool_idx=0,
            swa_dead_lo=lambda _: 0,
            get_evicted_seqlen=lambda _: previous,
        )
        kv.set_evicted_seqlen = lambda _, value: setattr(kv, "evicted", value)
        table = Mock()
        table.__getitem__ = Mock(return_value=object())
        allocator = SimpleNamespace(free_swa_segment=Mock())
        self.free_window(
            SimpleNamespace(kv=kv),
            length - 1,
            sliding_window_size=window,
            page_size=page,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            token_to_kv_pool_allocator=allocator,
            supports_prefix_sharing=sharing,
            retain_floor=retain_floor,
            reprefill_tail_tokens=replay,
        )
        return kv.evicted

    def test_first_replay_keeps_window_before_trimmed_match(self):
        for length in (32576, 32448, 269913, 1029986):
            with self.subTest(length=length):
                boundary = (length - 128) // 256 * 256
                self.assertLessEqual(self.frontier(length), boundary - 128)

    def test_all_page_alignments_preserve_replay_history(self):
        for page, window, replay in (
            (256, 128, 128),
            (64, 128, 128),
            (16, 8, 8),
            (1, 4, 4),
        ):
            for remainder in range(page):
                length = 64 * page + 2 * window + remainder
                with self.subTest(page=page, window=window, remainder=remainder):
                    boundary = (length - replay) // page * page
                    self.assertLessEqual(
                        self.frontier(length, window, page, replay), boundary - window
                    )

    def test_no_replay_keeps_existing_eviction_policy(self):
        for length in range(32000, 32512):
            expected = (length - 1 - 256) // 256 * 256
            self.assertEqual(self.frontier(length, replay=0), expected)

    def test_256_token_pages_retain_at_most_one_extra_page(self):
        for length in range(32000, 32512):
            before = self.frontier(length, replay=0)
            after = self.frontier(length)
            self.assertGreaterEqual(before - after, 0)
            self.assertLessEqual(before - after, 256)

    def test_bad_alignment_exposes_original_frontier_mismatch(self):
        self.assertEqual(self.frontier(32576, replay=0), 32256)
        self.assertEqual(self.frontier(32576), 32000)
        self.assertEqual(self.frontier(32448), 32000)

    def test_disabled_prefix_sharing_does_not_retain_replay_history(self):
        for replay in (0, 128):
            self.assertEqual(self.frontier(32576, replay=replay, sharing=False), 32256)

    def test_checkpoint_floor_and_eviction_cursor_still_apply(self):
        self.assertEqual(self.frontier(32576, retain_floor=30000), 29952)
        # Previously freed slots cannot be recovered by changing a frontier.
        self.assertEqual(self.frontier(32576, previous=32256), 32256)

    def test_scheduler_and_checkpoint_paths_pass_the_replay_guard(self):
        for relative, expected in (
            ("base_prefix_cache.py", "self.swa_reprefill_tail_tokens(req)"),
            (
                "unified_cache/components/swa.py",
                "self.cache.swa_reprefill_tail_tokens(req)",
            ),
        ):
            tree = ast.parse((self.source / relative).read_text())
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and ast.unparse(node.func) == "free_swa_out_of_window_slots"
            ]
            self.assertEqual(len(calls), 1)
            keyword = next(
                (kw for kw in calls[0].keywords if kw.arg == "reprefill_tail_tokens"),
                None,
            )
            self.assertIsNotNone(keyword, relative)
            self.assertEqual(ast.unparse(keyword.value), expected)


if __name__ == "__main__":
    unittest.main()
