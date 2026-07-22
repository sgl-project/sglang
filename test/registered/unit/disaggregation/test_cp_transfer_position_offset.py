"""Regression tests for CP-sharded disaggregation transfer indices."""

import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.common.conn import CommonKVSender
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCPTransferPositionOffset(CustomTestCase):
    @staticmethod
    def _make_sender(*, cp_rank: int, cp_size: int, total: int, current: int):
        sender = CommonKVSender.__new__(CommonKVSender)
        sender.kv_mgr = SimpleNamespace(
            enable_all_cp_ranks_for_transfer=True,
            attn_cp_rank=cp_rank,
            attn_cp_size=cp_size,
        )
        sender.num_kv_indices = total
        sender.curr_idx = current
        return sender

    @staticmethod
    def _bucket_locs(token_locs, compression_ratio, position_offset):
        return MooncakeKVManager._dsv4_bucket_locs_from_token_locs(
            None, token_locs, compression_ratio, position_offset
        )

    def test_c4_offset_tracks_cp_slice_with_noncontiguous_physical_locs(self):
        # CP=3 partitions ten request-local entries as [0:4], [4:7], [7:10].
        # This chunk spans [3:7], so rank 1 drops one entry from its front.
        # Physical locs are intentionally non-monotonic to cover radix reuse.
        sender = self._make_sender(cp_rank=1, cp_size=3, total=10, current=3)
        chunk_locs = np.array([12, 305, 2, 88], dtype=np.int32)

        locs, index_slice, is_last, should_skip, position_offset = (
            sender._prepare_send_indices(
                chunk_locs,
                token_position_offset=4,
            )
        )

        np.testing.assert_array_equal(locs, np.array([305, 2, 88], np.int32))
        self.assertEqual(index_slice, slice(4, 7))
        self.assertFalse(is_last)
        self.assertFalse(should_skip)
        self.assertEqual(position_offset, 5)
        np.testing.assert_array_equal(
            self._bucket_locs(locs, 4, position_offset),
            np.array([88 // 4], dtype=np.int32),
        )
        self.assertEqual(self._bucket_locs(locs, 4, 4).size, 0)

    def test_c128_offset_tracks_unaligned_cp_boundary(self):
        # CP rank 1 starts at request-local entry 129. The transfer chunk
        # starts at 128, so the filtered position offset must advance by one.
        # Without that adjustment, the c128 boundary selects the neighboring
        # physical loc and silently transfers the wrong compressed slot.
        sender = self._make_sender(cp_rank=1, cp_size=2, total=258, current=128)
        chunk_locs = (
            np.arange(130, dtype=np.int32) * np.int32(997) + np.int32(13)
        ) % np.int32(100003)

        locs, index_slice, is_last, should_skip, position_offset = (
            sender._prepare_send_indices(
                chunk_locs,
                token_position_offset=128,
            )
        )

        self.assertEqual(index_slice, slice(129, 258))
        self.assertTrue(is_last)
        self.assertFalse(should_skip)
        self.assertEqual(position_offset, 129)
        expected = np.array([locs[126] // 128], dtype=np.int32)
        np.testing.assert_array_equal(
            self._bucket_locs(locs, 128, position_offset), expected
        )
        self.assertFalse(
            np.array_equal(
                self._bucket_locs(locs, 128, 128),
                expected,
            )
        )


if __name__ == "__main__":
    unittest.main()
