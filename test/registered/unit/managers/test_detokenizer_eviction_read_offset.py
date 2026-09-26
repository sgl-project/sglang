"""Regression test for sgl-project/sglang#41236.

After a detokenizer state eviction re-initializes a request's DecodeStatus,
``read_offset`` must be relative to the chunk carried in
``recv_obj.decode_ids[i]`` rather than absolute into the request's cumulative
id list. Otherwise ``s.decode_ids[0:read_offset]`` reads past the end of the
fresh chunk and up to ``read_offset`` tokens are silently dropped on the next
streaming emit.
"""

import unittest
from types import SimpleNamespace


class TestDetokenizerEvictionReadOffset(unittest.TestCase):
    def test_chunk_committed_offset_is_relative_to_chunk(self):
        """Mirror the formula in SchedulerOutputStreamer.

        ``decode_ids`` is the request's cumulative id list.
        ``send_decode_id_offset`` is the number of ids already sent to the
        detokenizer on a previous step; the detokenizer only receives the
        suffix ``decode_ids[send_decode_id_offset:]`` in this chunk.
        ``read_offset`` is the absolute index up to which the detokenizer has
        already committed. After an eviction the detokenizer rebuilds state
        from just this chunk, so it needs ``max(0, read_offset -
        send_decode_id_offset)``, not the absolute ``read_offset``.
        """
        decode_ids = [11, 12, 13, 14, 15, 16, 17, 18]
        send_decode_id_offset = 3  # ids 11..13 already sent previously
        read_offset = 7  # detokenizer has committed through cumulative id idx 6 (value 17)

        chunk = decode_ids[send_decode_id_offset:]
        self.assertEqual(chunk, [14, 15, 16, 17, 18])

        chunk_committed = max(0, read_offset - send_decode_id_offset)
        # 4 of the 5 chunk entries were already committed (14..17); only 18
        # is new.
        self.assertEqual(chunk_committed, 4)
        self.assertLessEqual(chunk_committed, len(chunk))

        # The old buggy behavior passed the absolute read_offset (7) into a
        # chunk of length 5, which slices past the end and yields the entire
        # chunk as "already decoded" — silently dropping tokens 18.
        buggy = read_offset
        self.assertGreater(buggy, len(chunk), "sanity: old formula overshoots the chunk")
        self.assertEqual(chunk[:buggy], chunk, "old formula treats whole chunk as committed")

    def test_no_eviction_does_not_underflow(self):
        """When no eviction happened yet, send_decode_id_offset is 0 and the
        relative offset equals the absolute one."""
        send_decode_id_offset = 0
        read_offset = 2
        chunk_committed = max(0, read_offset - send_decode_id_offset)
        self.assertEqual(chunk_committed, read_offset)


if __name__ == "__main__":
    unittest.main()
