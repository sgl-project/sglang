import unittest

from sglang.srt.disaggregation.common.utils import (
    DSparkHiddenChunk,
    DSparkHiddenRequestState,
)


def _chunk(start: int, rows: int, is_last: bool = False) -> DSparkHiddenChunk:
    return DSparkHiddenChunk(
        room=1,
        prefill_rank=0,
        hidden_start=start,
        row_len=rows,
        is_last_hidden_chunk=is_last,
        dst_indices=list(range(rows)),
    )


class TestDSparkHiddenRequestState(unittest.TestCase):
    def test_disabled_state_is_done_for_hidden_but_not_kv(self):
        state = DSparkHiddenRequestState.disabled()

        self.assertFalse(state.enabled)
        self.assertFalse(state.streaming)
        self.assertTrue(state.hidden_request_done())
        self.assertFalse(state.kv_request_done())
        self.assertFalse(state.request_done())

        state.mark_kv_done()
        self.assertTrue(state.request_done())

    def test_full_state_waits_only_for_kv_done(self):
        state = DSparkHiddenRequestState.full(2, 6)

        self.assertTrue(state.enabled)
        self.assertFalse(state.streaming)
        self.assertEqual(state.start, 2)
        self.assertEqual(state.next_start, 2)
        self.assertEqual(state.end, 6)
        self.assertTrue(state.hidden_request_done())
        self.assertFalse(state.request_done())

        state.mark_kv_done()
        self.assertTrue(state.request_done())

    def test_streaming_hidden_done_is_separate_from_request_done(self):
        state = DSparkHiddenRequestState.streaming_state(0, 8)

        self.assertEqual(state.accept_chunk(_chunk(0, 4)), "accepted")
        self.assertFalse(state.hidden_request_done())
        self.assertFalse(state.request_done())

        self.assertEqual(state.accept_chunk(_chunk(4, 4, is_last=True)), "accepted")
        self.assertTrue(state.hidden_request_done())
        self.assertFalse(state.request_done())

        state.mark_kv_done()
        self.assertTrue(state.kv_request_done())
        self.assertTrue(state.request_done())

    def test_streaming_hidden_rejects_future_and_stale_chunks(self):
        state = DSparkHiddenRequestState.streaming_state(0, 8)

        self.assertEqual(state.accept_chunk(_chunk(4, 4)), "future")
        self.assertEqual(state.accept_chunk(_chunk(0, 4)), "accepted")
        self.assertEqual(state.accept_chunk(_chunk(0, 4)), "stale")

    def test_streaming_hidden_last_chunk_must_end_at_expected_offset(self):
        state = DSparkHiddenRequestState.streaming_state(0, 8)

        with self.assertRaisesRegex(RuntimeError, "unexpected offset"):
            state.accept_chunk(_chunk(0, 4, is_last=True))

    def test_streaming_hidden_chunk_cannot_exceed_expected_range(self):
        state = DSparkHiddenRequestState.streaming_state(0, 8)

        with self.assertRaisesRegex(RuntimeError, "exceeds request range"):
            state.accept_chunk(_chunk(0, 9))

    def test_streaming_hidden_reset_returns_to_disabled_state(self):
        state = DSparkHiddenRequestState.streaming_state(0, 8)
        self.assertEqual(state.accept_chunk(_chunk(0, 8, is_last=True)), "accepted")
        state.mark_kv_done()
        self.assertTrue(state.request_done())

        state.reset()

        self.assertFalse(state.enabled)
        self.assertFalse(state.streaming)
        self.assertEqual(state.start, 0)
        self.assertEqual(state.next_start, 0)
        self.assertEqual(state.end, 0)
        self.assertTrue(state.hidden_request_done())
        self.assertFalse(state.kv_request_done())
        self.assertFalse(state.request_done())

    def test_hidden_chunk_descriptor_keeps_ack_endpoint_metadata(self):
        chunk = DSparkHiddenChunk(
            room=3,
            prefill_rank=7,
            hidden_start=16,
            row_len=2,
            is_last_hidden_chunk=True,
            dst_indices=[4, 5],
            ack_host="127.0.0.1",
            ack_port=12345,
        )

        self.assertEqual(chunk.room, 3)
        self.assertEqual(chunk.prefill_rank, 7)
        self.assertEqual(chunk.hidden_start, 16)
        self.assertEqual(chunk.row_len, 2)
        self.assertTrue(chunk.is_last_hidden_chunk)
        self.assertEqual(chunk.dst_indices, [4, 5])
        self.assertEqual(chunk.ack_host, "127.0.0.1")
        self.assertEqual(chunk.ack_port, 12345)


if __name__ == "__main__":
    unittest.main()
