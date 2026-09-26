"""Unit tests for DetokenizerManager incremental-decoding offsets.

Pure CPU: builds the manager without __init__ (no IPC, no real tokenizer) and
drives _decode_batch_token_id_output directly with a byte-level stub tokenizer.
"""

import unittest

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Lone UTF-8 continuation byte: never decodes to a character on its own, so a
# byte-fallback vocabulary renders it as U+FFFD no matter what follows.
STRAY_BYTE = 0x80
# Three-byte UTF-8 sequence for U+4E2D, one byte per token.
CJK_BYTES = (0xE4, 0xB8, 0xAD)
TEXT_TOKEN = 256


class _ByteFallbackTokenizer:
    """Ids 0-255 are raw bytes, 256+ are ASCII letters.

    Mirrors the part of a byte-fallback vocabulary these tests depend on:
    undecodable bytes render as U+FFFD, and a multi-byte character only
    appears once all of its bytes have arrived.
    """

    is_fast = True

    def batch_decode(self, ids_list, skip_special_tokens=True, **kwargs):
        return [self._decode(ids) for ids in ids_list]

    @staticmethod
    def _decode(ids):
        buf = bytearray()
        for i in ids:
            if i < 256:
                buf.append(i)
            else:
                buf.append(ord("a") + (i - TEXT_TOKEN) % 26)
        return buf.decode("utf-8", errors="replace")


def _make_manager():
    m = DetokenizerManager.__new__(DetokenizerManager)
    m.tokenizer = _ByteFallbackTokenizer()
    m.vocab_size = None
    m.decode_status = {}
    m.disable_tokenizer_batch_decode = False
    m.is_tool_call_parser_gpt_oss = False
    return m


class _Recv:
    """The subset of BatchTokenIDOutput that the decode path reads."""

    def __init__(self, rid, decode_ids, read_offset):
        self.rids = [rid]
        self.finished_reasons = [None]
        self.decoded_texts = [""]
        self.decode_ids = [list(decode_ids)]
        self.read_offsets = [read_offset]
        self.skip_special_tokens = [True]
        self.spaces_between_special_tokens = [True]
        self.no_stop_trim = [False]


def _stream(manager, rid, prompt, steps):
    """Drive one streaming request and return its decode status.

    The scheduler sends the surrogate-context prompt tail together with the
    first output chunk, then one event per later chunk, so the prompt is
    context only and never part of the decoded output.
    """
    manager._decode_batch_token_id_output(
        _Recv(rid, list(prompt) + list(steps[0]), len(prompt))
    )
    for chunk in steps[1:]:
        manager._decode_batch_token_id_output(_Recv(rid, chunk, 0))
    return manager.decode_status[rid]


def _run_stray_byte_stream(n_steps, tokens_per_step):
    """Return (re-decoded window size, decoded text) after n_steps."""
    manager = _make_manager()
    s = _stream(
        manager,
        "rid",
        [TEXT_TOKEN] * 5,
        [[STRAY_BYTE] * tokens_per_step] * n_steps,
    )
    return len(s.decode_ids) - s.surr_offset, s.get_decoded_text()


class TestIncrementalDecodeWindow(unittest.TestCase):
    def test_replacement_char_stream_keeps_offsets_moving(self):
        """A stream of byte-fallback tokens must not freeze the offsets.

        The commit gate treats "decoded tail ends in U+FFFD" as "the trailing
        bytes are an incomplete character", but a byte-fallback token satisfies
        that forever. While the offsets are frozen, decode_ids[surr_offset:]
        grows with the output and is re-decoded on every step, so the request
        costs O(n^2) in its own output length and the client gets no text.
        """
        tokens_per_step = 4
        short_window, short_text = _run_stray_byte_stream(32, tokens_per_step)
        long_window, long_text = _run_stray_byte_stream(256, tokens_per_step)

        self.assertNotEqual(short_text, "", "no text ever reached the client")
        self.assertNotEqual(long_text, "", "no text ever reached the client")

        # The window is re-decoded every step, so it must not scale with the
        # output length.
        self.assertLess(
            long_window,
            256 * tokens_per_step // 4,
            f"re-decoded window grew to {long_window} tokens over "
            f"{256 * tokens_per_step} output tokens",
        )
        self.assertLessEqual(
            long_window,
            2 * short_window,
            f"re-decoded window grew with output length: {short_window} "
            f"tokens at 32 steps, {long_window} at 256",
        )

    def test_incomplete_character_is_still_held_back(self):
        """The stall is what keeps a split character intact, so it must stay.

        Guards the other direction of the same gate: committing on every step
        would emit U+FFFD for the leading bytes of a multi-byte character and
        lose the character itself.
        """
        manager = _make_manager()
        s = _stream(manager, "rid", [TEXT_TOKEN], [[b] for b in CJK_BYTES])
        self.assertEqual(s.get_decoded_text(), "中")

    def test_stall_counter_resets_on_a_clean_step(self):
        """Only *consecutive* stalls may force a commit.

        A stream that stalls briefly and recovers, over and over, is normal
        traffic: every multi-byte character stalls while its bytes arrive. If
        those stalls accumulated across characters, a long CJK response would
        eventually commit in the middle of one.
        """
        n_chars = 32
        steps = [[b] for _ in range(n_chars) for b in CJK_BYTES]
        manager = _make_manager()
        s = _stream(manager, "rid", [TEXT_TOKEN], steps)
        self.assertEqual(s.get_decoded_text(), "中" * n_chars)


if __name__ == "__main__":
    unittest.main()
