"""Regression test: incremental detokenization must never re-send streamed text.

`sent_offset` records how much of the decoded text the client already has. In
the "U+FFFD" recovery arm it is recomputed as
`decoded_text_len + len(find_printable_text(new_text))`, and
`find_printable_text` is not monotonic: a step whose tail is an incomplete
character can yield a *shorter* printable prefix than the step before it, so
`sent_offset` moves backwards, the next clean step under-counts
`pending = sent_offset - decoded_text_len`, and the client is sent text it
already received.

Three reachable shapes, all of them in the last-space branch of the heuristic
(`return text[: text.rfind(" ") + 1]`):

1. retreat to a space -- `"A 世a�"` yields `"A "` after `"A 世�"` yielded
   `"A 世"`;
2. collapse to `""` -- `"你好，�"` yields `""` (the fullwidth comma is not
   classified as CJK and there is no space to fall back to) after `"你�"`
   yielded `"你"`;
3. an empty delta while `pending > 0` -- `""` yields `""`, which would reset
   `sent_offset` to `decoded_text_len` and lose the uncommitted prefix.

Pure CPU: drives the real `_decode_batch_token_id_output` with a stub `self` and
a byte-level stub tokenizer, so no DetokenizerManager.__init__, no IPC and no
tokenizer download.
"""

import types
import unittest
from types import SimpleNamespace

from sglang.srt.managers import detokenizer_manager
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

# Ids >= TEXT_TOKEN decode to the byte (id - TEXT_TOKEN); the prompt tail uses
# the same encoding, so one stub covers both sides of the surrogate window.
TEXT_TOKEN = 256

# Byte-level pieces. "你好，世" splits so that two consecutive steps end on an
# incomplete character whose printable prefix shrinks (shape 2).
TAIL = b"Q: "
GEN_BYTES = "你好，世".encode("utf-8")
# 5 + 5 + 2 bytes: the first two steps end mid-character, the third completes it.
CHUNKS = [GEN_BYTES[:5], GEN_BYTES[5:10], GEN_BYTES[10:]]

# Shape 1: "A 世" then "A 世好a" + a dangling byte, whose printable prefix
# retreats from `"A 世"` to `"A "`.
SPACE_BYTES = "A 世好a世".encode("utf-8")
SPACE_CHUNKS = [SPACE_BYTES[:6], SPACE_BYTES[6:10], SPACE_BYTES[10:]]


class _ByteTokenizer:
    """Byte-fallback stub: one token per byte, decoded as UTF-8.

    Mirrors a byte-level BPE whose vocabulary holds single bytes, so a
    multi-byte character can arrive split across tokens and decode to U+FFFD
    until its final byte lands.
    """

    is_fast = True
    vocab_size = TEXT_TOKEN + 256

    @staticmethod
    def _text(ids):
        return bytes(i - TEXT_TOKEN for i in ids).decode("utf-8", errors="replace")

    def decode(self, ids, skip_special_tokens=True, spaces_between_special_tokens=True):
        return self._text(ids)

    def batch_decode(
        self, ids_list, skip_special_tokens=True, spaces_between_special_tokens=True
    ):
        return [self._text(ids) for ids in ids_list]


def _manager(tokenizer, disable_tokenizer_batch_decode=False):
    """A DetokenizerManager standing in for the real one, without __init__."""
    stub = SimpleNamespace(
        tokenizer=tokenizer,
        vocab_size=tokenizer.vocab_size,
        decode_status={},
        disable_tokenizer_batch_decode=disable_tokenizer_batch_decode,
    )
    stub.trim_matched_stop = types.MethodType(
        DetokenizerManager.trim_matched_stop, stub
    )
    stub._clamp_decode_ids = DetokenizerManager._clamp_decode_ids
    stub._grouped_batch_decode = types.MethodType(
        DetokenizerManager._grouped_batch_decode, stub
    )
    return stub


def _stream(chunks, disable_tokenizer_batch_decode=False, record_clamp=False):
    """Feed one BatchTokenIDOutput per chunk; return (stream, offsets, clamp).

    Follows the server: the first output carries the prompt tail plus the first
    generated tokens with `read_offset` marking the prompt boundary, and every
    later output carries only the ids added since (`send_decode_id_offset`).
    The final output is empty and finished, which materializes the request.

    With `record_clamp`, each call to `find_printable_text` records
    `(sent_offset before, decoded_text_len, len(printable))` so a test can tell
    which way the `max()` went on that step: the clamp is load-bearing exactly
    when `sent_offset > decoded_text_len + len(printable)`.
    """
    tokenizer = _ByteTokenizer()
    mgr = _manager(tokenizer, disable_tokenizer_batch_decode)
    ids = [TEXT_TOKEN + b for b in TAIL]
    read_offset = len(ids)
    out, seen, clamp = [], [], []

    real_find_printable_text = detokenizer_manager.find_printable_text

    def spy(text):
        result = real_find_printable_text(text)
        state = mgr.decode_status.get("r")
        if state is not None:
            clamp.append((state.sent_offset, state.decoded_text_len, len(result)))
        return result

    if record_clamp:
        detokenizer_manager.find_printable_text = spy
    try:
        for i, chunk in enumerate(list(chunks) + [b""]):
            finished = i == len(chunks)
            ids = ids + [TEXT_TOKEN + b for b in chunk]
            recv = SimpleNamespace(
                rids=["r"],
                decode_ids=[list(ids) if i == 0 else [TEXT_TOKEN + b for b in chunk]],
                read_offsets=[read_offset if i == 0 else 0],
                decoded_texts=[""],
                finished_reasons=[{} if finished else None],
                no_stop_trim=[False],
                skip_special_tokens=[True],
                spaces_between_special_tokens=[True],
            )
            out.append(DetokenizerManager._decode_batch_token_id_output(mgr, recv)[0])
            state = mgr.decode_status.get("r")
            if state is not None:
                seen.append((state.decoded_text_len, state.sent_offset))
    finally:
        detokenizer_manager.find_printable_text = real_find_printable_text
    return "".join(out), seen, clamp


def _reference(gen_bytes):
    """What the client should end up with: the generated text, once."""
    tail = TAIL.decode("utf-8")
    whole = (TAIL + gen_bytes).decode("utf-8")
    return whole[len(tail) :]


class TestStreamOffsets(unittest.TestCase):
    def test_incomplete_character_does_not_duplicate_text(self):
        stream, _, _ = _stream(CHUNKS)
        self.assertEqual(stream, _reference(GEN_BYTES))

    def test_incomplete_character_does_not_duplicate_text_without_batch_decode(self):
        # Same bookkeeping on the per-row decode path (gpt-oss style).
        stream, _, _ = _stream(CHUNKS, disable_tokenizer_batch_decode=True)
        self.assertEqual(stream, _reference(GEN_BYTES))

    def test_space_retreat_does_not_duplicate_text(self):
        # Shape 1: the printable prefix retreats to a space rather than
        # collapsing, so the under-count is one character rather than all of it.
        stream, _, _ = _stream(SPACE_CHUNKS)
        self.assertEqual(stream, _reference(SPACE_BYTES))

    def test_empty_streaming_step_does_not_reset_sent_offset(self):
        # Shape 3: a step with no new bytes must not reset `sent_offset` to
        # `decoded_text_len` while a printable prefix is still uncommitted.
        chunks = [GEN_BYTES[:5], b"", GEN_BYTES[5:]]
        stream, _, _ = _stream(chunks)
        self.assertEqual(stream, _reference(GEN_BYTES))

    def test_sent_offset_never_retreats(self):
        # The invariant the streaming arm relies on to compute `pending`.
        for chunks in (CHUNKS, SPACE_CHUNKS, [GEN_BYTES[:5], b"", GEN_BYTES[5:]]):
            with self.subTest(chunks=chunks):
                _, seen, _ = _stream(chunks)
                offsets = [sent for _, sent in seen]
                self.assertEqual(
                    offsets, sorted(offsets), f"sent_offset retreated: {seen}"
                )

    def test_clamp_is_exercised_in_both_directions(self):
        # The patched assignment is `max(sent_offset, decoded_text_len +
        # len(printable))`. This pins that the tests above reach both outcomes,
        # so the fix is not carried by a line that always evaluates one way:
        # the clamp is load-bearing when the heuristic retreats, and a no-op
        # when it grows.
        _, _, clamp = _stream(CHUNKS, record_clamp=True)
        fired = [c for c in clamp if c[0] > c[1] + c[2]]
        not_fired = [c for c in clamp if c[0] <= c[1] + c[2]]
        self.assertTrue(fired, f"clamp never fired; steps seen: {clamp}")
        self.assertTrue(not_fired, f"clamp always fired; steps seen: {clamp}")

    def test_plain_ascii_stream_is_unchanged(self):
        # The common case commits on every step and must stay byte-exact.
        chunks = [b"hello ", b"world ", b"bye"]
        stream, _, _ = _stream(chunks)
        self.assertEqual(stream, b"hello world bye".decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
