"""Regression test: incremental detokenization must never re-send streamed text.

`sent_offset` records how much of the decoded text the client already has. In
the "U+FFFD" recovery arm it is recomputed as
`decoded_text_len + len(find_printable_text(new_text))`, and
`find_printable_text` is not monotonic: a step whose tail is an incomplete
character can yield a *shorter* printable prefix than the step before it. For
example `"你�"` yields `"你"` (penultimate-CJK branch), while
`"你好，�"` yields `""` -- the fullwidth comma is not classified as CJK, so
the heuristic falls through to the last-space branch and there is no space to
fall back to. `sent_offset` then moves *backwards*, the next clean step
under-counts `pending = sent_offset - decoded_text_len`, and the client is sent
text it already received.

Pure CPU: drives the real `_decode_batch_token_id_output` with a stub `self` and
a byte-level stub tokenizer, so no DetokenizerManager.__init__, no IPC and no
tokenizer download.
"""

import types
import unittest
from types import SimpleNamespace

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Ids >= TEXT_TOKEN decode to the byte (id - TEXT_TOKEN); the prompt tail uses
# the same encoding, so one stub covers both sides of the surrogate window.
TEXT_TOKEN = 256

# Byte-level pieces. "你好，世" splits so that two consecutive steps end on an
# incomplete character whose printable prefix shrinks.
TAIL = b"Q: "
GEN_BYTES = "你好，世".encode("utf-8")
# 5 + 5 + 2 bytes: the first two steps end mid-character, the third completes it.
CHUNKS = [GEN_BYTES[:5], GEN_BYTES[5:10], GEN_BYTES[10:]]


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


def _stream(chunks, disable_tokenizer_batch_decode=False):
    """Feed one BatchTokenIDOutput per chunk; return (client stream, offsets).

    Follows the server: the first output carries the prompt tail plus the first
    generated tokens with `read_offset` marking the prompt boundary, and every
    later output carries only the ids added since (`send_decode_id_offset`).
    The final output is empty and finished, which materializes the request.
    """
    tokenizer = _ByteTokenizer()
    mgr = _manager(tokenizer, disable_tokenizer_batch_decode)
    ids = [TEXT_TOKEN + b for b in TAIL]
    read_offset = len(ids)
    out, seen = [], []
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
    return "".join(out), seen


class TestStreamOffsets(unittest.TestCase):
    def _reference(self):
        """What the client should end up with: the generated text, once."""
        tail = TAIL.decode("utf-8")
        whole = (TAIL + GEN_BYTES).decode("utf-8")
        return whole[len(tail) :]

    def test_incomplete_character_does_not_duplicate_text(self):
        stream, _ = _stream(CHUNKS)
        self.assertEqual(stream, self._reference())

    def test_incomplete_character_does_not_duplicate_text_without_batch_decode(self):
        # Same bookkeeping on the per-row decode path (gpt-oss style).
        stream, _ = _stream(CHUNKS, disable_tokenizer_batch_decode=True)
        self.assertEqual(stream, self._reference())

    def test_sent_offset_never_retreats(self):
        # The invariant the streaming arm relies on to compute `pending`.
        _, seen = _stream(CHUNKS)
        offsets = [sent for _, sent in seen]
        self.assertEqual(offsets, sorted(offsets), f"sent_offset retreated: {seen}")

    def test_plain_ascii_stream_is_unchanged(self):
        # The common case commits on every step and must stay byte-exact.
        chunks = [b"hello ", b"world ", b"bye"]
        stream, _ = _stream(chunks)
        self.assertEqual(stream, b"hello world bye".decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
