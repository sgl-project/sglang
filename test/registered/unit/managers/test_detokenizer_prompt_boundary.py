"""Incremental detokenization when a token-id prompt ends inside a UTF-8 character.

The output completes that character. The straddling character itself is not
emitted, but every generated character after it must be, with no U+FFFD for the
bytes the output completed. Byte-fallback tokenizers (SentencePiece <0xNN>
pieces, e.g. Gemma) decode each dangling byte as its own U+FFFD, which used to
make the detokenizer skip generated characters. Pure CPU: drives DetokenizeState
with in-memory tokenizers.
"""

import unittest
from array import array

from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from sglang.srt.managers.incremental_detokenize import (
    PROMPT_TAIL_LEN,
    DetokenizeHeader,
    DetokenizeState,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _byte_fallback_tokenizer():
    # Gemma's decoder chain; ids 0-255 are the <0xNN> byte pieces.
    vocab = {f"<0x{b:02X}>": b for b in range(256)}
    for piece in ["<unk>", "A", "▁ok", "▁hello", "\U0001d4f1", "\U0001d4ea"]:
        vocab[piece] = len(vocab)
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")


def _byte_level_tokenizer():
    # GPT-2 byte-to-unicode table; ids 0-255 are the single-byte tokens.
    printable = [*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    chars, extra = {}, 0
    for b in range(256):
        if b in printable:
            chars[b] = chr(b)
        else:
            chars[b] = chr(256 + extra)
            extra += 1
    vocab = {chars[b]: b for b in range(256)}
    vocab["<unk>"] = 256
    vocab[chars[0x20] + "ok"] = 257
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")


BF = _byte_fallback_tokenizer()
BL = _byte_level_tokenizer()


def _ids(tok, *pieces):
    return [p if isinstance(p, int) else tok.convert_tokens_to_ids(p) for p in pieces]


def _no_trim(output, finished_reason, no_stop_trim):
    return output


def _detokenize(tok, prompt_ids, output_ids, chunk_size=None):
    """Feed output_ids the way the detokenizer does; chunk_size=None is non-streaming."""
    header = DetokenizeHeader(
        prompt_tail_ids=array("q", prompt_ids[-PROMPT_TAIL_LEN:]),
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        no_stop_trim=False,
        http_worker_ipc=None,
    )
    state = DetokenizeState.from_header(header=header, vocab_size=len(tok))
    step = chunk_size or len(output_ids)
    chunks = [output_ids[i : i + step] for i in range(0, len(output_ids), step)]
    pieces, sent = [], 0
    for i, chunk in enumerate(chunks):
        sent += len(chunk)
        finished_reason = {"type": "length"} if i == len(chunks) - 1 else None
        state.append_row(
            rid="rid",
            output_ids=chunk,
            completion_tokens=sent,
            is_beam=False,
            vocab_size=len(tok),
        )
        read_ids, surr_ids = state.decode_inputs(
            finished_reason=finished_reason, trim_matched_stop=_no_trim
        )
        pieces.append(
            state.step(
                read_text=tok.decode(read_ids, skip_special_tokens=True),
                surr_text=tok.decode(surr_ids, skip_special_tokens=True),
                finished_reason=finished_reason,
                trim_matched_stop=_no_trim,
            )
        )
    return "".join(pieces)


class TestDetokenizerPromptBoundary(CustomTestCase):
    def _assert_all_modes(self, tok, prompt_ids, output_ids, expected):
        for chunk_size in (None, 1, 2):
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(
                    _detokenize(tok, prompt_ids, output_ids, chunk_size), expected
                )

    def test_byte_fallback_completed_char_keeps_following_chars(self):
        # Prompt ends with 3 of the 4 bytes of U+1D4E3; the output completes it.
        prompt = _ids(BF, "A", 0xF0, 0x9D, 0x93)
        output = _ids(BF, 0xA3, "\U0001d4f1", "\U0001d4ea", "▁ok")
        self._assert_all_modes(BF, prompt, output, "\U0001d4f1\U0001d4ea ok")

    def test_byte_fallback_prompt_chars_in_dangling_run_stay_skipped(self):
        # The byte run holds a whole prompt-side char (E0 B8 83) before the straddler.
        prompt = _ids(BF, "A", 0xE0, 0xB8, 0x83, 0xE0, 0xB8)
        output = _ids(BF, 0xA6, "▁ok")
        self._assert_all_modes(BF, prompt, output, " ok")

    def test_byte_fallback_byte_run_continuing_past_boundary(self):
        # After the straddler, the output keeps emitting byte pieces (U+0E03 U+0E26).
        prompt = _ids(BF, "A", 0xE0, 0xB8)
        output = _ids(BF, 0x83, 0xE0, 0xB8, 0x83, 0xE0, 0xB8, 0xA6, "▁ok")
        self._assert_all_modes(BF, prompt, output, "ฃฦ ok")

    def test_byte_fallback_dangling_bytes_never_completed(self):
        # The output does not continue the prompt's dangling bytes: emit no U+FFFD.
        prompt = _ids(BF, "A", 0xF0, 0x9D)
        output = _ids(BF, "▁hello", "▁ok")
        self._assert_all_modes(BF, prompt, output, " hello ok")

    def test_byte_level_boundary_unchanged(self):
        # Byte-level BPE decodes the dangling bytes as one U+FFFD; an invalid
        # byte before them must stay skipped too.
        for prompt in (_ids(BL, 0x41, 0xE0, 0xB8), _ids(BL, 0x41, 0x80, 0xE0, 0xB8)):
            with self.subTest(prompt=prompt):
                self._assert_all_modes(BL, prompt, _ids(BL, 0x81, 257), " ok")


if __name__ == "__main__":
    unittest.main()
