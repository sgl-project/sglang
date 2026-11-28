"""Regression for the streaming hold-back of stop strings.

While a request streams, a step whose decoded tail ends with a prefix of a stop
string is held back, so the client never sees the head of a stop string. When
the next token carries only the first bytes of a multi-byte character, the
decoded tail ends with U+FFFD, no longer ends with a stop-string prefix, and the
held step was released. The incremental detokenizer then sent the printable part
of the held text (up to the last space, or through a CJK character). One step
later the stop string matched and the final text was trimmed, but the head was
already sent: the streamed text was longer than the non-streamed text by the
head of the stop string ("Done ", "## ", ...).

Drives the real `Req` (`update_finish_state`, `check_match_stop_str_prefix`),
the real `_GenerationStreamAccumulator.accept` / `to_payload` and the real
`DetokenizerManager.handle_batch_token_id_out`, one token per decode step, with
a byte-level fake tokenizer in place of a model and its tokenizer; pure CPU. The
OpenAI chat and completions endpoints both relay these texts unchanged, so this
covers the part of their streaming path that they share.
"""

import unittest
from array import array

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.incremental_detokenize import DetokenizeStateTable
from sglang.srt.managers.io_struct import RequestOutputStats
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.output_streamer import (
    DEFAULT_FORCE_STREAM_INTERVAL,
    OutputSink,
    _GenerationStreamAccumulator,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PROMPT = b"Q:"
FFFD = "\N{REPLACEMENT CHARACTER}"

# name -> (output tokens as bytes, stop string, text a non-streaming client gets).
# Each split character is cut into two tokens the way byte-level BPE vocabularies
# cut it (Qwen3 encodes " U+2705" as two tokens that decode to " U+FFFD", "U+FFFD").
SPLIT_AFTER_PREFIX = {
    # Held "Done" is released by " \xe2\x9c" (first bytes of U+2705); leaked "Done ".
    "ascii_space_emoji": (
        [b"All", b" good", b".", b" Done", b" \xe2\x9c", b"\x85", b" bye"],
        "Done \u2705",
        "All good. ",
    ),
    # CJK head before the space and emoji; leaked the CJK head and the space.
    "cjk_space_emoji": (
        [
            "\u4eca\u5929".encode(),
            "\u7684".encode(),
            "\u4efb\u52a1".encode(),
            " \u5168\u90e8".encode(),
            "\u5b8c\u6210".encode(),
            b" \xe2\x9c",
            b"\x85",
            " \u518d\u89c1".encode(),
        ],
        "\u5b8c\u6210 \u2705",
        "\u4eca\u5929\u7684\u4efb\u52a1 \u5168\u90e8",
    ),
    # Markdown heading with a split CJK character (U+603B = e6 80 bb); leaked "## ".
    "markdown_heading": (
        [
            "\u7ed3\u8bba".encode(),
            "\u5982\u4e0b".encode(),
            b"\n",
            b"##",
            b" \xe6\x80",
            b"\xbb",
            "\u7ed3".encode(),
            " \u5168\u90e8".encode(),
        ],
        "## \u603b\u7ed3",
        "\u7ed3\u8bba\u5982\u4e0b\n",
    ),
    # List marker with a split CJK character (U+7B54 = e7 ad 94); leaked "- ".
    "list_marker": (
        [
            "\u8981\u70b9".encode(),
            b"\n",
            b"-",
            b" \xe7\xad",
            b"\x94",
            "\u6848".encode(),
            " \u662f".encode(),
        ],
        "- \u7b54\u6848",
        "\u8981\u70b9\n",
    ),
    # Two split CJK characters (U+9F98 = e9 be 98); leaked the first one.
    "split_cjk_after_cjk": (
        [
            "\u7b54".encode(),
            "\uff1a".encode(),
            b"\xe9\xbe",
            b"\x98",
            b"\xe9\xbe",
            b"\x98",
            b" end",
        ],
        "\u9f98\u9f98",
        "\u7b54\uff1a",
    ),
    # Stop string starting with a space; the space shares a token with the
    # first bytes of the emoji; leaked " ".
    "space_then_split_emoji": (
        [b"All", b" good", b" \xe2\x9c", b"\x85", b" bye"],
        " \u2705",
        "All good",
    ),
}

# Shapes that never leaked: nothing printable is held when the split arrives.
CONTROLS = {
    "ascii_only": (
        [b"All", b" good", b".", b" Done", b" now", b" bye"],
        "Done now",
        "All good. ",
    ),
    "no_space_before_split_emoji": (
        [b"All", b" good", b".", b" Done", b"\xe2\x9c", b"\x85", b" bye"],
        "Done\u2705",
        "All good. ",
    ),
    "split_char_first": (
        [b"x", b" \xe2\x9c", b"\x85", b" ok", b" bye"],
        "\u2705 ok",
        "x ",
    ),
    "no_space_before_split_cjk": (
        [b"All", b" good", b".", b" Done", b"\xe9\xbe", b"\x98", b" bye"],
        "Done\u9f98",
        "All good. ",
    ),
    "whole_token_cjk": (
        ["\u7b54".encode(), "\uff1a".encode(), "\u7b54\u6848".encode(), b" end"],
        "\u7b54\u6848",
        "\u7b54\uff1a",
    ),
    # The model emits a real U+FFFD (bytes ef bf bd) that is part of the stop string.
    "stop_contains_replacement_char": (
        [b"x", b" A", b"\xef\xbf\xbd", b"B", b" y"],
        "A" + FFFD + "B",
        "x ",
    ),
    "stop_ends_with_replacement_char": (
        [b"All", b" good", b".", b" Done", b" \xef\xbf\xbd", b" bye"],
        "Done " + FFFD,
        "All good. ",
    ),
}

_VOCAB = [PROMPT]
for _tokens, _, _ in [*SPLIT_AFTER_PREFIX.values(), *CONTROLS.values()]:
    _VOCAB += [t for t in _tokens if t not in _VOCAB]
_ID = {t: i for i, t in enumerate(_VOCAB)}

_REQUEST_STATS = RequestOutputStats(
    prompt_tokens=1,
    cached_tokens=0,
    cached_tokens_details=None,
    retraction_count=0,
    image_tokens=0,
    audio_tokens=0,
    video_tokens=0,
    time_stats=None,
)


class _ByteTokenizer:
    """Each id is a byte string; decode joins the bytes and decodes them as UTF-8,
    so a character split across ids decodes to U+FFFD until its last byte arrives,
    as byte-level BPE decoders do. errors="ignore" stands for a decoder that drops
    incomplete bytes instead."""

    is_fast = True
    eos_token_id = -1
    additional_stop_token_ids = None

    def __init__(self, errors="replace"):
        self.errors = errors

    def decode(self, ids, skip_special_tokens=True, spaces_between_special_tokens=True):
        return b"".join(_VOCAB[int(i)] for i in ids).decode("utf-8", self.errors)

    def batch_decode(self, ids_list, **kwargs):
        return [self.decode(ids, **kwargs) for ids in ids_list]

    def encode(self, text, add_special_tokens=False):
        # One id per UTF-8 byte: an upper bound on the tokens of any stop string,
        # so the stop-string window never misses the stop.
        return list(text.encode())


def _detokenizer(tokenizer):
    detok = DetokenizerManager.__new__(DetokenizerManager)
    detok.tokenizer = tokenizer
    detok.vocab_size = None
    detok.detokenize_state_table = DetokenizeStateTable(capacity=4)
    detok.disable_tokenizer_batch_decode = False
    detok.is_tool_call_parser_gpt_oss = False
    detok.tool_close_token_ids = frozenset()
    return detok


def _accumulator():
    return _GenerationStreamAccumulator(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        default_stream_interval=1,
        default_force_stream_interval=DEFAULT_FORCE_STREAM_INTERVAL,
        build_request_stats=lambda req: _REQUEST_STATS,
        is_request_level_customized_info_key=lambda key: False,
        current_weight_version="default",
        output_sink=OutputSink.DETOKENIZER,
    )


def _make_req(tokens, stop, *, stream, tokenizer):
    sp = SamplingParams(max_new_tokens=len(tokens), stop=[stop])
    sp.normalize(tokenizer=tokenizer)
    req = Req(
        rid="r",
        origin_input_text="",
        origin_input_ids=array("q", [_ID[PROMPT]]),
        sampling_params=sp,
        stream=stream,
        eos_token_ids=set(),
        vocab_size=len(_VOCAB),
    )
    req.tokenizer = tokenizer
    return req


def _client_text(tokens, stop, *, stream, tokenizer=None):
    """Decode one token per step; return the text the client assembles."""
    tokenizer = tokenizer or _ByteTokenizer()
    req = _make_req(tokens, stop, stream=stream, tokenizer=tokenizer)
    detok = _detokenizer(tokenizer)
    texts = []
    for token in tokens:
        req.output_ids.append(_ID[token])
        req.update_finish_state()
        accumulator = _accumulator()
        accumulator.accept(req=req)
        payload = accumulator.to_payload(dp_rank=0, is_idle_batch=False)
        if payload is not None:
            texts += detok.handle_batch_token_id_out(payload).output_strs
        if req.finished():
            break
    if req.finished_reason.to_json().get("matched") != stop:
        raise AssertionError(f"not finished on the stop string: {req.finished_reason}")
    return "".join(texts)


def _prefix_check(tokens, stop):
    """Whether a streaming step that just appended `tokens[-1]` is held back."""
    req = _make_req(tokens, stop, stream=True, tokenizer=_ByteTokenizer())
    req.output_ids.extend(_ID[t] for t in tokens)
    return req.check_match_stop_str_prefix()


class TestStopStrStreamHoldBack(CustomTestCase):
    def setUp(self):
        self.enterContext(get_context().override_server_args(kv_canary="none"))

    def test_stream_does_not_send_the_head_of_a_stop_string(self):
        """A character split across tokens after a held stop-string prefix must
        not release the hold: the streamed text equals the non-streamed one."""
        for name, (tokens, stop, want) in SPLIT_AFTER_PREFIX.items():
            with self.subTest(name):
                self.assertEqual(_client_text(tokens, stop, stream=True), want)

    def test_non_stream_text_is_trimmed_before_the_stop_string(self):
        for name, (tokens, stop, want) in {**SPLIT_AFTER_PREFIX, **CONTROLS}.items():
            with self.subTest(name):
                self.assertEqual(_client_text(tokens, stop, stream=False), want)

    def test_stream_matches_non_stream_without_a_split_after_the_prefix(self):
        for name, (tokens, stop, want) in CONTROLS.items():
            with self.subTest(name):
                self.assertEqual(_client_text(tokens, stop, stream=True), want)

    def test_stream_matches_non_stream_when_decode_drops_partial_bytes(self):
        """A decoder that drops incomplete bytes leaves the stop-string prefix at
        the end of the tail, so the hold is kept without any U+FFFD handling."""
        tokenizer = _ByteTokenizer(errors="ignore")
        for name, (tokens, stop, want) in SPLIT_AFTER_PREFIX.items():
            with self.subTest(name):
                self.assertEqual(
                    _client_text(tokens, stop, stream=True, tokenizer=tokenizer),
                    want,
                )

    def test_prefix_check_on_a_trailing_partial_character(self):
        tokens, stop, _ = SPLIT_AFTER_PREFIX["ascii_space_emoji"]
        # Tail "... Done U+FFFD": "Done " is still a prefix of the stop string.
        self.assertTrue(_prefix_check(tokens[:5], stop))
        # Tail "x U+FFFD": nothing before the partial character starts the stop.
        self.assertFalse(_prefix_check([b"x", b" \xe2\x9c"], stop))
        # A real U+FFFD at the start of a stop string still matches as before.
        self.assertTrue(_prefix_check([b"x", b" A", b"\xef\xbf\xbd"], FFFD + " end"))


if __name__ == "__main__":
    unittest.main()
