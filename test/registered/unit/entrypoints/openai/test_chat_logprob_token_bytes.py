"""Unit tests for the per-id byte cache behind chat completion logprobs.

Turns red if a cached ``bytes`` value differs from recovering it from the token
id (byte-level BPE) or from the token text, or if a byte-level id is converted
again after its first use.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.utils import _build_byte_decoder, token_id_to_bytes

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_BYTE_ENCODER = {b: ch for ch, b in _build_byte_decoder().items()}


def _piece(raw: bytes) -> str:
    return "".join(_BYTE_ENCODER[b] for b in raw)


# id -> (byte-level piece or None for a special id, display text)
_VOCAB = {
    0: (_piece(b"Hello"), "Hello"),
    1: (_piece(" é".encode()), " é"),
    2: (_piece(b"\xf0"), "�"),  # first byte of a 4-byte character
    3: (None, "<|special|>"),
}


class _FakeTokenizer:
    def __init__(self, is_byte_level: bool):
        self.is_byte_level = is_byte_level
        self.converted_ids = []

    def convert_ids_to_tokens(self, token_id):
        self.converted_ids.append(token_id)
        return _VOCAB[token_id][0]


# Kept alive for the whole module: the byte-level probe caches by id(tokenizer).
_BYTE_LEVEL_TOKENIZER = _FakeTokenizer(is_byte_level=True)
_PLAIN_TOKENIZER = _FakeTokenizer(is_byte_level=False)


class _ChatStub:
    _build_token_logprobs_from_raw = OpenAIServingChat._build_token_logprobs_from_raw
    _token_bytes = OpenAIServingChat._token_bytes

    def __init__(self, tokenizer):
        self.tokenizer_manager = SimpleNamespace(tokenizer=tokenizer)
        self._byte_level_token_bytes = {}


def _triple(logprob, token_id):
    return (logprob, token_id, _VOCAB[token_id][1])


def _raw_logprobs():
    tokens = [_triple(-0.1, 0), _triple(-0.2, 2), _triple(-0.3, 1), _triple(-0.4, 3)]
    tops = [
        [_triple(-0.1, 0), _triple(-2.0, 1), _triple(-3.0, 3)],
        [_triple(-0.2, 2), _triple(-1.5, 0)],
        None,
        [_triple(-0.4, 3), _triple(-0.9, 2), _triple(-1.0, 0)],
    ]
    return tokens, tops


def _expected_bytes(tokenizer, token_id, text):
    raw = token_id_to_bytes(tokenizer, token_id) if tokenizer.is_byte_level else None
    return raw if raw is not None else list(text.encode("utf-8"))


def _expected(tokenizer, tokens, tops):
    out = []
    for i, (logprob, token_id, text) in enumerate(tokens):
        out.append(
            {
                "token": text,
                "bytes": _expected_bytes(tokenizer, token_id, text),
                "logprob": logprob,
                "top_logprobs": [
                    {
                        "token": t,
                        "bytes": _expected_bytes(tokenizer, tid, t),
                        "logprob": lp,
                    }
                    for lp, tid, t in (tops[i] or [])
                ],
            }
        )
    return out


class TestChatLogprobTokenBytes(CustomTestCase):
    def _check(self, tokenizer):
        chat = _ChatStub(tokenizer)
        tokens, tops = _raw_logprobs()
        expected = _expected(tokenizer, tokens, tops)
        tokenizer.converted_ids.clear()
        for _ in range(2):
            got = chat._build_token_logprobs_from_raw(tokens, tops)
            self.assertEqual([x.model_dump() for x in got], expected)
        return chat

    def test_byte_level_bytes_come_from_the_id_once(self):
        chat = self._check(_BYTE_LEVEL_TOKENIZER)
        self.assertEqual(sorted(_BYTE_LEVEL_TOKENIZER.converted_ids), [0, 1, 2, 3])
        got = chat._build_token_logprobs_from_raw([_triple(-0.2, 2)], None)
        self.assertEqual(got[0].bytes, [0xF0])
        self.assertEqual(got[0].token, "�")

    def test_plain_tokenizer_uses_text_bytes(self):
        self._check(_PLAIN_TOKENIZER)
        self.assertEqual(_PLAIN_TOKENIZER.converted_ids, [])


if __name__ == "__main__":
    unittest.main()
