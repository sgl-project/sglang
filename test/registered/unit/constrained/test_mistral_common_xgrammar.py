import sys

import pytest

from sglang.srt.utils.hf_transformers.mistral_utils import (
    patch_mistral_common_tokenizer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

VOCAB_SIZE = 300
NUM_SPECIAL = 8


class _StubTekkenizer:
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        num_special=NUM_SPECIAL,
        fail_on_byte_piece=False,
    ):
        self.n_words = vocab_size
        self.num_special_tokens = num_special
        self.fail_on_byte_piece = fail_on_byte_piece

    def id_to_piece(self, token_id):
        return f"<special_{token_id}>"

    def id_to_byte_piece(self, token_id):
        if self.fail_on_byte_piece:
            raise RuntimeError("byte-piece conversion failed")
        if token_id == self.num_special_tokens:
            return b"\x00"
        return bytes([token_id % 256])


class _StubMistralTokenizer:
    def __init__(self, tekken=None):
        inner = type("InstructTokenizer", (), {"tokenizer": tekken})()
        self.tokenizer = type("MistralTokenizer", (), {"instruct_tokenizer": inner})()
        self.eos_token_id = 2
        self.chat_template = "x"

    def add_special_tokens(self, *args, **kwargs):
        return 0

    def convert_tokens_to_ids(self, val):
        return 0

    def decode(self, *args, **kwargs):
        return ""

    def batch_decode(self, *args, **kwargs):
        return []

    def apply_chat_template(self, *args, **kwargs):
        return []


class _MistralCommonStub(_StubMistralTokenizer):
    pass


def _patched(tekken):
    return patch_mistral_common_tokenizer(_MistralCommonStub(tekken))


def _is_allowed(mask, token_id):
    return bool((int(mask[0][token_id // 32]) >> (token_id % 32)) & 1)


def test_builds_tokenizer_info_over_full_vocab():
    info, stop_tokens = _patched(_StubTekkenizer()).init_xgrammar()

    assert info is not None
    assert info.vocab_size == VOCAB_SIZE
    assert stop_tokens == [2]


def test_json_schema_compiles_and_constrains():
    from xgrammar import GrammarCompiler, GrammarMatcher, allocate_token_bitmask

    info, _ = _patched(_StubTekkenizer()).init_xgrammar()
    grammar = GrammarCompiler(tokenizer_info=info).compile_json_schema(
        '{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}'
    )
    mask = allocate_token_bitmask(1, info.vocab_size)
    GrammarMatcher(grammar).fill_next_token_bitmask(mask)

    assert _is_allowed(mask, ord("{"))
    assert not _is_allowed(mask, ord("z"))


def test_returns_none_without_a_tekkenizer():
    info, stop_tokens = _patched(object()).init_xgrammar()

    assert info is None
    assert stop_tokens is None


def test_returns_none_when_vocab_extraction_fails():
    info, stop_tokens = _patched(
        _StubTekkenizer(fail_on_byte_piece=True)
    ).init_xgrammar()

    assert info is None
    assert stop_tokens is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
