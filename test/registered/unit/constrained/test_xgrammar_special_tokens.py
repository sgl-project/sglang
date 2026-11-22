"""JSON grammars must not let special tokens into the JSON text.

Covers python/sglang/srt/constrained/xgrammar_backend.py.
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that may pull in sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import json
import unittest
from types import SimpleNamespace

from xgrammar import TokenizerInfo, VocabType

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.test_utils import CustomTestCase

# RAW vocabulary: printable ASCII, then two special tokens and EOS.
VOCAB = [chr(i) for i in range(32, 127)] + ["<|im_start|>", "[TOOL_CALLS]", "<eos>"]
IM_START = VOCAB.index("<|im_start|>")
TOOL_CALLS = VOCAB.index("[TOOL_CALLS]")
EOS = VOCAB.index("<eos>")
SPECIAL = (IM_START, TOOL_CALLS, EOS)
SCHEMA = json.dumps(
    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
)


def _backend(vocab=VOCAB, vocab_type=VocabType.RAW, **tokenizer_attrs):
    info = TokenizerInfo(vocab, vocab_type, stop_token_ids=[vocab.index("<eos>")])
    tokenizer = SimpleNamespace(init_xgrammar=lambda: (info, None), **tokenizer_attrs)
    return XGrammarGrammarBackend(
        tokenizer, len(vocab), any_whitespace=False, cache_limit_bytes=1 << 20
    )


def _allowed(grammar, prefix: str, vocab=VOCAB) -> set:
    for ch in prefix:
        grammar.accept_token(vocab.index("Ġ" if ch == " " and "Ġ" in vocab else ch))
    mask = grammar.allocate_vocab_mask(len(vocab), 1, "cpu")
    grammar.fill_vocab_mask(mask, 0)
    return {i for i in range(len(vocab)) if (int(mask[0, i // 32]) >> (i % 32)) & 1}


class TestJsonExcludesSpecialTokens(CustomTestCase):
    def test_json_string_excludes_special_tokens(self):
        backend = _backend(all_special_ids=list(SPECIAL))
        allowed = _allowed(backend.dispatch_json(SCHEMA), '{"a": "')
        self.assertNotIn(IM_START, allowed)
        self.assertNotIn(TOOL_CALLS, allowed)
        self.assertIn(VOCAB.index("x"), allowed)

    def test_json_keeps_every_ordinary_token(self):
        plain = _allowed(_backend().dispatch_json(SCHEMA), '{"a": "')
        filtered = _allowed(
            _backend(all_special_ids=list(SPECIAL)).dispatch_json(SCHEMA), '{"a": "'
        )
        self.assertIn(IM_START, plain)
        self.assertEqual(filtered, plain - {IM_START, TOOL_CALLS})

    def test_json_still_ends_on_the_stop_token(self):
        grammar = _backend(all_special_ids=list(SPECIAL)).dispatch_json(SCHEMA)
        self.assertIn(EOS, _allowed(grammar, '{"a": "x"}'))
        grammar.accept_token(EOS)
        self.assertTrue(grammar.is_terminated())

    def test_builtin_json_object_excludes_special_tokens(self):
        backend = _backend(all_special_ids=list(SPECIAL))
        self.assertNotIn(
            IM_START, _allowed(backend.dispatch_json("$$ANY$$"), '{"a": "')
        )

    def test_special_added_tokens_are_excluded(self):
        # Most special tokens of real tokenizers are only in added_tokens_decoder.
        backend = _backend(
            added_tokens_decoder={
                IM_START: SimpleNamespace(special=True),
                TOOL_CALLS: SimpleNamespace(special=False),
            }
        )
        allowed = _allowed(backend.dispatch_json(SCHEMA), '{"a": "')
        self.assertNotIn(IM_START, allowed)
        self.assertIn(TOOL_CALLS, allowed)

    def test_added_tokens_decoder_method_is_ignored(self):
        # MistralCommonBackend exposes added_tokens_decoder as a method.
        backend = _backend(
            all_special_ids=[TOOL_CALLS, EOS, None], added_tokens_decoder=lambda: {}
        )
        self.assertNotIn(TOOL_CALLS, _allowed(backend.dispatch_json(SCHEMA), '{"a": "'))

    def test_byte_level_vocab_keeps_every_ordinary_token(self):
        vocab = [chr(i) for i in range(33, 127)] + ["Ġ", "Ġx", "<|im_start|>", "<eos>"]
        im_start = vocab.index("<|im_start|>")
        special = [im_start, vocab.index("<eos>")]
        prefix = '{"a": "'
        plain = _allowed(
            _backend(vocab, VocabType.BYTE_LEVEL).dispatch_json(SCHEMA), prefix, vocab
        )
        filtered = _allowed(
            _backend(
                vocab, VocabType.BYTE_LEVEL, all_special_ids=special
            ).dispatch_json(SCHEMA),
            prefix,
            vocab,
        )
        self.assertIn(vocab.index("Ġx"), filtered)
        self.assertEqual(filtered, plain - {im_start})


class TestOtherGrammarsKeepSpecialTokens(CustomTestCase):
    def test_regex_can_spell_special_tokens(self):
        # The Whisper language auto-detect regex spells <|en|><|transcribe|>.
        backend = _backend(all_special_ids=list(SPECIAL))
        grammar = backend.dispatch_regex(r"<\|im_start\|>[a-z]+")
        self.assertIn(IM_START, _allowed(grammar, ""))

    def test_ebnf_can_spell_special_tokens(self):
        # Tool-call EBNF (e.g. GLM-4.7) writes special tokens as literals.
        backend = _backend(all_special_ids=list(SPECIAL))
        grammar = backend.dispatch_ebnf('root ::= "<|im_start|>" [a-z]+')
        self.assertNotIsInstance(grammar, InvalidGrammarObject)
        self.assertIn(IM_START, _allowed(grammar, ""))

    def test_structural_tag_can_spell_special_tokens(self):
        tag = json.dumps(
            {
                "type": "structural_tag",
                "structures": [
                    {"begin": "[TOOL_CALLS]", "schema": {"type": "object"}, "end": ""}
                ],
                "triggers": ["[TOOL_CALLS]"],
            }
        )
        grammar = _backend(all_special_ids=list(SPECIAL)).dispatch_structural_tag(tag)
        self.assertNotIsInstance(grammar, InvalidGrammarObject)
        self.assertIn(TOOL_CALLS, _allowed(grammar, ""))


class TestJsonCompilerCache(CustomTestCase):
    def test_compilers_split_the_budget(self):
        backend = _backend(all_special_ids=list(SPECIAL))
        self.assertIsNot(backend.json_compiler, backend.grammar_compiler)
        self.assertEqual(backend.grammar_compiler.cache_limit_bytes, 1 << 19)
        self.assertEqual(backend.json_compiler.cache_limit_bytes, 1 << 19)

    def test_one_compiler_without_special_tokens(self):
        backend = _backend()
        self.assertIs(backend.json_compiler, backend.grammar_compiler)
        self.assertEqual(backend.grammar_compiler.cache_limit_bytes, 1 << 20)

    def test_reset_clears_both_compilers(self):
        backend = _backend(all_special_ids=list(SPECIAL))
        backend.dispatch_json(SCHEMA)
        backend.dispatch_ebnf('root ::= "<|im_start|>"')
        self.assertGreater(backend.json_compiler.get_cache_size_bytes(), 0)
        self.assertGreater(backend.grammar_compiler.get_cache_size_bytes(), 0)
        backend.reset()
        self.assertEqual(backend.json_compiler.get_cache_size_bytes(), 0)
        self.assertEqual(backend.grammar_compiler.get_cache_size_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
