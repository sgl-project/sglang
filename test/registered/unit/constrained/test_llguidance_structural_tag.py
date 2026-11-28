"""llguidance compiles xgrammar-native structural tags such as Kimi-K3's.

The expected verdicts were cross-checked against xgrammar on the same tags.
"""

import json
import os
import tempfile
import unittest

import tiktoken
import torch
from llguidance import LLMatcher, grammar_from
from llguidance.tiktoken import lltokenizer_from_encoding

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.llguidance_backend import GuidanceBackend, GuidanceGrammar
from sglang.srt.constrained.structural_tag_to_lark import structural_tag_to_lark
from sglang.srt.entrypoints.openai.protocol import Function, Tool, ToolChoice
from sglang.srt.environ import envs
from sglang.srt.function_call.kimik3_structural_tag import get_kimik3_structural_tag
from sglang.srt.tokenizer.tiktoken_tokenizer import PAT_STR_B, TiktokenTokenizer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

OPEN, SEP, CLOSE, EOS = "<|open|>", "<|sep|>", "<|close|>", "<|eos|>"
SPECIAL_TOKENS = {OPEN: 256, SEP: 257, CLOSE: 258, EOS: 259}
EOS_ID = SPECIAL_TOKENS[EOS]
BYTE_RANKS = {bytes([b]): b for b in range(256)}


def _encoding() -> tiktoken.Encoding:
    """A byte-level vocabulary plus the special tokens above."""
    return tiktoken.Encoding(
        name="byte-level-test",
        pat_str=PAT_STR_B,
        mergeable_ranks=BYTE_RANKS,
        special_tokens=dict(SPECIAL_TOKENS),
    )


class TikTokenTokenizer:
    """Stands in for the Kimi remote-code tokenizer, which wraps ``.model``."""

    def __init__(self):
        self.model = _encoding()
        self.eos_token_id = EOS_ID


TikTokenTokenizer.__module__ = "test_models.tokenization_kimi"

ENCODING = _encoding()


def _verdict(matcher: LLMatcher, text: str) -> str:
    for token in ENCODING.encode(text, allowed_special="all"):
        if matcher.try_consume_tokens([token]) != 1:
            return "reject"
    return "full" if matcher.is_accepting() else "prefix"


def _tools():
    weather = Function(
        name="get_weather",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}, "n": {"type": "number"}},
            "required": ["city"],
        },
    )
    f2 = Function(
        name="f2",
        strict=True,
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "enum": ["a", "Paris"]}},
            "required": ["city"],
            "additionalProperties": False,
        },
    )
    return [Tool(type="function", function=weather), Tool(type="function", function=f2)]


def _k3_call(name: str, key: str = "city", value: str = "Paris") -> str:
    argument = (
        f'{OPEN}argument key="{key}" type="string"{SEP}{value}{CLOSE}argument{SEP}'
    )
    return f'{OPEN}call tool="{name}" index="1"{SEP}{argument}{CLOSE}call{SEP}'


def _k3_tools(*calls: str) -> str:
    return f"{OPEN}tools{SEP}{''.join(calls)}{CLOSE}tools{SEP}"


CALL = _k3_tools(_k3_call("get_weather"))
TWO_CALLS = _k3_tools(_k3_call("get_weather"), _k3_call("get_weather"))
THINK_THEN_CALL = f"r{CLOSE}think{SEP}{CALL}"


class TestKimiK3StructuralTag(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = GuidanceBackend(TikTokenTokenizer())

    def _grammar(self, **kwargs) -> GuidanceGrammar:
        tag = get_kimik3_structural_tag(tools=_tools(), **kwargs)
        grammar = self.backend.dispatch_structural_tag(tag.model_dump_json())
        self.assertIsInstance(grammar, GuidanceGrammar)
        return grammar

    def test_tool_choice_verdicts(self):
        f2_choice = ToolChoice(type="function", function={"name": "f2"})
        cases = [
            ("auto", False, "hello", "full"),
            ("auto", False, "", "full"),
            ("auto", False, CALL, "full"),
            ("auto", False, TWO_CALLS, "reject"),
            ("auto", False, CALL + "bye", "reject"),
            ("auto", False, _k3_tools(_k3_call("nope")), "reject"),
            ("auto", False, _k3_tools(_k3_call("f2")), "full"),
            ("auto", False, _k3_tools(_k3_call("f2", value="Rome")), "reject"),
            ("auto", True, TWO_CALLS, "full"),
            ("auto", True, CALL + "bye", "full"),
            ("required", False, "hello", "prefix"),
            ("required", False, CALL, "full"),
            ("required", True, CALL + "bye", "reject"),
            ("none", False, CALL, "reject"),
            (f2_choice, False, _k3_tools(_k3_call("f2")), "full"),
        ]
        for tool_choice, parallel, text, expected in cases:
            with self.subTest(tool_choice=tool_choice, parallel=parallel, text=text):
                grammar = self._grammar(
                    tool_choice=tool_choice,
                    thinking_mode=False,
                    parallel_tool_calls=parallel,
                )
                self.assertEqual(_verdict(grammar.ll_matcher, text), expected)

    def test_thinking_mode_constrains_after_reasoning(self):
        for tool_choice in ("auto", "required"):
            with self.subTest(tool_choice=tool_choice):
                grammar = self._grammar(
                    tool_choice=tool_choice,
                    thinking_mode=True,
                    parallel_tool_calls=True,
                )
                fresh = grammar.copy()
                # A tool call inside the reasoning is just reasoning text.
                self.assertEqual(_verdict(grammar.ll_matcher, CALL), "prefix")
                self.assertEqual(_verdict(fresh.ll_matcher, THINK_THEN_CALL), "full")


def _tag(content, begin=f"{OPEN}v{SEP}", end=CLOSE):
    return {"type": "tag", "begin": begin, "content": content, "end": end}


def _token(name):
    return {"type": "token", "token": name}


def _qwen_xml(schema):
    return _tag({"type": "json_schema", "json_schema": schema, "style": "qwen_xml"})


def _value(text):
    return f"{OPEN}v{SEP}{text}{CLOSE}"


TRIGGERED_TAG = {
    "type": "tag",
    "begin": f"{OPEN}a{SEP}",
    "content": {"type": "const_string", "value": "x"},
    "end": CLOSE,
}
TRIGGERED_CALL = f"{OPEN}a{SEP}x{CLOSE}"


def _triggered(**flags):
    return {
        "type": "triggered_tags",
        "triggers": [OPEN],
        "tags": [TRIGGERED_TAG],
        **flags,
    }


class TestStructuralTagToLark(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = lltokenizer_from_encoding(ENCODING, eos_token=EOS_ID)

    def _translate(self, fmt, **kwargs) -> str:
        return structural_tag_to_lark(
            {"type": "structural_tag", "format": fmt},
            special_token_ids=SPECIAL_TOKENS,
            eos_token_ids=frozenset({EOS_ID}),
            **kwargs,
        )

    def _assert_verdicts(self, fmt, cases, **kwargs):
        lark = self._translate(fmt, **kwargs)
        for text, expected in cases:
            with self.subTest(text=text):
                matcher = LLMatcher(self.tokenizer, grammar_from("lark", lark))
                self.assertFalse(matcher.is_error(), matcher.get_error())
                self.assertEqual(_verdict(matcher, text), expected)

    def test_triggered_tags_forms(self):
        call = TRIGGERED_CALL
        forms = {
            "default": (
                _triggered(),
                [
                    ("", "full"),
                    (f"hi{call}bye", "full"),
                    (call + call, "full"),
                    # A trigger in the text must start a tag.
                    (f"{OPEN}b{SEP}", "reject"),
                ],
            ),
            "at_least_one": (
                _triggered(at_least_one=True),
                [("", "prefix"), ("hi", "reject"), (f"{call}bye{call}", "full")],
            ),
            "stop_after_first": (
                _triggered(stop_after_first=True),
                [("hi", "full"), (f"hi{call}", "full"), (f"hi{call}bye", "reject")],
            ),
            "both": (
                _triggered(at_least_one=True, stop_after_first=True),
                [(call, "full"), (f"{call}bye", "reject"), (f"hi{call}", "reject")],
            ),
        }
        for name, (fmt, cases) in forms.items():
            with self.subTest(form=name):
                self._assert_verdicts(fmt, cases)

    def test_any_text_excludes_strings_at_special_tokens(self):
        fmt = _tag({"type": "any_text", "excludes": [f"{OPEN}x"]})
        self._assert_verdicts(
            fmt,
            [(_value(f"hi{OPEN}y"), "full"), (_value(f"hi{OPEN}x"), "reject")],
        )

    def test_token_nodes_exclude_eos_and_tag_end(self):
        any_tokens = _tag(
            {"type": "any_tokens", "exclude_tokens": [SEP]},
            begin=_token(OPEN),
            end=_token(CLOSE),
        )
        self._assert_verdicts(
            any_tokens,
            [
                (f"{OPEN}abc{CLOSE}", "full"),
                (f"{OPEN}{CLOSE}", "full"),
                (f"{OPEN}a{SEP}{CLOSE}", "reject"),
                (f"{OPEN}a{EOS}", "reject"),
            ],
        )
        exclude_token = _tag(
            {"type": "exclude_token", "exclude_tokens": [SEP]},
            begin=_token(OPEN),
            end=_token(CLOSE),
        )
        self._assert_verdicts(
            exclude_token,
            [
                (f"{OPEN}a{CLOSE}", "full"),
                (f"{OPEN}{CLOSE}", "reject"),
                (f"{OPEN}{SEP}{CLOSE}", "reject"),
                (f"{OPEN}ab{CLOSE}", "reject"),
            ],
        )

    def test_regex_slash(self):
        for pattern in ("a/b", "a\\/b"):
            with self.subTest(pattern=pattern):
                self._assert_verdicts(
                    {"type": "regex", "pattern": pattern},
                    [("a/b", "full"), ("ab", "reject")],
                )

    def test_regex_class_escapes_are_ascii(self):
        # xgrammar matches \d and \w against ASCII only.
        cases = {
            r"\d+": [("12", "full"), ("\u0663", "reject")],
            r"[\d_]+": [("1_2", "full"), ("\u0663", "reject")],
            r"\w+": [("a_1", "full"), ("\u00e9", "reject")],
            r"\D": [("a", "full"), ("1", "reject"), ("\u0663", "full")],
            r"\\d": [("\\d", "full"), ("1", "reject")],
        }
        for pattern, verdicts in cases.items():
            with self.subTest(pattern=pattern):
                self._assert_verdicts({"type": "regex", "pattern": pattern}, verdicts)

    def test_qwen_xml_string_values(self):
        cases = {
            # xgrammar enforces only the pattern when both are present.
            "pattern_and_length": (
                {"type": "string", "pattern": "[a-z]+", "maxLength": 2},
                [(_value("abc"), "full"), (_value("ab1"), "reject")],
            ),
            "length": (
                {"type": "string", "maxLength": 2},
                [(_value("ab"), "full"), (_value("abc"), "reject")],
            ),
            "enum": (
                {"type": "string", "enum": ["a", "b c"]},
                [(_value("b c"), "full"), (_value("b"), "reject")],
            ),
            "free": ({}, [(_value('any "thing"'), "full")]),
        }
        for name, (schema, verdicts) in cases.items():
            with self.subTest(schema=name):
                self._assert_verdicts(_qwen_xml(schema), verdicts)

    def test_oversized_length_is_not_enforced(self):
        self._assert_verdicts(
            _qwen_xml({"type": "string", "maxLength": 2}),
            [(_value("abcdef"), "full")],
            max_enforced_string_length=1,
        )

    def test_json_schema_relaxations(self):
        def json_tag(schema):
            return _tag({"type": "json_schema", "json_schema": schema})

        # ``not`` is dropped instead of failing the whole grammar.
        self._assert_verdicts(
            json_tag({"type": "integer", "not": {"const": 3}}),
            [(_value("3"), "full"), (_value("x"), "reject")],
        )
        # ``enum`` wins over a conflicting sibling ``type``, as in xgrammar.
        self._assert_verdicts(
            json_tag({"type": "boolean", "enum": ["true"]}),
            [(_value('"true"'), "full"), (_value("true"), "reject")],
        )

    def test_json_schema_closed_like_xgrammar_strict_mode(self):
        def verdicts(schema, cases):
            with self.subTest(schema=schema):
                fmt = _tag({"type": "json_schema", "json_schema": schema})
                self._assert_verdicts(fmt, [(_value(t), v) for t, v in cases])

        a = {"a": {"type": "integer"}}
        b = {"b": {"type": "integer"}}
        both = '{"a": 1, "b": 2}'
        verdicts(
            {"type": "object", "properties": a},
            [('{"a": 1}', "full"), (both, "reject")],
        )
        verdicts(
            {"type": "object", "properties": a, "additionalProperties": True},
            [(both, "full")],
        )
        verdicts({"type": "object"}, [(both, "full")])
        verdicts(
            {"type": "array", "prefixItems": [{"type": "integer"}]},
            [("[1]", "full"), ("[1, 2]", "reject")],
        )
        # Closing each ``allOf`` member would reject the keys the other adds.
        verdicts(
            {"allOf": [{"properties": a}, {"properties": b}]},
            [(both, "full")],
        )

    def test_unsupported_shapes_raise(self):
        cases = {
            "plain_text_trigger": {
                "type": "triggered_tags",
                "triggers": ["x"],
                "tags": [],
            },
            "free_text_before_plain_text": {
                "type": "sequence",
                "elements": [
                    {"type": "any_text"},
                    {"type": "const_string", "value": "end"},
                ],
            },
            "unknown_token_name": _token("<|missing|>"),
            "unknown_node": {"type": "grammar", "grammar": "root ::= 'a'"},
        }
        for name, fmt in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                self._translate(fmt)


def _write_tiktoken_json(path: str) -> None:
    with open(path, "w") as f:
        json.dump(
            {
                "regular_tokens": [
                    {"bytes": list(b), "token": i} for b, i in BYTE_RANKS.items()
                ],
                "special_tokens": [
                    {"bytes": list(t.encode()), "token": i}
                    for t, i in SPECIAL_TOKENS.items()
                ],
                "word_split": "V1",
            },
            f,
        )


class TestGuidanceBackendTiktoken(CustomTestCase):
    def test_tiktoken_tokenizers_keep_special_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokenizer.json")
            _write_tiktoken_json(path)
            tokenizers = {
                "tiktoken": TiktokenTokenizer(path),
                "kimi": TikTokenTokenizer(),
            }
            for name, tokenizer in tokenizers.items():
                with self.subTest(tokenizer=name):
                    backend = GuidanceBackend(tokenizer)
                    self.assertEqual(backend.special_token_ids, SPECIAL_TOKENS)
                    self.assertEqual(backend.llguidance_tokenizer.eos_tokens, [EOS_ID])

    def test_legacy_structural_tag_still_compiles(self):
        backend = GuidanceBackend(TikTokenTokenizer())
        legacy = {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": f"{OPEN}f{SEP}",
                    "schema": {"type": "object"},
                    "end": CLOSE,
                }
            ],
            "triggers": [OPEN],
        }
        grammar = backend.dispatch_structural_tag(json.dumps(legacy))
        self.assertIsInstance(grammar, GuidanceGrammar)

    def test_untranslatable_tag_is_invalid_grammar(self):
        backend = GuidanceBackend(TikTokenTokenizer())
        tag = {"type": "structural_tag", "format": _token("<|missing|>")}
        grammar = backend.dispatch_structural_tag(json.dumps(tag))
        self.assertIsInstance(grammar, InvalidGrammarObject)

    def test_backend_mask_ops(self):
        backend = GuidanceBackend(TikTokenTokenizer())
        tag = {"type": "structural_tag", "format": _token(SEP)}
        grammar = backend.dispatch_structural_tag(json.dumps(tag))
        mask = backend.allocate_vocab_mask(vocab_size=260, batch_size=1, device="cpu")
        grammar.fill_vocab_mask(mask, 0)
        logits = torch.zeros(1, 260)
        backend.apply_vocab_mask(logits, backend.move_vocab_mask(mask, "cpu"))
        allowed = torch.isfinite(logits[0]).nonzero().flatten().tolist()
        self.assertEqual(allowed, [SPECIAL_TOKENS[SEP]])

    def test_max_enforced_string_length_env(self):
        tokenizer = TikTokenTokenizer()
        env = envs.SGLANG_LLGUIDANCE_MAX_ENFORCED_STRING_LENGTH
        with env.override(8):
            self.assertEqual(GuidanceBackend(tokenizer).max_enforced_string_length, 8)
        with env.override(-1):
            self.assertIsNone(GuidanceBackend(tokenizer).max_enforced_string_length)


if __name__ == "__main__":
    unittest.main()
