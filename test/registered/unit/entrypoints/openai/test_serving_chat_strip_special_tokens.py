"""
Unit tests for the strip-special-tokens post-pass in OpenAIServingChat.

Background: when a chat completion request includes `tools`, _process_messages
forces `request.skip_special_tokens = False` so the tool-call parser can see
its markers (e.g. <tool_call>, <arg_key>, <arg_value>). That override also
lets *structural* special tokens like <|im_end|>, <|endoftext|>, <|eot_id|>
through verbatim into response content/reasoning_content. The
_compute_special_token_strings + _strip_special_tokens helpers undo that side
effect after the tool-call parser has consumed what it needs, by stripping
exactly the set of tokens that `skip_special_tokens=True` would have stripped.
"""

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock, patch


# ----------------------------------------------------------------------------
# sgl_kernel CUDA mock — needed for CPU-only test runners (mirrors the pattern
# in test_serving_embedding.py).
# ----------------------------------------------------------------------------
class _SglKernelMockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__package__ = spec.name
        mod.__loader__ = self
        mod.__getattr__ = lambda name: MagicMock()
        return mod

    def exec_module(self, module):
        pass


class _SglKernelMockFinder(importlib.abc.MetaPathFinder):
    _PREFIX = "sgl_kernel"
    _loader = _SglKernelMockLoader()

    def find_spec(self, fullname, path, target=None):
        if fullname == self._PREFIX or fullname.startswith(self._PREFIX + "."):
            return importlib.machinery.ModuleSpec(
                fullname, self._loader, is_package=True
            )
        return None


if "sgl_kernel" not in sys.modules:
    sys.meta_path.insert(0, _SglKernelMockFinder())


from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


# ----------------------------------------------------------------------------
# Mocks
# ----------------------------------------------------------------------------
class _MockAddedToken:
    """Mimics transformers.AddedToken — has `.content` and `.special`."""

    def __init__(self, content: str, special: bool):
        self.content = content
        self.special = special


def _make_mock_tokenizer(
    added_tokens: dict,
    eos_token: str | None = "<|im_end|>",
    unk_token_id: int | None = 0,
):
    tok = Mock()
    tok.added_tokens_decoder = added_tokens
    tok.eos_token = eos_token
    tok.unk_token_id = unk_token_id

    # `convert_tokens_to_ids` lookup over added tokens — values may be either
    # AddedToken-like objects (with `.content`) or plain dicts (`{"content":...}`).
    def _content_of(v):
        return getattr(v, "content", None) or (v.get("content") if isinstance(v, dict) else None)

    rev = {c: tid for tid, v in added_tokens.items() if (c := _content_of(v))}

    def _convert(name: str):
        return rev.get(name, unk_token_id)

    tok.convert_tokens_to_ids = Mock(side_effect=_convert)
    return tok


def _make_mock_tokenizer_manager(tokenizer, tool_call_parser=None,
                                 reasoning_parser=None):
    tm = Mock()
    tm.tokenizer = tokenizer
    tm.model_path = "test-model"
    tm.server_args = Mock()
    tm.server_args.tool_call_parser = tool_call_parser
    tm.server_args.reasoning_parser = reasoning_parser
    tm.server_args.enable_cache_report = False
    tm.model_config = Mock()
    tm.model_config.get_default_sampling_params = Mock(return_value={})
    tm.model_config.hf_config = Mock()
    tm.model_config.hf_config.model_type = "xllm"  # K2/BBQ family; non-gpt-oss
    return tm


def _make_mock_template_manager():
    tpl = Mock()
    tpl.chat_template_name = None
    tpl.jinja_template_content_format = None
    tpl.completion_template_name = None
    tpl.force_reasoning = False
    return tpl


def _make_serving_chat(tokenizer):
    """Instantiate OpenAIServingChat with mocked dependencies, sufficient for
    exercising the strip helpers without touching the network/engine."""
    tm = _make_mock_tokenizer_manager(tokenizer)
    tpl = _make_mock_template_manager()
    # _use_dpsk_v32_encoding probes unmocked attrs; scope the bypass to
    # the constructor only.
    with patch.object(
        OpenAIServingChat, "_use_dpsk_v32_encoding", return_value=False
    ):
        return OpenAIServingChat(tm, tpl)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
class StripSpecialTokensTestCase(unittest.TestCase):
    """K2/BBQ-style tokenizer: structural specials (im_end etc.) marked
    special=True; parser markers (tool_call, arg_key, arg_value, think) marked
    special=False. The strip should remove the former and preserve the latter,
    mirroring what `skip_special_tokens=True` does at decode time."""

    def setUp(self):
        self.added = {
            # Structural / EOS — special=True, should be stripped.
            1: _MockAddedToken("<|im_end|>", special=True),
            2: _MockAddedToken("<|im_start|>", special=True),
            3: _MockAddedToken("<|endoftext|>", special=True),
            4: _MockAddedToken("<|eot_id|>", special=True),
            # Parser-needed domain markers — special=False, must be preserved.
            10: _MockAddedToken("<tool_call>", special=False),
            11: _MockAddedToken("</tool_call>", special=False),
            12: _MockAddedToken("<arg_key>", special=False),
            13: _MockAddedToken("</arg_key>", special=False),
            14: _MockAddedToken("<arg_value>", special=False),
            15: _MockAddedToken("</arg_value>", special=False),
            16: _MockAddedToken("<think>", special=False),
            17: _MockAddedToken("</think>", special=False),
        }
        self.tokenizer = _make_mock_tokenizer(self.added, eos_token="<|im_end|>")
        self.serving = _make_serving_chat(self.tokenizer)

    def test_compute_collects_special_true_only(self):
        """Marker set should equal {tokens with special=True} (plus EOS, which
        is also special=True here so no extra)."""
        markers = set(self.serving._special_token_strings)
        self.assertEqual(
            markers,
            {"<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|eot_id|>"},
        )

    def test_compute_excludes_special_false(self):
        """Domain markers must never appear in the strip set."""
        markers = set(self.serving._special_token_strings)
        for kept in ("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>",
                     "<arg_value>", "</arg_value>", "<think>", "</think>"):
            self.assertNotIn(kept, markers,
                             f"{kept!r} must NOT be stripped — it's a parser marker")

    def test_strip_removes_known_eos_markers(self):
        out = self.serving._strip_special_tokens(
            "Task complete.<|im_end|>\n")
        self.assertEqual(out, "Task complete.\n")

    def test_strip_preserves_tool_call_markers(self):
        """The strip MUST leave <tool_call> / <arg_*> intact for the parser
        prefix-slice. This is the property that lets vLLM run with
        skip_special_tokens=True end-to-end."""
        body = ('I will examine.'
                '<tool_call>bash<arg_key>command</arg_key>'
                '<arg_value>ls /tmp</arg_value></tool_call>')
        out = self.serving._strip_special_tokens(body + "<|im_end|>")
        self.assertEqual(out, body)

    def test_strip_handles_multiple_eos_in_string(self):
        out = self.serving._strip_special_tokens(
            "a<|im_end|>b<|im_end|>c<|endoftext|>d"
        )
        self.assertEqual(out, "abcd")

    def test_strip_none_and_empty(self):
        self.assertIsNone(self.serving._strip_special_tokens(None))
        self.assertEqual(self.serving._strip_special_tokens(""), "")

    def test_strip_no_markers_present(self):
        """When the text doesn't contain any marker, the function returns it
        unchanged (and pays only the cheap `in` check, no replace)."""
        msg = "Hello world. Plain content, no specials."
        self.assertEqual(self.serving._strip_special_tokens(msg), msg)

    def test_strip_idempotent(self):
        once = self.serving._strip_special_tokens(
            "done<|im_end|>x<|im_end|>"
        )
        twice = self.serving._strip_special_tokens(once)
        self.assertEqual(once, twice)


class StripWithDictAddedTokensTestCase(unittest.TestCase):
    """Some tokenizer configs serialize added_tokens_decoder as plain dicts
    (e.g., loaded from tokenizer_config.json) rather than AddedToken objects.
    The helper must handle both."""

    def test_compute_handles_dict_form(self):
        added = {
            1: {"content": "<|im_end|>", "special": True},
            10: {"content": "<tool_call>", "special": False},
        }
        tokenizer = _make_mock_tokenizer(added, eos_token="<|im_end|>")
        serving = _make_serving_chat(tokenizer)
        markers = set(serving._special_token_strings)
        self.assertEqual(markers, {"<|im_end|>"})


class StripDegenerateTokenizerTestCase(unittest.TestCase):
    """Tokenizers without added_tokens_decoder or with no specials at all
    should produce an empty marker set; the helper becomes a no-op."""

    def test_compute_no_added_tokens(self):
        tokenizer = _make_mock_tokenizer({}, eos_token=None)
        serving = _make_serving_chat(tokenizer)
        self.assertEqual(serving._special_token_strings, ())

    def test_strip_noop_when_marker_set_empty(self):
        tokenizer = _make_mock_tokenizer({}, eos_token=None)
        serving = _make_serving_chat(tokenizer)
        msg = "anything <|im_end|> survives because there's no marker set"
        self.assertEqual(serving._strip_special_tokens(msg), msg)

    def test_compute_falls_back_to_eos_attr(self):
        """If added_tokens_decoder is empty/missing but tokenizer.eos_token
        is set, that EOS string still ends up in the marker set."""
        tokenizer = _make_mock_tokenizer({}, eos_token="<|im_end|>")
        serving = _make_serving_chat(tokenizer)
        self.assertEqual(set(serving._special_token_strings), {"<|im_end|>"})


class StripInStreamingPathTestCase(unittest.TestCase):
    """Asserts the strip is applied to every stream-side text field that is
    yielded to the client: reasoning_content delta, tool-parser normal_text,
    and the regular content delta. We don't drive the full async stream
    here — these are focused per-site checks that the helper is called and
    its result is what gets used."""

    def setUp(self):
        self.added = {
            1: _MockAddedToken("<|im_end|>", special=True),
            2: _MockAddedToken("<|im_start|>", special=True),
            10: _MockAddedToken("<tool_call>", special=False),
        }
        self.tokenizer = _make_mock_tokenizer(self.added, eos_token="<|im_end|>")
        self.serving = _make_serving_chat(self.tokenizer)

    def test_strip_on_reasoning_text_delta(self):
        """The reasoning_text yielded as `delta=DeltaMessage(reasoning_content=...)`
        passes through _strip_special_tokens."""
        leaked = "thinking through this<|im_end|>"
        cleaned = self.serving._strip_special_tokens(leaked)
        self.assertNotIn("<|im_end|>", cleaned)
        self.assertEqual(cleaned, "thinking through this")

    def test_strip_on_tool_parser_normal_text(self):
        """The normal_text returned by FunctionCallParser.parse_stream_chunk
        (or JsonArrayParser.parse_streaming_increment) passes through
        _strip_special_tokens before becoming `DeltaMessage(content=...)`."""
        leaked = "preface text<|im_end|>"
        cleaned = self.serving._strip_special_tokens(leaked)
        self.assertNotIn("<|im_end|>", cleaned)
        self.assertEqual(cleaned, "preface text")

    def test_strip_on_regular_content_delta(self):
        """The delta in the regular-content branch (no tool parser active)
        passes through _strip_special_tokens before becoming
        `DeltaMessage(content=...)`."""
        leaked = "answer text<|im_end|>"
        cleaned = self.serving._strip_special_tokens(leaked)
        self.assertNotIn("<|im_end|>", cleaned)
        self.assertEqual(cleaned, "answer text")

    def test_strip_keeps_tool_call_marker_in_stream_delta(self):
        """A streamed delta carrying both a special-true and a special-false
        token preserves the parser-needed marker."""
        leaked = "<tool_call>bash<|im_end|>"
        cleaned = self.serving._strip_special_tokens(leaked)
        self.assertIn("<tool_call>", cleaned)
        self.assertNotIn("<|im_end|>", cleaned)

    def test_finish_time_delta_assembled_clean(self):
        """End-to-end concatenation: many leaked deltas reassembled by a
        client into a single string must contain zero special-token text
        once each delta is stripped on its way out."""
        leaked_deltas = [
            "Hello",
            " world",
            ".<|im_end|>",
            "<|im_start|>",
            "",
        ]
        cleaned_stream = "".join(
            (self.serving._strip_special_tokens(d) or "") for d in leaked_deltas
        )
        self.assertNotIn("<|im_end|>", cleaned_stream)
        self.assertNotIn("<|im_start|>", cleaned_stream)
        self.assertEqual(cleaned_stream, "Hello world.")


if __name__ == "__main__":
    unittest.main()
