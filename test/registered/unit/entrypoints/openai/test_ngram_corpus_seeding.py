import json
import unittest

from sglang.srt.entrypoints.openai.ngram_corpus_seeding import render_tool_call_seeds
from sglang.srt.sampling.sampling_params import (
    MAX_REQUEST_NGRAM_CORPUS_SEEDS_TOKENS,
    set_request_ngram_corpus_seeds,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _LineTokenizer:
    """One line per message, prefixed by the tool count, so a seed is exactly
    the appended line and both renders must have seen the same tools."""

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        return_dict=False,
        **_,
    ):
        lines = [f"tools={len(tools or [])}"]
        for message in messages:
            if message.get("tool_calls"):
                calls = ";".join(
                    f"{call['function']['name']}"
                    f"({json.dumps(call['function']['arguments'], sort_keys=True)})"
                    for call in message["tool_calls"]
                )
                lines.append(f"{message['role']}:{calls}")
            else:
                lines.append(f"{message['role']}:{message.get('content') or ''}")
        text = "\n".join(lines) + "\n"
        if add_generation_prompt:
            text += "assistant:"
        return text

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return [ord(char) for char in text]


class _RewritingTokenizer(_LineTokenizer):
    """Puts the message count first, so appending a message changes the prefix."""

    def apply_chat_template(self, messages, **kwargs):
        return f"{len(messages)}\n" + super().apply_chat_template(messages, **kwargs)


class _NoToolCallsTokenizer(_LineTokenizer):
    """Raises on an assistant tool_calls message, as many templates do."""

    def apply_chat_template(self, messages, **kwargs):
        if any(message.get("tool_calls") for message in messages):
            raise ValueError("tool calls are not supported by this template")
        return super().apply_chat_template(messages, **kwargs)


def _make_tool(name, properties):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": properties},
        },
    }


MESSAGES = [{"role": "user", "content": "What is the weather in Paris?"}]
TOOLS = [
    _make_tool("get_weather", {"city": {"type": "string"}}),
    _make_tool(
        "count_items", {"n": {"type": "integer"}, "strict": {"type": "boolean"}}
    ),
]


def _decode(seed_tokens):
    return "".join(chr(token) for token in seed_tokens)


class TestRenderToolCallSeeds(CustomTestCase):
    def _render(self, tokenizer, tools=TOOLS):
        return render_tool_call_seeds(
            tokenizer=tokenizer,
            messages=MESSAGES,
            tools=tools,
            template_kwargs={},
            encode_kwargs={"add_special_tokens": False},
        )

    def test_seed_is_the_rendered_call_after_the_conversation(self):
        seeds = self._render(_LineTokenizer())
        self.assertEqual(
            [_decode(seed_tokens) for seed_tokens in seeds],
            [
                'assistant:get_weather({"city": "X"})\n',
                'assistant:count_items({"n": 0, "strict": true})\n',
            ],
        )

    def test_template_that_does_not_extend_the_prompt_yields_no_seed(self):
        """If appending the call rewrites earlier text, the difference is not a
        continuation of the prompt and would never match."""
        self.assertEqual(self._render(_RewritingTokenizer()), [])

    def test_template_that_rejects_tool_calls_yields_no_seed(self):
        """The request is still served, just without seeds; the per-tool
        failure must be swallowed, not surfaced."""
        self.assertEqual(self._render(_NoToolCallsTokenizer()), [])

    def test_seeds_over_the_cap_are_skipped_not_truncated(self):
        """A truncated seed would draft a cut-off call. The cap is the flat
        list set_request_ngram_corpus_seeds builds, separators included, so two
        seeds that fit on their own may not fit together; whatever the renderer
        returns must attach without error."""
        half = MAX_REQUEST_NGRAM_CORPUS_SEEDS_TOKENS // 2
        # 'assistant:' + name + '({})\n' renders to exactly `half` characters.
        name = "x" * (half - len("assistant:({})\n"))
        tools = [_make_tool(name, {}), _make_tool(name.replace("x", "y"), {}), TOOLS[0]]
        seeds = self._render(_LineTokenizer(), tools=tools)
        weather = 'assistant:get_weather({"city": "X"})\n'
        self.assertEqual(
            [len(seed_tokens) for seed_tokens in seeds], [half, len(weather)]
        )
        sampling_params = {}
        set_request_ngram_corpus_seeds(sampling_params, seeds)


if __name__ == "__main__":
    unittest.main()
