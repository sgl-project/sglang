"""Incremental chat tokenization must return exactly what a full encode returns."""

import copy
import unittest
from types import SimpleNamespace

from transformers import AutoTokenizer

from sglang.srt.entrypoints.openai.incremental_chat_tokenizer import (
    IncrementalChatTokenizer,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

CONVERSATION = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 21C"},
    {"role": "assistant", "content": "Sunny and 21C."},
    {"role": "user", "content": "And tomorrow? Answer in one word."},
    {"role": "assistant", "content": "Rain."},
    {"role": "user", "content": "Thanks! Bye."},
]


def _render(tokenizer, messages, tools=None):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=tools
    )


class TestIncrementalChatTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def _replay(self, cache, tokenizer, turns, tools=None):
        for messages in turns:
            prompt = _render(tokenizer, messages, tools)
            got = cache.encode(
                messages,
                prompt,
                tools=tools,
                template_kwargs={},
                encode_kwargs={},
            )
            self.assertEqual(got, tokenizer.encode(prompt), msg=f"{len(messages)} msgs")

    def test_growing_conversation_reuses_history_and_matches_full_encode(self):
        """Each turn re-sends the history; ids must match and the history is reused."""
        cache = IncrementalChatTokenizer(self.tokenizer)
        turns = [CONVERSATION[:n] for n in range(2, len(CONVERSATION) + 1)]
        self._replay(cache, self.tokenizer, turns, tools=TOOLS)
        self.assertGreaterEqual(cache.stats.history_reuses, len(turns) - 1)

    def test_edited_history_falls_back_to_full_encode(self):
        """A rewritten earlier message must not reuse ids cached for the old text."""
        cache = IncrementalChatTokenizer(self.tokenizer)
        self._replay(cache, self.tokenizer, [CONVERSATION[:6]])
        edited = copy.deepcopy(CONVERSATION)
        edited[1]["content"] = "What is the weather in Rome?"
        self._replay(cache, self.tokenizer, [edited])
        self.assertEqual(cache.stats.history_reuses, 0)

    def test_history_not_a_text_prefix_falls_back(self):
        """A template that re-renders old turns differently (here: a turn count in
        the header) must encode in full instead of reusing a stale prefix."""
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.chat_template = (
            "<|im_start|>system\nturns={{ messages | length }}<|im_end|>\n"
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
            "<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        cache = IncrementalChatTokenizer(tokenizer)
        turns = [CONVERSATION[1:n] for n in (2, 3, 5, 6)]
        turns = [[m for m in t if m["role"] in ("user", "assistant")] for t in turns]
        self._replay(cache, tokenizer, turns)
        self.assertEqual(cache.stats.history_reuses, 0)

    def test_suffix_without_special_boundary_is_not_concatenated(self):
        """Text that merges across the join (no special token between messages)
        must be encoded in full: concatenating the two encodings differs."""
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m.content }}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>{% endif %}"
        )
        history = [{"role": "user", "content": "unbeliev"}]
        grown = history + [{"role": "assistant", "content": "able"}]
        naive = tokenizer.encode("unbeliev") + tokenizer.encode("able<|im_start|>")
        self.assertNotEqual(naive, tokenizer.encode(_render(tokenizer, grown)))

        cache = IncrementalChatTokenizer(tokenizer)
        self._replay(cache, tokenizer, [history, grown])

    def test_bounds_evict_least_recently_used(self):
        cache = IncrementalChatTokenizer(self.tokenizer, max_entries=2)
        for city in ("Paris", "Rome", "Oslo"):
            messages = [{"role": "user", "content": f"Weather in {city}?"}]
            cache.encode(
                messages,
                _render(self.tokenizer, messages),
                tools=None,
                template_kwargs={},
                encode_kwargs={},
            )
        self.assertEqual(len(cache._entries), 2)


class TestServingChatWiring(unittest.TestCase):
    """The serving path must use the cache when enabled and fail open when it raises."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def _server(self, incremental):
        return SimpleNamespace(
            tokenizer_manager=SimpleNamespace(tokenizer=self.tokenizer),
            _prompt_text_round_trip_is_lossy=False,
            _incremental_chat_tokenizer=incremental,
            _chat_template_cache={},
        )

    def _encode(self, server, messages):
        prompt_ids, _ = OpenAIServingChat._render_and_encode_chat_template(
            server,
            messages,
            tools=None,
            template_kwargs={},
            encode_kwargs={},
            use_cache=False,
        )
        return prompt_ids

    def test_enabled_path_matches_full_encode(self):
        cache = IncrementalChatTokenizer(self.tokenizer)
        server = self._server(cache)
        for n in (2, 5, 6):
            messages = [m for m in CONVERSATION[:n] if m["role"] != "tool"]
            messages = [m for m in messages if not m.get("tool_calls")]
            expected = self.tokenizer.encode(_render(self.tokenizer, messages))
            self.assertEqual(self._encode(server, messages), expected)
        self.assertEqual(cache.stats.calls, 3)

    def test_failure_falls_back_to_full_encode(self):
        broken = SimpleNamespace(encode=lambda *a, **k: 1 / 0)
        messages = CONVERSATION[:2]
        expected = self.tokenizer.encode(_render(self.tokenizer, messages))
        self.assertEqual(self._encode(self._server(broken), messages), expected)


if __name__ == "__main__":
    unittest.main()
