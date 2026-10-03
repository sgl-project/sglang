"""The chat prompt is encoded one message at a time and the ids are reused across turns.

An agent sends its whole history on every turn. These tests pin that the
per-message ids equal a full encode, that a turn only encodes its new messages,
that tokenizers for which the split is not exact get no cache, and that the
cache turns itself off when a full encode disagrees.
"""

import unittest
from types import SimpleNamespace

from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints.openai.prompt_segment_cache import PromptSegmentCache
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

render_and_encode = OpenAIServingChat._render_and_encode_chat_template

TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    # Mentions every role token so the cache finds its split points.
    "{# <|system|> <|user|> <|assistant|> <|observation|> #}"
)
ROLES = ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>")
CORPUS = [
    "the service returned errors after the deploy at noon",
    "query the logs for timeouts and connection resets",
    'tool result: {"status": "error", "count": 42, "host": "web-1"}',
    "the root cause is a connection pool that is too small",
]
# Message content that looks like the template: markers, parts of markers,
# whitespace at both ends, and text outside ASCII.
TEMPLATE_LIKE_CONTENT = [
    "the user wrote <|user|> and <|assistant|><|observation|> in a log line",
    "a cut marker <|user and <| and |> and <|system",
    "  leading and trailing whitespace \n\n",
    "région ünïcode 日本語 \U0001f600 <|system|>",
]


def _tokenizer(normalizer=None, template=TEMPLATE, extra_tokens=(), **role_flags):
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    if normalizer is not None:
        backend.normalizer = normalizer
    backend.train_from_iterator(
        CORPUS * 4, BpeTrainer(vocab_size=400, special_tokens=["[UNK]"])
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                AddedToken(role, special=True, **role_flags) for role in ROLES
            ]
        }
    )
    for token in extra_tokens:
        tokenizer.add_tokens([token], special_tokens=token.special)
    tokenizer.chat_template = template
    return tokenizer


class CountingTokenizer:
    """Forwards to a tokenizer and records the text of every encode call."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.encoded = []

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def encode(self, text, **kwargs):
        self.encoded.append(text)
        return self._tokenizer.encode(text, **kwargs)


class ContextDependentTokenizer(CountingTokenizer):
    """A full prompt gets one more id than the sum of its messages."""

    def encode(self, text, **kwargs):
        ids = super().encode(text, **kwargs)
        return ids + [0] if text.count("<|") > 1 else ids


def _conversation(turns, contents=CORPUS):
    messages = [
        {"role": "system", "content": contents[0]},
        {"role": "user", "content": contents[1]},
    ]
    for turn in range(turns):
        messages.append({"role": "assistant", "content": f"{contents[3]} {turn}"})
        messages.append({"role": "observation", "content": f"{contents[2]} {turn}"})
    return messages


def _render(tokenizer, turns, contents=CORPUS):
    return tokenizer.apply_chat_template(
        _conversation(turns, contents), tokenize=False, add_generation_prompt=True
    )


def _unchecked_cache(tokenizer, **kwargs):
    """A cache that never compares with a full encode, so the test does."""
    return PromptSegmentCache.create(
        tokenizer, check_first=0, check_every=10**9, **kwargs
    )


class TestPromptSegmentCache(CustomTestCase):
    def test_ids_equal_a_full_encode_on_every_turn(self):
        tokenizer = _tokenizer()
        cache = _unchecked_cache(tokenizer)
        for turns in range(6):
            text = _render(tokenizer, turns)
            self.assertEqual(cache.encode(text, {}), tokenizer.encode(text))
        self.assertTrue(cache.enabled)

    def test_ids_equal_a_full_encode_when_content_looks_like_the_template(self):
        tokenizer = _tokenizer()
        cache = _unchecked_cache(tokenizer)
        for turns in range(4):
            text = _render(tokenizer, turns, TEMPLATE_LIKE_CONTENT)
            self.assertEqual(cache.encode(text, {}), tokenizer.encode(text))

    def test_a_turn_only_encodes_its_new_messages(self):
        tokenizer = CountingTokenizer(_tokenizer())
        cache = _unchecked_cache(tokenizer)
        cache.encode(_render(tokenizer, 3), {})
        tokenizer.encoded.clear()

        cache.encode(_render(tokenizer, 4), {})

        # Turn 4 adds one assistant and one observation message. The generation
        # prompt at the end was already encoded on the previous turn.
        self.assertEqual(len(tokenizer.encoded), 2)
        self.assertTrue(all(text.startswith("<|") for text in tokenizer.encoded))

    def test_encode_kwargs_are_part_of_the_key(self):
        tokenizer = CountingTokenizer(_tokenizer())
        cache = _unchecked_cache(tokenizer)
        text = _render(tokenizer, 1)
        cache.encode(text, {})
        tokenizer.encoded.clear()

        cache.encode(text, {"add_special_tokens": False})

        self.assertGreater(len(tokenizer.encoded), 0)

    def test_a_disagreeing_full_encode_turns_the_cache_off(self):
        tokenizer = ContextDependentTokenizer(_tokenizer())
        cache = PromptSegmentCache.create(tokenizer, check_first=1)
        text = _render(tokenizer, 2)

        ids = cache.encode(text, {})

        self.assertEqual(ids, tokenizer.encode(text))
        self.assertFalse(cache.enabled)

    def test_the_check_repeats_at_the_interval(self):
        tokenizer = ContextDependentTokenizer(_tokenizer())
        cache = PromptSegmentCache.create(tokenizer, check_first=0, check_every=2)
        text = _render(tokenizer, 2)

        cache.encode(text, {})
        self.assertTrue(cache.enabled)
        cache.encode(text, {})

        self.assertFalse(cache.enabled)

    def test_old_messages_are_evicted_by_size(self):
        tokenizer = _tokenizer()
        cache = _unchecked_cache(tokenizer, max_chars=200)
        for turns in range(8):
            text = _render(tokenizer, turns)
            self.assertEqual(cache.encode(text, {}), tokenizer.encode(text))
        self.assertLessEqual(
            cache._chars, 200 + max(len(s) for s, _ in cache._segments)
        )

    def test_a_message_that_every_request_uses_is_not_evicted(self):
        tokenizer = CountingTokenizer(_tokenizer())
        system = {"role": "system", "content": CORPUS[0]}
        # Room for the system message and about two user messages.
        cache = _unchecked_cache(tokenizer, max_chars=200)

        for request in range(8):
            user = {"role": "user", "content": f"{CORPUS[1]} {request}"}
            text = tokenizer.apply_chat_template([system, user], tokenize=False)
            cache.encode(text, {})

        system_encodes = [t for t in tokenizer.encoded if t.startswith("<|system|>")]
        self.assertEqual(len(system_encodes), 1)


class TestUnsupportedTokenizers(CustomTestCase):
    def test_a_normalizer_gets_no_cache(self):
        tokenizer = _tokenizer(normalizer=normalizers.Lowercase())
        self.assertIsNone(PromptSegmentCache.create(tokenizer))

    def test_a_special_token_that_depends_on_its_neighbors_gets_no_cache(self):
        for flag in ("lstrip", "rstrip", "single_word"):
            with self.subTest(flag=flag):
                tokenizer = _tokenizer(**{flag: True})
                self.assertIsNone(PromptSegmentCache.create(tokenizer))

    def test_a_template_without_special_tokens_gets_no_cache(self):
        tokenizer = _tokenizer(template="{{ messages[0]['content'] }}")
        self.assertIsNone(PromptSegmentCache.create(tokenizer))

    def test_a_tokenizer_without_a_fast_backend_gets_no_cache(self):
        slow = SimpleNamespace(encode=lambda text, **kwargs: [1])
        self.assertIsNone(PromptSegmentCache.create(slow))

    def test_a_backend_that_is_not_huggingface_tokenizers_gets_no_cache(self):
        real = _tokenizer()
        other_backend = SimpleNamespace(
            backend_tokenizer=SimpleNamespace(normalizer=None),
            chat_template=real.chat_template,
            added_tokens_decoder=real.added_tokens_decoder,
            encode=real.encode,
        )
        self.assertIsNone(PromptSegmentCache.create(other_backend))

    def test_an_added_token_that_overlaps_a_marker_never_changes_the_ids(self):
        cases = {
            "ends with the start of a marker": (
                [AddedToken("=<", normalized=False)],
                TEMPLATE,
                "<|user|>\nx =<|assistant|>\n",
            ),
            "contains a marker": (
                [AddedToken("a<|user|>", normalized=False)],
                TEMPLATE,
                "<|system|>\nbanana<|user|>\nquery the logs",
            ),
            "marker that overlaps itself": (
                [AddedToken(">>", normalized=False, special=True)],
                TEMPLATE + "{# >> #}",
                "<|user|>\nshift >>> right<|assistant|>\n",
            ),
            "starts inside a marker that the backend matches later": (
                [
                    AddedToken("small", normalized=True, special=True),
                    AddedToken("mall", normalized=False),
                ],
                TEMPLATE + "{# small #}",
                "<|user|>\nthe pool is too small",
            ),
        }
        for name, (extra_tokens, template, text) in cases.items():
            with self.subTest(name):
                tokenizer = _tokenizer(template=template, extra_tokens=extra_tokens)
                cache = _unchecked_cache(tokenizer)
                # No cache is a correct answer. A cache with other ids is not.
                ids = cache.encode(text, {}) if cache else tokenizer.encode(text)
                self.assertEqual(ids, tokenizer.encode(text))


class TestRenderAndEncode(CustomTestCase):
    def _server(self, tokenizer, cache):
        return SimpleNamespace(
            tokenizer_manager=SimpleNamespace(tokenizer=tokenizer),
            _prompt_text_round_trip_is_lossy=False,
            _prompt_segment_cache=cache,
        )

    def _encode(self, server, turns):
        prompt_ids, _ = render_and_encode(
            server,
            _conversation(turns),
            tools=None,
            template_kwargs={},
            encode_kwargs={},
            use_cache=False,
        )
        return prompt_ids

    def test_the_chat_endpoint_reuses_message_ids(self):
        tokenizer = CountingTokenizer(_tokenizer())
        server = self._server(tokenizer, _unchecked_cache(tokenizer))
        first = self._encode(server, 2)
        tokenizer.encoded.clear()

        second = self._encode(server, 3)

        self.assertEqual(second, tokenizer._tokenizer.encode(_render(tokenizer, 3)))
        # Both prompts end with the generation prompt. The rest of the first is a prefix.
        tail = len(tokenizer._tokenizer.encode("<|assistant|>\n"))
        self.assertEqual(second[: len(first) - tail], first[:-tail])
        self.assertEqual(len(tokenizer.encoded), 2)

    def test_without_a_cache_the_prompt_is_encoded_whole(self):
        tokenizer = CountingTokenizer(_tokenizer())
        server = self._server(tokenizer, None)

        prompt_ids = self._encode(server, 2)

        self.assertEqual(prompt_ids, tokenizer._tokenizer.encode(_render(tokenizer, 2)))
        self.assertEqual(len(tokenizer.encoded), 1)

    def test_a_cache_that_turned_itself_off_encodes_the_prompt_whole(self):
        tokenizer = ContextDependentTokenizer(_tokenizer())
        cache = PromptSegmentCache.create(tokenizer, check_first=1)
        server = self._server(tokenizer, cache)
        self._encode(server, 2)
        self.assertFalse(cache.enabled)
        tokenizer.encoded.clear()

        self._encode(server, 3)

        self.assertEqual(tokenizer.encoded, [_render(tokenizer, 3)])


if __name__ == "__main__":
    unittest.main()
