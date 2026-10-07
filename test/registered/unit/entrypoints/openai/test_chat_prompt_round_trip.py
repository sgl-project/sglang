"""Chat prompts must not go through text on tokenizers that can't round-trip.

mistral_common tokenizers emit control tokens that have no text form, so
rendering the template to a string and re-encoding silently replaces them with
their literal characters (and adds a second BOS). These tests pin the probe that
detects such a tokenizer and the prompt dispatch that reacts to it.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

probe = OpenAIServingChat._probe_prompt_text_round_trip
engine_prompt = OpenAIServingChat._engine_prompt
render_and_encode = OpenAIServingChat._render_and_encode_chat_template


class FakeTokenizer:
    """Renders to text and encodes to ids independently, like the real thing."""

    def __init__(self, text_ids, template_ids, raises=False):
        self._text_ids = text_ids
        self._template_ids = template_ids
        self._raises = raises
        self.template_calls = []

    def encode(self, text, **kwargs):
        return [] if text == "" else list(self._text_ids)

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        self.template_calls.append(tokenize)
        if self._raises:
            raise ValueError("template needs kwargs this probe does not pass")
        return list(self._template_ids) if tokenize else "<s>[INST]x[/INST]"


def _server(tokenizer, auto_adds_specials=False):
    return SimpleNamespace(
        tokenizer_manager=SimpleNamespace(tokenizer=tokenizer),
        _tokenizer_auto_adds_specials=auto_adds_specials,
    )


def _messages(prompt_ids, prompt="rendered", **media):
    return SimpleNamespace(
        prompt=prompt,
        prompt_ids=prompt_ids,
        image_data=media.get("image_data"),
        video_data=media.get("video_data"),
        audio_data=media.get("audio_data"),
    )


class TestProbe(unittest.TestCase):
    def test_divergent_encodings_are_lossy(self):
        tok = FakeTokenizer(text_ids=[9, 9, 9, 9], template_ids=[1, 3, 4])
        self.assertTrue(probe(_server(tok)))

    def test_matching_encodings_are_not_lossy(self):
        tok = FakeTokenizer(text_ids=[1, 3, 4], template_ids=[1, 3, 4])
        self.assertFalse(probe(_server(tok)))

    def test_a_template_that_raises_keeps_the_text_path(self):
        tok = FakeTokenizer(text_ids=[1], template_ids=[1], raises=True)
        self.assertFalse(probe(_server(tok)))


class TestRenderAndEncode(unittest.TestCase):
    def test_lossy_tokenizer_encodes_straight_to_ids_without_a_text_render(self):
        tok = FakeTokenizer(text_ids=[9, 9, 9, 9], template_ids=[1, 3, 4])
        server = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(tokenizer=tok),
            _prompt_text_round_trip_is_lossy=True,
        )

        prompt_ids, _ = render_and_encode(
            server,
            [{"role": "user", "content": "x"}],
            tools=None,
            template_kwargs={},
            encode_kwargs={},
            use_cache=False,
        )

        self.assertEqual(prompt_ids, [1, 3, 4])
        self.assertEqual(tok.template_calls, [True])


class TestEnginePrompt(unittest.TestCase):
    def _server(self, lossy):
        return SimpleNamespace(
            _prompt_text_round_trip_is_lossy=lossy, chat_encoding_spec=None
        )

    def test_lossy_text_only_sends_ids(self):
        key, value = engine_prompt(
            self._server(True), _messages([1, 3, 4]), is_multimodal=True
        )
        self.assertEqual(key, "input_ids")
        self.assertEqual(value, [1, 3, 4])

    def test_lossy_with_an_image_still_sends_text(self):
        # The MM processor has to tokenize the text itself to expand placeholders.
        key, _ = engine_prompt(
            self._server(True),
            _messages([1, 3, 4], image_data=["img"]),
            is_multimodal=True,
        )
        self.assertEqual(key, "text")

    def test_non_lossy_multimodal_is_unchanged(self):
        key, value = engine_prompt(
            self._server(False), _messages([1, 3, 4]), is_multimodal=True
        )
        self.assertEqual(key, "text")
        self.assertEqual(value, "rendered")

    def test_lossy_without_ids_falls_back_to_text(self):
        key, _ = engine_prompt(self._server(True), _messages([]), is_multimodal=True)
        self.assertEqual(key, "text")

    def test_text_model_sends_ids_either_way(self):
        for lossy in (True, False):
            key, value = engine_prompt(
                self._server(lossy), _messages([1, 3, 4]), is_multimodal=False
            )
            self.assertEqual(key, "input_ids")
            self.assertEqual(value, [1, 3, 4])


if __name__ == "__main__":
    unittest.main()
