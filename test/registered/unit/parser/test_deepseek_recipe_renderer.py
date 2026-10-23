"""deepseek-recipe as the DeepSeek-V4 / V4.1 prompt renderer: the request
translation, the encoder and the sampling overlay."""

import unittest
from types import SimpleNamespace

from sglang.srt.parser import deepseek_recipe_renderer as renderer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def _user(content="x"):
    return {"role": "user", "content": content}


def _tokenizer_file(test):
    """The V4.1 tokenizer from the Hub cache or the Hub; skip when neither answers."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)
    except Exception as error:  # noqa: BLE001
        test.skipTest(f"the tokenizer is not available: {error}")


class TestPreNormalize(CustomTestCase):
    def test_task_is_rejected(self):
        with self.assertRaises(ValueError):
            renderer.pre_normalize({"messages": [_user()], "task": "title"})

    def test_developer_becomes_system(self):
        body = renderer.pre_normalize(
            {"messages": [{"role": "developer", "content": "Be terse."}, _user()]}
        )
        self.assertEqual(body["messages"][0]["role"], "system")

    def test_thinking_toggle_sources(self):
        kwargs = renderer.pre_normalize(
            {"messages": [_user()], "chat_template_kwargs": {"thinking": False}}
        )
        self.assertEqual(kwargs["thinking"], {"type": "disabled"})
        self.assertNotIn("chat_template_kwargs", kwargs)
        reasoning = renderer.pre_normalize(
            {"messages": [_user()], "reasoning": {"effort": "low", "enabled": True}}
        )
        self.assertEqual(reasoning["thinking"], {"type": "enabled"})
        self.assertEqual(reasoning["reasoning_effort"], "low")
        self.assertNotIn("reasoning", reasoning)
        none = renderer.pre_normalize(
            {"messages": [_user()], "reasoning_effort": "none"}
        )
        self.assertEqual(none["thinking"], {"type": "disabled"})
        overridden = renderer.pre_normalize(
            {
                "messages": [_user()],
                "reasoning_effort": "high",
                "chat_template_kwargs": {"thinking": False},
            }
        )
        self.assertEqual(overridden["thinking"], {"type": "disabled"})
        explicit = renderer.pre_normalize(
            {
                "messages": [_user()],
                "thinking": {"type": "enabled"},
                "chat_template_kwargs": {"thinking": False},
            }
        )
        self.assertEqual(explicit["thinking"], {"type": "enabled"})
        default = renderer.pre_normalize({"messages": [_user()]}, {"thinking": False})
        self.assertEqual(default["thinking"], {"type": "disabled"})
        self.assertNotIn("thinking", renderer.pre_normalize({"messages": [_user()]}))

    def test_effort_sources(self):
        body = renderer.pre_normalize(
            {
                "messages": [_user()],
                "reasoning_effort": "high",
                "chat_template_kwargs": {"reasoning_effort": 30},
            }
        )
        self.assertEqual(body["reasoning_effort"], 30)
        default = renderer.pre_normalize(
            {"messages": [_user()]}, {"reasoning_effort": "max"}
        )
        self.assertEqual(default["reasoning_effort"], "max")

    def test_tool_choice_renders_as_auto(self):
        tools = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]
        named = renderer.pre_normalize(
            {
                "messages": [_user()],
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": "b"}},
            }
        )
        self.assertEqual(named["tool_choice"], "auto")
        self.assertEqual(len(named["tools"]), 2)
        required = renderer.pre_normalize(
            {"messages": [_user()], "tools": tools, "tool_choice": "required"}
        )
        self.assertEqual(required["tool_choice"], "auto")
        for body in (
            {"messages": [_user()], "tools": tools, "tool_choice": "none"},
            {"messages": [_user()], "tools": [], "tool_choice": "auto"},
        ):
            off = renderer.pre_normalize(body)
            self.assertNotIn("tools", off)
            self.assertNotIn("tool_choice", off)

    def test_history_translation_and_host_fields(self):
        body = renderer.pre_normalize(
            {
                "messages": [
                    _user("weather?"),
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"location": "Paris", "days": 1.5},
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
                    {"role": "assistant", "content": "It is"},
                ],
                "continue_final_message": True,
                "n": 2,
                "response_format": {"type": "structural_tag", "structures": []},
            }
        )
        call = body["messages"][1]["tool_calls"][0]
        self.assertEqual(
            call["function"]["arguments"], '{"location":"Paris","days":1.5}'
        )
        self.assertTrue(body["messages"][3]["prefix"])
        self.assertNotIn("continue_final_message", body)
        self.assertNotIn("response_format", body)
        self.assertEqual(body["n"], 2)
        unpaired = renderer.pre_normalize(
            {"messages": [_user(), {"role": "tool", "content": "sunny"}]}
        )
        self.assertNotIn("tool_call_id", unpaired["messages"][1])

    def test_input_is_not_mutated(self):
        original = {"messages": [{"role": "developer", "content": "x"}], "n": 3}
        renderer.pre_normalize(original)
        self.assertEqual(original["messages"][0]["role"], "developer")
        self.assertEqual(original["n"], 3)


class TestOverlaySampling(CustomTestCase):
    def _render(self, **overrides):
        values = dict(
            prompt="",
            prompt_ids=[0, 1, 2],
            temperature=1.0,
            top_p=0.95,
            thinking=True,
            request=SimpleNamespace(resolve_max_tokens=lambda prompt, limit: 65536),
        )
        values.update(overrides)
        return renderer.RecipeRender(**values)

    def test_recipe_sampling_and_default_max_tokens(self):
        params = {"temperature": 0.0, "top_p": 0.1, "max_new_tokens": None}
        renderer.overlay_sampling(
            params, self._render(), prompt_tokens=100, context_length=1_000_000
        )
        self.assertEqual(params["temperature"], 1.0)
        self.assertEqual(params["top_p"], 0.95)
        self.assertEqual(params["max_new_tokens"], 65536)
        params = {"max_new_tokens": 5}
        renderer.overlay_sampling(
            params, self._render(), prompt_tokens=1, context_length=10
        )
        self.assertEqual(params["max_new_tokens"], 5)
        self.assertNotIn("json_schema", params)


class TestRecipeEncoder(CustomTestCase):
    def test_renders_and_encodes_with_the_served_vocabulary(self):
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(_tokenizer_file(self))
        encoder = renderer.RecipeEncoder("v41", tokenizer)
        render = encoder.convert(
            {"messages": [_user("")], "thinking": {"type": "disabled"}}
        )
        self.assertEqual(
            render.prompt, "<｜begin▁of▁sentence｜><｜User｜><｜Assistant｜></think>"
        )
        self.assertEqual(
            render.prompt_ids,
            [
                tokenizer.token_to_id(token)
                for token in (
                    "<｜begin▁of▁sentence｜>",
                    "<｜User｜>",
                    "<｜Assistant｜>",
                    "</think>",
                )
            ],
        )
        self.assertFalse(render.thinking)
        self.assertEqual(render.temperature, 1.0)
        self.assertEqual(render.top_p, 1.0)
        # The API's default: thinking on, with recipe's sampling for it.
        default = encoder.convert({"messages": [_user("hi")]}, encode=False)
        self.assertTrue(default.thinking)
        self.assertIsNone(default.prompt_ids)
        self.assertTrue(default.prompt.endswith("<think>"))
        self.assertGreaterEqual(default.top_p, 0.95)
        with self.assertRaisesRegex(ValueError, "n = 1"):
            encoder.convert({"messages": [_user("hi")], "n": 2})


if __name__ == "__main__":
    unittest.main()
