import json
import unittest

from sglang.srt.parser import deepseek_recipe_renderer as renderer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def _tokenizer_file():
    """The V4.1 tokenizer; only outside CI does an unreachable Hub skip the test."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)
    except Exception as error:  # noqa: BLE001
        if is_in_ci():
            raise
        raise unittest.SkipTest(f"the tokenizer is not available: {error}")


class TestPreNormalize(CustomTestCase):
    def test_developer_becomes_system(self):
        body = renderer.pre_normalize(
            {
                "messages": [
                    {"role": "developer", "content": "Be terse."},
                    {"role": "user", "content": "x"},
                ],
                "thinking": False,
            }
        )
        self.assertEqual(body["messages"][0]["role"], "system")

    def test_tool_choice_renders_as_auto(self):
        tools = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]
        named = renderer.pre_normalize(
            {
                "messages": [{"role": "user", "content": "x"}],
                "thinking": False,
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": "b"}},
            }
        )
        self.assertEqual(named["tool_choice"], "auto")
        self.assertEqual(len(named["tools"]), 2)
        required = renderer.pre_normalize(
            {
                "messages": [{"role": "user", "content": "x"}],
                "thinking": False,
                "tools": tools,
                "tool_choice": "required",
            }
        )
        self.assertEqual(required["tool_choice"], "auto")
        for tools_field, choice in ((tools, "none"), ([], "auto")):
            off = renderer.pre_normalize(
                {
                    "messages": [{"role": "user", "content": "x"}],
                    "thinking": False,
                    "tools": tools_field,
                    "tool_choice": choice,
                }
            )
            self.assertNotIn("tools", off)
            self.assertNotIn("tool_choice", off)

    def test_history_translation(self):
        body = renderer.pre_normalize(
            {
                "messages": [
                    {"role": "user", "content": "weather?"},
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
                "thinking": False,
                "continue_final_message": True,
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


class TestEffortBudget(CustomTestCase):
    def test_integer_and_float_budgets(self):
        self.assertEqual(renderer.parse_effort_budget(30), 30)
        self.assertEqual(renderer.parse_effort_budget(0.5), 50)
        self.assertEqual(renderer.parse_effort_budget(0.0), 1)
        for value in (0, 101, 1.0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                renderer.parse_effort_budget(value)


class TestDeepSeekRecipe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from tokenizers import Tokenizer

        cls.tokenizer = Tokenizer.from_file(_tokenizer_file())
        cls.deepseek_recipe = renderer.DeepSeekRecipe("dsv41", cls.tokenizer)

    def _ids(self, *tokens):
        return [self.tokenizer.token_to_id(token) for token in tokens]

    def test_encodes_with_the_served_vocabulary(self):
        self.assertEqual(
            self.deepseek_recipe.encode(
                {"messages": [{"role": "user", "content": ""}], "thinking": False}
            ),
            self._ids(
                "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>"
            ),
        )
        thinking = self.deepseek_recipe.encode(
            {"messages": [{"role": "user", "content": "hi"}], "thinking": True}
        )
        self.assertEqual(thinking[-2:], self._ids("<｜Assistant｜>", "<think>"))

    def test_tasks_replace_or_follow_the_generation_header(self):
        for thinking in (False, True):
            body = {
                "messages": [{"role": "user", "content": "hello"}],
                "thinking": thinking,
            }
            plain = self.deepseek_recipe.encode(body)
            for task, token in renderer.TASK_TOKENS.items():
                with self.subTest(thinking=thinking, task=task):
                    expected = (
                        plain + self._ids(token)
                        if task == "action"
                        else plain[:-2] + self._ids(token)
                    )
                    self.assertEqual(
                        self.deepseek_recipe.encode(body, task=task), expected
                    )
        continued = {
            "messages": [
                {"role": "user", "content": "count"},
                {"role": "assistant", "content": "1, 2,"},
            ],
            "thinking": False,
            "continue_final_message": True,
        }
        with self.assertRaisesRegex(ValueError, "end with a user message"):
            self.deepseek_recipe.encode(continued, task="query")

    def test_effort_budget_rewrites_the_effort_line(self):
        body = {"messages": [{"role": "user", "content": "hi"}], "thinking": True}
        high = self.tokenizer.decode(
            self.deepseek_recipe.encode({**body, "reasoning_effort": "high"}),
            skip_special_tokens=False,
        )
        self.assertIn("Reasoning Effort: 75 (range 1-100", high)
        for budget in (1, 30, 100):
            with self.subTest(budget=budget):
                expected = high.replace(
                    "Reasoning Effort: 75 ", f"Reasoning Effort: {budget} ", 1
                )
                self.assertEqual(
                    self.deepseek_recipe.encode(body, effort_budget=budget),
                    self.tokenizer.encode(expected, add_special_tokens=False).ids,
                )
        # Chat mode renders no effort line, so the budget changes nothing.
        chat = {"messages": [{"role": "user", "content": "hi"}], "thinking": False}
        self.assertEqual(
            self.deepseek_recipe.encode(chat, effort_budget=30),
            self.deepseek_recipe.encode(chat),
        )

    def test_sliced_conversation_matches_full_prompt_prefix(self):
        """Dropped prompt configuration breaks prefix parity for sliced histories."""
        recipe = renderer.load_deepseek_recipe()
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
        for thinking in (True, False):
            body = {
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {"role": "assistant", "content": "", "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                ],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "thinking": thinking,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "weather",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    },
                },
            }
            with self.subTest(thinking=thinking):
                conversation = self.deepseek_recipe.convert(body, effort_budget=30)
                full_ids = self.deepseek_recipe.encode_conversation(
                    conversation, effort_budget=30
                )
                self.assertEqual(
                    full_ids, self.deepseek_recipe.encode(body, effort_budget=30)
                )
                history = recipe.Conversation(
                    conversation.messages[:2],
                    thinking_mode=conversation.thinking_mode,
                    tools=conversation.tools,
                    tool_choice=conversation.tool_choice,
                    reasoning_effort=conversation.reasoning_effort,
                    response_schema=conversation.response_schema,
                    response_regex=conversation.response_regex,
                )
                prefix_ids = self.deepseek_recipe.encode_conversation(
                    history, effort_budget=30
                )
                think_token = "<think>" if thinking else "</think>"
                self.assertEqual(
                    prefix_ids[-2:], self._ids("<｜Assistant｜>", think_token)
                )
                self.assertEqual(full_ids[: len(prefix_ids) - 2], prefix_ids[:-2])

    def test_v4_rejects_tools_declared_in_messages(self):
        """V4's encoding has no place for message-declared tools and would drop them."""
        tool = {"type": "function", "function": {"name": "f"}}
        with self.assertRaisesRegex(ValueError, "tools declared in messages"):
            renderer.DeepSeekRecipe("dsv4", self.tokenizer).encode(
                {
                    "messages": [
                        {"role": "system", "content": "", "tools": [tool]},
                        {"role": "user", "content": "x"},
                    ],
                    "thinking": False,
                    "tool_choice": "auto",
                }
            )

    def test_conversion_errors_locate_the_request_message(self):
        """An error names the message's index in the request, whatever number of
        earlier messages declare tools."""
        tools = [{"function": {"name": "f"}}]
        bad = [{"type": "image_url", "image_url": {"url": "ftp://example.com/a.png"}}]
        hi = {"role": "user", "content": "hi"}
        failing = {"role": "user", "content": bad}
        for index, messages in (
            (1, [{"role": "system", "content": "s", "tools": tools}, failing]),
            (1, [hi, failing]),
            (
                3,
                [
                    {"role": "system", "content": "s", "tools": tools},
                    hi,
                    {
                        "role": "developer",
                        "content": "",
                        "tools": [{"function": {"name": "g"}}],
                    },
                    failing,
                ],
            ),
            (1, [hi, {"role": "system", "content": bad, "tools": tools}, hi]),
        ):
            with self.subTest(index=index, count=len(messages)):
                with self.assertRaisesRegex(
                    ValueError, rf"^\.messages\[{index}\]: Unsupported image_url"
                ):
                    self.deepseek_recipe.encode(
                        {"messages": messages, "thinking": False, "tool_choice": "auto"}
                    )

    def test_unlocatable_conversion_error_keeps_its_text(self):
        """An error naming an index the request does not hold stays a ValueError
        with recipe's own text."""
        from unittest import mock

        recipe = renderer.load_deepseek_recipe()
        tools = [{"function": {"name": "f"}}]
        hi = {"role": "user", "content": "hi"}
        declaring = {"role": "system", "content": "s", "tools": tools}
        other = {
            "role": "developer",
            "content": "",
            "tools": [{"function": {"name": "g"}}],
        }
        for messages, rewritten in (
            ([], 0),
            ([hi], 1),
            ([declaring, hi], 4),
            ([declaring, hi, other, hi], 8),
        ):
            for index in (rewritten, 999999):
                text = f".messages[{index}]: boundary; keep .messages[0]"
                error = recipe.ConversionError()
                error.body = json.dumps({"error": {"message": text}})
                request = mock.Mock()
                request.convert.side_effect = error
                with self.subTest(count=len(messages), index=index):
                    with mock.patch.object(
                        recipe, "ChatCompletionRequest", return_value=request
                    ):
                        with self.assertRaises(ValueError) as raised:
                            self.deepseek_recipe.convert(
                                {
                                    "messages": messages,
                                    "thinking": False,
                                    "tool_choice": "auto",
                                }
                            )
                    self.assertEqual(str(raised.exception), text)

    def test_api_validation_surfaces_as_value_error(self):
        with self.assertRaises(ValueError):
            self.deepseek_recipe.encode(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "thinking": True,
                    "reasoning_effort": "x",
                }
            )

    def test_fastokens_preserves_prompt_ids(self):
        try:
            from fastokens import Tokenizer, _ConfiguredTokenizerShim
        except ImportError:
            self.skipTest("fastokens is not installed")
        if not hasattr(Tokenizer, "encode_segments"):
            self.skipTest("fastokens has no segmented encoder")
        candidate = renderer.DeepSeekRecipe(
            "dsv41", _ConfiguredTokenizerShim(self.tokenizer)
        )
        self.assertIsNotNone(candidate._fast_prompt_tokenizer)
        message = {"role": "user", "content": "中文 👩🏽‍💻 <think>x</think>"}
        for thinking in (False, True):
            body = {"messages": [message], "thinking": thinking}
            with self.subTest(thinking=thinking):
                self.assertEqual(
                    candidate.encode(body), self.deepseek_recipe.encode(body)
                )


if __name__ == "__main__":
    unittest.main()
