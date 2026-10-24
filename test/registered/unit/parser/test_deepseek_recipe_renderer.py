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

    def test_api_validation_surfaces_as_value_error(self):
        with self.assertRaises(ValueError):
            self.deepseek_recipe.encode(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "thinking": True,
                    "reasoning_effort": "x",
                }
            )


if __name__ == "__main__":
    unittest.main()
