"""The deepseek-recipe chat encoding backend: the request translation, the
sampling overlay, the adapter registration and, with the deepseek-recipe package
installed, the prompt the parity fixture records."""

import hashlib
import json
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.entrypoints.openai import serving_deepseek_recipe as recipe
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FIXTURE = (
    Path(__file__).resolve().parents[5]
    / "rust/sglang-processor/tests/fixtures/parity/deepseek-v4.1-flash-1017-recipe.json"
)


def _user(content="x"):
    return {"role": "user", "content": content}


class TestPreNormalize(CustomTestCase):
    def test_task_is_rejected(self):
        with self.assertRaises(ValueError):
            recipe.pre_normalize({"messages": [_user()], "task": "title"})

    def test_developer_becomes_system(self):
        body = recipe.pre_normalize(
            {"messages": [{"role": "developer", "content": "Be terse."}, _user()]}
        )
        self.assertEqual(body["messages"][0]["role"], "system")

    def test_thinking_toggle_sources(self):
        kwargs = recipe.pre_normalize(
            {"messages": [_user()], "chat_template_kwargs": {"thinking": False}}
        )
        self.assertEqual(kwargs["thinking"], {"type": "disabled"})
        self.assertNotIn("chat_template_kwargs", kwargs)
        # protocol.normalize_reasoning_inputs: reasoning.effort sets the effort and
        # the kwargs flag; an explicit kwargs flag keeps its value.
        reasoning = recipe.pre_normalize(
            {"messages": [_user()], "reasoning": {"effort": "low", "enabled": True}}
        )
        self.assertEqual(reasoning["thinking"], {"type": "enabled"})
        self.assertEqual(reasoning["reasoning_effort"], "low")
        self.assertNotIn("reasoning", reasoning)
        none = recipe.pre_normalize({"messages": [_user()], "reasoning_effort": "none"})
        self.assertEqual(none["thinking"], {"type": "disabled"})
        overridden = recipe.pre_normalize(
            {
                "messages": [_user()],
                "reasoning_effort": "high",
                "chat_template_kwargs": {"thinking": False},
            }
        )
        self.assertEqual(overridden["thinking"], {"type": "disabled"})
        explicit = recipe.pre_normalize(
            {
                "messages": [_user()],
                "thinking": {"type": "enabled"},
                "chat_template_kwargs": {"thinking": False},
            }
        )
        self.assertEqual(explicit["thinking"], {"type": "enabled"})
        default = recipe.pre_normalize({"messages": [_user()]}, {"thinking": False})
        self.assertEqual(default["thinking"], {"type": "disabled"})
        self.assertNotIn("thinking", recipe.pre_normalize({"messages": [_user()]}))

    def test_effort_sources(self):
        # serving_chat: the kwargs effort replaces the request's verbatim (recipe
        # rejects a number, as the API does); the server default fills in last.
        body = recipe.pre_normalize(
            {
                "messages": [_user()],
                "reasoning_effort": "high",
                "chat_template_kwargs": {"reasoning_effort": 30},
            }
        )
        self.assertEqual(body["reasoning_effort"], 30)
        default = recipe.pre_normalize(
            {"messages": [_user()]}, {"reasoning_effort": "max"}
        )
        self.assertEqual(default["reasoning_effort"], "max")

    def test_tool_choice_renders_as_auto_with_the_chosen_tool(self):
        tools = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]
        named = recipe.pre_normalize(
            {
                "messages": [_user()],
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": "b"}},
            }
        )
        self.assertEqual(named["tool_choice"], "auto")
        self.assertEqual([t["function"]["name"] for t in named["tools"]], ["b"])
        required = recipe.pre_normalize(
            {"messages": [_user()], "tools": tools, "tool_choice": "required"}
        )
        self.assertEqual(required["tool_choice"], "auto")
        self.assertEqual(len(required["tools"]), 2)
        for body in (
            {"messages": [_user()], "tools": tools, "tool_choice": "none"},
            {"messages": [_user()], "tools": [], "tool_choice": "auto"},
        ):
            off = recipe.pre_normalize(body)
            self.assertNotIn("tools", off)
            self.assertNotIn("tool_choice", off)

    def test_history_translation_and_host_fields(self):
        body = recipe.pre_normalize(
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
        # recipe's own fields and validation stand: `n` passes through, and a tool
        # message without tool_call_id is recipe's to reject.
        self.assertEqual(body["n"], 2)
        unpaired = recipe.pre_normalize(
            {"messages": [_user(), {"role": "tool", "content": "sunny"}]}
        )
        self.assertNotIn("tool_call_id", unpaired["messages"][1])

    def test_input_is_not_mutated(self):
        original = {"messages": [{"role": "developer", "content": "x"}], "n": 3}
        recipe.pre_normalize(original)
        self.assertEqual(original["messages"][0]["role"], "developer")
        self.assertEqual(original["n"], 3)


class TestApplySampling(CustomTestCase):
    def _render(self, **overrides):
        values = dict(
            prompt="",
            prompt_ids=[0, 1, 2],
            temperature=1.0,
            top_p=0.95,
            request=SimpleNamespace(resolve_max_tokens=lambda prompt, limit: 65536),
        )
        values.update(overrides)
        return recipe.RecipeRender(**values)

    def test_recipe_sampling_and_default_max_tokens(self):
        params = {"temperature": 0.0, "top_p": 0.1, "max_new_tokens": None}
        recipe.overlay_sampling(
            params, self._render(), prompt_tokens=100, context_length=1_000_000
        )
        self.assertEqual(params["temperature"], 1.0)
        self.assertEqual(params["top_p"], 0.95)
        self.assertEqual(params["max_new_tokens"], 65536)
        params = {"max_new_tokens": 5}
        recipe.overlay_sampling(
            params, self._render(), prompt_tokens=1, context_length=10
        )
        self.assertEqual(params["max_new_tokens"], 5)
        self.assertNotIn("json_schema", params)


class TestAdapterRegistry(CustomTestCase):
    """The recipe backend is the `recipe` builder of the dsv4 / dsv41 specs."""

    def _context(self, spec, backend, tokenizer=None, default_kwargs=None):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        return registry.ChatEncodingAdapterContext(
            chat_encoding_spec=spec,
            tokenizer=tokenizer,
            reasoning_parser=None,
            tool_call_parser=None,
            chat_hooks=registry.ChatOutputHooks(
                tool_call_parsing_active=lambda request: False,
                history_tool_calls_count=lambda request: 0,
                tool_call_id=lambda item, index: "",
            ),
            hf_config=SimpleNamespace(
                model_type="deepseek_v41", architectures=["DeepseekV41ForCausalLM"]
            ),
            model_path="",
            revision=None,
            default_chat_template_kwargs=default_kwargs,
            backend=backend,
        )

    def test_builtin_backend_keeps_the_inline_deepseek_encoders(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        self.assertTrue(registry.has_chat_encoding_adapter("dsv41"))
        self.assertTrue(registry.has_chat_encoding_adapter("dsv4"))
        self.assertIsNone(
            registry.build_chat_encoding_adapter(self._context("dsv41", "builtin"))
        )

    def test_recipe_backend_is_deepseek_only(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        with self.assertRaises(ValueError):
            registry.build_chat_encoding_adapter(self._context("kimi_k3", "recipe"))

    def test_default_effort_must_be_a_recipe_effort(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        with self.assertRaisesRegex(ValueError, "reasoning_effort=60"):
            registry.build_chat_encoding_adapter(
                self._context(
                    "dsv41", "recipe", default_kwargs={"reasoning_effort": 60}
                )
            )

    @unittest.skipUnless(recipe.is_available(), "deepseek-recipe is not installed")
    def test_recipe_adapter_encodes_like_the_fixture_and_overlays_sampling(self):
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        try:
            tokenizer_file = hf_hub_download(
                fixture["model"],
                "tokenizer.json",
                revision=fixture["revision"],
                local_files_only=True,
            )
        except Exception:  # noqa: BLE001 - not cached: nothing to encode with
            self.skipTest("the fixture tokenizer is not cached")
        served_tokenizer = SimpleNamespace(
            backend_tokenizer=Tokenizer.from_file(tokenizer_file)
        )
        adapter = registry.build_chat_encoding_adapter(
            self._context("dsv41", "recipe", tokenizer=served_tokenizer)
        )
        self.assertFalse(adapter.strict_tool_call_arguments)
        self.assertIsNone(adapter.output)
        case = next(c for c in fixture["cases"] if c["name"] == "a1_empty_chat")
        # The client's model name is SGLang's business; recipe sees its own.
        request = ChatCompletionRequest(**{**case["request"], "model": "anything"})
        ids = adapter.encode_messages(messages=[], request=request, tools=None)
        digest = hashlib.sha256()
        for token_id in ids:
            digest.update(struct.pack("<I", token_id))
        self.assertEqual(len(ids), case["token_count"])
        self.assertEqual(digest.hexdigest(), case["token_sha256"])
        # The overlay is a function of the request alone: no encode_messages needed.
        params = {"max_new_tokens": None}
        adapter.apply_sampling(
            request=ChatCompletionRequest(
                messages=[{"role": "user", "content": ""}],
                thinking={"type": "disabled"},
            ),
            sampling_params=params,
            prompt_tokens=len(ids),
            context_length=100_000,
        )
        self.assertEqual(params["max_new_tokens"], 8192)
        # The handler's tool dumps carry None for unset fields; recipe must see the
        # tool as the client sent it.
        tool = {
            "type": "function",
            "function": {
                "name": "f",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        dumped = {
            "type": "function",
            "function": {**tool["function"], "strict": None},
            "defer_loading": None,
        }
        with_tools = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "disabled"},
        )
        self.assertEqual(
            adapter.encode_messages(messages=[], request=with_tools, tools=[dumped]),
            adapter.encode_messages(messages=[], request=with_tools, tools=[tool]),
        )


if __name__ == "__main__":
    unittest.main()
