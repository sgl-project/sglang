"""GLM-5.3 / GLM-5.3-Flash Python chat templates.

The expected prompts below were rendered with the checkpoints' chat_template.jinja.
Set SGLANG_TEST_GLM53_CHECKPOINT / SGLANG_TEST_GLM53_FLASH_CHECKPOINT to model
directories to also compare against their Jinja templates directly.
"""

import inspect
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.utils.chat_template_utils import _compile_jinja_template

from sglang.srt.entrypoints.openai.parallel_prompt_encode import encode_prompt_segments
from sglang.srt.parser.glm53_chat_template import render_glm53, render_glm53_flash
from sglang.srt.parser.jinja_template_utils import detect_jinja_template_content_format
from sglang.srt.parser.python_chat_template import (
    GLM53,
    GLM53_FLASH,
    get_python_chat_template,
    install_python_chat_template,
    render_chat_segments,
    resolve_python_chat_template,
)
from sglang.srt.parser.template_detection import (
    detect_inline_system_support,
    resolve_auto_parsers,
)
from sglang.srt.parser.template_manager import TemplateManager
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3.0, suite="base-a-test-cpu")

PREFIX = "[gMASK]<sop><|system|>Reasoning Effort: Max"
TOOLS_HEADER = (
    "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user "
    "query.\n\nYou are provided with function signatures within <tools></tools> XML "
    "tags:\n<tools>\n"
)
TOOLS_FOOTER = (
    "</tools>\n\nFor each function call, output the function name and arguments "
    "within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}"
    "</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>"
    "<arg_value>{arg-value-2}</arg_value>...</tool_call>"
)
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather.",
        "strict": True,
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
WEATHER_JSON = (
    '{"name": "get_weather", "description": "Get weather.", "parameters": '
    '{"type": "object", "properties": {"city": {"type": "string"}}}}'
)
DEFERRED = {"type": "function", "function": {"name": "later", "defer_loading": True}}
CALLS = [
    {
        "id": "a",
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}},
    },
    {
        "id": "b",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": {"city": "Rome", "opts": {"metric": True}},
        },
    },
]
CALLS_TEXT = (
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
    "<arg_key>days</arg_key><arg_value>2</arg_value></tool_call>"
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value>"
    '<arg_key>opts</arg_key><arg_value>{"metric": true}</arg_value></tool_call>'
)

# name -> (render kwargs, expected GLM-5.3 prompt, expected GLM-5.3-Flash prompt)
CASES = {
    "basic": (
        dict(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            add_generation_prompt=True,
        ),
        PREFIX + "<|system|>You are helpful.<|user|>Hi<|assistant|><think>",
        None,
    ),
    "reasoning_history": (
        dict(
            messages=[
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "<think>r2</think> a2 "},
            ],
            reasoning_effort="low",
        ),
        "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>q1<|assistant|>"
        "<think>r1</think>a1<|user|>q2<|assistant|><think>r2</think>a2",
        None,
    ),
    "reasoning_effort_fallback": (
        dict(
            messages=[{"role": "user", "content": "q"}],
            reasoning_effort="medium",
        ),
        PREFIX + "<|user|>q",
        None,
    ),
    "clear_thinking": (
        dict(
            messages=[
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
            ],
            clear_thinking=True,
            reasoning_effort="high",
        ),
        "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>q1<|assistant|>"
        "<think></think>a1<|user|>q2<|assistant|><think>r2</think>a2",
        None,
    ),
    "tool_results_follow_tool_calls": (
        dict(
            messages=[
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "", "tool_calls": CALLS},
                {"role": "tool", "tool_call_id": "b", "content": "Rome: sun"},
                {"role": "tool", "tool_call_id": "a", "content": "Paris: rain"},
            ],
            tools=[WEATHER, DEFERRED],
            add_generation_prompt=True,
        ),
        PREFIX
        + TOOLS_HEADER
        + WEATHER_JSON
        + "\n"
        + TOOLS_FOOTER
        + "<|user|>Weather?<|assistant|><think></think>"
        + CALLS_TEXT
        + "<|observation|><tool_response>Paris: rain</tool_response>"
        "<tool_response>Rome: sun</tool_response><|assistant|><think>",
        None,
    ),
    "unmatched_tool_result_keeps_order": (
        dict(
            messages=[
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "", "tool_calls": CALLS},
                {"role": "tool", "tool_call_id": "b", "content": "Rome: sun"},
                {"role": "tool", "tool_call_id": "zzz", "content": "?"},
            ],
        ),
        PREFIX
        + "<|user|>Weather?<|assistant|><think></think>"
        + CALLS_TEXT
        + "<|observation|><tool_response>Rome: sun</tool_response>"
        "<tool_response>?</tool_response>",
        None,
    ),
    "media": (
        dict(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look:"},
                        {"type": "image"},
                        {"type": "video_url"},
                        {"type": "input_audio"},
                    ],
                }
            ],
        ),
        PREFIX
        + "<|user|>Look:"
        + "".join(
            f"<reminder>You are unable to process this {media} because you don't "
            "have multi-modal input ability. Try different methods.</reminder>"
            for media in ("image", "video", "audio")
        ),
        PREFIX + "<|user|>Look:<|begin_of_image|><|image|><|end_of_image|>"
        "<|begin_of_video|><|video|><|end_of_video|><|begin_of_audio|><|end_of_audio|>",
    ),
    "tool_reference": (
        dict(
            messages=[
                {"role": "user", "content": "q"},
                {
                    "role": "tool",
                    "content": [
                        {
                            "output": [
                                {"type": "tool_reference", "name": "get_weather"}
                            ],
                            "tool_call_id": "a",
                        }
                    ],
                },
            ],
            tools=[WEATHER],
        ),
        PREFIX
        + TOOLS_HEADER
        + WEATHER_JSON
        + "\n"
        + TOOLS_FOOTER
        + "<|user|>q<|observation|><tool_response><tools>\n"
        + WEATHER_JSON
        + "\n</tools></tool_response>",
        None,
    ),
    # The template only assigns reasoning_content in its own branches, so a
    # template variable of that name fills in for turns without reasoning.
    "reasoning_content_variable": (
        dict(
            messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "A1"},
            ],
            reasoning_content="X",
        ),
        PREFIX + "<|user|>q<|assistant|><think>X</think>A1",
        None,
    ),
}


def _render(render, kwargs):
    kwargs = dict(kwargs)
    return render(kwargs.pop("messages"), **kwargs)


def _glm53_like_template(multimodal):
    """A reduced GLM-5.3 template with every construct template detection keys on."""
    concat = "~" if multimodal else "+"
    image = (
        "{%- elif item.type == 'image' -%}<|begin_of_image|><|image|><|end_of_image|>"
        if multimodal
        else ""
    )
    return (
        "[gMASK]<sop>\n"
        "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is "
        "defined and reasoning_effort in ['low', 'high'] else 'max' -%}"
        "<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>"
        "{%- if m.content is string -%}{{ m.content }}{%- else -%}"
        "{%- for item in m.content -%}{%- if item.type == 'text' -%}{{ item.text }}"
        f"{image}{{%- endif -%}}{{%- endfor -%}}{{%- endif -%}}"
        "{%- elif m.role == 'tool' -%}<|observation|>{{ m.tool_call_id }}"
        "{%- elif m.role == 'assistant' -%}<|assistant|>"
        "{% for tc in m.tool_calls %}"
        f"{{{{- '<tool_call>' {concat} tc.name -}}}}"
        "{% for k, v in tc.arguments.items() %}<arg_key>{{ k }}</arg_key>"
        "<arg_value>{{ v }}</arg_value>{% endfor %}</tool_call>{% endfor %}"
        "{%- endif -%}{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|assistant|>{{- '<think>' -}}{%- endif -%}"
    )


GLM52_LIKE_TEMPLATE = (
    "[gMASK]<sop>{%- if enable_thinking is defined and not enable_thinking -%}"
    "/nothink{%- endif -%}<tool_call><arg_key><arg_value>"
)


def _hf_config(architecture):
    return SimpleNamespace(architectures=[architecture])


def _tiny_tokenizer(chat_template=None):
    backend = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="<|endoftext|>"
    )
    tokenizer.add_tokens(["Hi", "q", "<|user|>", "<|assistant|>", "<think>"])
    tokenizer.chat_template = chat_template
    return tokenizer


class TestGlm53ChatTemplate(CustomTestCase):
    def test_renders_like_the_jinja_templates(self):
        for name, (kwargs, text_prompt, flash_prompt) in CASES.items():
            with self.subTest(case=name, model="GLM-5.3"):
                self.assertEqual(_render(render_glm53, kwargs), text_prompt)
            with self.subTest(case=name, model="GLM-5.3-Flash"):
                self.assertEqual(
                    _render(render_glm53_flash, kwargs), flash_prompt or text_prompt
                )

    def test_tool_keys_are_interpolated_raw(self):
        tool = {"name": "f", 'we"ird': 1, "strict": False}
        prompt = render_glm53([{"role": "user", "content": "q"}], tools=[tool])
        self.assertIn('{"name": "f", "we"ird": 1}\n', prompt)

    def test_tool_call_name_concatenation(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": None, "arguments": {}}}],
            }
        ]
        # GLM-5.3 concatenates with `+`, GLM-5.3-Flash with `~`.
        with self.assertRaisesRegex(TypeError, "can only concatenate str"):
            render_glm53(messages)
        self.assertTrue(
            render_glm53_flash(messages).endswith("<tool_call>None</tool_call>")
        )

    def test_malformed_inputs_raise_like_the_template(self):
        user = {"role": "user", "content": "q"}
        reference = {
            "role": "tool",
            "content": [{"type": "tool_reference", "name": "f"}],
        }
        malformed = {
            "string arguments": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}],
                }
            ],
            "missing name": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"arguments": {}}}],
                }
            ],
            "tool_reference without tools": [user, reference],
            "mapping tool content": [user, {"role": "tool", "content": {"a": 1}}],
        }
        for name, messages in malformed.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    render_glm53(messages)
        with self.assertRaises(ValueError):
            render_glm53([user], tools=["not a tool"])


class TestPythonChatTemplateSelection(CustomTestCase):
    def test_features_match_jinja_template_detection(self):
        for multimodal, template in ((False, GLM53), (True, GLM53_FLASH)):
            jinja_template = _glm53_like_template(multimodal)
            detected = TemplateManager()
            detected._run_template_detection(jinja_template, None)
            with self.subTest(template=template.name):
                self.assertEqual(
                    detect_jinja_template_content_format(jinja_template),
                    template.content_format,
                )
                self.assertEqual(detected.force_reasoning, template.force_reasoning)
                self.assertEqual(detected.reasoning_config, template.reasoning_config)
                self.assertEqual(
                    detected.jinja_template_may_reorder_tool_results,
                    template.may_reorder_tool_results,
                )
                self.assertEqual(
                    detected.suggested_reasoning_parser, template.reasoning_parser
                )
                self.assertEqual(
                    detected.suggested_tool_call_parser, template.tool_call_parser
                )
                self.assertEqual(
                    detect_inline_system_support(jinja_template),
                    template.supports_inline_system,
                )

    def test_resolves_by_architecture_and_template(self):
        text, flash = _glm53_like_template(False), _glm53_like_template(True)
        cases = [
            ("GlmMoeDsaForCausalLM", text, GLM53),
            ("Glm5NextForConditionalGeneration", flash, GLM53_FLASH),
            # GLM-5.2 shares the architecture but not the template.
            ("GlmMoeDsaForCausalLM", GLM52_LIKE_TEMPLATE, None),
            # The multimodal block must match the architecture.
            ("GlmMoeDsaForCausalLM", flash, None),
            ("Glm5NextForConditionalGeneration", text, None),
            ("GlmMoeDsaForCausalLM", None, None),
            ("LlamaForCausalLM", text, None),
        ]
        for architecture, jinja_template, expected in cases:
            with self.subTest(
                architecture=architecture, expected=expected and expected.name
            ):
                self.assertIs(
                    resolve_python_chat_template(
                        _hf_config(architecture), jinja_template
                    ),
                    expected,
                )

    def test_builtin_names(self):
        self.assertIs(get_python_chat_template("glm-5.3"), GLM53)
        self.assertIs(get_python_chat_template("glm-5.3-flash"), GLM53_FLASH)
        self.assertIsNone(get_python_chat_template("chatml"))


class TestTemplateManagerPythonChatTemplate(CustomTestCase):
    def _tokenizer_manager(self, architecture, chat_template):
        return SimpleNamespace(
            tokenizer=_tiny_tokenizer(chat_template),
            processor=None,
            model_config=SimpleNamespace(hf_config=_hf_config(architecture)),
        )

    def test_replaces_the_jinja_template(self):
        tokenizer_manager = self._tokenizer_manager(
            "GlmMoeDsaForCausalLM", _glm53_like_template(False)
        )
        manager = TemplateManager()
        manager.load_chat_template(tokenizer_manager, None, "zai-org/GLM-5.3")

        self.assertIs(manager.python_chat_template, GLM53)
        self.assertIsNone(manager.chat_template_name)
        self.assertEqual(manager.jinja_template_content_format, "openai")
        self.assertTrue(manager.force_reasoning)
        self.assertEqual(manager.reasoning_config, GLM53.reasoning_config)
        self.assertTrue(manager.jinja_template_may_reorder_tool_results)
        self.assertEqual(manager.suggested_reasoning_parser, "glm45")
        self.assertEqual(manager.suggested_tool_call_parser, "glm47")

        messages = CASES["media"][0]["messages"]
        self.assertEqual(
            tokenizer_manager.tokenizer.apply_chat_template(messages, tokenize=False),
            render_glm53(messages),
        )

    def test_other_templates_keep_the_jinja_template(self):
        # GLM-5.2 shares GLM-5.3's architecture but not its template.
        tokenizer_manager = self._tokenizer_manager(
            "GlmMoeDsaForCausalLM", GLM52_LIKE_TEMPLATE
        )
        manager = TemplateManager()
        manager.load_chat_template(tokenizer_manager, None, "zai-org/GLM-5.2")

        self.assertIsNone(manager.python_chat_template)
        messages = [{"role": "user", "content": "q"}]
        self.assertEqual(
            tokenizer_manager.tokenizer.apply_chat_template(messages, tokenize=False),
            _compile_jinja_template(GLM52_LIKE_TEMPLATE).render(
                messages=messages, add_generation_prompt=False
            ),
        )

    def test_explicit_builtin_name(self):
        tokenizer_manager = self._tokenizer_manager("LlamaForCausalLM", None)
        manager = TemplateManager()
        manager.load_chat_template(tokenizer_manager, "glm-5.3-flash", "some/model")

        self.assertIs(manager.python_chat_template, GLM53_FLASH)
        messages = CASES["media"][0]["messages"]
        self.assertEqual(
            tokenizer_manager.tokenizer.apply_chat_template(messages, tokenize=False),
            CASES["media"][2],
        )


def _patch_hf_transformers_utils(get_tokenizer, get_config):
    module = ModuleType("sglang.srt.utils.hf_transformers_utils")
    module.get_tokenizer = get_tokenizer
    module.get_config = get_config
    return patch.dict(sys.modules, {module.__name__: module})


class TestAutoParsersFromPythonChatTemplate(CustomTestCase):
    def _resolve(self, architecture, jinja_template, chat_template=None):
        server_args = ServerArgs(
            model_path="dummy",
            reasoning_parser="auto",
            tool_call_parser="auto",
            trust_remote_code=False,
            chat_template=chat_template,
        )
        tokenizer = SimpleNamespace(chat_template=jinja_template, get_vocab=dict)
        self.get_tokenizer = Mock(return_value=tokenizer)
        self.get_config = Mock(return_value=_hf_config(architecture))
        with (
            _patch_hf_transformers_utils(self.get_tokenizer, self.get_config),
            self.assertLogs("sglang.srt.parser.template_detection", "INFO") as logs,
        ):
            resolve_auto_parsers(server_args)

        from sglang.srt.arg_groups.overrides import resolution_result

        parsers = tuple(
            resolution_result(server_args, field)
            for field in ("reasoning_parser", "tool_call_parser")
        )
        return parsers, "\n".join(logs.output)

    def test_auto_parsers_come_from_the_python_template(self):
        for architecture, multimodal, name in (
            ("GlmMoeDsaForCausalLM", False, "glm-5.3"),
            ("Glm5NextForConditionalGeneration", True, "glm-5.3-flash"),
        ):
            with self.subTest(template=name):
                parsers, logs = self._resolve(
                    architecture, _glm53_like_template(multimodal)
                )
                self.assertEqual(parsers, ("glm45", "glm47"))
                self.assertIn(f"from Python chat template '{name}'", logs)

    def test_explicit_builtin_name_loads_nothing(self):
        parsers, logs = self._resolve("LlamaForCausalLM", None, "glm-5.3-flash")
        self.assertEqual(parsers, ("glm45", "glm47"))
        self.assertIn("from Python chat template 'glm-5.3-flash'", logs)
        self.get_tokenizer.assert_not_called()
        self.get_config.assert_not_called()

    def test_other_templates_use_jinja_detection(self):
        _, logs = self._resolve("GlmMoeDsaForCausalLM", GLM52_LIKE_TEMPLATE)
        self.assertNotIn("Python chat template", logs)


class TestInstallPythonChatTemplate(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = _tiny_tokenizer("{{ messages[0]['content'] }}!")
        install_python_chat_template(self.tokenizer, GLM53)
        self.messages = [{"role": "user", "content": "Hi"}]
        self.rendered = render_glm53(self.messages, add_generation_prompt=True)

    def test_tokenizes_like_transformers(self):
        ids = self.tokenizer(self.rendered, add_special_tokens=False)["input_ids"]
        apply = self.tokenizer.apply_chat_template
        self.assertEqual(
            apply(self.messages, add_generation_prompt=True, tokenize=False),
            self.rendered,
        )
        self.assertEqual(
            apply(self.messages, add_generation_prompt=True, return_dict=False), ids
        )
        self.assertEqual(
            apply(self.messages, add_generation_prompt=True)["input_ids"], ids
        )
        self.assertEqual(
            apply(
                [self.messages, self.messages],
                add_generation_prompt=True,
                tokenize=False,
            ),
            [self.rendered, self.rendered],
        )
        with self.assertRaises(ValueError):
            apply([], tokenize=False)

    def test_converts_callable_tools(self):
        def get_weather(city: str):
            """Get the weather.

            Args:
                city: The city.
            """

        prompt = self.tokenizer.apply_chat_template(
            self.messages, tools=[get_weather], tokenize=False
        )
        self.assertIn('"name": "get_weather"', prompt)

    def test_jinja_only_features_use_the_original_implementation(self):
        apply = self.tokenizer.apply_chat_template
        self.assertEqual(
            apply(
                self.messages,
                chat_template="{{ messages[0]['content'] }}?",
                tokenize=False,
            ),
            "Hi?",
        )
        self.assertEqual(
            apply(self.messages, continue_final_message=True, tokenize=False), "Hi"
        )

    def test_signature_matches_transformers(self):
        original = inspect.signature(PreTrainedTokenizerFast.apply_chat_template)
        installed = inspect.signature(self.tokenizer.apply_chat_template)
        expected = [
            (name, parameter.default)
            for name, parameter in original.parameters.items()
            if name != "self"
        ]
        self.assertEqual(
            [
                (name, parameter.default)
                for name, parameter in installed.parameters.items()
            ],
            expected,
        )

    def test_segments_are_what_apply_chat_template_renders(self):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": CALLS[:1]},
            {"role": "tool", "tool_call_id": "a", "content": "Sunny"},
            {"role": "user", "content": "Thanks"},
        ]
        segments = render_chat_segments(
            self.tokenizer,
            GLM53,
            messages,
            tools=[WEATHER],
            add_generation_prompt=True,
            template_kwargs={"reasoning_effort": "low"},
        )
        self.assertEqual(
            "".join(segments),
            self.tokenizer.apply_chat_template(
                messages,
                tools=[WEATHER],
                add_generation_prompt=True,
                tokenize=False,
                reasoning_effort="low",
            ),
        )
        turn_tokens = ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>")
        for token in ("[gMASK]", "<sop>", "<think>") + turn_tokens:
            self.assertIn(token, segments)
        self.assertFalse(
            [
                s
                for s in segments
                if s not in turn_tokens and any(t in s for t in turn_tokens)
            ]
        )

        for key in ("chat_template", "continue_final_message", "tokenize", "documents"):
            self.assertIsNone(
                render_chat_segments(
                    self.tokenizer, GLM53, self.messages, template_kwargs={key: None}
                ),
                key,
            )
        with self.assertRaises(ValueError):
            render_chat_segments(self.tokenizer, GLM53, [])


@unittest.skipUnless(
    os.environ.get("SGLANG_TEST_GLM53_CHECKPOINT")
    or os.environ.get("SGLANG_TEST_GLM53_FLASH_CHECKPOINT"),
    "set SGLANG_TEST_GLM53_CHECKPOINT / SGLANG_TEST_GLM53_FLASH_CHECKPOINT",
)
class TestAgainstCheckpointTemplates(CustomTestCase):
    def test_matches_checkpoint_jinja(self):
        for env, render, template in (
            ("SGLANG_TEST_GLM53_CHECKPOINT", render_glm53, GLM53),
            ("SGLANG_TEST_GLM53_FLASH_CHECKPOINT", render_glm53_flash, GLM53_FLASH),
        ):
            model_dir = os.environ.get(env)
            if not model_dir:
                continue
            with open(
                os.path.join(model_dir, "chat_template.jinja"), encoding="utf-8"
            ) as f:
                compiled = _compile_jinja_template(f.read())
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            for name, (kwargs, _, _) in CASES.items():
                for add_generation_prompt in (False, True):
                    with self.subTest(
                        model=model_dir, case=name, gen=add_generation_prompt
                    ):
                        kwargs = {
                            **kwargs,
                            "add_generation_prompt": add_generation_prompt,
                        }
                        prompt = compiled.render(**kwargs)
                        self.assertEqual(_render(render, kwargs), prompt)
                        variables = dict(kwargs)
                        segments = render_chat_segments(
                            tokenizer,
                            template,
                            variables.pop("messages"),
                            tools=variables.pop("tools", None),
                            add_generation_prompt=variables.pop(
                                "add_generation_prompt"
                            ),
                            template_kwargs=variables,
                        )
                        self.assertEqual(
                            encode_prompt_segments(tokenizer, segments),
                            tokenizer.encode(prompt),
                        )


if __name__ == "__main__":
    unittest.main()
