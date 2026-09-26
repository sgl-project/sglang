"""CPU coverage of chat request, server, and model sampling precedence."""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that can pull in sgl_kernel

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    LegacyStructuralTagResponseFormat,
    MessageProcessingResult,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestChatPreferredSamplingParams(CustomTestCase):
    @staticmethod
    def request(**kwargs):
        return ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            **kwargs,
        )

    def params(self, preferred=None, model=None, **kwargs):
        return self.request(**kwargs).to_sampling_params(
            stop=[],
            model_generation_config=model or {},
            preferred_sampling_params=preferred,
        )

    def test_sampler_precedence_including_null_and_explicit_defaults(self):
        # The last value is the protocol default, intentionally different from
        # both the server and checkpoint. Sending it explicitly must still win.
        cases = (
            ("temperature", 0.7, 0.5, 1.0),
            ("top_p", 0.8, 0.6, 1.0),
            ("top_k", 20, 10, -1),
            ("min_p", 0.1, 0.2, 0.0),
            ("repetition_penalty", 1.1, 1.2, 1.0),
        )
        for key, server, model, default in cases:
            for preferred in (None, {}, {key: server}):
                for config in ({}, {key: model}):
                    expected = server if preferred else config.get(key, default)
                    for request_fields in ({}, {key: None}, {key: default}):
                        with self.subTest(
                            key=key,
                            preferred=preferred,
                            config=config,
                            request=request_fields,
                        ):
                            params = self.params(preferred, config, **request_fields)
                            self.assertEqual(
                                params[key],
                                default
                                if request_fields.get(key) is not None
                                else expected,
                            )

    def test_penalties_use_server_defaults_and_explicit_zero(self):
        for key in ("presence_penalty", "frequency_penalty"):
            with self.subTest(key=key):
                self.assertEqual(self.params({key: 1.5})[key], 1.5)
                self.assertEqual(self.params({key: 1.5}, **{key: 0.0})[key], 0.0)
                self.assertEqual(self.params()[key], 0.0)
                with self.assertRaises(ValidationError):
                    self.request(**{key: None})

    def test_decoder_options_preserve_explicit_false_and_null(self):
        for key, preferred, explicit in (
            ("ignore_eos", True, False),
            ("no_stop_trim", True, False),
            ("skip_special_tokens", False, True),
            ("stop_token_ids", [7], None),
            ("stop_regex", "server-stop", None),
            ("logit_bias", {"7": 1.0}, None),
            ("custom_params", {"tag": "server"}, None),
        ):
            with self.subTest(key=key):
                self.assertEqual(self.params({key: preferred})[key], preferred)
                self.assertEqual(
                    self.params({key: preferred}, **{key: explicit})[key], explicit
                )
        self.assertEqual(self.params({"sampling_seed": 42})["sampling_seed"], 42)
        self.assertEqual(self.params({"sampling_seed": 42}, seed=0)["sampling_seed"], 0)
        self.assertIsNone(
            self.params({"sampling_seed": 42}, seed=None)["sampling_seed"]
        )
        self.assertEqual(self.params({"min_new_tokens": 8})["min_new_tokens"], 8)
        self.assertEqual(
            self.params({"min_new_tokens": 8}, min_tokens=0)["min_new_tokens"], 0
        )

    def test_max_token_aliases_and_unlimited_default(self):
        for preferred in (None, {}, {"max_new_tokens": 256}):
            for fields, expected in (
                ({}, 256 if preferred else None),
                ({"max_tokens": None}, None),
                ({"max_completion_tokens": None}, None),
                ({"max_tokens": 0}, 0),
                ({"max_completion_tokens": 0}, 0),
                ({"max_tokens": 32}, 32),
                ({"max_tokens": 32, "max_completion_tokens": 0}, 0),
                ({"max_tokens": 32, "max_completion_tokens": None}, 32),
                ({"max_tokens": 32, "max_completion_tokens": 64}, 64),
            ):
                with self.subTest(preferred=preferred, fields=fields):
                    self.assertEqual(
                        self.params(preferred, **fields)["max_new_tokens"], expected
                    )

    def test_preferred_stop_adds_to_template_stops_without_mutating_inputs(self):
        for preferred_stop in ("server-stop", ["server-stop", "template-stop"]):
            preferred = {"stop": preferred_stop}
            template_stop = ["template-stop"]
            before = json.dumps(preferred)
            params = self.request().to_sampling_params(
                stop=template_stop,
                model_generation_config={},
                preferred_sampling_params=preferred,
            )
            self.assertEqual(params["stop"], ["server-stop", "template-stop"])
            self.assertEqual(template_stop, ["template-stop"])
            self.assertEqual(json.dumps(preferred), before)
        for explicit in (None, [], "client-stop"):
            with self.subTest(explicit=explicit):
                params = self.request(stop=explicit).to_sampling_params(
                    stop=["template-stop"],
                    model_generation_config={},
                    preferred_sampling_params={"stop": "server-stop"},
                )
                self.assertEqual(params["stop"], ["template-stop"])

    def test_nested_decoder_option_and_parser_override_win(self):
        preferred = {
            "spaces_between_special_tokens": False,
            "skip_special_tokens": True,
        }
        self.assertFalse(self.params(preferred)["spaces_between_special_tokens"])
        for kwargs in ({}, {"enable_thinking": False}):
            self.assertFalse(
                self.params(preferred, chat_template_kwargs=kwargs)[
                    "spaces_between_special_tokens"
                ]
            )
        self.assertTrue(
            self.params(
                preferred, chat_template_kwargs={"spaces_between_special_tokens": True}
            )["spaces_between_special_tokens"]
        )
        request = self.request()
        request.skip_special_tokens = False  # the serving parser requires delimiters
        params = request.to_sampling_params(
            stop=[], model_generation_config={}, preferred_sampling_params=preferred
        )
        self.assertFalse(params["skip_special_tokens"])

    def test_request_and_tool_grammars_are_not_replaced_by_server_defaults(self):
        preferred = {"regex": "server", "ebnf": 'root ::= "server"', "temperature": 0.7}
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        params = self.params(
            preferred,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            },
        )
        self.assertEqual(json.loads(params["json_schema"]), schema)
        self.assertIsNone(params["regex"])
        self.assertIsNone(params["ebnf"])
        params = self.request(tool_choice="required").to_sampling_params(
            stop=["tool-end"],
            model_generation_config={},
            tool_call_constraint=("json_schema", schema),
            preferred_sampling_params=preferred,
        )
        self.assertEqual(json.loads(params["json_schema"]), schema)
        self.assertEqual(params["stop"], ["tool-end"])
        self.assertIsNone(params["regex"])
        self.assertIsNone(params["ebnf"])
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(self.params({"regex": "server"})["regex"], "server")
        self.assertEqual(self.params(preferred, regex="client")["regex"], "client")
        self.assertIsNone(self.params(preferred, regex="client")["ebnf"])

    def test_no_preferred_params_keep_existing_protocol_defaults(self):
        params = self.params()
        self.assertIsNone(params["max_new_tokens"])
        self.assertEqual(params["min_new_tokens"], 0)
        self.assertEqual(params["n"], 1)
        self.assertEqual(params["stop"], [])
        self.assertEqual(params["temperature"], 1.0)
        self.assertEqual(params["top_k"], -1)
        self.assertTrue(params["skip_special_tokens"])
        self.assertTrue(params["spaces_between_special_tokens"])
        self.assertEqual(params, self.params({}))
        self.assertEqual(params, self.params({"not_a_sampling_param": 123}))

    def test_preferred_grammar_merge_preserves_request_and_tool_constraints(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        tag = LegacyStructuralTagResponseFormat(
            type="structural_tag",
            structures=[{"begin": "<answer>", "schema": schema, "end": "</answer>"}],
            triggers=["<answer>"],
        )
        grammars = {
            "regex": "server",
            "ebnf": 'root ::= "server"',
            "json_schema": json.dumps({"type": "string"}),
            "structural_tag": tag.model_dump_json(by_alias=True),
        }
        requests = (
            ("regex", {"regex": "client"}, None),
            ("ebnf", {"ebnf": 'root ::= "client"'}, None),
            (
                "json_schema",
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "answer", "schema": schema},
                    }
                },
                None,
            ),
            (
                "structural_tag",
                {"response_format": tag.model_dump(by_alias=True)},
                None,
            ),
            ("json_schema", {"tool_choice": "required"}, ("json_schema", schema)),
            ("structural_tag", {"tool_choice": "required"}, ("structural_tag", tag)),
        )
        for preferred_key, preferred_value in grammars.items():
            preferred = {preferred_key: preferred_value}
            # Each default is a supported, valid SamplingParams setting alone.
            SamplingParams(**preferred).verify(128)
            for request_key, fields, tool_constraint in requests:
                with self.subTest(
                    preferred=preferred_key, request=request_key, fields=fields
                ):
                    request = self.request(**fields)
                    kwargs = {
                        "stop": [],
                        "model_generation_config": {},
                        "tool_call_constraint": tool_constraint,
                    }
                    expected = request.to_sampling_params(**kwargs)
                    converted = request.to_sampling_params(
                        **kwargs, preferred_sampling_params=preferred
                    )
                    # Exercise the same final merge used by TokenizerManager,
                    # then run the actual mutual-exclusion validation.
                    effective = SamplingParams(**{**preferred, **converted})
                    effective.verify(128)
                    self.assertEqual(
                        getattr(effective, request_key), expected[request_key]
                    )
                    for other_key in grammars.keys() - {request_key}:
                        self.assertIsNone(getattr(effective, other_key))
                    # Only mask an active configured default, never add a
                    # blanket set of None grammar fields to ordinary requests.
                    self.assertLessEqual(
                        set(converted) - set(expected), {preferred_key}
                    )
        for empty in (None, ""):
            self.assertEqual(
                self.params({key: empty for key in grammars}, regex="client"),
                self.params(regex="client"),
            )

    def test_serving_conversion_uses_preferred_params_and_choice_count(self):
        # Exercise the production conversion and batch expansion; only message
        # rendering is mocked, so no tokenizer, model weights, or GPU are needed.
        chat = object.__new__(OpenAIServingChat)
        chat.tokenizer_manager = SimpleNamespace(
            model_config=SimpleNamespace(is_multimodal=False),
            preferred_sampling_params={"temperature": 0.7, "n": 2},
        )
        chat.template_manager = SimpleNamespace(reasoning_config=None)
        chat.reasoning_parser = None
        chat.is_gpt_oss = False
        chat.chat_encoding_spec = None
        chat.default_sampling_params = {"temperature": 0.5}
        chat.allowed_custom_labels = None
        chat._should_return_input_ids = Mock(return_value=False)
        chat._process_messages = Mock(
            return_value=MessageProcessingResult(
                prompt="Hello",
                prompt_ids=[1, 2],
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=[],
                stop=["template-stop"],
            )
        )
        for fields, expected_n in (({}, 2), ({"n": 1}, 1), ({"n": 3}, 3)):
            with self.subTest(fields=fields):
                internal, request = chat._convert_to_internal_request(
                    self.request(input_ids=[1, 2], **fields)
                )
                self.assertEqual(internal.sampling_params["temperature"], 0.7)
                self.assertEqual(internal.sampling_params["stop"], ["template-stop"])
                self.assertEqual(request.n, expected_n)
                internal.normalize_batch_and_arguments()
                self.assertEqual(internal.parallel_sample_num, expected_n)
                if expected_n > 1:
                    self.assertEqual(len(internal.sampling_params), expected_n)
                    for params in internal.sampling_params:
                        self.assertEqual(params["temperature"], 0.7)
                        self.assertEqual(params["n"], expected_n)
                results = [
                    {
                        "text": f"choice-{index}",
                        "meta_info": {
                            "id": "chatcmpl-preferred",
                            "prompt_tokens": 10,
                            "completion_tokens": index + 1,
                            "cached_tokens": 4,
                            "image_tokens": 3,
                            "finish_reason": {"type": "stop"},
                            "weight_version": "default",
                        },
                    }
                    for index in range(expected_n)
                ]
                with get_context().override_server_args(enable_cache_report=True):
                    response = chat._build_chat_response(request, results, created=123)
                self.assertEqual(
                    [choice.index for choice in response.choices],
                    list(range(expected_n)),
                )
                self.assertEqual(response.usage.prompt_tokens, 10)
                self.assertEqual(
                    response.usage.completion_tokens, sum(range(1, expected_n + 1))
                )
                self.assertEqual(response.usage.prompt_tokens_details.cached_tokens, 4)
                self.assertEqual(response.usage.prompt_tokens_details.image_tokens, 3)


if __name__ == "__main__":
    unittest.main()
