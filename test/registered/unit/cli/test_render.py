import json
import unittest
from types import SimpleNamespace

from sglang.cli.render import (
    build_renderer_config,
    extract_engine_url,
    write_renderer_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestStandaloneRendererCli(unittest.TestCase):
    def test_engine_url_is_required_and_removed_from_server_arguments(self):
        engine_url, remaining = extract_engine_url(
            ["model", "--engine-url", "http://engine:30000", "--port", "8000"]
        )
        self.assertEqual(engine_url, "http://engine:30000")
        self.assertEqual(remaining, ["model", "--port", "8000"])

        with self.assertRaisesRegex(ValueError, "requires --engine-url"):
            extract_engine_url(["model"])

    def test_engine_url_equals_form_is_supported(self):
        engine_url, remaining = extract_engine_url(
            ["model", "--engine-url=http://engine:30000"]
        )
        self.assertEqual(engine_url, "http://engine:30000")
        self.assertEqual(remaining, ["model"])

    def test_config_file_contains_the_exact_launch_payload(self):
        payload = {"engine_url": "http://engine:30000", "renderer": {"model": "m"}}
        path = write_renderer_config(payload)
        try:
            self.assertEqual(json.loads(path.read_text()), payload)
        finally:
            path.unlink(missing_ok=True)

    def test_renderer_config_forwards_default_chat_template_kwargs(self):
        server_args = SimpleNamespace(
            served_model_name="model",
            tokenizer_path="tokenizer",
            revision=None,
            model_path="model-path",
            chat_template=None,
            tool_call_parser=None,
            reasoning_parser=None,
            default_chat_template_kwargs={"enable_thinking": False},
            stream_response_default_include_usage=False,
            allow_auto_truncate=False,
            enable_return_hidden_states=False,
        )
        model_config = SimpleNamespace(vocab_size=128, context_len=4096)

        config = build_renderer_config(
            server_args,
            model_config,
            {"temperature": 0.7, "top_p": 0.9},
            32,
        )

        self.assertEqual(
            config["default_chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertNotIn("skip_tokenizer_init", config)
        self.assertNotIn("skip_tokenizer_init", config["limits"])
        self.assertNotIn("vocab_size", config)
        self.assertEqual(config["limits"]["vocab_size"], 128)


if __name__ == "__main__":
    unittest.main()
