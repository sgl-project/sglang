import json
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers.rust_renderer import (
    RustRendererHost,
    build_renderer_args,
    connect_host,
    validate_embedded_renderer,
)
from sglang.srt.utils.network import NetworkAddress
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def server_args(**overrides):
    values = {
        "served_model_name": "served-model",
        "tokenizer_path": "tokenizer",
        "tokenizer_worker_num": 3,
        "revision": "revision",
        "model_path": "model-path",
        "chat_template": "template.jinja",
        "tool_call_parser": "parser",
        "reasoning_parser": "reasoner",
        "default_chat_template_kwargs": {"thinking": False},
        "sampling_defaults": "model",
        "stream_response_default_include_usage": True,
        "allow_auto_truncate": True,
        "enable_return_hidden_states": True,
        "skip_tokenizer_init": False,
        "ssl_keyfile": None,
        "ssl_certfile": None,
        "enable_http2": False,
        "hf_chat_template_name": None,
        "completion_template": None,
        "enable_cache_report": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def flag_values(args):
    return {
        args[index]: args[index + 1]
        for index in range(1, len(args) - 1)
        if args[index].startswith("--") and not args[index + 1].startswith("--")
    }


class TestRustRendererHost(unittest.TestCase):
    def test_host_owns_sidecar_lifecycle(self):
        sidecar = mock.Mock()
        with (
            mock.patch(
                "sglang.srt.managers.rust_renderer.get_free_port",
                return_value=31000,
            ),
            mock.patch(
                "sglang.srt.managers.rust_renderer.RustRendererSidecar.launch",
                return_value=sidecar,
            ) as launch,
        ):
            host = RustRendererHost(
                server_args(),
                SimpleNamespace(
                    vocab_size=128,
                    context_len=4096,
                    is_multimodal=False,
                ),
                "0.0.0.0:30000",
                32,
            )
            host.start([2, 3])
            host.stop()
            host.stop()

        self.assertEqual(host.internal_server_addr.to_host_port_str(), "127.0.0.1:31000")
        launch.assert_called_once_with(
            args=host.args,
            public_addr=host.public_addr,
            internal_server_url="http://127.0.0.1:31000",
            cores=[2, 3],
        )
        sidecar.stop.assert_called_once_with()

    def test_embedded_args_use_resolved_state_and_internal_server(self):
        args = build_renderer_args(
            server_args(),
            SimpleNamespace(vocab_size=128, context_len=4096),
            NetworkAddress("0.0.0.0", 30000),
            internal_server_url="http://127.0.0.1:31000",
            num_reserved_tokens=32,
        )
        values = flag_values(args)

        self.assertEqual(args[0], "model-path")
        self.assertEqual(values["--engine-url"], "http://127.0.0.1:31000")
        self.assertEqual(values["--fallback-url"], "http://127.0.0.1:31000")
        self.assertEqual(values["--host"], "0.0.0.0")
        self.assertEqual(values["--port"], "30000")
        self.assertEqual(values["--tokenizer-path"], "tokenizer")
        self.assertEqual(values["--served-model-name"], "served-model")
        self.assertEqual(values["--revision"], "revision")
        self.assertEqual(values["--context-length"], "4096")
        self.assertEqual(values["--vocab-size"], "128")
        self.assertEqual(values["--num-reserved-tokens"], "32")
        self.assertEqual(
            json.loads(values["--default-chat-template-kwargs"]),
            {"thinking": False},
        )
        self.assertIn("--allow-auto-truncate", args)
        self.assertIn("--enable-return-hidden-states", args)
        self.assertIn("--stream-response-default-include-usage", args)

    def test_rejects_modes_the_embedded_renderer_cannot_preserve(self):
        text_model = SimpleNamespace(is_multimodal=False)
        cases = [
            (server_args(skip_tokenizer_init=True), text_model, "requires a tokenizer"),
            (server_args(ssl_certfile="cert"), text_model, "implement TLS"),
            (server_args(enable_http2=True), text_model, "implement HTTP/2"),
            (
                server_args(hf_chat_template_name="tool_use"),
                text_model,
                "hf-chat-template-name",
            ),
            (
                server_args(completion_template="template"),
                text_model,
                "completion-template",
            ),
            (
                server_args(enable_cache_report=True),
                text_model,
                "enable-cache-report",
            ),
            (
                server_args(),
                SimpleNamespace(is_multimodal=True),
                "text-only models",
            ),
        ]

        for args, model, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_embedded_renderer(args, model)

    def test_wildcard_listener_uses_loopback_for_readiness(self):
        self.assertEqual(connect_host("0.0.0.0"), "127.0.0.1")
        self.assertEqual(connect_host("::"), "::1")
        self.assertEqual(connect_host("127.0.0.2"), "127.0.0.2")


if __name__ == "__main__":
    unittest.main()
