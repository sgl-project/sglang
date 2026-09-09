import json
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.managers.rust_renderer import (
    RustRendererSidecar,
    build_renderer_args,
    connect_host,
    validate_embedded_renderer,
)
from sglang.srt.rust_server.server import RustServer
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
        "preferred_sampling_params": None,
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


def model_config(**overrides):
    values = {
        "model_path": "resolved-model-path",
        "vocab_size": 128,
        "context_len": 4096,
        "is_multimodal": False,
        "get_default_sampling_params": lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "min_p": 0.1,
            "repetition_penalty": 1.05,
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def flag_values(args):
    return {
        args[index]: args[index + 1]
        for index in range(1, len(args) - 1)
        if args[index].startswith("--") and not args[index + 1].startswith("--")
    }


class TestRustRendererSidecar(unittest.TestCase):
    def test_launch_preserves_public_dp_address_and_uses_internal_engine_address(self):
        scheduler = SimpleNamespace(
            server_args=server_args(),
            model_config=model_config(),
            ps=SimpleNamespace(dp_size=2, attn_dp_rank=1),
        )
        for enabled in (False, True):
            with (
                self.subTest(renderer=enabled),
                envs.SGLANG_RUST_RENDERER.override(enabled),
                mock.patch(
                    "sglang.srt.rust_server.server.get_serving",
                    return_value=SimpleNamespace(host="0.0.0.0", port=30000),
                ),
                mock.patch(
                    "sglang.srt.rust_server.server.resolving_view",
                    return_value=scheduler.server_args,
                ),
                mock.patch(
                    "sglang.srt.rust_server.server.compute_num_reserved_tokens",
                    return_value=32,
                ),
                mock.patch(
                    "sglang.srt.rust_server.server._partition_cores",
                    return_value=(None, None),
                ),
                mock.patch("sglang.srt.rust_server.server._build_server_args") as build,
                mock.patch(
                    "sglang.srt.rust_extensions.load_rust_extension"
                ) as extension,
                mock.patch(
                    "sglang.srt.managers.rust_renderer.RustRendererSidecar"
                ) as sidecar,
            ):
                sidecar.return_value.internal_server_addr = NetworkAddress(
                    "127.0.0.1", 31000
                )
                server = RustServer.launch(scheduler)
                if enabled:
                    sidecar.assert_called_once_with(
                        scheduler.server_args,
                        scheduler.model_config,
                        NetworkAddress("0.0.0.0", 30001),
                        32,
                    )
                    build.assert_called_once_with(
                        scheduler, host="127.0.0.1", port=31000
                    )
                    sidecar.return_value.start.assert_called_once_with(None)
                else:
                    sidecar.assert_not_called()
                    build.assert_called_once_with(scheduler)
                extension.return_value.Server.assert_called_once_with(
                    build.return_value,
                    cores=None,
                    port_offset=None if enabled else 1,
                )
                server.close()
                extension.return_value.Server.return_value.shutdown.assert_called_once()
                if enabled:
                    sidecar.return_value.stop.assert_called_once()

    def test_sidecar_owns_topology_and_process_lifecycle(self):
        process = mock.Mock(pid=123, exitcode=0)
        process.is_alive.return_value = False
        context = mock.Mock()
        context.Process.return_value = process
        watchdog = mock.Mock()
        with (
            mock.patch(
                "sglang.srt.managers.rust_renderer.get_free_port",
                return_value=31000,
            ),
            mock.patch(
                "sglang.srt.managers.rust_renderer.find_renderer_binary",
                return_value="/bin/sglang-renderer",
            ),
            mock.patch(
                "sglang.srt.managers.rust_renderer.mp.get_context",
                return_value=context,
            ),
            mock.patch(
                "sglang.srt.managers.rust_renderer.SubprocessWatchdog",
                return_value=watchdog,
            ),
            mock.patch.object(RustRendererSidecar, "_wait_until_listening"),
        ):
            sidecar = RustRendererSidecar(
                server_args(),
                model_config(),
                NetworkAddress("0.0.0.0", 30000),
                32,
            )
            sidecar.start([2, 3])
            sidecar.stop()
            sidecar.stop()

        self.assertEqual(
            sidecar.internal_server_addr.to_host_port_str(), "127.0.0.1:31000"
        )
        context.Process.assert_called_once_with(
            name="sglang_renderer",
            target=mock.ANY,
            args=("/bin/sglang-renderer", sidecar.args, [2, 3]),
        )
        process.start.assert_called_once_with()
        watchdog.start.assert_called_once_with()
        watchdog.stop.assert_called_once_with()
        self.assertIsNone(sidecar.process)

    def test_embedded_args_use_resolved_state_and_internal_server(self):
        args = build_renderer_args(
            server_args(
                model_path="s3://bucket/model",
                tokenizer_path="s3://bucket/model",
            ),
            model_config(),
            NetworkAddress("0.0.0.0", 30000),
            internal_server_url="http://127.0.0.1:31000",
            num_reserved_tokens=32,
        )
        values = flag_values(args)

        self.assertEqual(args[0], "resolved-model-path")
        self.assertEqual(values["--engine-url"], "http://127.0.0.1:31000")
        self.assertIn("--proxy-unhandled-routes", args)
        self.assertNotIn("--fallback-url", args)
        self.assertEqual(values["--host"], "0.0.0.0")
        self.assertEqual(values["--port"], "30000")
        self.assertEqual(values["--tokenizer-path"], "resolved-model-path")
        self.assertEqual(values["--served-model-name"], "served-model")
        self.assertEqual(values["--revision"], "revision")
        self.assertEqual(values["--context-length"], "4096")
        self.assertEqual(values["--vocab-size"], "128")
        self.assertEqual(values["--num-reserved-tokens"], "32")
        self.assertEqual(
            json.loads(values["--resolved-sampling-params"]),
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 20,
                "min_p": 0.1,
                "repetition_penalty": 1.05,
            },
        )
        self.assertNotIn("--sampling-defaults", args)
        self.assertEqual(
            json.loads(values["--default-chat-template-kwargs"]),
            {"thinking": False},
        )
        self.assertIn("--allow-auto-truncate", args)
        self.assertIn("--enable-return-hidden-states", args)
        self.assertIn("--stream-response-default-include-usage", args)

    def test_readiness_rejects_an_unrelated_listener(self):
        unrelated = mock.Mock(status=200)
        unrelated.getheader.return_value = None
        renderer = mock.Mock(status=204)
        renderer.getheader.return_value = "ready"
        connections = []
        for response in (unrelated, renderer):
            connection = mock.Mock()
            connection.getresponse.return_value = response
            connections.append(connection)

        with (
            mock.patch(
                "sglang.srt.managers.rust_renderer.get_free_port",
                return_value=31000,
            ),
            mock.patch(
                "sglang.srt.managers.rust_renderer.http.client.HTTPConnection",
                side_effect=connections,
            ) as connect,
            mock.patch(
                "sglang.srt.managers.rust_renderer.time.monotonic", return_value=0
            ),
            mock.patch("sglang.srt.managers.rust_renderer.time.sleep"),
        ):
            sidecar = RustRendererSidecar(
                server_args(), model_config(), NetworkAddress("0.0.0.0", 30000), 32
            )
            sidecar.process = mock.Mock()
            sidecar.process.is_alive.return_value = True
            sidecar._wait_until_listening()

        self.assertEqual(connect.call_count, 2)
        for connection in connections:
            connection.request.assert_called_once_with("GET", "/_sglang_renderer/ready")
            connection.close.assert_called_once_with()

    def test_rejects_modes_the_embedded_renderer_cannot_preserve(self):
        text_model = SimpleNamespace(is_multimodal=False)
        cases = [
            (server_args(skip_tokenizer_init=True), text_model, "requires a tokenizer"),
            (server_args(ssl_certfile="cert"), text_model, "implement TLS"),
            (server_args(enable_http2=True), text_model, "implement HTTP/2"),
            (
                server_args(preferred_sampling_params={"temperature": 0.2}),
                text_model,
                "preferred-sampling-params",
            ),
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
