"""Rust bootstrap preserves advertised ports and each backend's wire protocol."""

import unittest
from unittest.mock import Mock, patch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.environ import envs
from sglang.srt.rust_server import disaggregation
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRustBootstrap(unittest.TestCase):
    def test_http_bootstrap_preserves_default_shared_and_separate_ports(self):
        for port in (8998, 31000, 33000):
            with self.subTest(port=port), envs.SGLANG_RUST_SERVER.override(True):
                args = ServerArgs(
                    model_path="dummy",
                    port=31000,
                    disaggregation_mode="prefill",
                    disaggregation_bootstrap_port=port,
                )
                handle_pd_disaggregation(args)
                self.assertEqual(
                    resolving_view(args).disaggregation_bootstrap_port, port
                )
                backend = Mock(bootstrap_protocol="http")
                with patch.object(
                    disaggregation, "_bootstrap_class", return_value=backend
                ):
                    self.assertEqual(disaggregation.http_bootstrap_port(args), port)
                    with disaggregation.bootstrap_service(args):
                        backend.assert_not_called()

    def test_binary_service_starts_before_workers_and_closes_after_failure(self):
        args = ServerArgs(
            model_path="dummy",
            host="::",
            port=31000,
            disaggregation_mode="prefill",
            disaggregation_bootstrap_port=8998,
        )
        backend = Mock(bootstrap_protocol="binary")
        with (
            envs.SGLANG_RUST_SERVER.override(True),
            patch.object(
                disaggregation.disaggregation_utils,
                "get_kv_class",
                return_value=backend,
            ),
        ):
            self.assertIsNone(disaggregation.http_bootstrap_port(args))
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                with disaggregation.bootstrap_service(args) as server:
                    backend.assert_called_once_with("::", 8998)
                    self.assertIs(server, backend.return_value)
                    raise RuntimeError("worker failed")
            backend.return_value.close.assert_called_once_with()

            args.port = 8998
            with self.assertRaisesRegex(ValueError, "different.*bootstrap-port"):
                with disaggregation.bootstrap_service(args):
                    self.fail("binary bootstrap cannot share the HTTP port")
            self.assertEqual(backend.call_count, 1)

    def test_only_the_prefill_launch_owner_starts_external_bootstrap(self):
        with (
            envs.SGLANG_RUST_SERVER.override(True),
            patch.object(disaggregation, "_bootstrap_class") as backend,
        ):
            for role, node in (("null", 0), ("decode", 0), ("prefill", 1)):
                with self.subTest(role=role, node=node):
                    args = ServerArgs(
                        model_path="dummy", disaggregation_mode=role, node_rank=node
                    )
                    with disaggregation.bootstrap_service(args):
                        backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
