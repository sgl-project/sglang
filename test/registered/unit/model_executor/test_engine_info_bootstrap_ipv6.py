"""Regression test: bootstrap registration URL must respect --host bind family.

Bug: ModelRunner._register_to_engine_info_bootstrap hardcoded
    bootstrap_host = "127.0.0.1"
for the single-node path. When the SGLang server is launched with `--host ::`
(IPv6-only pods, V6ONLY=1), the EngineInfoBootstrapServer only accepts IPv6
connections, so the IPv4 PUT is refused → registration silently fails → the
seed's transfer_engine_info dict stays empty → R-Fork client gets 404 from
/get_remote_instance_transfer_engine_info and weight transfer hangs.

Fix: reuse server_args.engine_info_bootstrap_url, which already maps
`::` → `::1` and `0.0.0.0` → `127.0.0.1` (see ServerArgs.url() in
server_args.py). This matches what http_server.py:1122 already does for
/get_transfer_engine_info, so the seed-side registration and lookup now agree
on the loopback family.
"""

import inspect
import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

# CPU CI runners don't have CUDA — match the precedent in test_server_args.py.
_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


def _make_server_args(host: str) -> ServerArgs:
    return ServerArgs(model_path="dummy", host=host, port=30000)


class TestEngineInfoBootstrapURLLoopbackFamily(CustomTestCase):
    """ServerArgs.engine_info_bootstrap_url must produce a URL whose loopback
    matches the bind family that EngineInfoBootstrapServer listens on."""

    def test_ipv6_host_produces_ipv6_loopback_url(self):
        sa = _make_server_args(host="::")
        url = sa.engine_info_bootstrap_url
        self.assertIn("[::1]", url, f"expected [::1] in URL, got: {url}")
        self.assertNotIn("127.0.0.1", url)

    def test_ipv4_wildcard_produces_ipv4_loopback_url(self):
        sa = _make_server_args(host="0.0.0.0")
        url = sa.engine_info_bootstrap_url
        self.assertIn("127.0.0.1", url)
        self.assertNotIn("[::1]", url)

    def test_specific_ipv4_host_used_verbatim(self):
        sa = _make_server_args(host="192.0.2.10")
        url = sa.engine_info_bootstrap_url
        self.assertIn("192.0.2.10", url)

    def test_url_uses_bootstrap_port_not_server_port(self):
        sa = _make_server_args(host="::")
        url = sa.engine_info_bootstrap_url
        self.assertIn(f":{sa.engine_info_bootstrap_port}", url)
        self.assertNotIn(":30000", url)


class TestRegisterUsesBootstrapURL(CustomTestCase):
    """Source-level guard: _register_to_engine_info_bootstrap must build its URL
    via engine_info_bootstrap_url for the single-node path. Catches a regression
    that re-introduces the hardcoded "127.0.0.1"."""

    def test_register_method_references_engine_info_bootstrap_url(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        src = inspect.getsource(ModelRunner._register_to_engine_info_bootstrap)
        self.assertIn(
            "engine_info_bootstrap_url",
            src,
            "Single-node path must use server_args.engine_info_bootstrap_url; "
            "hardcoding 127.0.0.1 breaks IPv6-only pods.",
        )
        self.assertNotIn(
            '"127.0.0.1"',
            src,
            "Hardcoded 127.0.0.1 was re-introduced — IPv6-only pods will hang.",
        )


if __name__ == "__main__":
    unittest.main()
