"""Unit tests for the benchmark --insecure flag (disable TLS verification).

These guard the bug where a self-signed / internal-CA server behind a gateway
made every `sglang.benchmark` health check fail with a swallowed SSLError:
the ``wait_for_endpoint`` loop caught ``requests.exceptions.RequestException``
(which subsumes ``SSLError``) and printed only "Server did not become ready",
so the real cause was invisible.  They also pin the behavior that:

- ``--insecure`` must propagate `verify=False` to every requests call that
  touches the server (health check, model discovery, server_info, cache flush,
  bench session), and the default (no flag) must keep ``verify=True``;
- ``wait_for_endpoint`` must surface the underlying error when it times out.
"""

import argparse
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests
import urllib3

from sglang.benchmark.endpoint import server_is_up
from sglang.benchmark.serving import (
    _create_bench_client_session,
    _suppress_insecure_warnings,
    wait_for_endpoint,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-b-test-cpu")

_SECURE = SimpleNamespace(insecure=False)
_INSECURE = SimpleNamespace(insecure=True)


def _patch_args(namespace):
    """Patch the module-global ``args``, creating it if unset.

    ``serving.args`` is only assigned by ``set_global_args``/``run_benchmark``,
    so a bare ``patch`` on an un-imported-run process raises AttributeError.
    """
    return patch("sglang.benchmark.serving.args", namespace, create=True)


class TestInsecureSslRequests(CustomTestCase):
    @patch("sglang.benchmark.serving.requests.get")
    def test_wait_for_endpoint_insecure_disables_verification(self, mock_get):
        """--insecure must translate to verify=False on the health poll."""
        mock_get.return_value.status_code = 200
        with _patch_args(_INSECURE):
            wait_for_endpoint("https://gw.example/v1/models", timeout_sec=1)
        mock_get.assert_called_once()
        self.assertFalse(mock_get.call_args.kwargs["verify"])

    @patch("sglang.benchmark.serving.requests.get")
    def test_wait_for_endpoint_default_verifies(self, mock_get):
        """Without --insecure the health poll must keep verify=True."""
        mock_get.return_value.status_code = 200
        with _patch_args(_SECURE):
            wait_for_endpoint("https://gw.example/v1/models", timeout_sec=1)
        self.assertTrue(mock_get.call_args.kwargs["verify"])

    @patch("sglang.benchmark.serving.requests.get")
    def test_wait_for_endpoint_surfaces_ssl_error_on_timeout(self, mock_get):
        """Timeout must not swallow the SSLError that caused it.

        The poll loop raises SSLError on every attempt; the caller sees the
        exception class and message in the printed diagnostics (mocked here
        by asserting the message is emitted before the False return).
        """
        mock_get.side_effect = requests.exceptions.SSLError("certificate verify failed")
        with _patch_args(_SECURE), patch(
            "sglang.benchmark.serving.print"
        ) as mock_print:
            result = wait_for_endpoint("https://gw.example/v1/models", timeout_sec=1)
        self.assertFalse(result)
        output = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("SSLError", output)
        self.assertIn("certificate verify failed", output)

    @patch("sglang.benchmark.serving.requests.get")
    def test_wait_for_endpoint_ready_after_transient_failure(self, mock_get):
        """A transient connection error must not fail the poll outright."""
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("refused"),
            MagicMock(status_code=200),
        ]
        with _patch_args(_INSECURE):
            ok = wait_for_endpoint("https://gw.example/v1/models", timeout_sec=3)
        self.assertTrue(ok)

    @patch("sglang.benchmark.endpoint.requests.get")
    def test_server_is_up_passes_verify_ssl(self, mock_get):
        """server_is_up must forward verify_ssl to requests."""
        mock_get.return_value.status_code = 200
        server_is_up("https://gw.example", timeout=1, verify_ssl=False)
        self.assertFalse(mock_get.call_args.kwargs["verify"])
        server_is_up("https://gw.example", timeout=1, verify_ssl=True)
        self.assertTrue(mock_get.call_args.kwargs["verify"])

    @patch("sglang.benchmark.serving.aiohttp.ClientSession")
    @patch("sglang.benchmark.serving.aiohttp.TCPConnector")
    def test_bench_session_respects_insecure(
        self, mock_connector_cls, mock_session_cls
    ):
        """--insecure must disable TLS verification on the aiohttp connector."""
        with _patch_args(_INSECURE):
            _create_bench_client_session()
        mock_connector_cls.assert_called_once_with(ssl=False)

    @patch("sglang.benchmark.serving.aiohttp.ClientSession")
    @patch("sglang.benchmark.serving.aiohttp.TCPConnector")
    def test_bench_session_default_verifies(
        self, mock_connector_cls, mock_session_cls
    ):
        """Without --insecure the aiohttp connector must verify TLS."""
        with _patch_args(_SECURE):
            _create_bench_client_session()
        mock_connector_cls.assert_called_once_with(ssl=True)

    def test_suppress_insecure_warnings_prepends_ignore(self):
        """--insecure must put the InsecureRequestWarning ignore at the front.

        Python ships a built-in ``default`` rule for ``UserWarning`` (the
        parent of ``InsecureRequestWarning``). It matches InsecureRequestWarning
        first, so an ``append=True`` filter sits behind it and never wins -- the
        warning keeps spamming the log. Reproduce that rule here and assert the
        ignore filter lands in front of it.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("default", UserWarning)
            with _patch_args(_INSECURE):
                _suppress_insecure_warnings()
            for action, _, category, _, _ in warnings.filters:
                if category is not None and issubclass(
                    urllib3.exceptions.InsecureRequestWarning, category
                ):
                    self.assertEqual(
                        action,
                        "ignore",
                        "ignore filter must precede the built-in UserWarning "
                        "default rule to actually silence the warning",
                    )
                    return
        self.fail("no filter matches InsecureRequestWarning after --insecure")

    def test_suppress_insecure_warnings_keeps_default_secure(self):
        """Without --insecure, no InsecureRequestWarning ignore filter is added."""
        with warnings.catch_warnings():
            warnings.simplefilter("default", UserWarning)
            with _patch_args(_SECURE):
                _suppress_insecure_warnings()
            for action, _, category, _, _ in warnings.filters:
                if (
                    action == "ignore"
                    and category is not None
                    and issubclass(
                        urllib3.exceptions.InsecureRequestWarning, category
                    )
                ):
                    self.fail("secure run must not install an ignore filter")


class TestInsecureSslOneBatch(CustomTestCase):
    def test_bench_args_insecure_default_off(self):
        """--insecure must default to off and parse from the CLI flag."""
        from sglang.benchmark.one_batch_server import BenchArgs

        self.assertFalse(BenchArgs().insecure)
        parser = argparse.ArgumentParser()
        BenchArgs.add_cli_args(parser)
        parsed = parser.parse_args(["--insecure"])
        self.assertTrue(parsed.insecure)


if __name__ == "__main__":
    unittest.main(verbosity=2)
