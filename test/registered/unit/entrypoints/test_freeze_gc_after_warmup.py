import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from sglang.srt.entrypoints.http_server import _freeze_gc_after_server_warmup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MODULE = "sglang.srt.entrypoints.http_server"


class TestFreezeGCAfterWarmup(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.server_args = SimpleNamespace(
            api_key=None,
            admin_api_key=None,
            url=lambda: "http://localhost:30000",
            ssl_ca_certs=None,
            ssl_certfile=None,
        )
        self.post = self.enterContext(patch(f"{_MODULE}.requests.post"))
        self.sleep = self.enterContext(patch(f"{_MODULE}.time.sleep"))
        self.warning = self.enterContext(patch(f"{_MODULE}.logger.warning"))

    def test_retries_connection_failure_then_succeeds(self):
        response = Mock()
        self.post.side_effect = [requests.ConnectionError("not ready"), response]

        _freeze_gc_after_server_warmup(self.server_args)

        self.assertEqual(self.post.call_count, 2)
        self.sleep.assert_called_once_with(1)
        response.raise_for_status.assert_called_once_with()
        self.warning.assert_not_called()

    def test_warns_once_after_connection_retries_are_exhausted(self):
        self.post.side_effect = requests.ConnectionError("not ready")

        _freeze_gc_after_server_warmup(self.server_args)

        self.assertEqual(self.post.call_count, 15)
        self.assertEqual(self.sleep.call_count, 14)
        self.warning.assert_called_once_with(
            "post-warmup freeze_gc failed", exc_info=True
        )

    def test_does_not_retry_non_connection_error(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("503")
        self.post.return_value = response

        _freeze_gc_after_server_warmup(self.server_args)

        self.post.assert_called_once()
        self.sleep.assert_not_called()
        self.warning.assert_called_once_with(
            "post-warmup freeze_gc failed", exc_info=True
        )


if __name__ == "__main__":
    unittest.main()
