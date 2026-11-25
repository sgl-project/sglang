"""Unit tests for srt/connector/redis.py credential handling"""

import unittest
from unittest.mock import patch

from sglang.srt.connector.redis import RedisConnector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _fake_base_init(self, url):
    """Minimal stand-in for BaseConnector.__init__: skip the temp
    directory and signal-handler registration. Marking the connector
    closed makes __del__ a no-op (the connection is a mock anyway)."""
    self.closed = True


def _make_connector(url):
    """Build a RedisConnector with redis.Redis mocked out; returns
    (connector, mock_client)."""
    with patch(
        "sglang.srt.connector.base_connector.BaseConnector.__init__",
        _fake_base_init,
    ):
        with patch("redis.Redis") as mock_client:
            connector = RedisConnector(url)
    return connector, mock_client


class TestRedisCredentials(CustomTestCase):
    def test_password_is_passed_to_the_client(self):
        """redis://:password@host is the standard way to pass an AUTH
        password (empty username)."""
        _, client = _make_connector("redis://:secret@localhost:6379/model")

        self.assertEqual(client.call_args.kwargs["password"], "secret")
        self.assertIsNone(client.call_args.kwargs["username"])

    def test_username_and_password_are_passed_to_the_client(self):
        """ACL credentials (Redis 6+) travel as username:password."""
        _, client = _make_connector("redis://alice:secret@localhost:6379/model")

        self.assertEqual(client.call_args.kwargs["username"], "alice")
        self.assertEqual(client.call_args.kwargs["password"], "secret")

    def test_credentials_are_percent_decoded(self):
        """urlparse does not decode credentials; special characters such
        as "@" or ":" must be percent-encoded in the URL and decoded
        before reaching the client, matching redis-py's from_url."""
        _, client = _make_connector("redis://alice:p%40ss%3Aword@localhost:6379/model")

        self.assertEqual(client.call_args.kwargs["password"], "p@ss:word")

    def test_urls_without_credentials_pass_none(self):
        """Backwards compatibility: bare URLs must not authenticate."""
        connector, client = _make_connector("redis://localhost:6379/model")

        self.assertIsNone(client.call_args.kwargs["username"])
        self.assertIsNone(client.call_args.kwargs["password"])
        self.assertEqual(connector.model_name, "model")

    def test_host_port_and_model_name_are_unchanged(self):
        """Credentials must not disturb the host, port, or the model
        namespace taken from the URL path."""
        connector, client = _make_connector(
            "redis://alice:secret@localhost:6379/my-model"
        )

        self.assertEqual(client.call_args.kwargs["host"], "localhost")
        self.assertEqual(client.call_args.kwargs["port"], 6379)
        self.assertEqual(connector.model_name, "my-model")


if __name__ == "__main__":
    unittest.main()
