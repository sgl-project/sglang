import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="default")


class TestCreateRemoteConnector(unittest.TestCase):
    """Verify that create_remote_connector works without an explicit device arg."""

    @patch("sglang.srt.connector.S3Connector")
    def test_s3_without_device(self, mock_s3):
        from sglang.srt.connector import create_remote_connector

        mock_s3.return_value = MagicMock()
        client = create_remote_connector("s3://bucket/model")
        mock_s3.assert_called_once_with("s3://bucket/model")
        self.assertIsNotNone(client)

    @patch("sglang.srt.connector.RedisConnector")
    def test_redis_without_device(self, mock_redis):
        from sglang.srt.connector import create_remote_connector

        mock_redis.return_value = MagicMock()
        client = create_remote_connector("redis://localhost:6379")
        mock_redis.assert_called_once_with("redis://localhost:6379")
        self.assertIsNotNone(client)

    @patch("sglang.srt.connector.RemoteInstanceConnector")
    def test_instance_with_device(self, mock_instance):
        from sglang.srt.connector import create_remote_connector

        mock_instance.return_value = MagicMock()
        client = create_remote_connector("instance://host:1234", "cuda:0")
        mock_instance.assert_called_once_with("instance://host:1234", "cuda:0")
        self.assertIsNotNone(client)

    @patch("sglang.srt.connector.RemoteInstanceConnector")
    def test_instance_default_device(self, mock_instance):
        from sglang.srt.connector import create_remote_connector

        mock_instance.return_value = MagicMock()
        client = create_remote_connector("instance://host:1234")
        mock_instance.assert_called_once_with("instance://host:1234", "cpu")
        self.assertIsNotNone(client)

    def test_invalid_url(self):
        from sglang.srt.connector import create_remote_connector

        with self.assertRaises(ValueError):
            create_remote_connector("unknown://foo")


if __name__ == "__main__":
    unittest.main()
