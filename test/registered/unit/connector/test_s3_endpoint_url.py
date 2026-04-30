"""Unit tests for S3Connector endpoint_url / region / credential plumbing."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestS3ConnectorEndpointUrl(unittest.TestCase):
    """Verify endpoint_url / region / creds flow through to boto3.client."""

    def _build(self, url, **kwargs):
        # Patch boto3 before importing S3Connector so the import is cheap.
        fake_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            from sglang.srt.connector.s3 import S3Connector

            connector = S3Connector(url, **kwargs)
        return fake_boto3, connector

    def test_query_string_endpoint_url(self):
        url = "s3://b/k?endpoint_url=https://s3.us-west-004.backblazeb2.com&region=us-west-004"
        fake_boto3, conn = self._build(url)
        fake_boto3.client.assert_called_once_with(
            "s3",
            endpoint_url="https://s3.us-west-004.backblazeb2.com",
            region_name="us-west-004",
        )
        # URL is cleaned of query string for downstream parsing.
        self.assertEqual(conn.url, "s3://b/k")

    def test_env_var_fallback(self):
        with patch.dict(
            "os.environ",
            {"AWS_ENDPOINT_URL": "https://minio:9000", "AWS_DEFAULT_REGION": "us-east-1"},
        ):
            fake_boto3, _ = self._build("s3://bucket/path")
        fake_boto3.client.assert_called_once_with(
            "s3", endpoint_url="https://minio:9000", region_name="us-east-1"
        )

    def test_explicit_kwargs_win(self):
        url = "s3://b/k?endpoint_url=https://from-url"
        fake_boto3, _ = self._build(url, endpoint_url="https://explicit")
        fake_boto3.client.assert_called_once_with("s3", endpoint_url="https://explicit")

    def test_no_overrides_passes_no_kwargs(self):
        with patch.dict("os.environ", {}, clear=False):
            for var in ("AWS_ENDPOINT_URL", "AWS_DEFAULT_REGION"):
                # Defensive: ensure env doesn't leak into this case.
                import os

                os.environ.pop(var, None)
            fake_boto3, _ = self._build("s3://bucket/path")
        fake_boto3.client.assert_called_once_with("s3")


if __name__ == "__main__":
    unittest.main()
