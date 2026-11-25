"""Unit tests for srt/connector/s3.py download path handling"""

import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from sglang.srt.connector.s3 import S3Connector, list_files
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_connector(mock_client, url, local_dir):
    """Build an S3Connector without running __init__, which creates a
    temp directory and installs SIGINT/SIGTERM handlers that unit tests
    should not leak. pull_files() only needs these three attributes."""
    connector = S3Connector.__new__(S3Connector)
    connector.url = url
    connector.local_dir = local_dir
    connector.client = mock_client
    # __del__ calls close(), which needs this flag from __init__.
    connector.closed = False
    return connector


class TestListFilesPrefix(CustomTestCase):
    """These tests mock the S3 protocol, so they must follow however
    list_files() talks to S3 (currently a single list_objects_v2 call)."""

    def _list(self, url):
        client = MagicMock()
        client.list_objects_v2.return_value = {
            "Contents": [{"Key": "model/config.json"}]
        }
        _, prefix, _ = list_files(client, url)
        return client, prefix

    def test_prefix_is_matched_as_a_directory(self):
        """The listing prefix must end with "/" so that sibling keys like
        "model-other/..." are not treated as part of "model"."""
        client, prefix = self._list("s3://bucket/model")

        self.assertEqual(prefix, "model/")
        self.assertEqual(client.list_objects_v2.call_args.kwargs["Prefix"], "model/")

    def test_prefix_with_trailing_slash_is_not_duplicated(self):
        client, prefix = self._list("s3://bucket/model/")

        self.assertEqual(prefix, "model/")

    def test_bucket_root_has_empty_prefix(self):
        client, prefix = self._list("s3://bucket")

        self.assertEqual(prefix, "")


class TestPullFiles(CustomTestCase):
    """These tests patch list_files() itself, so they exercise only the
    download-path logic and stay independent of how listing is
    implemented (single request vs paginator)."""

    def _make_patched(self, url, prefix, keys):
        """Create a connector whose list_files() returns `keys`; returns
        (client, connector, local_dir)."""
        client = MagicMock()
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        connector = _make_connector(client, url, tmp.name)
        patcher = patch(
            "sglang.srt.connector.s3.list_files",
            return_value=("bucket", prefix, keys),
        )
        self.addCleanup(patcher.stop)
        self.list_files_mock = patcher.start()
        return client, connector, tmp.name

    def test_files_are_downloaded_inside_local_dir(self):
        """Without the trailing-slash prefix, stripping "model" from
        "model/config.json" leaves "/config.json", and os.path.join drops
        local_dir entirely on Linux."""
        client, connector, local_dir = self._make_patched(
            "s3://bucket/model", "model/", ["model/config.json", "model/sub/a.bin"]
        )

        connector.pull_files()

        destinations = [call.args[2] for call in client.download_file.call_args_list]
        self.assertEqual(
            destinations,
            [
                os.path.join(local_dir, "config.json"),
                os.path.join(local_dir, "sub", "a.bin"),
            ],
        )

    def test_url_with_trailing_slash_downloads_normally(self):
        client, connector, local_dir = self._make_patched(
            "s3://bucket/model/", "model/", ["model/config.json"]
        )

        connector.pull_files()

        destination = client.download_file.call_args.args[2]
        self.assertEqual(destination, os.path.join(local_dir, "config.json"))

    def test_nested_subdirectories_are_created(self):
        client, connector, local_dir = self._make_patched(
            "s3://bucket/model", "model/", ["model/sub/deep/a.bin"]
        )

        connector.pull_files()

        client.download_file.assert_called_once()
        destination = client.download_file.call_args.args[2]
        self.assertEqual(destination, os.path.join(local_dir, "sub", "deep", "a.bin"))
        self.assertTrue(os.path.isdir(os.path.join(local_dir, "sub", "deep")))

    def test_parent_directory_traversal_is_rejected(self):
        """A key containing ".." must not write outside the local dir."""
        client, connector, _ = self._make_patched(
            "s3://bucket/model", "model/", ["model/../evil.bin"]
        )

        with self.assertRaises(ValueError):
            connector.pull_files()
        client.download_file.assert_not_called()

    def test_absolute_path_after_prefix_stripping_is_rejected(self):
        """A key like "model//evil" strips to "/evil", which os.path.join
        treats as an absolute path on Linux."""
        client, connector, _ = self._make_patched(
            "s3://bucket/model", "model/", ["model//evil.bin"]
        )

        with self.assertRaises(ValueError):
            connector.pull_files()
        client.download_file.assert_not_called()

    def test_bucket_root_listing_downloads_into_local_dir(self):
        client, connector, local_dir = self._make_patched("s3://bucket", "", ["a.bin"])

        connector.pull_files()

        destination = client.download_file.call_args.args[2]
        self.assertEqual(destination, os.path.join(local_dir, "a.bin"))

    def test_patterns_are_forwarded_to_list_files(self):
        """pull_files() must forward its pattern arguments to list_files()."""
        client, connector, _ = self._make_patched(
            "s3://bucket/model", "model/", ["model/config.json"]
        )

        connector.pull_files(allow_pattern=["*.safetensors"], ignore_pattern=["*.bin"])

        args = self.list_files_mock.call_args.args
        self.assertEqual(args[0], client)
        self.assertEqual(args[1], "s3://bucket/model")
        self.assertEqual(args[2], ["*.safetensors"])
        self.assertEqual(args[3], ["*.bin"])


if __name__ == "__main__":
    unittest.main()
