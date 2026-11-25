"""Unit tests for srt/connector/s3.py download path handling"""

import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from sglang.srt.connector.s3 import S3Connector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _contents(*keys):
    return {"Contents": [{"Key": key} for key in keys]}


def _make_connector(mock_client, url):
    """Build an S3Connector whose boto3 client is replaced by a mock."""
    with patch("boto3.client", return_value=mock_client):
        return S3Connector(url)


class TestPullFiles(CustomTestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.list_objects_v2.return_value = _contents()

    def test_files_are_downloaded_inside_local_dir(self):
        """Without the trailing-slash normalization, stripping the prefix
        "model" from "model/config.json" leaves "/config.json", and
        os.path.join drops local_dir entirely on Linux."""
        self.client.list_objects_v2.return_value = _contents(
            "model/config.json", "model/sub/a.bin"
        )
        connector = _make_connector(self.client, "s3://bucket/model")

        with TemporaryDirectory() as local_dir:
            connector.local_dir = local_dir
            connector.pull_files()

            destinations = [
                call.args[2] for call in self.client.download_file.call_args_list
            ]
            self.assertEqual(
                destinations,
                [
                    os.path.join(local_dir, "config.json"),
                    os.path.join(local_dir, "sub", "a.bin"),
                ],
            )

    def test_prefix_is_matched_as_a_directory(self):
        """The listing prefix must end with "/" so that sibling keys like
        "model-other/..." are not treated as part of "model"."""
        self.client.list_objects_v2.return_value = _contents("model/config.json")
        connector = _make_connector(self.client, "s3://bucket/model")

        connector.pull_files()

        kwargs = self.client.list_objects_v2.call_args.kwargs
        self.assertEqual(kwargs["Prefix"], "model/")

    def test_nested_subdirectories_are_created(self):
        self.client.list_objects_v2.return_value = _contents("model/sub/deep/a.bin")
        connector = _make_connector(self.client, "s3://bucket/model")

        with TemporaryDirectory() as local_dir:
            connector.local_dir = local_dir
            connector.pull_files()

            self.client.download_file.assert_called_once()
            destination = self.client.download_file.call_args.args[2]
            self.assertEqual(
                destination, os.path.join(local_dir, "sub", "deep", "a.bin")
            )
            self.assertTrue(os.path.isdir(os.path.join(local_dir, "sub", "deep")))

    def test_parent_directory_traversal_is_rejected(self):
        """A key containing ".." must not write outside the local dir."""
        self.client.list_objects_v2.return_value = _contents("model/../evil.bin")
        connector = _make_connector(self.client, "s3://bucket/model")

        with TemporaryDirectory() as local_dir:
            connector.local_dir = local_dir
            with self.assertRaises(ValueError):
                connector.pull_files()
            self.client.download_file.assert_not_called()

    def test_absolute_path_after_prefix_stripping_is_rejected(self):
        """A key like "model//evil" strips to "/evil", which os.path.join
        treats as an absolute path on Linux."""
        self.client.list_objects_v2.return_value = _contents("model//evil.bin")
        connector = _make_connector(self.client, "s3://bucket/model")

        with TemporaryDirectory() as local_dir:
            connector.local_dir = local_dir
            with self.assertRaises(ValueError):
                connector.pull_files()
            self.client.download_file.assert_not_called()

    def test_bucket_root_listing_downloads_into_local_dir(self):
        """A bare bucket URL (empty prefix) must still download into the
        local dir."""
        self.client.list_objects_v2.return_value = _contents("a.bin")
        connector = _make_connector(self.client, "s3://bucket")

        with TemporaryDirectory() as local_dir:
            connector.local_dir = local_dir
            connector.pull_files()

            kwargs = self.client.list_objects_v2.call_args.kwargs
            self.assertEqual(kwargs["Prefix"], "")
            destination = self.client.download_file.call_args.args[2]
            self.assertEqual(destination, os.path.join(local_dir, "a.bin"))


if __name__ == "__main__":
    unittest.main()
