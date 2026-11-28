"""Local destinations must be independent of the input URL scheme."""

import os
import unittest
from tempfile import TemporaryDirectory
from unittest import mock

from sglang.srt.connector.azure import AzureBlobConnector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAzureDownloadDestination(unittest.TestCase):
    def _assert_config_destination(self, url):
        # blobfile v3's default output_az_paths=True canonicalizes entries
        # returned by an HTTPS listing to az:// URLs as well.
        remote = mock.MagicMock()
        remote.glob.return_value = ["az://account/container/model/config.json"]
        remote.isdir.return_value = False

        connector = AzureBlobConnector.__new__(AzureBlobConnector)
        connector.url = url
        connector.local_dir = None
        connector.closed = False
        connector.bf = remote

        with TemporaryDirectory() as local_dir:
            connector.local_dir = local_dir
            connector.pull_files(allow_pattern=["*config.json"])

            remote.copy.assert_called_once()
            destination = remote.copy.call_args.args[1]
            self.assertEqual(destination, os.path.join(local_dir, "config.json"))

    def test_az_input_preserves_relative_path(self):
        self._assert_config_destination("az://account/container/model")

    def test_https_input_preserves_relative_path(self):
        self._assert_config_destination(
            "https://account.blob.core.windows.net/container/model"
        )


if __name__ == "__main__":
    unittest.main()
