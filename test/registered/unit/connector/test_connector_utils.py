"""Unit tests for srt/connector/utils.py"""

import os
import shutil
import tempfile
import unittest

from sglang.srt.connector.utils import pull_files_from_db
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _KVStore:
    def __init__(self, local_dir, files):
        self.local_dir = local_dir
        self.files = files

    def get_local_dir(self):
        return self.local_dir

    def list(self, prefix):
        return [key for key in self.files if key.startswith(prefix)]

    def getstr(self, key):
        return self.files[key]


class TestPullFilesFromDb(CustomTestCase):
    def test_a_nested_key_does_not_move_the_files_after_it(self):
        """Every file lands at its own path under the connector's directory;
        a key with a subdirectory must not redirect the keys listed after it."""
        local_dir = tempfile.mkdtemp(prefix="connector_utils_")
        self.addCleanup(shutil.rmtree, local_dir, ignore_errors=True)
        files = {
            "model/files/config.json": "{}",
            "model/files/sub/nested.json": "[1]",
            "model/files/tokenizer.json": "[2]",
            "other/files/skipped.json": "[3]",
        }

        pull_files_from_db(_KVStore(local_dir, files), "model")

        pulled = {}
        for root, _, names in os.walk(local_dir):
            for name in names:
                path = os.path.join(root, name)
                with open(path) as handle:
                    pulled[os.path.relpath(path, local_dir)] = handle.read()
        self.assertEqual(
            pulled,
            {
                "config.json": "{}",
                os.path.join("sub", "nested.json"): "[1]",
                "tokenizer.json": "[2]",
            },
        )


if __name__ == "__main__":
    unittest.main()
