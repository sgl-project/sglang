"""Unit tests for srt/connector/utils.py"""

import os
import shutil
import tempfile
import unittest

from sglang.srt.connector.utils import pull_files_from_db
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


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


class _RecordingKVStore(_KVStore):
    """_KVStore that records every getstr() call."""

    def __init__(self, local_dir, files):
        super().__init__(local_dir, files)
        self.getstr_keys = []

    def getstr(self, key):
        self.getstr_keys.append(key)
        return super().getstr(key)


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

    def _store(self, files):
        local_dir = tempfile.mkdtemp(prefix="connector_utils_")
        self.addCleanup(shutil.rmtree, local_dir, ignore_errors=True)
        return _RecordingKVStore(local_dir, files), local_dir

    def _pulled(self, local_dir):
        pulled = {}
        for root, _, names in os.walk(local_dir):
            for name in names:
                path = os.path.join(root, name)
                with open(path) as handle:
                    pulled[os.path.relpath(path, local_dir)] = handle.read()
        return pulled

    def test_allow_pattern_limits_what_is_downloaded(self):
        """allow_pattern must be honored: excluded files are neither
        fetched via getstr() nor written to disk."""
        store, local_dir = self._store(
            {
                "model/files/config.json": "{}",
                "model/files/tokenizer.json": "[2]",
            }
        )

        pull_files_from_db(store, "model", allow_pattern=["*config.json"])

        self.assertEqual(self._pulled(local_dir), {"config.json": "{}"})
        self.assertEqual(store.getstr_keys, ["model/files/config.json"])

    def test_ignore_pattern_excludes_files(self):
        """ignore_pattern must be honored (e.g. the tokenizer loader
        excludes *.pt/*.safetensors/*.bin)."""
        store, local_dir = self._store(
            {
                "model/files/config.json": "{}",
                "model/files/weights.bin": "[2]",
            }
        )

        pull_files_from_db(store, "model", ignore_pattern=["*.bin"])

        self.assertEqual(self._pulled(local_dir), {"config.json": "{}"})
        self.assertEqual(store.getstr_keys, ["model/files/config.json"])

    def test_allow_and_ignore_combine(self):
        store, local_dir = self._store(
            {
                "model/files/config.json": "{}",
                "model/files/spy-config.json": "[3]",
                "model/files/weights.bin": "[2]",
            }
        )

        pull_files_from_db(
            store,
            "model",
            allow_pattern=["*.json"],
            ignore_pattern=["*spy*"],
        )

        self.assertEqual(self._pulled(local_dir), {"config.json": "{}"})
        self.assertEqual(store.getstr_keys, ["model/files/config.json"])

    def test_empty_allow_pattern_downloads_nothing(self):
        """An empty allow list matches nothing; the `is not None` check
        must keep this distinct from `None` (unrestricted)."""
        store, local_dir = self._store(
            {
                "model/files/config.json": "{}",
                "model/files/tokenizer.json": "[2]",
            }
        )

        pull_files_from_db(store, "model", allow_pattern=[])

        self.assertEqual(self._pulled(local_dir), {})
        self.assertEqual(store.getstr_keys, [])

    def test_empty_ignore_pattern_downloads_everything(self):
        """An empty ignore list excludes nothing."""
        store, local_dir = self._store(
            {
                "model/files/config.json": "{}",
                "model/files/tokenizer.json": "[2]",
            }
        )

        pull_files_from_db(store, "model", ignore_pattern=[])

        self.assertEqual(
            self._pulled(local_dir),
            {"config.json": "{}", "tokenizer.json": "[2]"},
        )

    def test_no_patterns_downloads_everything(self):
        """Without patterns the behavior is unchanged (backwards compat)."""
        store, local_dir = self._store(
            {
                "model/files/config.json": "{}",
                "model/files/sub/nested.json": "[1]",
            }
        )

        pull_files_from_db(store, "model")

        self.assertEqual(
            self._pulled(local_dir),
            {
                "config.json": "{}",
                os.path.join("sub", "nested.json"): "[1]",
            },
        )

    def test_nested_keys_are_filtered_and_landed_correctly(self):
        store, local_dir = self._store(
            {
                "model/files/sub/tokenizer.json": "[1]",
                "model/files/sub/weights.bin": "[2]",
                "model/files/config.json": "{}",
            }
        )

        pull_files_from_db(store, "model", ignore_pattern=["*.bin"])

        self.assertEqual(
            self._pulled(local_dir),
            {
                "config.json": "{}",
                os.path.join("sub", "tokenizer.json"): "[1]",
            },
        )

    def test_parent_directory_traversal_is_rejected(self):
        """A key containing ".." must not write outside the local dir."""
        store, local_dir = self._store({"model/files/../evil.txt": "[1]"})

        with self.assertRaises(ValueError):
            pull_files_from_db(store, "model")
        self.assertEqual(store.getstr_keys, [])
        self.assertEqual(self._pulled(local_dir), {})

    def test_absolute_path_after_prefix_stripping_is_rejected(self):
        """A key like "model/files//evil" strips to "/evil", which
        os.path.join treats as an absolute path on Linux."""
        store, local_dir = self._store({"model/files//evil.txt": "[1]"})

        with self.assertRaises(ValueError):
            pull_files_from_db(store, "model")
        self.assertEqual(store.getstr_keys, [])
        self.assertEqual(self._pulled(local_dir), {})


if __name__ == "__main__":
    unittest.main()
