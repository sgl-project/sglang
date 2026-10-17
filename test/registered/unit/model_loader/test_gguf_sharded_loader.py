"""Unit tests for sharded GGUF discovery in GGUFModelLoader._get_gguf_files.

A sharded GGUF checkpoint is split into ``<name>-00001-of-000NN.gguf`` files
kept side by side. Given any single shard, the loader must rebuild the full,
ordered set so that every tensor gets loaded; an unsharded file must be
returned unchanged.
"""

import os
import tempfile
import unittest

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import GGUFModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGGUFShardedDiscovery(CustomTestCase):
    def setUp(self):
        self.loader = GGUFModelLoader(LoadConfig(load_format="gguf"))

    def _touch(self, path):
        open(path, "w").close()
        return path

    def test_discovers_full_shard_set_from_any_shard(self):
        with tempfile.TemporaryDirectory() as d:
            # An unsharded path is returned unchanged.
            single = self._touch(os.path.join(d, "model.gguf"))
            self.assertEqual(self.loader._get_gguf_files(single), [single])

            shards = [
                self._touch(os.path.join(d, f"model-{i:05d}-of-00003.gguf"))
                for i in range(1, 4)
            ]
            # A sibling set with a different total, or one that merely shares a
            # prefix, must not be pulled in.
            self._touch(os.path.join(d, "model-00001-of-00005.gguf"))
            self._touch(os.path.join(d, "model-2-00001-of-00003.gguf"))

            # Any shard rebuilds exactly its own ordered set.
            self.assertEqual(self.loader._get_gguf_files(shards[0]), shards)
            self.assertEqual(self.loader._get_gguf_files(shards[2]), shards)

    def test_discovers_variable_width_padding(self):
        with tempfile.TemporaryDirectory() as d:
            shards = [
                self._touch(os.path.join(d, f"model-{i:02d}-of-03.gguf"))
                for i in range(1, 4)
            ]
            self.assertEqual(self.loader._get_gguf_files(shards[1]), shards)

    def test_rejects_incomplete_shard_set(self):
        with tempfile.TemporaryDirectory() as d:
            first = self._touch(os.path.join(d, "model-00001-of-00003.gguf"))
            self._touch(os.path.join(d, "model-00002-of-00003.gguf"))
            # third shard absent
            with self.assertRaises(ValueError):
                self.loader._get_gguf_files(first)


if __name__ == "__main__":
    unittest.main()
