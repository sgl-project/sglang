"""Unit tests for sharded GGUF discovery in GGUFModelLoader._get_gguf_files.

A sharded GGUF checkpoint is split into ``<name>-00001-of-000NN.gguf`` files
kept side by side. Given any single shard, the loader must discover the full,
sorted set; an unsharded file must be returned unchanged.
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
            # A shard set with a different total must not be mixed in.
            self._touch(os.path.join(d, "model-00001-of-00005.gguf"))

            # Given any shard, the full matching set is discovered, sorted.
            self.assertEqual(self.loader._get_gguf_files(shards[0]), sorted(shards))
            self.assertEqual(self.loader._get_gguf_files(shards[2]), sorted(shards))

    def test_rejects_incomplete_shard_set(self):
        with tempfile.TemporaryDirectory() as d:
            first = self._touch(os.path.join(d, "model-00001-of-00003.gguf"))
            self._touch(os.path.join(d, "model-00002-of-00003.gguf"))
            # third shard absent -> declared count does not match
            with self.assertRaises(ValueError):
                self.loader._get_gguf_files(first)


if __name__ == "__main__":
    unittest.main()
