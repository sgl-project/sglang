"""Unit tests for EmbeddingStore, EmbeddingStoreFactory."""

import unittest

from sglang.srt.mem_cache.embedding_store import EmbeddingStoreFactory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEmbeddingStoreFactory(unittest.TestCase):
    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            EmbeddingStoreFactory.create_backend("unknown_backend")

    def test_mooncake_registered(self):
        self.assertIn("mooncake", EmbeddingStoreFactory._registry)


if __name__ == "__main__":
    unittest.main()
