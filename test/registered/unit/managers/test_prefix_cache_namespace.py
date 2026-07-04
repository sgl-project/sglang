import unittest

from sglang.srt.managers.prefix_cache_namespace import compose_prefix_cache_extra_key


class TestPrefixCacheNamespace(unittest.TestCase):
    def test_preserves_existing_namespace_without_lora(self):
        self.assertIsNone(compose_prefix_cache_extra_key(None, None))
        self.assertEqual(compose_prefix_cache_extra_key("tenant", None), "tenant")

    def test_adds_lora_namespace(self):
        self.assertEqual(
            compose_prefix_cache_extra_key(None, "adapter"),
            "lora_id:7:adapter",
        )
        self.assertEqual(
            compose_prefix_cache_extra_key("tenant", "adapter"),
            "extra_key:6:tenant|lora_id:7:adapter",
        )

    def test_avoids_string_concatenation_collisions(self):
        self.assertNotEqual(
            compose_prefix_cache_extra_key("ab", "c"),
            compose_prefix_cache_extra_key("a", "bc"),
        )


if __name__ == "__main__":
    unittest.main()
