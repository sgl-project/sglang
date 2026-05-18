import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
PREFIX_CACHE_KEY_PATH = (
    REPO_ROOT / "python" / "sglang" / "srt" / "managers" / "prefix_cache_key.py"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

prefix_cache_key = _load_module("prefix_cache_key", PREFIX_CACHE_KEY_PATH)
build_prefix_cache_extra_key = prefix_cache_key.build_prefix_cache_extra_key
encode_prefix_cache_key_parts = prefix_cache_key.encode_prefix_cache_key_parts
escape_prefix_cache_user_key = prefix_cache_key.escape_prefix_cache_user_key
STRUCTURED_PREFIX_CACHE_KEY_PREFIX = prefix_cache_key.STRUCTURED_PREFIX_CACHE_KEY_PREFIX
ESCAPED_PREFIX_CACHE_KEY_PREFIX = prefix_cache_key.ESCAPED_PREFIX_CACHE_KEY_PREFIX


class TestPrefixCacheKey(unittest.TestCase):
    def test_no_extra_key_and_no_lora_stays_none(self):
        self.assertIsNone(build_prefix_cache_extra_key(None, None, None))

    def test_base_extra_key_is_preserved_without_lora(self):
        self.assertEqual(
            build_prefix_cache_extra_key(None, "adapter-a", None), "adapter-a"
        )

    def test_base_cache_salt_is_preserved_without_extra_key_or_lora(self):
        self.assertEqual(build_prefix_cache_extra_key("salt", None, None), "salt")

    def test_reserved_base_extra_key_is_escaped_without_lora(self):
        user_key = STRUCTURED_PREFIX_CACHE_KEY_PREFIX + '[["lora_id","adapter-a"]]'

        self.assertEqual(
            build_prefix_cache_extra_key(None, user_key, None),
            ESCAPED_PREFIX_CACHE_KEY_PREFIX + user_key,
        )

    def test_lora_id_is_namespaced_without_extra_key(self):
        self.assertEqual(
            build_prefix_cache_extra_key(None, None, "adapter-a"),
            encode_prefix_cache_key_parts([("lora_id", "adapter-a")]),
        )

    def test_lora_id_does_not_collide_with_base_extra_key(self):
        base_key = build_prefix_cache_extra_key(None, "adapter-a", None)
        lora_key = build_prefix_cache_extra_key(None, None, "adapter-a")

        self.assertNotEqual(base_key, lora_key)

    def test_encoded_lora_namespace_does_not_collide_with_base_extra_key(self):
        lora_key = build_prefix_cache_extra_key(None, None, "adapter-a")
        base_key = build_prefix_cache_extra_key(None, lora_key, None)

        self.assertNotEqual(base_key, lora_key)

    def test_extra_key_and_lora_id_boundaries_are_preserved(self):
        first = build_prefix_cache_extra_key(None, "ab", "c")
        second = build_prefix_cache_extra_key(None, "a", "bc")

        self.assertNotEqual(first, second)

    def test_cache_salt_and_extra_key_boundaries_are_preserved(self):
        first = build_prefix_cache_extra_key("ab", "c", None)
        second = build_prefix_cache_extra_key("a", "bc", None)

        self.assertNotEqual(first, second)

    def test_cache_salt_namespace_does_not_alias_crafted_extra_key(self):
        composed = build_prefix_cache_extra_key("salt", "key", None)
        crafted = build_prefix_cache_extra_key(None, composed, None)

        self.assertNotEqual(composed, crafted)

    def test_cache_salt_lora_namespace_does_not_alias_crafted_extra_key(self):
        composed = build_prefix_cache_extra_key("salt", "key", "adapter-a")
        crafted = build_prefix_cache_extra_key(None, composed, "adapter-a")

        self.assertNotEqual(composed, crafted)

    def test_escaped_user_key_cannot_alias_escaped_internal_key(self):
        internal_key = encode_prefix_cache_key_parts([("lora_id", "adapter-a")])
        escaped_internal_key = escape_prefix_cache_user_key(internal_key)

        self.assertNotEqual(
            escape_prefix_cache_user_key(escaped_internal_key),
            escaped_internal_key,
        )

    def test_type_error_identifies_component_name(self):
        with self.assertRaisesRegex(TypeError, "Prefix-cache namespace part lora_id"):
            encode_prefix_cache_key_parts([("lora_id", ["not", "a", "str"])])
        with self.assertRaisesRegex(TypeError, "Prefix-cache namespace part extra_key"):
            build_prefix_cache_extra_key(None, ["not", "a", "str"], "adapter-a")
        with self.assertRaisesRegex(
            TypeError, "Prefix-cache namespace part cache_salt"
        ):
            build_prefix_cache_extra_key(["not", "a", "str"], None, "adapter-a")

    def test_no_parts_stays_none(self):
        self.assertIsNone(encode_prefix_cache_key_parts([]))
        self.assertIsNone(
            encode_prefix_cache_key_parts([("extra_key", None), ("lora_id", None)])
        )


if __name__ == "__main__":
    unittest.main()
