import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.mem_cache.storage.mooncake_store.legacy_fallback import (
    LegacyCacheFallback,
    validate_legacy_fallback,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLegacyCacheFallback(unittest.TestCase):
    def setUp(self):
        self.primary = Mock()
        self.legacy = Mock()
        self.primary.put.return_value = 0
        self.fallback = LegacyCacheFallback(self.primary, self.legacy)
        self.buffer = ctypes.create_string_buffer(b"xxxx", 4)
        self.ptr = ctypes.addressof(self.buffer)

    def read(self, result=-1):
        return self.fallback.read(["key"], [self.ptr], [4], [result])

    def test_disabled_and_format_validation(self):
        cfg = SimpleNamespace(kv_cache_dtype="bfloat16")
        pool = SimpleNamespace(layout="page_first", page_size=16)
        options = dict(
            kv_cache_dtype="bfloat16",
            layout="page_first",
            page_size=16,
            verified_compatible=True,
        )
        self.assertFalse(validate_legacy_fallback(None, cfg, pool))
        self.assertTrue(validate_legacy_fallback(options, cfg, pool))
        for key, value in (
            ("kv_cache_dtype", "float16"),
            ("layout", "layer_first"),
            ("page_size", 32),
            ("page_size", True),
            ("verified_compatible", False),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_legacy_fallback({**options, key: value}, cfg, pool)
        for invalid in (True, {}, {**options, "typo": True}):
            with self.assertRaises(ValueError):
                validate_legacy_fallback(invalid, cfg, pool)
        with self.assertRaises(ValueError):
            validate_legacy_fallback(options, SimpleNamespace(), pool)

    def test_primary_hit_does_not_touch_legacy(self):
        self.assertEqual(self.read(4), [4])
        self.assertEqual(self.fallback.exists(["key"], [1]), [1])
        self.assertEqual(self.legacy.mock_calls, [])
        self.assertEqual(self.primary.mock_calls, [])

    def test_only_confirmed_missing_objects_fall_back(self):
        for status in (1, -1):
            self.primary.batch_is_exist.return_value = [status]
            self.assertEqual(self.read(), [-1])
        self.legacy.get.assert_not_called()
        self.primary.batch_is_exist.side_effect = RuntimeError("unavailable")
        with self.assertLogs(level="WARNING"):
            self.assertEqual(self.read(), [-1])
        self.legacy.get.assert_not_called()

    def test_restore_and_promote(self):
        self.primary.batch_is_exist.return_value = [0]
        self.legacy.get.return_value = b"abcd"
        self.assertEqual(self.read(), [4])
        self.assertEqual(self.buffer.raw, b"abcd")
        self.primary.put.assert_called_once_with("key", b"abcd")
        self.legacy.put.assert_not_called()

    def test_missing_or_wrong_length_never_copies_or_promotes(self):
        self.primary.batch_is_exist.return_value = [0]
        for value in (None, b"", b"abc", b"abcde"):
            self.legacy.get.return_value = value
            self.assertEqual(self.read(), [-1])
            self.assertEqual(self.buffer.raw, b"xxxx")
        self.primary.put.assert_not_called()

    def test_legacy_evicted_after_exists_is_a_miss(self):
        self.legacy.batch_is_exist.return_value = [1]
        self.assertEqual(self.fallback.exists(["key"], [0]), [1])
        self.primary.batch_is_exist.return_value = [0]
        self.legacy.get.return_value = None
        self.assertEqual(self.read(), [-1])

    def test_legacy_error_is_a_miss(self):
        self.primary.batch_is_exist.return_value = [0]
        self.legacy.get.side_effect = RuntimeError("unavailable")
        with self.assertLogs(level="WARNING"):
            self.assertEqual(self.read(), [-1])
        self.assertEqual(self.buffer.raw, b"xxxx")

    def test_promotion_failure_preserves_successful_read(self):
        self.primary.batch_is_exist.return_value = [0]
        self.legacy.get.return_value = b"abcd"
        for failure in (-1, RuntimeError("quota")):
            self.primary.put.side_effect = (
                failure if isinstance(failure, Exception) else None
            )
            self.primary.put.return_value = failure
            with self.assertLogs(level="WARNING"):
                self.assertEqual(self.read(), [4])
            self.assertEqual(self.buffer.raw, b"abcd")

    def test_multibuffer_scatter(self):
        a, b = ctypes.create_string_buffer(2), ctypes.create_string_buffer(2)
        self.primary.batch_is_exist.return_value = [0]
        self.legacy.get.return_value = b"abcd"
        result = self.fallback.read(
            ["key"], [[ctypes.addressof(a), ctypes.addressof(b)]], [[2, 2]], [-1]
        )
        self.assertEqual(result, [4])
        self.assertEqual((a.raw, b.raw), (b"ab", b"cd"))

    def test_mixed_batch_preserves_order_and_errors(self):
        self.legacy.batch_is_exist.return_value = [1, 0]
        self.assertEqual(
            self.fallback.exists(["a", "b", "c", "d"], [1, 0, -1, 0]), [1, 1, -1, 0]
        )
        self.legacy.batch_is_exist.assert_called_once_with(["b", "d"])
        self.primary.batch_is_exist.return_value = [0, 1]
        self.legacy.get.return_value = b"abcd"
        self.assertEqual(
            self.fallback.read(
                ["a", "b", "c"], [0, self.ptr, 0], [4, 4, 4], [4, -1, -1]
            ),
            [4, 4, -1],
        )
        self.legacy.get.assert_called_once_with("b")

    def test_legacy_existence_error_preserves_primary_results(self):
        self.legacy.batch_is_exist.side_effect = RuntimeError("unavailable")
        with self.assertLogs(level="WARNING"):
            self.assertEqual(self.fallback.exists(["key"], [0]), [0])


if __name__ == "__main__":
    unittest.main()
