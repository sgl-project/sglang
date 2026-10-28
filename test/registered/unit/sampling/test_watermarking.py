"""Unit tests for srt/sampling/watermarking — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import json
import math
import os
import tempfile
import unittest

from sglang.srt.sampling.watermarking.config import (
    MAX_WATERMARK_CONTEXT_WINDOW,
    WatermarkConfigError,
    WatermarkServerConfig,
    load_watermark_config,
    parse_watermark_key,
)
from sglang.srt.sampling.watermarking.detector import (
    WatermarkDetection,
    WatermarkDetector,
    WatermarkStatistics,
    detect,
    hash_context,
    watermark_hash,
)
from sglang.test.test_utils import CustomTestCase


# ---------------------------------------------------------------------------
# parse_watermark_key
# ---------------------------------------------------------------------------


class TestParseWatermarkKey(CustomTestCase):
    def test_hex_string_with_prefix(self):
        # 0xDEADBEEF -> 3735928559 (fits in 63 bits, no sign flip)
        self.assertEqual(parse_watermark_key("0xDEADBEEF"), 3735928559)

    def test_hex_string_without_prefix(self):
        self.assertEqual(parse_watermark_key("DEADBEEF"), 3735928559)

    def test_case_insensitive(self):
        self.assertEqual(parse_watermark_key("deadbeef"), parse_watermark_key("DEADBEEF"))

    def test_single_digit(self):
        self.assertEqual(parse_watermark_key("0x1"), 1)

    def test_max_key_wraps_to_negative(self):
        # All-F (>= 2**63) wraps to -1 via two's-complement reinterpretation.
        self.assertEqual(parse_watermark_key("FFFFFFFFFFFFFFFF"), -1)

    def test_rejects_non_string(self):
        for bad in (123, None, 3.14, []):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    parse_watermark_key(bad)

    def test_rejects_empty_digits(self):
        with self.assertRaisesRegex(ValueError, "1 to 16 hex digits"):
            parse_watermark_key("0x")

    def test_rejects_too_long(self):
        with self.assertRaisesRegex(ValueError, "1 to 16 hex digits"):
            parse_watermark_key("0x" + "F" * 17)

    def test_rejects_non_hex_characters(self):
        with self.assertRaisesRegex(ValueError, "hex digits"):
            parse_watermark_key("ZZZZZZZZ")

    def test_rejects_hex_with_non_hex_suffix(self):
        with self.assertRaisesRegex(ValueError, "hex digits"):
            parse_watermark_key("DEAD_BEEF")


# ---------------------------------------------------------------------------
# load_watermark_config — inline JSON path
# ---------------------------------------------------------------------------


class TestLoadWatermarkConfigInline(CustomTestCase):
    def test_minimal_empty_object(self):
        cfg = load_watermark_config("{}")
        self.assertIsInstance(cfg, WatermarkServerConfig)
        self.assertIsNone(cfg.key)
        self.assertIsNone(cfg.context_window)

    def test_full_valid_config(self):
        payload = json.dumps(
            {
                "key": "0xDEADBEEF",
                "context_window": 4,
                "mixing_probability": 0.5,
                "max_probability": 0.9,
                "default_enabled": True,
                "enforce_all": False,
            }
        )
        cfg = load_watermark_config(payload)
        self.assertEqual(cfg.key, "0xDEADBEEF")
        self.assertEqual(cfg.context_window, 4)
        self.assertEqual(cfg.mixing_probability, 0.5)
        self.assertTrue(cfg.default_enabled)
        self.assertFalse(cfg.enforce_all)

    def test_context_window_boundary_values(self):
        load_watermark_config(json.dumps({"context_window": 1}))
        load_watermark_config(json.dumps({"context_window": MAX_WATERMARK_CONTEXT_WINDOW}))

    def test_rejects_context_window_zero(self):
        with self.assertRaises(WatermarkConfigError):
            load_watermark_config(json.dumps({"context_window": 0}))

    def test_rejects_context_window_above_max(self):
        with self.assertRaises(WatermarkConfigError):
            load_watermark_config(
                json.dumps({"context_window": MAX_WATERMARK_CONTEXT_WINDOW + 1})
            )

    def test_rejects_context_window_boolean(self):
        # bool is a subclass of int in Python; the validator must reject it explicitly.
        with self.assertRaises(WatermarkConfigError):
            load_watermark_config(json.dumps({"context_window": True}))

    def test_rejects_non_dict_json(self):
        for payload in ('["a", "b"]', '"just a string"', "42"):
            with self.subTest(payload=payload):
                with self.assertRaises(WatermarkConfigError):
                    load_watermark_config(payload)

    def test_rejects_unknown_fields(self):
        with self.assertRaises(WatermarkConfigError):
            load_watermark_config(json.dumps({"unknown_field": 1}))

    def test_rejects_invalid_json(self):
        with self.assertRaises(WatermarkConfigError):
            load_watermark_config("{bad json")

    def test_rejects_probability_as_boolean(self):
        # bool subclass guard: mixing_probability must be a real number, not True/False.
        with self.assertRaises(WatermarkConfigError):
            load_watermark_config(json.dumps({"mixing_probability": True}))

    def test_rejects_default_enabled_as_non_bool(self):
        with self.assertRaises(WatermarkConfigError):
            load_watermark_config(json.dumps({"default_enabled": 1}))


# ---------------------------------------------------------------------------
# load_watermark_config — file path
# ---------------------------------------------------------------------------


class TestLoadWatermarkConfigFile(CustomTestCase):
    def test_reads_valid_config_from_file(self):
        payload = json.dumps({"context_window": 8})
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        try:
            cfg = load_watermark_config(tmp_path)
            self.assertEqual(cfg.context_window, 8)
        finally:
            os.unlink(tmp_path)

    def test_rejects_missing_file(self):
        with self.assertRaises(WatermarkConfigError):
            load_watermark_config("/nonexistent/path/watermark.json")

    def test_rejects_oversized_config(self):
        # A payload larger than 4096 bytes must be rejected.
        oversized = json.dumps({"key": "A" * 4097})
        with self.assertRaises(WatermarkConfigError):
            load_watermark_config(oversized)

    def test_rejects_directory_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(WatermarkConfigError):
                load_watermark_config(tmpdir)


# ---------------------------------------------------------------------------
# WatermarkServerConfig repr redacts keys
# ---------------------------------------------------------------------------


class TestWatermarkServerConfigRepr(CustomTestCase):
    def test_repr_redacts_key(self):
        cfg = WatermarkServerConfig(key="0xDEADBEEF", context_window=4)
        rep = repr(cfg)
        self.assertIn("<redacted>", rep)
        self.assertNotIn("DEADBEEF", rep)
        self.assertIn("context_window=4", rep)


# ---------------------------------------------------------------------------
# hash_context — determinism and collision
# ---------------------------------------------------------------------------


class TestHashContext(CustomTestCase):
    # Derived property: the murmur3 hash of fixed inputs must be stable
    # across Python versions and refactors of the implementation.

    def test_known_hash_is_stable(self):
        # This value was computed from the reference implementation and pins
        # the hash output against future refactors that could silently change
        # the watermark detection result for previously-generated text.
        self.assertEqual(hash_context([1, 2, 3]), 3923217110)

    def test_empty_context_is_zero(self):
        self.assertEqual(hash_context([]), 0)

    def test_deterministic(self):
        self.assertEqual(hash_context([10, 20, 30]), hash_context([10, 20, 30]))

    def test_order_matters(self):
        self.assertNotEqual(hash_context([1, 2]), hash_context([2, 1]))

    def test_token_id_truncated_to_uint32(self):
        # Token IDs are masked to 32 bits before hashing.
        self.assertEqual(hash_context([2**32 + 1]), hash_context([1]))

    def test_result_is_uint32(self):
        h = hash_context([42, 43, 44])
        self.assertGreaterEqual(h, 0)
        self.assertLessEqual(h, 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# watermark_hash — determinism
# ---------------------------------------------------------------------------


class TestWatermarkHash(CustomTestCase):
    def test_known_hash_is_stable(self):
        key = 0xDEADBEEF
        ctx = hash_context([1, 2, 3])
        self.assertEqual(watermark_hash(key, ctx, 42), 4099755571)

    def test_result_is_uint32(self):
        h = watermark_hash(0xCAFEBABE, 0x12345678, 99)
        self.assertGreaterEqual(h, 0)
        self.assertLessEqual(h, 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# WatermarkStatistics.from_scores
# ---------------------------------------------------------------------------


class TestWatermarkStatistics(CustomTestCase):
    def test_empty_scores_returns_null_hypothesis(self):
        stats = WatermarkStatistics.from_scores([])
        self.assertEqual(stats.num_contexts, 0)
        self.assertEqual(stats.score, 0.0)
        self.assertEqual(stats.z_score, 0.0)
        self.assertAlmostEqual(stats.p_value, 1.0)
        self.assertEqual(stats.log_p_value, 0.0)

    def test_log_p_value_equals_log_of_p_value(self):
        # Derived property: log_p_value must stay consistent with p_value
        # as long as p_value has not underflowed.
        scores = [0.5, 1.0, 1.5, 2.0]
        stats = WatermarkStatistics.from_scores(scores)
        if stats.p_value > 0:
            self.assertAlmostEqual(stats.log_p_value, math.log(stats.p_value), places=10)

    def test_p_value_in_unit_interval(self):
        for scores in ([0.1], [1.0, 2.0], [0.01, 0.01, 0.01]):
            with self.subTest(scores=scores):
                stats = WatermarkStatistics.from_scores(scores)
                self.assertGreaterEqual(stats.p_value, 0.0)
                self.assertLessEqual(stats.p_value, 1.0)

    def test_z_score_formula(self):
        # z = (sum - n) / sqrt(n)
        scores = [1.0, 1.0, 1.0, 1.0]
        stats = WatermarkStatistics.from_scores(scores)
        self.assertAlmostEqual(stats.z_score, 0.0)

    def test_num_contexts_matches_score_count(self):
        scores = [0.3, 0.7, 1.1]
        stats = WatermarkStatistics.from_scores(scores)
        self.assertEqual(stats.num_contexts, 3)


# ---------------------------------------------------------------------------
# WatermarkDetector — basic contract
# ---------------------------------------------------------------------------


class TestWatermarkDetectorInit(CustomTestCase):
    def test_rejects_invalid_mixing_probability(self):
        for bad in (0.0, 1.0, -0.1, 1.5, True, "0.5"):
            with self.subTest(value=bad):
                with self.assertRaises((ValueError, TypeError)):
                    WatermarkDetector("0xDEADBEEF", mixing_probability=bad)

    def test_rejects_invalid_context_window(self):
        for bad in (0, MAX_WATERMARK_CONTEXT_WINDOW + 1, -1, True, 1.5):
            with self.subTest(value=bad):
                with self.assertRaises((ValueError, TypeError)):
                    WatermarkDetector("0xDEADBEEF", context_window=bad)


class TestWatermarkDetectorTokens(CustomTestCase):
    KEY = "0xDEADBEEF"

    def _detector(self, **kwargs):
        return WatermarkDetector(self.KEY, **kwargs)

    def test_empty_completion_returns_zero_contexts(self):
        det = self._detector(context_window=4)
        result = det.detect_tokens([])
        self.assertEqual(result.key_a_all_positions.num_contexts, 0)
        self.assertFalse(result.prefix_limit_reached)

    def test_short_completion_skips_initial_tokens_without_prompt(self):
        # Without prompt_token_ids, the first context_window tokens are skipped
        # because their context would include unknown prompt tokens.
        det = self._detector(context_window=4)
        result = det.detect_tokens([1, 2, 3])
        # Only 3 tokens, context_window=4, so all are skipped
        self.assertEqual(result.key_a_all_positions.num_contexts, 0)

    def test_detect_tokens_returns_detection_object(self):
        det = self._detector(context_window=2)
        result = det.detect_tokens([100, 200, 300, 400, 500])
        self.assertIsInstance(result, WatermarkDetection)
        self.assertIsInstance(result.key_a_all_positions, WatermarkStatistics)

    def test_known_context_count(self):
        # [100,200,300,400,500], cw=2, no prompt ->
        #   first = min(2,5)=2 => positions 2,3,4 each have unique context
        det = self._detector(context_window=2)
        result = det.detect_tokens([100, 200, 300, 400, 500])
        self.assertEqual(result.key_a_all_positions.num_contexts, 3)

    def test_repeated_context_is_deduped(self):
        # [10,10,10,20] with cw=1: only context (10,) is unique, positions 1,2,3
        # but (10,) seen at pos=1, repeated at pos=2; (10,) repeated at pos=3? No:
        # pos=1 ctx=(10,)token=10, pos=2 ctx=(10,)token=10 repeated,
        # pos=3 ctx=(10,)token=20 -- (10,) already in seen -> repeated
        det = self._detector(context_window=1)
        result = det.detect_tokens([10, 10, 10, 20])
        self.assertEqual(result.key_a_all_positions.num_contexts, 1)
        self.assertEqual(result.repeated_contexts, 2)

    def test_prompt_shifts_context_window(self):
        # Providing prompt_token_ids changes the context of the first token.
        # Without prompt: first completion token has context drawn from completion.
        # With prompt:    first completion token has context drawn from prompt tail.
        # The score for the first token therefore differs.
        det = self._detector(context_window=2)
        result_no_prompt = det.detect_tokens([3, 4, 5])
        result_with_prompt = det.detect_tokens([3, 4, 5], prompt_token_ids=[1, 2])
        # With prompt, start=2 (min(len([1,2]),2)) and first=2, so we get
        # 3 contexts instead of 1 (min(2,3)=2 skipped without prompt).
        self.assertGreater(
            result_with_prompt.key_a_all_positions.num_contexts,
            result_no_prompt.key_a_all_positions.num_contexts,
        )

    def test_no_key_b_stats_are_none(self):
        det = self._detector(context_window=2)
        result = det.detect_tokens([10, 20, 30, 40])
        self.assertIsNone(result.key_b_all_positions)
        self.assertIsNone(result.key_a_partition)
        self.assertIsNone(result.key_b_partition)

    def test_dual_key_stats_are_populated(self):
        det = WatermarkDetector(self.KEY, key_b="0xCAFEBABE", context_window=2)
        result = det.detect_tokens([10, 20, 30, 40])
        self.assertIsNotNone(result.key_b_all_positions)
        self.assertIsNotNone(result.key_a_partition)
        self.assertIsNotNone(result.key_b_partition)
        # combined must draw from partition_a + partition_b
        n_combined = result.combined.num_contexts
        n_a = result.key_a_partition.num_contexts
        n_b = result.key_b_partition.num_contexts
        self.assertEqual(n_combined, n_a + n_b)

    def test_prefix_limit_reached_flag(self):
        # max_contexts=2 with 5 unique contexts => limit is reached
        det = WatermarkDetector(self.KEY, context_window=1, max_contexts=2)
        result = det.detect_tokens([1, 2, 3, 4, 5])
        self.assertTrue(result.prefix_limit_reached)
        self.assertLessEqual(result.key_a_all_positions.num_contexts, 2)

    def test_invalid_token_ids_raise_value_error(self):
        det = self._detector()
        with self.assertRaises(ValueError):
            det.detect_tokens([-1])
        with self.assertRaises(ValueError):
            det.detect_tokens([2**32 + 1])
        with self.assertRaises(ValueError):
            det.detect_tokens([True])

    def test_tokens_examined_accounts_for_prompt_skip(self):
        # Without prompt: first min(cw,len) tokens are skipped.
        # [1,2,3,4,5], cw=2 -> first=2, examined counts positions 2..4 -> 3
        det = self._detector(context_window=2)
        result = det.detect_tokens([1, 2, 3, 4, 5])
        self.assertEqual(result.tokens_examined, 3)


# ---------------------------------------------------------------------------
# detect() convenience wrapper
# ---------------------------------------------------------------------------


class TestDetectFunction(CustomTestCase):
    def test_matches_detector_class(self):
        tokens = [10, 20, 30, 40, 50]
        key = "0xDEADBEEF"
        via_fn = detect(tokens, key=key, context_window=3)
        via_cls = WatermarkDetector(key, context_window=3).detect_tokens(tokens)
        self.assertEqual(
            via_fn.key_a_all_positions.num_contexts,
            via_cls.key_a_all_positions.num_contexts,
        )
        self.assertAlmostEqual(
            via_fn.key_a_all_positions.score,
            via_cls.key_a_all_positions.score,
        )


if __name__ == "__main__":
    unittest.main()
