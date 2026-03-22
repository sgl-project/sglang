"""Unit tests for schedule_batch.py finish reasons, enums, and pad utilities."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest

from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    FINISHED_MATCHED_REGEX,
    MM_PAD_SHIFT_VALUE,
    BaseFinishReason,
    Modality,
    _compute_pad_value,
    sanity_check_mm_pad_shift_value,
)


class TestBaseFinishReason(unittest.TestCase):

    def test_default_is_not_error(self):
        self.assertFalse(BaseFinishReason().is_error)

    def test_can_set_error(self):
        self.assertTrue(BaseFinishReason(is_error=True).is_error)


class TestFinishMatchedToken(unittest.TestCase):

    def test_to_json_with_int(self):
        result = FINISH_MATCHED_TOKEN(matched=42).to_json()
        self.assertEqual(result, {"type": "stop", "matched": 42})

    def test_to_json_with_list(self):
        result = FINISH_MATCHED_TOKEN(matched=[1, 2, 3]).to_json()
        self.assertEqual(result["matched"], [1, 2, 3])

    def test_not_an_error(self):
        self.assertFalse(FINISH_MATCHED_TOKEN(matched=0).is_error)


class TestFinishMatchedStr(unittest.TestCase):

    def test_to_json(self):
        result = FINISH_MATCHED_STR(matched="</s>").to_json()
        self.assertEqual(result, {"type": "stop", "matched": "</s>"})


class TestFinishedMatchedRegex(unittest.TestCase):

    def test_to_json(self):
        result = FINISHED_MATCHED_REGEX(matched="pattern.*").to_json()
        self.assertEqual(result, {"type": "stop", "matched": "pattern.*"})


class TestFinishLength(unittest.TestCase):

    def test_to_json(self):
        result = FINISH_LENGTH(length=100).to_json()
        self.assertEqual(result, {"type": "length", "length": 100})

    def test_not_an_error(self):
        self.assertFalse(FINISH_LENGTH(length=50).is_error)


class TestFinishAbort(unittest.TestCase):

    def test_default_message(self):
        result = FINISH_ABORT().to_json()
        self.assertEqual(result["type"], "abort")
        self.assertEqual(result["message"], "Aborted")
        self.assertIsNone(result["status_code"])
        self.assertIsNone(result["err_type"])

    def test_custom_fields(self):
        reason = FINISH_ABORT(message="OOM", status_code=500, err_type="InternalError")
        result = reason.to_json()
        self.assertEqual(result["message"], "OOM")
        self.assertEqual(result["status_code"], 500)
        self.assertEqual(result["err_type"], "InternalError")

    def test_is_always_error(self):
        self.assertTrue(FINISH_ABORT().is_error)


class TestModality(unittest.TestCase):

    def test_from_str_valid(self):
        self.assertEqual(Modality.from_str("image"), Modality.IMAGE)
        self.assertEqual(Modality.from_str("video"), Modality.VIDEO)
        self.assertEqual(Modality.from_str("audio"), Modality.AUDIO)

    def test_from_str_case_insensitive(self):
        self.assertEqual(Modality.from_str("IMAGE"), Modality.IMAGE)
        self.assertEqual(Modality.from_str("Video"), Modality.VIDEO)

    def test_from_str_invalid_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Modality.from_str("text")
        self.assertIn("text", str(ctx.exception))

    def test_all_returns_image_video_audio(self):
        result = Modality.all()
        self.assertIn(Modality.IMAGE, result)
        self.assertIn(Modality.VIDEO, result)
        self.assertIn(Modality.AUDIO, result)
        self.assertEqual(len(result), 3)


class TestComputePadValue(unittest.TestCase):

    def test_deterministic(self):
        self.assertEqual(_compute_pad_value(12345), _compute_pad_value(12345))

    def test_different_hashes_differ(self):
        self.assertNotEqual(_compute_pad_value(1), _compute_pad_value(2))

    def test_result_above_mm_pad_shift(self):
        self.assertGreaterEqual(_compute_pad_value(0), MM_PAD_SHIFT_VALUE)


class TestSanityCheckMmPadShiftValue(unittest.TestCase):

    def test_valid_vocab_size_passes(self):
        sanity_check_mm_pad_shift_value.cache_clear()
        sanity_check_mm_pad_shift_value(32000)

    def test_oversized_vocab_raises(self):
        sanity_check_mm_pad_shift_value.cache_clear()
        with self.assertRaises(ValueError) as ctx:
            sanity_check_mm_pad_shift_value(MM_PAD_SHIFT_VALUE + 1)
        self.assertIn("MM_PAD_SHIFT_VALUE", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
