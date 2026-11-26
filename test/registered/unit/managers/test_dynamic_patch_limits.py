"""Request-supplied dynamic-tiling caps must stay below the server limit.

Dynamic tiling enumerates aspect-ratio candidates in a triple loop whose
cost grows quadratically with the requested patch cap, so an unbounded
``max_dynamic_patch`` lets one request burn unbounded CPU before the
expanded token count is ever checked against the context length.
"""

import unittest

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.multimodal.media_io import (
    MAX_DYNAMIC_PATCH_LIMIT,
    validate_dynamic_patch_limit,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DYNAMIC_PATCH_FIELDS = (
    "max_dynamic_patch",
    "min_dynamic_patch",
    "image_max_dynamic_patch",
    "video_max_dynamic_patch",
)


class TestValidateDynamicPatchLimit(CustomTestCase):
    def test_none_is_admitted(self):
        validate_dynamic_patch_limit(None, "max_dynamic_patch")

    def test_values_at_and_below_limit_are_admitted(self):
        for value in (1, 12, MAX_DYNAMIC_PATCH_LIMIT):
            with self.subTest(value=value):
                validate_dynamic_patch_limit(value, "max_dynamic_patch")

    def test_non_positive_values_are_left_to_the_processor_clamp(self):
        # The InternVL processor resolves caps with max(1, int(...)), so a
        # non-positive request value is inert rather than expensive.
        for value in (0, -5):
            with self.subTest(value=value):
                validate_dynamic_patch_limit(value, "max_dynamic_patch")

    def test_values_above_limit_are_rejected(self):
        for value in (MAX_DYNAMIC_PATCH_LIMIT + 1, 200, 999999):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "max_dynamic_patch.*exceeds the maximum"
                ):
                    validate_dynamic_patch_limit(value, "max_dynamic_patch")


class TestGenerateReqInputDynamicPatchLimits(CustomTestCase):
    def test_unset_fields_are_admitted(self):
        req = GenerateReqInput(text="hello")
        req.normalize_batch_and_arguments()

    def test_each_field_at_limit_is_admitted(self):
        for field_name in _DYNAMIC_PATCH_FIELDS:
            with self.subTest(field=field_name):
                req = GenerateReqInput(
                    text="hello", **{field_name: MAX_DYNAMIC_PATCH_LIMIT}
                )
                req.normalize_batch_and_arguments()

    def test_each_field_above_limit_is_rejected(self):
        for field_name in _DYNAMIC_PATCH_FIELDS:
            with self.subTest(field=field_name):
                req = GenerateReqInput(
                    text="hello", **{field_name: MAX_DYNAMIC_PATCH_LIMIT + 1}
                )
                with self.assertRaisesRegex(ValueError, field_name):
                    req.normalize_batch_and_arguments()

    def test_absurd_value_is_rejected(self):
        req = GenerateReqInput(text="hello", max_dynamic_patch=999999)
        with self.assertRaisesRegex(ValueError, "max_dynamic_patch"):
            req.normalize_batch_and_arguments()


if __name__ == "__main__":
    unittest.main()
