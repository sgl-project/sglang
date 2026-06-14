"""Unit tests for tokenizer-level MIS item length validation."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import _validate_mis_item_lengths

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTokenizerManagerMISItemLength(CustomTestCase):
    def test_below_boundary_passes(self):
        _validate_mis_item_lengths([4, 65005], 65006)

    def test_at_boundary_passes(self):
        _validate_mis_item_lengths([4, 65540], 65541)

    def test_above_boundary_raises(self):
        with self.assertRaisesRegex(ValueError, "uint16 metadata limit"):
            _validate_mis_item_lengths([4, 70005], 70006)

    def test_long_tail_item_raises(self):
        with self.assertRaisesRegex(ValueError, "uint16 metadata limit"):
            _validate_mis_item_lengths([4], 70005)

    def test_multi_item_total_large_but_each_item_ok(self):
        _validate_mis_item_lengths(
            [
                4,
                7005,
                14006,
                21007,
                28008,
                35009,
                42010,
                49011,
                56012,
                63013,
                70014,
            ],
            70015,
        )

    def test_multi_item_with_one_long_item_raises(self):
        with self.assertRaisesRegex(ValueError, "uint16 metadata limit"):
            _validate_mis_item_lengths([4, 21, 70022, 70039], 70040)


if __name__ == "__main__":
    unittest.main()
