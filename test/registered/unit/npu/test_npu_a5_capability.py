import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.utils import has_npu_a5_support


class TestNPUA5Capability(unittest.TestCase):
    def tearDown(self):
        has_npu_a5_support.cache_clear()

    def test_ascend950_is_supported(self):
        with (
            patch("sglang.srt.hardware_backend.npu.utils.is_npu", return_value=True),
            patch("torch_npu.npu.get_device_name", return_value="Ascend950_9599"),
        ):
            self.assertTrue(has_npu_a5_support())

    def test_pre_a5_npu_is_not_supported(self):
        with (
            patch("sglang.srt.hardware_backend.npu.utils.is_npu", return_value=True),
            patch("torch_npu.npu.get_device_name", return_value="Ascend910_9392"),
        ):
            self.assertFalse(has_npu_a5_support())

    @patch("sglang.srt.hardware_backend.npu.utils.is_npu", return_value=False)
    def test_non_npu_is_not_supported(self, _):
        self.assertFalse(has_npu_a5_support())


if __name__ == "__main__":
    unittest.main()
