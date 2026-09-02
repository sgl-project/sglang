import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.utils import is_npu_arch35


class TestArch35Capability(unittest.TestCase):
    def tearDown(self):
        is_npu_arch35.cache_clear()

    def test_arch35_is_supported_from_acl_device_info(self):
        with (
            patch("sglang.srt.hardware_backend.npu.utils.is_npu", return_value=True),
            patch.dict(
                "sys.modules",
                {
                    "acl": SimpleNamespace(
                        rt=SimpleNamespace(get_device_info=lambda *_: (3510, 0))
                    )
                },
            ),
        ):
            self.assertTrue(is_npu_arch35())

    def test_non_arch35_npu_is_not_supported(self):
        with (
            patch("sglang.srt.hardware_backend.npu.utils.is_npu", return_value=True),
            patch.dict(
                "sys.modules",
                {
                    "acl": SimpleNamespace(
                        rt=SimpleNamespace(get_device_info=lambda *_: (2901, 0))
                    )
                },
            ),
        ):
            self.assertFalse(is_npu_arch35())

    @patch("sglang.srt.hardware_backend.npu.utils.is_npu", return_value=False)
    def test_non_npu_is_not_supported(self, _):
        self.assertFalse(is_npu_arch35())


if __name__ == "__main__":
    unittest.main()
