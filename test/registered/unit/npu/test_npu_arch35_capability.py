import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.utils import is_npu_a3, is_npu_arch35


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


class TestA3Capability(unittest.TestCase):
    def tearDown(self):
        is_npu_a3.cache_clear()

    def test_a3_93_series_is_supported(self):
        for name in ("Ascend910_9362", "Ascend910_9392"):
            with (
                self.subTest(name=name),
                patch(
                    "sglang.srt.hardware_backend.npu.utils.is_npu", return_value=True
                ),
                patch(
                    "sglang.srt.hardware_backend.npu.utils.torch",
                    SimpleNamespace(npu=SimpleNamespace(get_device_name=lambda: name)),
                ),
            ):
                is_npu_a3.cache_clear()
                self.assertTrue(is_npu_a3())

    def test_other_npu_series_do_not_enable_a3(self):
        for name in ("Ascend910B2", "Ascend950PR_9599", "Ascend910"):
            with (
                self.subTest(name=name),
                patch(
                    "sglang.srt.hardware_backend.npu.utils.is_npu", return_value=True
                ),
                patch(
                    "sglang.srt.hardware_backend.npu.utils.torch",
                    SimpleNamespace(npu=SimpleNamespace(get_device_name=lambda: name)),
                ),
            ):
                is_npu_a3.cache_clear()
                self.assertFalse(is_npu_a3())

    def test_non_npu_does_not_query_device_name(self):
        with (
            patch("sglang.srt.hardware_backend.npu.utils.is_npu", return_value=False),
            patch("sglang.srt.hardware_backend.npu.utils.torch", SimpleNamespace()),
        ):
            self.assertFalse(is_npu_a3())


if __name__ == "__main__":
    unittest.main()
