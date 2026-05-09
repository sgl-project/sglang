import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.utils.common import empty_device_cache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestEmptyDeviceCache(unittest.TestCase):
    def test_uses_explicit_device_module(self):
        device_module = MagicMock()

        self.assertTrue(empty_device_cache(device_module))

        device_module.empty_cache.assert_called_once_with()

    def test_returns_false_when_backend_has_no_empty_cache(self):
        device_module = object()

        self.assertFalse(empty_device_cache(device_module))

    def test_uses_current_device_module_by_default(self):
        device_module = MagicMock()

        with patch(
            "sglang.srt.utils.common.torch.get_device_module",
            return_value=device_module,
        ):
            self.assertTrue(empty_device_cache())

        device_module.empty_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
