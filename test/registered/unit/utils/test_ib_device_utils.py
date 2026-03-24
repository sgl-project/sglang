import json
import os
import tempfile
import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.utils import get_ib_devices_for_gpu

MOCK_SYSFS = "/sys/class/infiniband"
AVAILABLE_DEVICES = {"mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3", "mlx5_4", "mlx5_5", "mlx5_6", "mlx5_7"}


def _mock_sysfs(isdir_return=True, listdir_return=AVAILABLE_DEVICES):
    """Return decorators that mock sysfs for IB device validation."""

    def decorator(func):
        @patch("os.listdir", return_value=listdir_return)
        @patch("os.path.isdir", return_value=isdir_return)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


class TestGetIbDevicesForGpu(unittest.TestCase):
    # --- None / empty input ---

    def test_none_returns_none(self):
        self.assertIsNone(get_ib_devices_for_gpu(None, 0))

    def test_whitespace_returns_none(self):
        self.assertIsNone(get_ib_devices_for_gpu("  ", 0))

    # --- Comma-separated format ---

    @_mock_sysfs()
    def test_single_device(self, _isdir, _listdir):
        self.assertEqual(get_ib_devices_for_gpu("mlx5_0", 0), "mlx5_0")

    @_mock_sysfs()
    def test_csv_with_whitespace(self, _isdir, _listdir):
        result = get_ib_devices_for_gpu("  mlx5_0 , mlx5_1  ", 0)
        self.assertEqual(result, "mlx5_0,mlx5_1")

    # --- JSON GPU mapping format ---

    @_mock_sysfs()
    def test_json_mapping_shared_devices_across_gpus(self, _isdir, _listdir):
        """Regression: commas inside JSON values must not cause false duplicate detection."""
        mapping = json.dumps({
            "0": "mlx5_0,mlx5_1", "1": "mlx5_0,mlx5_1",
            "2": "mlx5_2,mlx5_3", "3": "mlx5_2,mlx5_3",
            "4": "mlx5_4,mlx5_5", "5": "mlx5_4,mlx5_5",
            "6": "mlx5_6,mlx5_7", "7": "mlx5_6,mlx5_7",
        })
        for gpu_id in range(8):
            result = get_ib_devices_for_gpu(mapping, gpu_id)
            self.assertIsNotNone(result)
        self.assertEqual(get_ib_devices_for_gpu(mapping, 0), "mlx5_0,mlx5_1")
        self.assertEqual(get_ib_devices_for_gpu(mapping, 7), "mlx5_6,mlx5_7")
        with self.assertRaises(ValueError, msg="No IB devices configured for GPU 8"):
            get_ib_devices_for_gpu(mapping, 8)
        # Duplicates *within* a single GPU must still be rejected
        bad_mapping = json.dumps({"0": "mlx5_0,mlx5_0"})
        with self.assertRaises(ValueError, msg="Duplicate"):
            get_ib_devices_for_gpu(bad_mapping, 0)

    @_mock_sysfs()
    def test_json_mapping_invalid_value_type(self, _isdir, _listdir):
        mapping = json.dumps({"0": ["mlx5_0"]})
        with self.assertRaises(ValueError):
            get_ib_devices_for_gpu(mapping, 0)

    # --- JSON file format ---

    @_mock_sysfs()
    def test_json_file(self, _isdir, _listdir):
        mapping = {"0": "mlx5_0,mlx5_1", "1": "mlx5_2,mlx5_3"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(mapping, f)
            f.flush()
            path = f.name
        try:
            self.assertEqual(get_ib_devices_for_gpu(path, 0), "mlx5_0,mlx5_1")
            self.assertEqual(get_ib_devices_for_gpu(path, 1), "mlx5_2,mlx5_3")
        finally:
            os.unlink(path)

    def test_json_file_not_found(self):
        with self.assertRaises(RuntimeError, msg="does not exist"):
            get_ib_devices_for_gpu("/nonexistent/path.json", 0)

    @_mock_sysfs()
    def test_json_file_invalid_content(self, _isdir, _listdir):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json{{{")
            f.flush()
            path = f.name
        try:
            with self.assertRaises(RuntimeError, msg="Failed to parse JSON"):
                get_ib_devices_for_gpu(path, 0)
        finally:
            os.unlink(path)

    # --- Validation errors ---

    @_mock_sysfs()
    def test_duplicate_devices(self, _isdir, _listdir):
        with self.assertRaises(ValueError, msg="Duplicate"):
            get_ib_devices_for_gpu("mlx5_0,mlx5_0", 0)

    @_mock_sysfs()
    def test_invalid_device_name(self, _isdir, _listdir):
        with self.assertRaises(ValueError, msg="Invalid IB devices"):
            get_ib_devices_for_gpu("mlx5_99", 0)

    @_mock_sysfs(isdir_return=False)
    def test_no_sysfs_dir(self, _isdir, _listdir):
        with self.assertRaises(RuntimeError, msg="sysfs path not found"):
            get_ib_devices_for_gpu("mlx5_0", 0)

    @_mock_sysfs(listdir_return=set())
    def test_no_devices_in_sysfs(self, _isdir, _listdir):
        with self.assertRaises(RuntimeError, msg="No IB devices found"):
            get_ib_devices_for_gpu("mlx5_0", 0)

    @_mock_sysfs()
    def test_json_mapping_with_invalid_device(self, _isdir, _listdir):
        mapping = json.dumps({"0": "mlx5_99"})
        with self.assertRaises(ValueError, msg="Invalid IB devices"):
            get_ib_devices_for_gpu(mapping, 0)


if __name__ == "__main__":
    unittest.main()
