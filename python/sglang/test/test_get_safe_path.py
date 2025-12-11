import unittest
from pathlib import Path

from sglang.utils import get_safe_path


class TestGetSafePath(unittest.TestCase):

    def test_relative_path_within_base(self):
        """resolve relative path within base_dir"""
        base_path = Path("/base").resolve()
        path_str = "subdir/file.txt"
        expected_path = base_path / "subdir" / "file.txt"
        result_path = get_safe_path(path_str, str(base_path))
        self.assertEqual(result_path, expected_path)

    def test_absolute_path_within_tmp(self):
        """resolve absolute path within /tmp"""
        tmp_path = Path("/tmp").resolve()
        path_str = "/tmp/somefile.txt"
        expected_path = tmp_path / "somefile.txt"
        result_path = get_safe_path(path_str)
        self.assertEqual(result_path, expected_path)

    def test_empty_path_raises_error(self):
        """raise ValueError for empty path"""
        with self.assertRaises(ValueError) as context:
            get_safe_path("")
        self.assertIn("Empty path is not allowed", str(context.exception))

    def test_outside_allowed_path_raises_error(self):
        """raise ValueError for path outside allowed directories"""
        path_str = "../test"
        base_str = "/base"
        with self.assertRaises(ValueError) as context:
            get_safe_path(path_str, base_str)
        self.assertIn("is not allowed", str(context.exception))

    def test_absolute_path_not_in_allowed_dirs_raises_error(self):
        """raise ValueError for absolute path not in allowed directories"""
        path_str = "/unauthorized/path"
        base_str = "/base"
        with self.assertRaises(ValueError) as context:
            get_safe_path(path_str, base_str)
        self.assertIn("is not allowed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
