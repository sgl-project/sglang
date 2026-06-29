#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from bump_flashinfer_version import (
    read_current_flashinfer_version,
    replace_flashinfer_version,
)


class TestBumpFlashInferVersion(unittest.TestCase):
    def test_pyproject_cuda_extra_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            pyproject = repo_root / "python" / "pyproject.toml"
            pyproject.parent.mkdir()
            pyproject.write_text(
                "dependencies = [\n"
                '  "flashinfer_cubin==0.6.12",\n'
                '  "flashinfer_python[cu13]==0.6.12",\n'
                "]\n"
            )

            self.assertEqual(read_current_flashinfer_version(repo_root), "0.6.12")
            self.assertTrue(
                replace_flashinfer_version(pyproject, "0.6.12", "0.6.14rc1")
            )
            self.assertIn('"flashinfer_python[cu13]==0.6.14rc1"', pyproject.read_text())
            self.assertIn('"flashinfer_cubin==0.6.14rc1"', pyproject.read_text())


if __name__ == "__main__":
    unittest.main()
