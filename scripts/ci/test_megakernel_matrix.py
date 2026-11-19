# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for model selection and the GPU-runner allowlist."""

import tempfile
import unittest
from pathlib import Path

import yaml
from megakernel_matrix import LEGACY, build_matrix


class TestMegakernelMatrix(unittest.TestCase):
    def test_routing_selection_and_fail_closed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                *LEGACY["deepseek_v41"]["tests"],
                "test_kimi.py",
                "test_qwen.py",
            ):
                (root / name).touch()
            self.assertEqual(
                build_matrix(root, [])["include"][0]["runner"], "4-gpu-gb300"
            )
            models = {
                **LEGACY,
                "kimi": {"gpu": "B300", "gpus": 8, "tests": ["test_kimi.py"]},
                "qwen": {"gpu": "GB300", "gpus": 8, "tests": ["test_qwen.py"]},
            }
            manifest = root / "ci_models.yaml"
            manifest.write_text(yaml.safe_dump(models))
            entries = build_matrix(root, [])["include"]
            self.assertEqual(
                {row["runner"] for row in entries}, {"4-gpu-gb300", "8-gpu-b300"}
            )
            self.assertEqual(
                build_matrix(root, ["kimi"])["include"],
                [
                    {
                        "runner": "8-gpu-b300",
                        "models": "kimi",
                        "tests": "test/e2e/test_kimi.py",
                    }
                ],
            )
            for requested in (["missing"], ["qwen"]):
                with self.subTest(requested=requested), self.assertRaises(ValueError):
                    build_matrix(root, requested)
            for invalid in (
                {"gpu": "B300", "gpus": True, "tests": ["test_kimi.py"]},
                {"gpu": "B300", "gpus": 8, "tests": []},
                {"gpu": "B300", "gpus": 8, "tests": ["../test_kimi.py"]},
                {"gpu": "B300", "gpus": 8, "tests": ["test_missing.py"]},
                {"gpu": "privileged-runner", "gpus": 8, "tests": ["test_kimi.py"]},
            ):
                manifest.write_text(yaml.safe_dump({"kimi": invalid}))
                with self.subTest(model=invalid), self.assertRaises(ValueError):
                    build_matrix(root, ["kimi"])


if __name__ == "__main__":
    unittest.main()
