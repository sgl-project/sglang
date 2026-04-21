"""Unit tests for diffusion CI case parsing."""

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = REPO_ROOT / "python"
PARSER_DIR = REPO_ROOT / "scripts/ci/utils/diffusion"

for path in (PYTHON_DIR, PARSER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diffusion_case_parser

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestDiffusionCaseParser(CustomTestCase):
    def test_parse_appended_diffusion_cases(self):
        source = textwrap.dedent("""
            ONE_GPU_CASES_A: list[DiffusionTestCase] = [
                DiffusionTestCase("base_case", None, None),
            ]

            if not current_platform.is_hip():
                ONE_GPU_CASES_A.append(
                    DiffusionTestCase("appended_case", None, None)
                )

            IGNORED_CASES = []
            IGNORED_CASES.append(DiffusionTestCase("ignored_case", None, None))
            """)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "gpu_cases.py"
            config_path.write_text(source, encoding="utf-8")

            cases = diffusion_case_parser.parse_testcase_configs(config_path)

        self.assertEqual(cases["ONE_GPU_CASES_A"], ["base_case", "appended_case"])
        self.assertNotIn("IGNORED_CASES", cases)


class TestDiffusionTorchFallbacks(CustomTestCase):
    def test_rotary_embedding_native_accepts_full_interleaved_cache(self):
        from sglang.jit_kernel.diffusion.triton.torch_fallback import (
            apply_rotary_embedding_native,
        )

        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        cos = torch.ones(2, 4, dtype=torch.float32)
        sin = torch.full((2, 4), 0.5, dtype=torch.float32)

        out = apply_rotary_embedding_native(x, cos, sin, interleaved=True)

        cos_half = cos[..., ::2].unsqueeze(-2)
        sin_half = sin[..., ::2].unsqueeze(-2)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        expected = torch.stack(
            (x1 * cos_half - x2 * sin_half, x2 * cos_half + x1 * sin_half),
            dim=-1,
        ).flatten(-2)
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()
