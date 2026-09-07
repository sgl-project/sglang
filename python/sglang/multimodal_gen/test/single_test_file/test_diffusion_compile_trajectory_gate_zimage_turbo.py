"""GPU integration test for the compile-trajectory promotion gate (RFC #35333).

Builds a real manifest by running the small test model both eagerly and
regionally-compiled (``build_compile_trajectory_manifest``), then confirms
``sglang generate`` actually uses the gate at request time: a covered
signature compiles, an uncovered one (different resolution) falls back to
eager instead of silently compiling anyway.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.multimodal_gen.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST
from sglang.test.test_utils import CustomTestCase

PROMPT = "a red cube on a white table, centered product photo"
COVERED_RESOLUTION = (512, 512)
UNCOVERED_RESOLUTION = (384, 384)
NUM_INFERENCE_STEPS = 4
SEED = 0


class TestDiffusionCompileTrajectoryGateZImageTurbo(CustomTestCase):
    def _build_manifest(self, manifest_path: Path) -> None:
        width, height = COVERED_RESOLUTION
        cmd = [
            sys.executable,
            "-m",
            "sglang.multimodal_gen.tools.build_compile_trajectory_manifest",
            "--model-path",
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            "--prompt",
            PROMPT,
            "--width",
            str(width),
            "--height",
            str(height),
            "--num-inference-steps",
            str(NUM_INFERENCE_STEPS),
            "--seed",
            str(SEED),
            # Empirically measured on this model/step-count: eager vs. whole-module
            # compile lands at cosine ~0.9991 and max_abs ~1.39 at the terminal
            # checkpoint (compiler kernel fusion/reordering drift compounding over
            # a 4-step turbo schedule); these give headroom above that, not an
            # arbitrary guess.
            "--cosine-min",
            "0.99",
            "--max-abs-max",
            "3.0",
            "--output-manifest",
            str(manifest_path),
        ]
        env = os.environ.copy()
        env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"manifest build failed:\n{result.stdout[-4000:]}",
        )
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 1)
        self.assertEqual(
            manifest[0]["status"],
            "validated",
            f"manifest not validated: {json.dumps(manifest[0], indent=2)}",
        )

    def _generate(self, *, width: int, height: int, manifest_path: Path) -> str:
        cmd = [
            "sglang",
            "generate",
            "--backend",
            "sglang",
            "--model-path",
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            "--prompt",
            PROMPT,
            "--width",
            str(width),
            "--height",
            str(height),
            "--seed",
            str(SEED),
            "--num-inference-steps",
            str(NUM_INFERENCE_STEPS),
            "--no-save-output",
            "--guidance-scale",
            "0.0",
            "--enable-torch-compile",
            "--compile-trajectory-gate-manifest",
            str(manifest_path),
        ]
        env = os.environ.copy()
        env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"generate failed:\n{result.stdout[-4000:]}",
        )
        self.assertIn("Pixel data generated successfully", result.stdout)
        return result.stdout

    def test_covered_signature_compiles_uncovered_falls_back_to_eager(self):
        artifact_dir = Path(
            os.environ.get(
                "SGLANG_DIFFUSION_ARTIFACT_DIR",
                tempfile.mkdtemp(prefix="sglang_compile_trajectory_gate_"),
            )
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact_dir / "compile_trajectory_manifest.json"

        self._build_manifest(manifest_path)

        covered_width, covered_height = COVERED_RESOLUTION
        covered_log = self._generate(
            width=covered_width, height=covered_height, manifest_path=manifest_path
        )
        self.assertIn("compile-trajectory-gate: signature", covered_log)
        self.assertIn("covered by validated plan", covered_log)
        self.assertNotIn("falling back to eager", covered_log)

        uncovered_width, uncovered_height = UNCOVERED_RESOLUTION
        uncovered_log = self._generate(
            width=uncovered_width,
            height=uncovered_height,
            manifest_path=manifest_path,
        )
        self.assertIn("no validated plan covers signature", uncovered_log)
        self.assertIn("falling back to eager", uncovered_log)


if __name__ == "__main__":
    unittest.main()
