import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.multimodal_gen.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST
from sglang.test.test_utils import CustomTestCase


class TestDiffusionBCGZImageTurbo(CustomTestCase):
    def test_zimage_turbo_true_bcg_generate(self):
        artifact_dir = Path(
            os.environ.get(
                "SGLANG_DIFFUSION_ARTIFACT_DIR",
                tempfile.mkdtemp(prefix="sglang_diffusion_bcg_"),
            )
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        log_path = artifact_dir / "zimage_turbo_bcg.log"
        perf_path = artifact_dir / "zimage_turbo_bcg_perf.json"

        cmd = [
            "sglang",
            "generate",
            "--backend",
            "sglang",
            "--model-path",
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            "--prompt",
            (
                "A detailed cinematic scene of a glass observatory above a quiet "
                "lake at sunrise, with soft mist, warm reflections, and crisp "
                "architectural detail"
            ),
            "--width",
            "512",
            "--height",
            "512",
            "--seed",
            "42",
            "--num-inference-steps",
            "9",
            "--warmup-resolutions",
            "512x512",
            "--no-save-output",
            "--guidance-scale",
            "0.0",
            "--enable-breakable-cuda-graph",
            "--bcg-text-buckets",
            "128",
            "--enable-torch-compile",
            "false",
            "--dit-layerwise-offload",
            "false",
            "--dit-cpu-offload",
            "false",
            "--perf-dump-path",
            str(perf_path),
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
        log_path.write_text(result.stdout, encoding="utf-8")

        self.assertEqual(
            result.returncode,
            0,
            f"Z-Image-Turbo BCG command failed. Log: {log_path}\n"
            f"{result.stdout[-4000:]}",
        )
        self.assertNotIn("Falling back to diffusers backend", result.stdout)
        self.assertNotIn("Using diffusers backend", result.stdout)
        self.assertNotIn("Loaded diffusers pipeline", result.stdout)
        self.assertNotIn("[Diffusion BCG] capture failed", result.stdout)
        self.assertIn("[Diffusion BCG] captured", result.stdout)
        self.assertIn("Pixel data generated successfully", result.stdout)

        self.assertTrue(perf_path.exists(), f"perf dump not found: {perf_path}")
        perf = json.loads(perf_path.read_text(encoding="utf-8"))
        stage_names = {
            step.get("name") for step in perf.get("steps", []) if isinstance(step, dict)
        }
        self.assertIn("DenoisingStage", stage_names)
        self.assertGreater(len(perf.get("denoise_steps_ms", [])), 0)


if __name__ == "__main__":
    unittest.main()
