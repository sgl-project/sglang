import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from openai import OpenAI

from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    get_dynamic_server_port,
)
from sglang.test.test_utils import CustomTestCase

IMAGE_SIZE = "512x512"
PROMPT_SWITCH_NUM_INFERENCE_STEPS = 4
PROMPT_SWITCH_SEED = 0
BCG_CAPTURE_MARKER = "[Diffusion BCG] captured"


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

    def test_zimage_turbo_bcg_prompt_switch_reuses_captured_graph(self):
        port = get_dynamic_server_port()
        extra_args = (
            "--model-type diffusion "
            "--num-gpus 1 "
            "--strict-ports "
            "--enable-breakable-cuda-graph "
            f"--warmup-resolutions {IMAGE_SIZE} "
            "--bcg-text-buckets 256"
        )
        manager = ServerManager(
            model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            port=port,
            wait_deadline=float(os.environ.get("SGLANG_TEST_WAIT_SECS", "1200")),
            extra_args=extra_args,
        )
        ctx = manager.start()
        try:
            client = OpenAI(
                api_key="sglang-anything",
                base_url=f"http://localhost:{ctx.port}/v1",
                timeout=float(
                    os.environ.get("SGLANG_TEST_OPENAI_REQUEST_TIMEOUT_SECS", "600")
                ),
                max_retries=0,
            )
            self._generate_prompt_switch_image(
                client,
                "a red cube on a white table, centered product photo",
            )
            self._generate_prompt_switch_image(
                client,
                "a blue butterfly flying above green grass, watercolor illustration",
            )
        finally:
            server_log = ctx.stdout_file.read_text(encoding="utf-8", errors="ignore")
            ctx.cleanup()

        capture_count = server_log.count(BCG_CAPTURE_MARKER)
        self.assertEqual(
            capture_count,
            1,
            "BCG should be captured during warmup and then reused by different "
            f"prompt requests. Observed {capture_count} capture log line(s).\n"
            f"Server log tail:\n{ctx.log_tail(lines=400)}",
        )

    def _generate_prompt_switch_image(self, client: OpenAI, prompt: str):
        response = client.images.with_raw_response.generate(
            model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            prompt=prompt,
            n=1,
            size=IMAGE_SIZE,
            response_format="b64_json",
            extra_body={
                "num_inference_steps": PROMPT_SWITCH_NUM_INFERENCE_STEPS,
                "seed": PROMPT_SWITCH_SEED,
            },
        )
        parsed = response.parse()
        self.assertTrue(parsed.data[0].b64_json)


if __name__ == "__main__":
    unittest.main()
