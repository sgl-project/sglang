import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1800, suite="stage-a-tp-test-cpu-intel")

DEFAULT_WAN_MODEL_NAME_FOR_TEST = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
GENERATE_TIMEOUT = int(os.environ.get("WAN_GENERATE_TIMEOUT", "1800"))


class TestDiffusionCPUParallel(CustomTestCase):
    model = os.environ.get("WAN_MODEL_PATH", DEFAULT_WAN_MODEL_NAME_FOR_TEST)

    def setUp(self):
        super().setUp()

    def run_parallel_test(self, *, tp_size: int, sp_degree: int):
        with tempfile.TemporaryDirectory(prefix="sglang_cpu_diffusion_") as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir()
            output_file = output_dir / "test.mp4"
            log_path = Path(tmp) / "generate.log"
            command = [
                "sglang",
                "generate",
                "--model-path",
                self.model,
                "--tp-size",
                str(tp_size),
                "--sp-degree",
                str(sp_degree),
                "--ulysses-degree",
                str(sp_degree),
                "--ring-degree",
                "1",
                "--num-gpus",
                str(tp_size * sp_degree),
                "--prompt",
                "A cat walking in a garden.",
                "--width",
                "256",
                "--height",
                "256",
                "--num-frames",
                "5",
                "--num-inference-steps",
                "2",
                "--seed",
                "42",
                "--save-output",
                "--output-path",
                str(output_dir),
                "--output-file-name",
                output_file.name,
            ]

            env = os.environ.copy()
            env.update(
                SGLANG_DIFFUSION_PLATFORM_OVERRIDE="cpu",
                SGLANG_DIFFUSION_TARGET_DEVICE="cpu",
            )

            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                timed_out = False
                try:
                    returncode = process.wait(timeout=GENERATE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:
                    kill_process_tree(process.pid)

            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            log_tail = log_text[-16000:]

            if timed_out:
                self.fail(
                    f"Diffusion generation timed out after {GENERATE_TIMEOUT}s.\n"
                    f"Command: {' '.join(command)}\n"
                    f"Generation log:\n{log_tail}"
                )

            self.assertEqual(
                returncode,
                0,
                msg=(
                    f"Diffusion generation failed with return code {returncode}.\n"
                    f"Command: {' '.join(command)}\n"
                    f"Generation log:\n{log_tail}"
                ),
            )

            self.assertTrue(
                output_file.is_file(),
                msg=(
                    f"Expected output video not found: {output_file}\n"
                    f"Actual generated files: {[str(p) for p in output_dir.rglob('*')]}\n"
                    f"Command: {' '.join(command)}\n"
                    f"Generation log:\n{log_tail}"
                ),
            )

            self.assertGreater(
                output_file.stat().st_size,
                0,
                msg=(
                    f"Generated video is empty: {output_file}\n"
                    f"Command: {' '.join(command)}\n"
                    f"Generation log:\n{log_tail}"
                ),
            )

    def test_cpu_tp_only(self):
        self.run_parallel_test(tp_size=2, sp_degree=1)

    def test_cpu_sp_only(self):
        self.run_parallel_test(tp_size=1, sp_degree=2)

    def test_cpu_tp2_sp2(self):
        self.run_parallel_test(tp_size=2, sp_degree=2)


if __name__ == "__main__":
    unittest.main()
