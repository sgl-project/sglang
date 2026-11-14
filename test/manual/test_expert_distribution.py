import tempfile
import unittest
from pathlib import Path

import requests
import torch

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestExpertDistribution(CustomTestCase):
    def test_expert_distribution_record(self):
        # TODO: Add tests for DeepEP gatherer (currently our CI cannot run that)
        for info in [
            dict(model_path="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"),
            dict(model_path="Qwen/Qwen1.5-MoE-A2.7B"),
            dict(model_path="Qwen/Qwen1.5-MoE-A2.7B", tp_size=2),
            dict(model_path="Qwen/Qwen1.5-MoE-A2.7B", mode="per_pass"),
            dict(model_path="Qwen/Qwen1.5-MoE-A2.7B", mode="per_token"),
        ]:
            with self.subTest(info=info):
                self._execute_core(**info)

    def _execute_core(self, model_path: str, mode: str = "stat", tp_size: int = 1):
        """Test expert distribution record endpoints"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.set(tmp_dir)

            process = popen_launch_server(
                model_path,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp-size",
                    str(tp_size),
                    "--expert-distribution-recorder-mode",
                    mode,
                    "--disable-cuda-graph",
                    "--disable-overlap-schedule",
                ],
            )

            try:
                # Start recording
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/start_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                # Make some requests to generate expert distribution data
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": "The capital of France is",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 32,
                        },
                    },
                )
                self.assertEqual(response.status_code, 200)

                # Stop recording
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/stop_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                # Dump the recorded data
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/dump_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                # Check data rows
                data = torch.load(
                    list(Path(tmp_dir).glob("*.pt"))[0], weights_only=True
                )
                print(f"{data=}")

                if mode in ["per_pass", "per_token"]:
                    self.assertGreater(len(data), 0, "Should contain data rows")
                else:
                    logical_count = data["logical_count"]
                    print(f"{logical_count.sum()=} {logical_count=}")
                    self.assertTrue(logical_count.sum() > 0)

            finally:
                kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
