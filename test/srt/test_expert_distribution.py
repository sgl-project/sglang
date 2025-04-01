import glob
import os
import unittest

import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestExpertDistribution(CustomTestCase):
    def setUp(self):
        # Clean up any existing expert distribution files before each test
        for f in glob.glob("expert_distribution_*.csv"):
            os.remove(f)

    def tearDown(self):
        # Clean up any expert distribution files after each test
        for f in glob.glob("expert_distribution_*.csv"):
            os.remove(f)

    def test_expert_distribution_record(self):
        for model_path in [
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "Qwen/Qwen1.5-MoE-A2.7B",
        ]:
            with self.subTest(model_path=model_path):
                self._execute_core(model_path=model_path)

    def _execute_core(self, model_path: str):
        """Test expert distribution record endpoints"""
        process = popen_launch_server(
            model_path,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
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
                        "max_new_tokens": 3,
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
            rows = list(csv_reader)
            self.assertGreater(len(rows), 0, "CSV file should contain data rows")

            for row in rows:
                # Verify each row has 3 columns
                self.assertEqual(
                    len(row),
                    3,
                    "Each row should have layer_id, expert_id and count",
                )

                # Verify data types
                layer_id, expert_id, count = row
                self.assertTrue(
                    layer_id.isdigit(),
                    f"layer_id should be an integer {row=} {rows=}",
                )
                self.assertTrue(
                    expert_id.isdigit(),
                    f"expert_id should be an integer {row=} {rows=}",
                )
                self.assertTrue(
                    count.isdigit(), f"count should be an integer {row=} {rows=}"
                )

        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
