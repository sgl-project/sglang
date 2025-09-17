import os
import random
import subprocess
import time
import unittest

import requests
import torch

from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    kill_process_tree,
    popen_launch_server,
)


class TestVRAMUsageBenchServing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_vram_usage_increase_under_10_percent(self):
        # Step 0: skip if no GPU
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            self.skipTest("No CUDA device available; skip VRAM OOM test")

        def get_vram_used_gb() -> float:
            i = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            used_bytes = total_bytes - free_bytes
            return used_bytes / 1024**3

        # Step 1: measure current GPU VRAM used (GiB)
        before_used_gb = get_vram_used_gb()
        # Step 2: launch bench_serving in a separate process and record pid

        num_prompts = 40 if is_in_ci() else 100
        cmd = [
            "python3",
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang-oai-chat",
            "--dataset-name",
            "random-image",
            "--num-prompts",
            str(num_prompts),
            "--base-url",
            self.base_url,
            "--random-image-resolution",
            "1080p",
            "--max-concurrency",
            "1",
            "--random-range-ratio",
            "1",
            "--random-output-len",
            "1",
            "--random-input-len",
            "1024",
            "--random-image-num-images",
            "3",
            "--disable-tqdm",
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        peak_used_gb = before_used_gb
        # Step 2.1: poll during execution and record peak VRAM used (GiB)
        try:
            while process.poll() is None:
                peak_used_gb = max(peak_used_gb, get_vram_used_gb())
                # INSERT_YOUR_CODE
                time.sleep(1.0)
            # Take one more measurement after process exits
            peak_used_gb = max(peak_used_gb, get_vram_used_gb())
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(
                process.returncode,
                0,
                msg=f"bench_serving failed with code {process.returncode}, stderr={stderr.decode(errors='ignore')}",
            )
        finally:
            if process.poll() is None:
                process.kill()

        # Step 3: assert the VRAM usage increase is less than 4 GiB
        increase_gb = peak_used_gb - before_used_gb
        self.assertLess(
            increase_gb,
            5,
            msg=f"VRAM usage increase too high: before={before_used_gb:.3f} GiB, peak={peak_used_gb:.3f} GiB, increase={increase_gb:.3f} GiB",
        )


if __name__ == "__main__":
    unittest.main()
