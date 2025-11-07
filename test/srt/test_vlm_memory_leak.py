"""
Memory Leak test for VLM server.

"""

import gc
import subprocess
import time
import unittest

import openai
import pytest

# Import common utilities
from test_vision_openai_server_common import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    IMAGE_MAN_IRONING_URL,
    kill_process_tree,
    popen_launch_server,
)


def get_gpu_memory_gb():
    """
    Runs nvidia-smi and parses the output to get the current GPU memory usage.
    This is the reliable way to measure the *server's* memory from this script.
    """
    try:
        # Query for GPU memory used, in MiB, no header
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Output is like "37285" (in MiB)
        memory_mib = int(result.stdout.strip())
        return memory_mib / 1024  # Convert MiB to GiB
    # Catch specific, expected errors instead of a broad Exception
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not get GPU memory from nvidia-smi: {e}")
        return -1  # Return an invalid value to show failure


# --- Base class for our stress tests ---
class VLMMemoryLeakTestMixin(unittest.TestCase):

    # --- Default test parameters ---
    MODEL_NAME = None
    ITERATIONS = 200
    MAX_INCREASE_GB = 0.5  # Default threshold
    LOG_INTERVAL = 20

    # --- Constants for sleep timers ---
    SERVER_STABILIZE_TIME_S = 10
    POST_LOOP_WAIT_TIME_S = 2

    @classmethod
    def setUpClass(cls):
        if cls.MODEL_NAME is None:
            return  # Don't run the base class

        print(f"\nLaunching server (Model: {cls.MODEL_NAME})...")
        cls.process = popen_launch_server(
            cls.MODEL_NAME,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )
        cls.base_url = DEFAULT_URL_FOR_TEST + "/v1"
        cls.api_key = "sk-123"
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url)
        print(f"Server launched (PID: {cls.process.pid}).")

        # Allow time for model to fully load before measuring memory
        print(f"Waiting {cls.SERVER_STABILIZE_TIME_S}s for server to stabilize...")
        time.sleep(cls.SERVER_STABILIZE_TIME_S)

    @classmethod
    def tearDownClass(cls):
        if cls.MODEL_NAME is None:
            return
        print("Terminating server...")
        kill_process_tree(cls.process.pid)
        print("Server terminated.")

    def test_vlm_stress_loop(self):
        """
        Runs a VLM chat task in a loop, measuring memory via nvidia-smi.
        """
        # If this is the base class (MODEL_NAME is None), skip this test.
        if self.MODEL_NAME is None:
            pytest.skip("Skipping base class template test")

        print(f"\n--- Starting VLM Stress Test ({self.MODEL_NAME}) ---")
        gc.collect()  # Clean up our *own* process

        # 1. Get memory *after* server has loaded the model
        memory_before = get_gpu_memory_gb()
        self.assertGreater(memory_before, 0, "nvidia-smi failed to read memory")
        print(f"Memory allocated AFTER load: {memory_before:.4f} GB")

        for i in range(self.ITERATIONS):
            if (i + 1) % self.LOG_INTERVAL == 0:
                current_mem = get_gpu_memory_gb()
                print(
                    f"Iteration {i+1}/{self.ITERATIONS}... (Current Mem: {current_mem:.4f} GB)"
                )

            try:
                self.client.chat.completions.create(
                    model="default",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What is in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": IMAGE_MAN_IRONING_URL},
                                },
                            ],
                        }
                    ],
                    temperature=0,
                    max_tokens=16,
                )
            except Exception as e:
                print(f"Error during iteration {i+1}: {e}")
                self.fail(f"Test failed at iteration {i+1}")

        # 3. Force GC and record memory after the loop
        print(f"--- Stress test loop finished ({self.MODEL_NAME}) ---")
        gc.collect()

        # Wait a moment for any final requests to clear
        time.sleep(self.POST_LOOP_WAIT_TIME_S)

        memory_after = get_gpu_memory_gb()
        print(f"Memory allocated after loop: {memory_after:.4f} GB")

        # 4. Assert that the memory increase is within an acceptable threshold
        memory_increase_gb = memory_after - memory_before
        print(f"Memory increase: {memory_increase_gb:.4f} GB")

        self.assertTrue(
            memory_increase_gb < self.MAX_INCREASE_GB,
            f"Potential memory leak detected on {self.MODEL_NAME}! "
            f"Grew from {memory_before:.4f} GB to {memory_after:.4f} GB.",
        )


# --- Test Case 1: Lightweight Model (0.5B) ---
class TestVLMMemoryLeak_0_5B(VLMMemoryLeakTestMixin):
    MODEL_NAME = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
    # All other attributes (ITERATIONS, MAX_INCREASE_GB, LOG_INTERVAL)
    # are inherited from the base class.


# --- Test Case 2: Larger Model (7B) ---
class TestVLMMemoryLeak_7B(VLMMemoryLeakTestMixin):
    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    # ITERATIONS is inherited (200)
    MAX_INCREASE_GB = 1.0  # Override the default 0.5
    LOG_INTERVAL = 10  # Override the default 20


if __name__ == "__main__":
    unittest.main()
