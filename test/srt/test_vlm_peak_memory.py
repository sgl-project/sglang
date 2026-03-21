"""
Peak Memory Stress Test for VLM server.
Focus: Measuring High-Water Mark (Peak) VRAM usage using a background monitoring thread.
(Dynamic VRAM Detection Version)
"""

import subprocess
import threading
import time
import unittest

import openai

# Import common utilities from SGLang test suite
from test_vision_openai_server_common import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    IMAGE_MAN_IRONING_URL,
    kill_process_tree,
    popen_launch_server,
)


class PeakMemoryMonitor(threading.Thread):
    """
    A background thread that:
    1. Detects the Total GPU VRAM capacity at startup.
    2. Continuously polls nvidia-smi to capture the highest VRAM usage.
    """

    def __init__(self, interval_s: float = 0.1):
        super().__init__()
        self.interval = interval_s
        self.stop_event = threading.Event()

        self.peak_memory_gb = 0.0
        self.total_memory_gb = 0.0
        self.daemon = True  # Ensure thread dies if main process crashes

        # --- Auto-detect Total Memory on init ---
        self.total_memory_gb = self._get_total_memory_gb()
        print(f"[Monitor] Detected Total GPU Memory: {self.total_memory_gb:.2f} GB")

    def _get_total_memory_gb(self) -> float:
        """Query nvidia-smi for the total memory of the first visible GPU."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            # Output is like "81920" (in MiB)
            # If multiple GPUs, take the first one or the max (assuming homogeneous cluster)
            usages = [int(x) for x in result.stdout.strip().split("\n") if x.strip()]
            if usages:
                return max(usages) / 1024
            return 0.0
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(
                f"[Monitor Error] Failed to run nvidia-smi to detect total memory: {e}"
            )
            return 1.0  # Avoid division by zero later, safe fallback
        except ValueError as e:
            print(
                f"[Monitor Error] Failed to parse memory size from nvidia-smi output: {e}"
            )
            return 1.0  # Avoid division by zero later, safe fallback

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Query used memory
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    usages = [
                        int(x) for x in result.stdout.strip().split("\n") if x.strip()
                    ]
                    if usages:
                        current_max_mib = max(usages)
                        current_max_gb = current_max_mib / 1024

                        # Update the High-Water Mark
                        if current_max_gb > self.peak_memory_gb:
                            self.peak_memory_gb = current_max_gb
            except Exception as e:
                print(
                    f"[Monitor Error] Polling nvidia-smi failed: {e}. Stopping monitor."
                )
                break

            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

    def get_peak_memory(self) -> float:
        return self.peak_memory_gb

    def get_total_memory(self) -> float:
        return self.total_memory_gb


class TestVLMPeakMemory(unittest.TestCase):

    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

    # Constants
    SERVER_STABILIZE_TIME_S = 10
    LOAD_ITERATIONS = 50

    # Instead of hardcoded, we use a RATIO (Percentage)
    # Alert if usage exceeds 95% of the GPU's capacity
    MAX_MEMORY_USAGE_RATIO = 0.95

    @classmethod
    def setUpClass(cls):
        print(f"\n[Setup] Launching server (Model: {cls.MODEL_NAME})...")
        cls.process = popen_launch_server(
            cls.MODEL_NAME,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )
        cls.base_url = DEFAULT_URL_FOR_TEST + "/v1"
        cls.api_key = "sk-123"
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url)
        print(f"[Setup] Server launched (PID: {cls.process.pid}).")

        print(f"[Setup] Waiting {cls.SERVER_STABILIZE_TIME_S}s for stabilization...")
        time.sleep(cls.SERVER_STABILIZE_TIME_S)

    @classmethod
    def tearDownClass(cls):
        print("[Teardown] Terminating server...")
        kill_process_tree(cls.process.pid)
        print("[Teardown] Server terminated.")

    def test_peak_memory_under_load(self):
        """
        Sends requests while a background thread monitors peak memory usage.
        Validates that peak usage does not exceed a safe ratio of total VRAM.
        """
        print(f"\n--- Starting Peak Memory Stress Test ({self.MODEL_NAME}) ---")

        # 1. Start the background monitor (It will auto-detect Total VRAM)
        monitor = PeakMemoryMonitor(interval_s=0.1)
        monitor.start()

        # Get the detected total memory for logging
        total_vram = monitor.get_total_memory()
        safe_limit_gb = total_vram * self.MAX_MEMORY_USAGE_RATIO

        print(f"[Monitor] Detected Total VRAM: {total_vram:.2f} GB")
        print(
            f"[Monitor] Safety Threshold ({self.MAX_MEMORY_USAGE_RATIO*100}%): {safe_limit_gb:.2f} GB"
        )

        try:
            # 2. Generate Load
            print(
                f"[Load] Starting {self.LOAD_ITERATIONS} iterations of VLM inference..."
            )
            for i in range(self.LOAD_ITERATIONS):
                if (i + 1) % 10 == 0:
                    print(f"  Processing request {i+1}/{self.LOAD_ITERATIONS}...")

                try:
                    self.client.chat.completions.create(
                        model="default",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Describe this image in detail.",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": IMAGE_MAN_IRONING_URL},
                                    },
                                ],
                            }
                        ],
                        temperature=0,
                        max_tokens=64,
                    )
                except Exception as e:
                    self.fail(f"Request failed at iteration {i}: {e}")

        finally:
            # 3. Stop Monitor
            monitor.stop()
            monitor.join()

        # 4. Analyze Results
        peak_usage = monitor.get_peak_memory()
        utilization_ratio = peak_usage / total_vram if total_vram > 0 else 0

        print(f"\nReport:")
        print(f"   Total VRAM:   {total_vram:.4f} GB")
        print(f"   Peak Usage:   {peak_usage:.4f} GB")
        print(f"   Utilization:  {utilization_ratio:.2%}")

        # 5. Assertions
        self.assertGreater(peak_usage, 0.0, "Monitor failed to capture any VRAM usage.")

        # Check against the Ratio (Dynamic Threshold)
        if peak_usage > safe_limit_gb:
            print(f"WARNING: Peak usage exceeded safe threshold!")
            self.fail(f"Memory usage {peak_usage:.2f}GB > {safe_limit_gb:.2f}GB")
        else:
            print(
                f"Memory usage is within safe limits (< {self.MAX_MEMORY_USAGE_RATIO*100}%)."
            )


if __name__ == "__main__":
    unittest.main()
