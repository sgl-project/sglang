"""
Usage:
# From the test/srt directory:
cd test/srt
python3 -m unittest test_start_profile.TestStartProfile
python3 -m unittest test_start_profile.TestStartProfileWithNsys

# Run specific tests:
python3 -m unittest test_start_profile.TestStartProfile.test_start_profile_1
python3 -m unittest test_start_profile.TestStartProfileWithNsys.test_start_profile_cuda_profiler
"""

import os
import shutil
import subprocess
import time
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=41, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=60, suite="stage-b-test-small-1-gpu-amd")

OUTPUT_DIR = "./profiler_dir"


def _is_nsys_available():
    """Check if nsys (Nsight Systems) is available on the system."""
    try:
        result = subprocess.run(["nsys", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class TestStartProfile(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_TORCH_PROFILER_DIR.set(OUTPUT_DIR)
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def setUp(self):
        self._clear_profile_dir()

    def test_start_profile_1(self):
        """Test /start_profile with start_step and num_steps argument. This have to be the first test for start_step to work"""
        response = self._start_profile(start_step="15", num_steps=5)

        self._post_request()

        self._check_non_empty_profile_dir()

    def test_start_profile_2(self):
        """Test /start_profile with no argument"""
        response = self._start_profile()

        self._post_request()

        # Before /stop_profile, the profile directory should be empty
        self._check_empty_profile_dir()

        # Post /stop_profile and check the profile directory is non-empty
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/stop_profile",
        )
        self._check_non_empty_profile_dir()

    def test_start_profile_3(self):
        """Test /start_profile with num_steps argument"""
        response = self._start_profile(num_steps=5)

        self._post_request()

        self._check_non_empty_profile_dir()

    def _start_profile(self, **kwargs):
        """Start profiling with optional parameters."""
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/start_profile",
            json=kwargs if kwargs else None,
        )
        self.assertEqual(response.status_code, 200)

    def _post_request(self):
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

    def _clear_profile_dir(self):
        if os.path.isdir(OUTPUT_DIR):
            # Remove the directory and all its contents
            shutil.rmtree(OUTPUT_DIR)

    def _check_non_empty_profile_dir(self):
        self.assertTrue(os.path.isdir(OUTPUT_DIR), "Output directory does not exist.")
        self.assertNotEqual(
            len(os.listdir(OUTPUT_DIR)), 0, "Output directory is empty!"
        )

    def _check_empty_profile_dir(self):
        if os.path.isdir(OUTPUT_DIR):
            self.assertEqual(
                len(os.listdir(OUTPUT_DIR)), 0, "Output directory is non-empty!"
            )


class TestStartProfileWithNsys(CustomTestCase):
    """Test /start_profile with CUDA_PROFILER (requires nsys wrapper)

    Each test starts its own clean server instance with nsys profiling.
    """

    @classmethod
    def setUpClass(cls):
        if not _is_nsys_available():
            raise unittest.SkipTest("nsys (Nsight Systems) is not available")

        envs.SGLANG_TORCH_PROFILER_DIR.set(OUTPUT_DIR)
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        # Use a different port to avoid conflicts with other tests
        cls.base_url = "http://127.0.0.1:21100"

    def setUp(self):
        """Start a clean server with nsys for each test"""
        # Kill any existing processes on this port
        self._kill_existing_server()

        # Clean up old profile files for this test
        test_name = self.id().split(".")[-1]  # Get test method name
        self.nsys_output_file = f"nsys_profile_{test_name}"

        if os.path.isdir(OUTPUT_DIR):
            profile_file = os.path.join(OUTPUT_DIR, f"{self.nsys_output_file}.nsys-rep")
            if os.path.exists(profile_file):
                try:
                    os.remove(profile_file)
                except OSError:
                    pass

        # Launch server with nsys wrapper
        self.process = self._popen_launch_server_with_nsys(
            self.model,
            self.base_url,
            self.nsys_output_file,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    def tearDown(self):
        """Kill server and verify profile was created"""

        # Kill server first to let nsys finalize the .nsys-rep file
        kill_process_tree(self.process.pid)

        # Also ensure nsys agent processes are killed
        try:
            subprocess.run(
                ["pkill", "-f", "nsys.*--start-agent"],
                timeout=5,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Give nsys a moment to finalize the report
        time.sleep(3)

        # Verify the .nsys-rep file was created
        self._verify_nsys_profile_created()

    def _kill_existing_server(self):
        """Kill any existing server process on our port and orphaned nsys agents"""
        try:
            # Kill server on our port
            subprocess.run(["lsof", "-ti", ":21100"], capture_output=True, timeout=5)
            subprocess.run(["pkill", "-f", "sglang.launch_server.*21100"], timeout=5)

            # Kill any orphaned nsys agent processes
            subprocess.run(
                ["pkill", "-f", "nsys.*--start-agent"],
                timeout=5,
                stderr=subprocess.DEVNULL,  # Suppress "no process found" errors
            )

            time.sleep(2)  # Wait for cleanup
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def _popen_launch_server_with_nsys(self, model, base_url, output_file, timeout):
        """Launch server wrapped with nsys profile -c cudaProfilerApi

        Each test gets its own output file for complete isolation.
        """
        _, host, port = base_url.split(":")
        host = host[2:]

        # Build the server launch command
        command = [
            "nsys",
            "profile",
            "-c",
            "cudaProfilerApi",
            "--capture-range-end",
            "stop",  # Stop after first cudaProfilerStop()
            "-o",
            os.path.join(OUTPUT_DIR, output_file),
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            host,
            "--port",
            port,
        ]

        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Launch the process - capture output to keep test output clean
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready
        start_time = time.perf_counter()
        elapsed = 0
        with requests.Session() as session:
            while elapsed < timeout:
                elapsed = time.perf_counter() - start_time

                return_code = process.poll()
                if return_code is not None:
                    raise Exception(
                        f"Server process exited with code {return_code}. "
                        "Check server logs above for errors."
                    )

                try:
                    response = session.get(f"{base_url}/health_generate", timeout=5)
                    if response.status_code == 200:
                        return process
                except (requests.RequestException, requests.Timeout):
                    pass

                time.sleep(5)

        # Timeout reached
        kill_process_tree(process.pid)
        raise TimeoutError(
            f"Server failed to start within {timeout} seconds. "
            f"Check the server logs above for more information."
        )

    def _verify_nsys_profile_created(self):
        """Verify that the .nsys-rep file was created after server shutdown."""
        if not os.path.isdir(OUTPUT_DIR):
            raise AssertionError("Output directory does not exist.")

        expected_file = f"{self.nsys_output_file}.nsys-rep"
        profile_path = os.path.join(OUTPUT_DIR, expected_file)

        if not os.path.exists(profile_path):
            files = os.listdir(OUTPUT_DIR)
            raise AssertionError(
                f"Expected profile file '{expected_file}' not found. "
                f"Files present: {files}"
            )

    def test_start_profile_cuda_profiler_with_start_step(self):
        """Test /start_profile with CUDA_PROFILER, start_step, and num_steps"""
        # Use start_step to let server warm up before profiling
        response = self._start_profile(
            activities=["CUDA_PROFILER"], start_step=10, num_steps=3
        )

        self._post_request()

        # Profile verification happens in tearDown()

    def test_start_profile_cuda_profiler(self):
        """Test /start_profile with CUDA_PROFILER activity (no start_step)"""
        # Simple num_steps test - profiling starts immediately
        response = self._start_profile(activities=["CUDA_PROFILER"], num_steps=5)

        self._post_request()

        # Profile verification happens in tearDown()

    def _start_profile(self, **kwargs):
        """Start profiling with optional parameters."""
        response = requests.post(
            f"{self.base_url}/start_profile",
            json=kwargs if kwargs else None,
        )
        self.assertEqual(response.status_code, 200)
        return response

    def _post_request(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
