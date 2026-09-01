"""Test --mm-io-worker-num and --mm-processor-worker-num parameter on NPU.
Verify mm-io/mm-processor thread counts while in high-concurrency graph processing case.
Verify the parameters do not affect MMMU dataset accuracy.
Logs are printed to screen in real time while also captured for assertion.
"""

import os
import re
import subprocess
import sys
import threading
import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=1800, suite="full-2-npu-a3", nightly=True)

MODEL = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH

BASE_SERVER_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--mem-fraction-static",
    "0.6",
    "--mm-attention-backend",
    "ascend_attn",
    "--mm-io-worker-num",
    "16",
    "--mm-processor-worker-num",
    "2",
]

BENCHMARK_ARGS = [
    sys.executable,
    "-m",
    "sglang.bench_serving",
    "--dataset-name",
    "image",
    "--model",
    MODEL,
    "--num-prompts",
    "50",
    "--image-count",
    "32",
    "--image-resolution",
    "720p",
    "--request-rate",
    "inf",
    "--max-concurrency",
    "8",
    "--random-output-len",
    "128",
]

EXPECTED_MM_IO_WORKER_NUM = 16
EXPECTED_MM_PROCESSOR_WORKER_NUM = 2
MMMU_ACCURACY_THRESHOLD = 0.40


def _find_sglang_server_pids(main_pid=None):
    """Find sglang server Python process PIDs via ps -ef.

    The ``sglang serve`` command used by popen_launch_server may be a wrapper
    that spawns ``python -m sglang.launch_server`` as a child. We search by
    command-line content rather than parent-child relationship so we find the
    actual Python process even if it was reparented to init.
    """
    result = subprocess.run(
        ["ps", "-ef"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    candidates = []
    if main_pid is not None:
        candidates.append(str(main_pid))
    for line in result.stdout.splitlines():
        if "grep" in line:
            continue
        if "bench_serving" in line:
            continue
        if "launch_server" in line or ("python" in line and "sglang" in line):
            parts = line.split()
            if len(parts) >= 2 and parts[1] not in candidates:
                candidates.append(parts[1])
    return candidates


def _pyspy_dump(pid):
    """Run py-spy dump on a single PID and return the combined output."""
    try:
        result = subprocess.run(
            ["py-spy", "dump", "--pid", str(pid)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            print(f"py-spy dump (pid={pid}) exited with code {result.returncode}")
        return output
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"py-spy dump failed for PID {pid}: {e}")
        return ""


def _count_threads(pyspy_output, pattern):
    """Count lines in py-spy dump output matching a pattern.

    Thread header lines in py-spy dump have the format:
        Thread <tid> (idle): "<thread_name>"
    The thread name (e.g. "sglang-mm-io_0") only appears in header lines,
    not in stack frames, so counting all matching lines is safe.
    """
    count = 0
    for line in pyspy_output.splitlines():
        if re.search(pattern, line, re.IGNORECASE):
            count += 1
    return count


class TestNpuMmWorkerNumAndMmmu(CustomTestCase):
    """Verify mm-io/mm-processor worker threads, image benchmark throughput,
    and MMMU accuracy for Qwen2.5-VL-3B-Instruct on Ascend NPU.

    Server: tp=2, ascend backend, mm-attention-backend=ascend_attn,
    mm-io-worker-num=16, mm-processor-worker-num=2.

    [Test Category] VLM
    [Test Target] --mm-io-worker-num; --mm-processor-worker-num; MMMU
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=BASE_SERVER_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_image_benchmark_and_thread_count(self):
        """Run image benchmark and verify mm-io/mm-processor thread counts.

        Thread monitoring runs in a background thread while the image benchmark
        is actively processing. This is required because:
        1. mm-io/mm-processor threads are created on demand during image
           processing and may not exist when the server is idle.
        2. The ``sglang serve`` wrapper PID may differ from the actual Python
           server PID, so we search by command-line content via ps -ef.
        """
        host, port = self.base_url.replace("http://", "").split(":")
        cmd = BENCHMARK_ARGS + ["--host", host, "--port", port]

        benchmark_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        monitor_result = {}

        def _monitor():
            """Background monitor: run py-spy repeatedly while the benchmark is
            actively processing images, keeping the result with the most threads.

            mm-io/mm-processor threads are created on demand. When the server is
            idle only ``_0`` of each exists. The full 16 mm-io and 2
            mm-processor threads appear once the server starts loading images.
            We retry py-spy until we catch the server mid-processing.
            """
            # Wait for the benchmark to start sending image requests
            time.sleep(15)

            best_output = ""
            best_mm_io = 0
            best_mm_proc = 0
            best_pids = []

            for attempt in range(40):
                if benchmark_process.poll() is not None:
                    print("[monitor] Benchmark exited, stopping")
                    break

                candidate_pids = _find_sglang_server_pids(main_pid=self.process.pid)
                if not candidate_pids:
                    time.sleep(5)
                    continue

                attempt_output = ""
                for pid in candidate_pids:
                    attempt_output += _pyspy_dump(pid)

                mm_io = _count_threads(attempt_output, r"sglang-mm-io")
                mm_proc = _count_threads(attempt_output, r"sglang-mm-processor")

                print(
                    f"[monitor #{attempt + 1}] PIDs={candidate_pids} "
                    f"mm-io={mm_io} mm-processor={mm_proc}"
                )

                if mm_io > best_mm_io or mm_proc > best_mm_proc:
                    best_output = attempt_output
                    best_mm_io = mm_io
                    best_mm_proc = mm_proc
                    best_pids = candidate_pids

                if (
                    mm_io >= EXPECTED_MM_IO_WORKER_NUM
                    and mm_proc >= EXPECTED_MM_PROCESSOR_WORKER_NUM
                ):
                    print(f"[monitor] Target reached on attempt {attempt + 1}")
                    break

                time.sleep(5)

            monitor_result["output"] = best_output
            monitor_result["pids"] = best_pids
            monitor_result["mm_io"] = best_mm_io
            monitor_result["mm_proc"] = best_mm_proc

        monitor_thread = threading.Thread(target=_monitor, daemon=True)
        monitor_thread.start()

        completed = None
        mean_ttft = None
        try:
            for line in benchmark_process.stdout:
                if line.strip():
                    print(line, end="")
                stripped = line.strip()
                if "Successful requests:" in stripped:
                    parts = stripped.split()
                    if parts:
                        completed = int(parts[-1])
                if "Mean TTFT" in stripped:
                    parts = stripped.split()
                    if len(parts) >= 4:
                        mean_ttft = float(parts[3])
        finally:
            if benchmark_process.stdout and not benchmark_process.stdout.closed:
                benchmark_process.stdout.close()
            benchmark_process.wait()

        monitor_thread.join(timeout=120)

        # --- Verify benchmark results ---
        self.assertIsNotNone(
            completed,
            "Failed to extract 'Successful requests' from benchmark output",
        )
        self.assertEqual(
            completed,
            50,
            f"Expected 50 successful requests, got {completed}",
        )
        if mean_ttft is not None:
            print(f"\n[Image Benchmark] Mean TTFT: {mean_ttft:.2f} ms")

        # --- Verify thread counts ---
        pyspy_output = monitor_result.get("output", "")
        self.assertTrue(
            pyspy_output.strip(),
            "py-spy dump produced no output — is py-spy installed?",
        )

        thread_lines = [
            line
            for line in pyspy_output.splitlines()
            if line.strip().startswith("Thread")
        ]
        print(f"\n--- py-spy thread headers ({len(thread_lines)} threads) ---")
        for line in thread_lines:
            print(line)
        print("--- end ---\n")

        mm_io_count = _count_threads(pyspy_output, r"sglang-mm-io")
        mm_processor_count = _count_threads(pyspy_output, r"sglang-mm-processor")

        print(f"mm-io threads found: {mm_io_count}")
        print(f"mm-processor threads found: {mm_processor_count}")

        self.assertGreaterEqual(
            mm_io_count,
            EXPECTED_MM_IO_WORKER_NUM,
            f"Expected at least {EXPECTED_MM_IO_WORKER_NUM} mm-io threads, "
            f"found {mm_io_count}",
        )
        self.assertGreaterEqual(
            mm_processor_count,
            EXPECTED_MM_PROCESSOR_WORKER_NUM,
            f"Expected at least {EXPECTED_MM_PROCESSOR_WORKER_NUM} "
            f"mm-processor threads, found {mm_processor_count}",
        )

    def test_mmmu_accuracy(self):
        """Run MMMU eval — verify accuracy >= 0.40."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=MODEL,
            eval_name="mmmu",
            num_examples=100,
            num_threads=64,
            max_tokens=2048,
            return_latency=True,
        )

        metrics, latency = run_eval(args)

        score = round(metrics["score"], 4)
        print(
            f"\n{'=' * 42}\n"
            f"{MODEL} - MMMU score={score}, latency={round(latency, 2)}s\n"
            f"{'=' * 42}\n"
        )

        self.assertGreaterEqual(
            score,
            MMMU_ACCURACY_THRESHOLD,
            f"MMMU accuracy ({score}) below threshold ({MMMU_ACCURACY_THRESHOLD})",
        )


if __name__ == "__main__":
    unittest.main()
