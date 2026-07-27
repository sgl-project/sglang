import os
import subprocess
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="full-1-npu-a3", nightly=True)

_COMMON_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--dtype",
    "bfloat16",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.78",
    "--weight-loader-disable-mmap",
]


def _vmtouch_file(filepath):
    """Run vmtouch and return (resident_pages, percentage, raw_stdout, raw_stderr).
    Format: Resident Pages: 910157/968961 36/36 93.9%
    """
    try:
        result = subprocess.run(
            ["vmtouch", filepath],
            capture_output=True,
            text=True,
            timeout=30,
        )
        stdout = result.stdout
        stderr = result.stderr
        resident = -1
        pct = ""
        for line in stdout.split("\n"):
            if "Resident Pages" in line:
                # Parse: " Resident Pages: 910157/968961 36/36 93.9%"
                parts = line.split("Resident Pages:")[1].strip().split()
                resident = int(parts[0].split("/")[0])
                pct = parts[2] if len(parts) >= 3 else ""
                break
        return resident, pct, stdout, stderr
    except Exception as e:
        return -1, "", "", str(e)


def _print_vmtouch(safetensor_files, weight_dir, label):
    """Print vmtouch results for all safetensor files."""
    print(f"\n=== vmtouch {label} ===")
    for fname in safetensor_files:
        fpath = os.path.join(weight_dir, fname)
        resident, pct, stdout, stderr = _vmtouch_file(fpath)
        if resident >= 0:
            print(f"vmtouch: {fname} Resident Pages={resident} ({pct})")
        else:
            print(f"vmtouch: {fname} FAILED stderr={stderr[:200]}")
            if stdout:
                for line in stdout.split("\n")[:5]:
                    if line.strip():
                        print(f"  DEBUG: {line.strip()}")
    print(f"=== end vmtouch {label} ===\n")


class TestWeightLoaderDropCache(CustomTestCase):
    """--weight-loader-drop-cache-after-load — verify page cache after
    drop-cache.  vmtouch runs while server is still alive.

    [Test Category] Parameter
    [Test Target] --weight-loader-drop-cache-after-load
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.out_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.err_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *_COMMON_ARGS,
                "--weight-loader-drop-cache-after-load",
                "--log-level",
                "info",
            ],
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_file.close()
        cls.err_file.close()
        os.unlink(cls.out_file.name)
        os.unlink(cls.err_file.name)

    def test_drop_cache_after_load(self):
        """Launch with drop-cache-after-load, vmtouch right after server ready — no generate."""
        resp = requests.get(self.base_url + "/health", timeout=30)
        self.assertEqual(resp.status_code, 200)

        weight_dir = QWEN3_8B_WEIGHTS_PATH
        safetensor_files = sorted(
            f for f in os.listdir(weight_dir) if f.endswith(".safetensors")
        )
        _print_vmtouch(safetensor_files, weight_dir, "drop-cache ON")


class TestWeightLoaderDropCacheOff(CustomTestCase):
    """Default (no drop-cache) — baseline for comparing page cache with
    TestWeightLoaderDropCache.  vmtouch runs after server shutdown for
    a fair comparison.

    [Test Category] Parameter
    [Test Target] --weight-loader-drop-cache-after-load (off, baseline)
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.out_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.err_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *_COMMON_ARGS,
                "--log-level",
                "info",
            ],
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_file.close()
        cls.err_file.close()
        os.unlink(cls.out_file.name)
        os.unlink(cls.err_file.name)

    def test_drop_cache_off_baseline(self):
        """Launch without drop-cache, vmtouch right after server ready — no generate."""
        resp = requests.get(self.base_url + "/health", timeout=30)
        self.assertEqual(resp.status_code, 200)

        weight_dir = QWEN3_8B_WEIGHTS_PATH
        safetensor_files = sorted(
            f for f in os.listdir(weight_dir) if f.endswith(".safetensors")
        )
        _print_vmtouch(safetensor_files, weight_dir, "drop-cache OFF")


if __name__ == "__main__":
    unittest.main()
