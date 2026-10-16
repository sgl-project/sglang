"""
Test the --sidecar / --sidecar-args server options on NPU.

Launches the server with a sidecar module next to the native gRPC server and
verifies that:
  - the sidecar receives the provider argv (with --sidecar-shutdown-timeout
    stripped by SGLang itself) and the loopback gRPC endpoint
  - the sidecar startup is logged
  - inference keeps working alongside the sidecar

Prerequisites (nightly CI image):
  The image installs sglang from source (editable), so the prebuilt Rust
  extension .so files from the official wheel (matching the image's Python
  ABI, cp312) must be placed on the shared volume at
  ``/root/.cache/.cache/npu-pkgs/rust`` (override with the
  PREBUILT_RUST_EXT_DIR env var). setUpClass copies the .so files into the
  editable source tree and the server is launched with
  SGLANG_RUST_BUILD_MODE=never so sglang never tries to compile the Rust
  extensions at runtime.
"""

import json
import os
import shutil
import time
import unittest
from pathlib import Path

import requests
import sglang

from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)

# Directory on the shared volume holding the prebuilt _*.so files extracted
# from the official aarch64 wheel. Overridable for local runs.
PREBUILT_RUST_EXT_DIR = Path(
    os.environ.get(
        "PREBUILT_RUST_EXT_DIR", "/root/.cache/.cache/npu-pkgs/rust"
    )
)

# The sidecar is resolved by module name through sys.path, so the probe module
# is written next to the test and its directory is exported via PYTHONPATH.
GRPC_PORT = 40000
SIDECAR_MODULE_NAME = "sglang_npu_sidecar_probe"
SIDECAR_MODULE_PATH = os.path.join(os.getcwd(), f"{SIDECAR_MODULE_NAME}.py")
SIDECAR_MARKER_PATH = os.path.join(os.getcwd(), "sglang_npu_sidecar_probe_marker.json")
OUT_LOG_PATH = "./out_log.txt"
ERR_LOG_PATH = "./err_log.txt"

SIDECAR_MODULE_SOURCE = '''\
import json
import os
import time
def main(argv):
    """Record what the server actually handed to the sidecar, then stay alive.
    The main loop must block: the server-side watchdog treats a sidecar that
    exits non-zero as a crash and SIGQUITs the whole service.
    """
    with open(os.environ["SIDECAR_MARKER_PATH"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "argv": list(argv),
                "endpoint": os.environ.get("SGLANG_GRPC_ENDPOINT", ""),
                "pid": os.getpid(),
            },
            f,
        )
    while True:
        time.sleep(1)
'''


def _deploy_rust_extensions():
    """Stage the prebuilt Rust extension .so files into the editable source tree.

    The nightly CI image installs sglang with `pip install -e`, so
    `import sglang.srt.rust_extensions._grpc` resolves inside the source tree
    at /sgl-workspace/sglang/python/sglang/. Copy the .so files from the
    shared volume there (idempotent: skipped once the .so files are in place,
    so repeated runs and LPT-shard siblings in the same pod are no-ops).
    """
    target_dir = Path(sglang.__file__).resolve().parent / "srt" / "rust_extensions"
    if any(target_dir.glob("_*.so")):
        return
    so_files = sorted(PREBUILT_RUST_EXT_DIR.glob("_*.so")) if PREBUILT_RUST_EXT_DIR.is_dir() else []
    if not so_files:
        raise FileNotFoundError(
            f"No prebuilt _*.so files found in {PREBUILT_RUST_EXT_DIR}. "
            "Stage the sglang/srt/rust_extensions/*.so files from the official "
            "wheel there first, or point PREBUILT_RUST_EXT_DIR at them."
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    for f in so_files:
        shutil.copy2(f, target_dir / f.name)


class TestSidecar(CustomTestCase):
    """Testcase: Launch the server with --sidecar/--sidecar-args next to the native gRPC server and verify the sidecar receives the provider argv (with --sidecar-shutdown-timeout stripped) and the loopback gRPC endpoint, while inference keeps working.
    [Test Category] Parameter
    [Test Target] --sidecar, --sidecar-args
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        # Stage the wheel's prebuilt Rust extensions before the server import
        # chain touches sglang.srt.rust_extensions.
        _deploy_rust_extensions()

        with open(SIDECAR_MODULE_PATH, "w", encoding="utf-8") as f:
            f.write(SIDECAR_MODULE_SOURCE)
        env = {
            "PYTHONPATH": os.getcwd() + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "SIDECAR_MARKER_PATH": SIDECAR_MARKER_PATH,
            # Never build the Rust extensions at runtime; use the prebuilt
            # .so files staged above.
            "SGLANG_RUST_BUILD_MODE": "never",
        }
        other_args = [
            "--grpc-port",
            str(GRPC_PORT),
            "--sidecar",
            SIDECAR_MODULE_NAME,
            "--sidecar-args",
            json.dumps(["--flag", "value", "--sidecar-shutdown-timeout", "10"]),
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
        ]
        cls.out_log_file = open(OUT_LOG_PATH, "w+", encoding="utf-8")
        cls.err_log_file = open(ERR_LOG_PATH, "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        for attr in ("out_log_file", "err_log_file"):
            if hasattr(cls, attr) and getattr(cls, attr):
                getattr(cls, attr).close()
        for path in (OUT_LOG_PATH, ERR_LOG_PATH, SIDECAR_MODULE_PATH, SIDECAR_MARKER_PATH):
            if os.path.exists(path):
                os.remove(path)

    def _wait_for_sidecar_marker(self, timeout=60.0):
        # The sidecar writes its marker asynchronously right after the gRPC
        # server starts; poll instead of assuming it already happened.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if os.path.exists(SIDECAR_MARKER_PATH):
                try:
                    with open(SIDECAR_MARKER_PATH, encoding="utf-8") as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    pass  # caught the file mid-write; retry
            time.sleep(1.0)
        self.fail(f"sidecar marker {SIDECAR_MARKER_PATH} not written within {timeout}s")

    def test_sidecar_receives_args_and_endpoint(self):
        marker = self._wait_for_sidecar_marker()
        # --sidecar-shutdown-timeout is consumed by SGLang itself; only the
        # provider argv reaches main().
        print(f"#### {marker=}")
        self.assertEqual(marker["argv"], ["--flag", "value"])
        self.assertEqual(marker["endpoint"], f"http://127.0.0.1:{GRPC_PORT}")

    def test_sidecar_startup_logged(self):
        # SGLang's Python logging goes to stderr while Rust tracing and
        # uvicorn access logs go to stdout, so scan both streams.
        log_content = ""
        for log_file in (self.out_log_file, self.err_log_file):
            log_file.seek(0)
            log_content += log_file.read()
        self.assertIn(f"Sidecar module {SIDECAR_MODULE_NAME} started pid=", log_content)

    def test_inference_works_alongside_sidecar(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)


if __name__ == "__main__":
    unittest.main()
