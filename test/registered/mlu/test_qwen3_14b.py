import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=900, suite="nightly-test-mlu", nightly=True)


def _mlu_available() -> bool:
    try:
        import torch_mlu  # noqa: F401

        return bool(torch.mlu.is_available())
    except Exception:
        return False


def _find_reachable_port(base_port: int = 30000) -> int:
    for port in range(base_port, base_port + 2000):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", port))
            listener.listen(1)
            probe = socket.create_connection(("127.0.0.1", port), timeout=1)
            conn, _ = listener.accept()
            probe.close()
            conn.close()
            return port
        except OSError:
            continue
        finally:
            listener.close()
    raise RuntimeError("No reachable localhost port found for MLU E2E test")


def _tail_log(log_file: Path, lines: int = 200) -> str:
    if not log_file.exists():
        return f"Log file does not exist: {log_file}"
    return "\n".join(log_file.read_text(errors="replace").splitlines()[-lines:])


def _request_json(url: str, payload: dict, timeout: int = 120) -> str:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected status {response.status}")
        return response.read().decode(errors="replace")


@unittest.skipUnless(_mlu_available(), "MLU device is not available")
class TestMLUQwen314BE2E(CustomTestCase):
    def test_qwen3_14b_server_smoke(self):
        """Run an end-to-end Qwen3-14B MLU server smoke test."""
        repo_root = Path(__file__).resolve().parents[3]
        model_path = os.environ.get("MLU_QWEN3_14B_MODEL_PATH")
        if model_path is None:
            default_model_path = Path("/data/models/Qwen3-14B/")
            model_path = (
                str(default_model_path) if default_model_path.is_dir() else "Qwen3-14B"
            )

        host = "127.0.0.1"
        port = _find_reachable_port(30000)
        base_url = f"http://{host}:{port}"
        readiness_timeout = int(os.environ.get("READINESS_TIMEOUT", "900"))
        log_file_handle = tempfile.NamedTemporaryFile(
            prefix="sglang-mlu-qwen3-14b-", suffix=".log", delete=False
        )
        log_file = Path(log_file_handle.name)
        log_file_handle.close()

        env = os.environ.copy()
        env.setdefault("MLU_VISIBLE_DEVICES", "2")
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--device",
            "mlu",
            "--trust-remote-code",
            "--host",
            host,
            "--port",
            str(port),
            "--skip-server-warmup",
        ]

        with log_file.open("w") as server_log:
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        try:
            deadline = time.monotonic() + readiness_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail(
                        "MLU Qwen3-14B server exited before readiness.\n"
                        f"returncode: {process.returncode}\n"
                        f"log_file: {log_file}\n"
                        f"log_tail:\n{_tail_log(log_file)}"
                    )
                try:
                    with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                        if r.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                self.fail(
                    "Timed out waiting for MLU Qwen3-14B server readiness.\n"
                    f"log_file: {log_file}\n"
                    f"log_tail:\n{_tail_log(log_file)}"
                )

            single = _request_json(
                f"{base_url}/generate",
                {
                    "text": "The future of AI is",
                    "sampling_params": {"max_new_tokens": 16, "temperature": 0},
                },
            )
            print(f"single generate output: {single}")

            prompts = [
                "The future of AI is",
                "Machine learning helps",
                "Cambricon MLU accelerators are",
                "SGLang is designed for",
            ]

            def request_once(item):
                idx, prompt = item
                body = _request_json(
                    f"{base_url}/generate",
                    {
                        "text": prompt,
                        "sampling_params": {
                            "max_new_tokens": 64,
                            "temperature": 0,
                        },
                    },
                )
                return idx, prompt, body

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(request_once, enumerate(prompts)))

            for idx, prompt, body in results:
                print(f"concurrent request {idx} prompt={prompt!r} output: {body}")
                self.assertTrue(body)

            log = log_file.read_text(errors="replace")
            required = [
                "Capture mlu graph begin",
                "Capture mlu graph end",
                "cuda graph: True",
            ]
            missing = [item for item in required if item not in log]
            self.assertFalse(
                missing,
                f"Missing graph evidence in log {log_file}: {missing}\n"
                f"log_tail:\n{_tail_log(log_file)}",
            )
        finally:
            if process.poll() is None:
                kill_process_tree(process.pid)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
