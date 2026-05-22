from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import ClassVar, Dict, List, Optional

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mock_model_utils import MOCK_MODEL_PATH, mock_model_server_args
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=600, stage="extra-a", runner_config="2-gpu-large")

# DO NOT pass --disable-cuda-graph or --disable-piecewise-cuda-graph in any
# canary e2e test. The canary kernel must run inside the cuda graph alongside
# the real attn kernel; disabling the graph silently bypasses the only path
# that exercises that invariant end-to-end.
_NUM_PROMPTS = 32
_INPUT_LEN = 6144
_OUTPUT_LEN = 1024


def _send_parallel_requests(
    base_url: str,
    *,
    n: int,
    max_new_tokens: int,
    timeout: float = 60.0,
    max_workers: int = 16,
) -> List[Dict[str, object]]:
    """Fire N /generate requests concurrently; return raw response dicts."""

    def _one(i: int) -> Dict[str, object]:
        payload = {
            "input_ids": _make_input_ids(seed=i, length=_INPUT_LEN),
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        }
        try:
            resp = requests.post(base_url + "/generate", json=payload, timeout=timeout)
            return {"index": i, "status_code": resp.status_code, "text": resp.text}
        except requests.exceptions.RequestException as exc:
            return {"index": i, "error": repr(exc)}

    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, i) for i in range(n)]
        for fut in as_completed(futures):
            results.append(fut.result())
    results.sort(key=lambda r: r["index"])
    return results


def _make_input_ids(*, seed: int, length: int) -> List[int]:
    return [((seed + i) % 2048) + 1 for i in range(length)]


def _tee_stream(src: object, sinks: List[object]) -> None:
    """Read lines from src and write to all sinks (thread target)."""
    for line in iter(src.readline, ""):
        for sink in sinks:
            sink.write(line)
            sink.flush()
    src.close()


def _popen_pd_with_capture(
    model: str,
    base_url: str,
    other_args: List[str],
    env: Optional[dict],
    stdout_buf: io.StringIO,
    stderr_buf: io.StringIO,
) -> subprocess.Popen:
    """Launch a PD server process with tee'd stdout/stderr capture."""
    _, host, port = base_url.split(":")
    host = host[2:]
    command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        *[str(x) for x in other_args],
        "--host",
        host,
        "--port",
        port,
    ]
    print(f"command={' '.join(command)}")
    child_env = {**os.environ, **(env or {})}
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=_tee_stream,
        args=(proc.stdout, [stdout_buf, sys.stdout]),
        daemon=True,
    ).start()
    threading.Thread(
        target=_tee_stream,
        args=(proc.stderr, [stderr_buf, sys.stderr]),
        daemon=True,
    ).start()
    return proc


class _MockModelPDBase(PDDisaggregationServerBase):
    """PD fixture for mock-model + canary e2e tests."""

    model: ClassVar[str] = MOCK_MODEL_PATH
    extra_prefill_args: ClassVar[List[str]] = mock_model_server_args()
    extra_decode_args: ClassVar[List[str]] = mock_model_server_args()
    _stdout_bufs: ClassVar[List[io.StringIO]]
    _stderr_bufs: ClassVar[List[io.StringIO]]

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SGLANG_KV_CANARY_INPUT_CHECK"] = "1"
        os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = "1"
        cls._stdout_bufs = []
        cls._stderr_bufs = []
        super().setUpClass()
        cls.launch_all()

    @classmethod
    def start_prefill(cls) -> None:
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--skip-server-warmup",
        ] + list(cls.extra_prefill_args)
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = cls._popen_with_capture(cls.prefill_url, prefill_args)

    @classmethod
    def start_decode(cls) -> None:
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--skip-server-warmup",
        ] + list(cls.extra_decode_args)
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = cls._popen_with_capture(cls.decode_url, decode_args)

    @classmethod
    def _popen_with_capture(
        cls, base_url: str, other_args: List[str]
    ) -> subprocess.Popen:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        cls._stdout_bufs.append(stdout_buf)
        cls._stderr_bufs.append(stderr_buf)
        env = {
            "SGLANG_KV_CANARY_INPUT_CHECK": "1",
            "SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE": "1",
        }
        return _popen_pd_with_capture(
            cls.model,
            base_url,
            other_args,
            env,
            stdout_buf,
            stderr_buf,
        )

    def assert_no_canary_violation(self) -> None:
        time.sleep(2)
        log_text = "".join(buf.getvalue() for buf in self._stdout_bufs) + "".join(
            buf.getvalue() for buf in self._stderr_bufs
        )
        self.assertNotIn("kv_canary violation:", log_text)


class TestPdTransferCanaryClean(_MockModelPDBase, unittest.TestCase):
    """PD standard scenario + canary all-on, no violation expected."""

    def test_pd_transfer_canary_clean(self) -> None:
        # Step 1: send parallel requests through the LB to exercise PD transfer path.
        results = _send_parallel_requests(
            self.lb_url,
            n=_NUM_PROMPTS,
            max_new_tokens=_OUTPUT_LEN,
            timeout=240.0,
            max_workers=_NUM_PROMPTS,
        )

        # Step 2: every request must complete with status 200.
        for result in results:
            self.assertEqual(result.get("status_code"), 200, result)

        # Step 3: servers must stay alive.
        self.assertIsNone(self.process_prefill.poll(), "Prefill server died")
        self.assertIsNone(self.process_decode.poll(), "Decode server died")
        self.assert_no_canary_violation()


class TestPdTransferChecksumFullRealData(_MockModelPDBase, unittest.TestCase):
    """--kv-canary-real-data=all + sweep every step, no perturb, no violation."""

    extra_prefill_args: ClassVar[List[str]] = mock_model_server_args(
        "--kv-canary-real-data",
        "all",
        "--kv-canary-sweep-interval",
        "1",
    )
    extra_decode_args: ClassVar[List[str]] = mock_model_server_args(
        "--kv-canary-real-data",
        "all",
        "--kv-canary-sweep-interval",
        "1",
        "--disaggregation-decode-enable-radix-cache",
    )

    def test_pd_transfer_checksum_full_real_data(self) -> None:
        # Step 1: drive traffic through the PD path with full real-KV hashing.
        results = _send_parallel_requests(
            self.lb_url,
            n=_NUM_PROMPTS,
            max_new_tokens=_OUTPUT_LEN,
            timeout=240.0,
            max_workers=_NUM_PROMPTS,
        )

        # Step 2: all requests must succeed.
        for result in results:
            self.assertEqual(result.get("status_code"), 200, result)

        # Step 3: servers must stay healthy.
        self.assertIsNone(self.process_prefill.poll(), "Prefill server died")
        self.assertIsNone(self.process_decode.poll(), "Decode server died")
        self.assert_no_canary_violation()


if __name__ == "__main__":
    unittest.main()
