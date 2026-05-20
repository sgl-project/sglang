from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import ClassVar, Dict, Iterable, List, Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

register_cuda_ci(est_time=600, stage="extra-a", runner_config="2-gpu-large")

# DO NOT pass --disable-cuda-graph or --disable-piecewise-cuda-graph in any
# canary e2e test. user-instruction.md b 段 requires the canary kernel to run
# inside the cuda graph alongside the real attn kernel; disabling the graph
# silently bypasses the only path that exercises that invariant end-to-end.

_MODEL = "Qwen/Qwen3-0.6B"

_MOCK_PD_COMMON_ARGS: List[str] = [
    "--load-format",
    "dummy",
    "--json-model-override-args",
    '{"num_hidden_layers": 1}',
    "--sampling-backend",
    "oracle",
    "--kv-canary",
    "raise",
]

_DEFAULT_PROMPTS: List[str] = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Explain in one sentence what a transformer is.",
    "1 + 1 =",
]


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
        prompt = _DEFAULT_PROMPTS[i % len(_DEFAULT_PROMPTS)]
        payload = {
            "text": f"{prompt} {i}",
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        }
        try:
            resp = requests.post(
                base_url + "/generate", json=payload, timeout=timeout
            )
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

    model: ClassVar[str] = _MODEL
    extra_prefill_args: ClassVar[List[str]] = _MOCK_PD_COMMON_ARGS
    extra_decode_args: ClassVar[List[str]] = _MOCK_PD_COMMON_ARGS

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SGLANG_KV_CANARY_INPUT_CHECK"] = "1"
        super().setUpClass()
        cls.launch_all()


class TestPdTransferCanaryClean(_MockModelPDBase, unittest.TestCase):
    """PD standard scenario + canary all-on, no violation expected."""

    def test_pd_transfer_canary_clean(self) -> None:
        # Step 1: send parallel requests through the LB to exercise PD transfer path.
        results = _send_parallel_requests(
            self.lb_url, n=16, max_new_tokens=32, timeout=60.0
        )

        # Step 2: every request must complete with status 200.
        for result in results:
            self.assertEqual(result.get("status_code"), 200, result)

        # Step 3: servers must stay alive.
        self.assertIsNone(self.process_prefill.poll(), "Prefill server died")
        self.assertIsNone(self.process_decode.poll(), "Decode server died")


class TestPdTransferChecksumFullRealData(_MockModelPDBase, unittest.TestCase):
    """--kv-canary-real-data=all + sweep every step, no perturb, no violation."""

    extra_prefill_args: ClassVar[List[str]] = _MOCK_PD_COMMON_ARGS + [
        "--kv-canary-real-data",
        "all",
        "--kv-canary-sweep-interval",
        "1",
    ]
    extra_decode_args: ClassVar[List[str]] = _MOCK_PD_COMMON_ARGS + [
        "--kv-canary-real-data",
        "all",
        "--kv-canary-sweep-interval",
        "1",
    ]

    def test_pd_transfer_checksum_full_real_data(self) -> None:
        # Step 1: drive traffic through the PD path with full real-KV hashing.
        results = _send_parallel_requests(
            self.lb_url, n=16, max_new_tokens=32, timeout=60.0
        )

        # Step 2: all requests must succeed.
        for result in results:
            self.assertEqual(result.get("status_code"), 200, result)

        # Step 3: servers must stay healthy.
        self.assertIsNone(self.process_prefill.poll(), "Prefill server died")
        self.assertIsNone(self.process_decode.poll(), "Decode server died")


class TestPdTransferCorruptedByteDetected(PDDisaggregationServerBase, unittest.TestCase):
    """Inject byte corruption; canary sweep must report a violation."""

    model: ClassVar[str] = _MODEL
    extra_prefill_args: ClassVar[List[str]] = _MOCK_PD_COMMON_ARGS + [
        "--kv-canary-real-data",
        "all",
        "--kv-canary-sweep-interval",
        "1",
    ]
    extra_decode_args: ClassVar[List[str]] = _MOCK_PD_COMMON_ARGS + [
        "--kv-canary-real-data",
        "all",
        "--kv-canary-sweep-interval",
        "1",
    ]

    _prefill_stdout: ClassVar[io.StringIO]
    _prefill_stderr: ClassVar[io.StringIO]
    _decode_stdout: ClassVar[io.StringIO]
    _decode_stderr: ClassVar[io.StringIO]
    _launch_failed: ClassVar[bool] = False

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SGLANG_KV_CANARY_INPUT_CHECK"] = "1"
        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB"] = "0.5"
        super().setUpClass()

        cls._prefill_stdout = io.StringIO()
        cls._prefill_stderr = io.StringIO()
        cls._decode_stdout = io.StringIO()
        cls._decode_stderr = io.StringIO()
        cls._launch_failed = False

        try:
            cls.launch_all()
        except Exception:
            cls._launch_failed = True

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB", None)
        super().tearDownClass()

    @classmethod
    def start_prefill(cls) -> None:
        from sglang.srt.environ import envs
        from sglang.test.test_utils import is_in_ci

        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
        ] + list(cls.extra_prefill_args)
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = _popen_pd_with_capture(
            model=cls.model,
            base_url=cls.prefill_url,
            other_args=prefill_args,
            env=None,
            stdout_buf=cls._prefill_stdout,
            stderr_buf=cls._prefill_stderr,
        )

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
        ] + list(cls.extra_decode_args)
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = _popen_pd_with_capture(
            model=cls.model,
            base_url=cls.decode_url,
            other_args=decode_args,
            env=None,
            stdout_buf=cls._decode_stdout,
            stderr_buf=cls._decode_stderr,
        )

    def _assert_sweep_violation_logged(
        self,
        kind_prefixes: Iterable[str],
        flush_wait_seconds: float = 2.0,
    ) -> None:
        """Assert at least one canary sweep_ violation appears in captured server output."""
        if flush_wait_seconds > 0:
            time.sleep(flush_wait_seconds)
        haystack = "".join([
            self._prefill_stdout.getvalue(),
            self._prefill_stderr.getvalue(),
            self._decode_stdout.getvalue(),
            self._decode_stderr.getvalue(),
        ])
        prefixes = list(kind_prefixes)
        hits = [p for p in prefixes if f"canary_kind:       {p}" in haystack]
        if hits:
            return
        excerpt = haystack[-2000:] if len(haystack) > 2000 else haystack
        self.fail(
            f"Expected a 'canary_kind: <kind>' line with kind in {prefixes} in "
            f"captured prefill/decode output, but none found. "
            f"Tail of combined output (last 2000 chars):\n{excerpt}"
        )

    def test_pd_transfer_corrupted_byte_detected(self) -> None:
        # Step 1: if perturb fired during warmup and the server crashed, violation is
        # already in the captured output — assert and return early.
        if self._launch_failed:
            self._assert_sweep_violation_logged(["sweep_"], flush_wait_seconds=2.0)
            return

        # Step 2: drive heavy traffic to maximize perturb trigger probability.
        _send_parallel_requests(
            self.lb_url, n=32, max_new_tokens=32, timeout=60.0
        )

        # Step 3: assert that the sweep path caught a real-KV byte corruption.
        self._assert_sweep_violation_logged(["sweep_"], flush_wait_seconds=2.0)


if __name__ == "__main__":
    unittest.main()
