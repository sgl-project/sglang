from __future__ import annotations

import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import ClassVar, Dict, List

import requests

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.violation_log_utils import assert_no_violation_in_log
from sglang.test.mock_model.utils import (
    MOCK_MODEL_PATH,
    mock_model_server_args,
    mock_model_server_env,
)
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=170, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=165, stage="extra-a", runner_config="2-gpu-large-amd")

# DO NOT pass --disable-cuda-graph in canary e2e tests.  The canary kernel
# must run inside the cuda graph alongside the real attn kernel; disabling the
# full graph silently bypasses the only path that exercises that invariant
# end-to-end.
#
# --disable-piecewise-cuda-graph is REQUIRED by canary: install_canary
# (api.py) asserts it, and the SingleForwardManager design depends on it.
# mock_model_server_args() already passes it; do not remove it.
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


class _MockModelPDBase(PDDisaggregationServerBase):
    """PD fixture for mock-model + canary e2e tests."""

    capture_per_side_logs = True
    model: ClassVar[str] = MOCK_MODEL_PATH
    extra_prefill_args: ClassVar[List[str]] = mock_model_server_args(
        "--skip-server-warmup"
    )
    extra_decode_args: ClassVar[List[str]] = mock_model_server_args(
        "--skip-server-warmup"
    )
    extra_prefill_env: ClassVar[Dict[str, str]] = mock_model_server_env(
        input_check_enabled=True
    )
    extra_decode_env: ClassVar[Dict[str, str]] = mock_model_server_env(
        input_check_enabled=True
    )

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.launch_all()

    def assert_no_canary_violation(self) -> None:
        time.sleep(2)
        log_text = "".join(
            buf.getvalue()
            for buf in (
                self._prefill_stdout_buf,
                self._prefill_stderr_buf,
                self._decode_stdout_buf,
                self._decode_stderr_buf,
            )
            if buf is not None
        )
        assert_no_violation_in_log(log_text)


class TestPdTransferCanaryClean(_MockModelPDBase, unittest.TestCase):
    """PD standard scenario + baseline canary (input-check, no real-KV checksum); no violation expected."""

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


@unittest.skipIf(
    is_hip(),
    "ROCm: PD full-real-data KV checksum intermittently trips a "
    "verify_real_kv_hash canary violation on the decode-side transferred prefix "
    "(see https://github.com/sgl-project/sglang/issues/28971). The baseline PD "
    "canary test above stays enabled on AMD.",
)
class TestPdTransferChecksumFullRealData(_MockModelPDBase, unittest.TestCase):
    """--kv-canary-real-data=all + sweep every step, no perturb, no violation."""

    extra_prefill_args: ClassVar[List[str]] = mock_model_server_args(
        "--skip-server-warmup",
        "--kv-canary-real-data",
        "all",
        "--kv-canary-sweep-interval",
        "1",
    )
    extra_decode_args: ClassVar[List[str]] = mock_model_server_args(
        "--skip-server-warmup",
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
