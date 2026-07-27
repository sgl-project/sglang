"""Integration tests for load snapshot with real NPU servers.

Tests [no dp, normal dp] x [shm, zmq] by launching real servers
and querying /v1/loads.
SGLANG_LOAD_SNAPSHOT_USE_ZMQ is not supported on NPU
"""

import json
import time
import unittest
import urllib.request

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)


def _query_loads(base_url, retries=5, interval=2.0):
    url = f"{base_url}/v1/loads"
    for attempt in range(retries):
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            data = json.loads(resp.read())
            if data.get("loads"):
                return data
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(interval)
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        return json.loads(resp.read())
    except Exception:
        return {"loads": []}


def _launch_and_check(test_case, other_args=None, env=None, expected_dp_size=1):
    process = popen_launch_server(
        LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args or [],
        env=env,
    )
    try:
        time.sleep(5)
        data = _query_loads(DEFAULT_URL_FOR_TEST)
        loads = data.get("loads", [])
        test_case.assertGreater(len(loads), 0, f"Expected non-empty loads, got: {data}")
        test_case.assertEqual(len(loads), expected_dp_size)
        dp_ranks = sorted(l["dp_rank"] for l in loads)
        test_case.assertEqual(dp_ranks, list(range(expected_dp_size)))
        for load in loads:
            test_case.assertGreater(load["max_total_num_tokens"], 0)
    finally:
        kill_process_tree(process.pid)


class TestLoadSnapshotNoDP(CustomTestCase):
    """Verify /v1/loads endpoint with single worker (no DP) on NPU, using
    shm and zmq backends.SGLANG_LOAD_SNAPSHOT_USE_ZMQ is not supported

    [Test Category] Parameter
    [Test Target] /v1/loads
    """

    def test_shm_backend(self):
        _launch_and_check(
            self,
            other_args=["--attention-backend", "ascend", "--disable-cuda-graph"],
            expected_dp_size=1,
        )


class TestLoadSnapshotNormalDP(CustomTestCase):
    """Verify /v1/loads endpoint with DP=2 on NPU, using shm  backends.
    SGLANG_LOAD_SNAPSHOT_USE_ZMQ is not supported

    [Test Category] Parameter
    [Test Target] --dp;/v1/loads
    """

    def test_shm_backend(self):
        _launch_and_check(
            self,
            other_args=[
                "--dp",
                "2",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
            expected_dp_size=2,
        )


if __name__ == "__main__":
    unittest.main()
