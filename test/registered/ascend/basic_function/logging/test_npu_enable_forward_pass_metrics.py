import os
import time
import unittest
from pathlib import Path

import msgpack
import requests
import zmq

from sglang.test.ascend.test_ascend_utils import logger
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=120, suite="full-1-npu-a3", nightly=True)


class TestNPUMetricsMFUEnabled(TestNPULoggingBase):
    """Testcase: Verify forward-pass metrics emission over ZMQ IPC with configured worker_id.

    [Test Category] Parameter
    [Test Target] --enable-forward-pass-metrics, --forward-pass-metrics-worker-id, --forward-pass-metrics-ipc-name
    """

    # -----------------------------
    # Constants
    # -----------------------------
    IPC_NAME_PREFIX = f"/tmp/sglang-test-fwd-metrics-{os.getpid()}"
    IPC_ENDPOINT = f"ipc://{IPC_NAME_PREFIX}"
    IPC_SOCKET_PATH = Path(f"{IPC_NAME_PREFIX}.0")
    IPC_SUBSCRIBE_ENDPOINT = f"{IPC_ENDPOINT}.0"

    ZMQ_RCV_TIMEOUT_MS = 5000
    METRIC_RECV_TIMEOUT_SEC = 20
    FPM_SAMPLE_COUNT = 10

    EXPECTED_WORKER_ID = "should-be-overridden"

    METRICS_ARGS = [
        "--enable-forward-pass-metrics",
        "--forward-pass-metrics-worker-id",
        EXPECTED_WORKER_ID,
        "--forward-pass-metrics-ipc-name",
        IPC_ENDPOINT,
    ]

    # -----------------------------
    # Class-level state
    # -----------------------------
    zmq_ctx = None
    zmq_sub = None

    # -----------------------------
    # Setup / Teardown
    # -----------------------------
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.other_args.extend(cls.METRICS_ARGS)
        cls.launch_server()

        cls.zmq_ctx = zmq.Context()
        cls.zmq_sub = cls.zmq_ctx.socket(zmq.SUB)
        cls.zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        cls.zmq_sub.setsockopt(zmq.RCVTIMEO, cls.ZMQ_RCV_TIMEOUT_MS)
        cls.zmq_sub.connect(cls.IPC_SUBSCRIBE_ENDPOINT)

        logger.info(
            "ZMQ SUB connected to %s (timeout=%dms)",
            cls.IPC_SUBSCRIBE_ENDPOINT,
            cls.ZMQ_RCV_TIMEOUT_MS,
        )

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.zmq_sub is not None:
                cls.zmq_sub.close(linger=0)
            if cls.zmq_ctx is not None:
                cls.zmq_ctx.term()

            try:
                cls.IPC_SOCKET_PATH.unlink(missing_ok=True)
                logger.info("Cleaned up IPC socket: %s", cls.IPC_SOCKET_PATH)
            except Exception as e:
                logger.warning(
                    "Could not remove IPC socket %s: %s", cls.IPC_SOCKET_PATH, e
                )
        finally:
            super().tearDownClass()

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _is_msgpack_map(first_byte: int) -> bool:
        return 0x80 <= first_byte <= 0x8F or first_byte == 0xDE or first_byte == 0xDF

    def _recv_fpm_metric(self, timeout=METRIC_RECV_TIMEOUT_SEC):
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                frames = self.zmq_sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.02)
                continue

            for frame in frames:
                if not frame or not self._is_msgpack_map(frame[0]):
                    continue

                try:
                    metric = msgpack.unpackb(frame, raw=False)
                    logger.info("FPM metric decoded: %s", metric)
                    return metric
                except Exception as e:
                    logger.info("MsgPack decode failed: %s", e)

        self.fail(
            f"No forward-pass metric received on "
            f"{self.IPC_SOCKET_PATH} within {timeout}s"
        )

    # -----------------------------
    # Test case
    # -----------------------------
    def test_forward_pass_metrics_all_args_configured(self):
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        self.assertIn(f"FPM: ZMQ PUB bound on {self.IPC_SUBSCRIBE_ENDPOINT}", content)

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": ["The capital of France is"] * 2,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
                "stream": False,
                "ignore_eos": True,
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200, response.text)

        seen_worker_ids = []
        last_metric = None

        for _ in range(self.FPM_SAMPLE_COUNT):
            last_metric = self._recv_fpm_metric(timeout=2)
            seen_worker_ids.append(last_metric.get("worker_id"))

        # -----------------------------
        # Core assertions
        # -----------------------------
        self.assertTrue(
            self.IPC_SOCKET_PATH.is_socket(),
            "Forward-pass metrics IPC socket should exist",
        )

        self.assertIsInstance(last_metric, dict)
        self.assertIn("worker_id", last_metric)

        worker_id = last_metric["worker_id"]
        self.assertIsInstance(worker_id, str)
        self.assertTrue(worker_id)
        self.assertEqual(
            worker_id,
            self.EXPECTED_WORKER_ID,
            "worker_id should match the configured value on Ascend/NPU",
        )

        self.assertTrue(
            all(wid == self.EXPECTED_WORKER_ID for wid in seen_worker_ids),
            f"worker_id should be stable across samples; seen: {seen_worker_ids}",
        )

        self.assertIn(
            "counter_id",
            last_metric,
            "FPM must contain counter_id as per-step identifier on Ascend/NPU",
        )

        counter_id = last_metric["counter_id"]
        self.assertIsInstance(counter_id, int)
        self.assertGreaterEqual(counter_id, 0)


if __name__ == "__main__":
    unittest.main()
