"""Integration test: the EPD encoder server exports sglang:encoder_* metrics."""

import unittest
import uuid
from typing import Dict, List
from urllib.parse import urlparse

import requests
import zmq
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from sglang.srt.disaggregation.encode_http_server import MINIMUM_PNG_PICTURE_BASE64
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_zmq_socket_on_host
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

_MODEL_NAME = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST


def _parse_prometheus_metrics(metrics_text: str) -> Dict[str, List[Sample]]:
    result: Dict[str, List[Sample]] = {}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            result.setdefault(sample.name, []).append(sample)
    return result


class TestEncoderServerMetrics(CustomTestCase):
    def test_encoder_metrics_exported(self):
        process = popen_launch_server(
            _MODEL_NAME,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--encoder-only",
                "--trust-remote-code",
                "--enable-metrics",
                "--tp",
                "1",
            ],
        )
        self.addCleanup(kill_process_tree, process.pid)
        base_host = urlparse(DEFAULT_URL_FOR_TEST).hostname
        context = zmq.Context()
        recv_port, recv_socket = get_zmq_socket_on_host(
            context, zmq.PULL, host=base_host
        )
        try:
            health = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(health.status_code, 200)

            req_id = f"metrics-probe-{uuid.uuid4().hex}"
            requests.post(
                f"{DEFAULT_URL_FOR_TEST}/scheduler_receive_url",
                json={
                    "req_id": req_id,
                    "receive_url": f"{base_host}:{recv_port}",
                    "receive_count": 1,
                },
            )
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/encode",
                json={
                    "req_id": req_id,
                    "modality": "IMAGE",
                    "mm_items": [f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"],
                    "num_parts": 1,
                    "part_idx": 0,
                    "embedding_port": None,
                },
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            )
            self.assertEqual(response.status_code, 200)

            metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
            self.assertEqual(metrics_response.status_code, 200)
            metrics_text = metrics_response.text

            self.assertIn("sglang:encoder_requests_received_total", metrics_text)
            self.assertIn(f'model_name="{_MODEL_NAME}"', metrics_text)

            metrics = _parse_prometheus_metrics(metrics_text)
            received = metrics.get("sglang:encoder_requests_received_total", [])
            self.assertGreater(sum(s.value for s in received), 0)
        finally:
            recv_socket.close()
            context.term()


if __name__ == "__main__":
    unittest.main()
