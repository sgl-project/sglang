"""Integration test: embedded Rust server exports frontend Prometheus metrics."""

import json
import time
import unittest
from typing import Dict, List

import requests
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_rust_server_built,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")


def _parse_prometheus_metrics(metrics_text: str) -> Dict[str, List[Sample]]:
    result: Dict[str, List[Sample]] = {}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            result.setdefault(sample.name, []).append(sample)
    return result


def _sum_samples(
    metrics: Dict[str, List[Sample]], name: str, labels: Dict[str, str]
) -> float:
    return sum(
        sample.value
        for sample in metrics.get(name, [])
        if all(sample.labels.get(k) == v for k, v in labels.items())
    )


def _all_samples_zero(metrics: Dict[str, List[Sample]], name: str) -> bool:
    return all(sample.value == 0 for sample in metrics.get(name, []))


@unittest.skipUnless(
    is_rust_server_built(),
    "embedded rust server extension not built",
)
class TestRustServerMetrics(CustomTestCase):
    def test_rust_server_metrics_exported(self):
        process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={
                "SGLANG_RUST_SERVER": "1",
                "SGLANG_USE_PICKLE_IPC": "0",
                "SGLANG_USE_AITER": "0",
            },
            other_args=[
                "--enable-metrics",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
                "--grammar-backend",
                "none",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--skip-server-warmup",
                "--mem-fraction-static",
                "0.2",
                "--dtype",
                "bfloat16",
            ],
        )
        self.addCleanup(kill_process_tree, process.pid)

        health = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(health.status_code, 200, health.text)

        unary = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
        )
        self.assertEqual(unary.status_code, 200, unary.text)

        stream = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "Today I learned",
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                "stream": True,
            },
            stream=True,
        )
        self.assertEqual(stream.status_code, 200)
        for line in stream.iter_lines():
            if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                json.loads(line[6:])

        deadline = time.time() + 10
        metrics_text = ""
        metrics: Dict[str, List[Sample]] = {}
        while True:
            metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
            self.assertEqual(metrics_response.status_code, 200)
            self.assertIn("text/plain", metrics_response.headers["content-type"])
            metrics_text = metrics_response.text
            metrics = _parse_prometheus_metrics(metrics_text)
            if _all_samples_zero(
                metrics, "sglang:http_requests_active"
            ) and _all_samples_zero(metrics, "sglang:rust_server_inflight_requests"):
                break
            if time.time() >= deadline:
                break
            time.sleep(0.2)

        for name in [
            "sglang:http_requests_total",
            "sglang:http_responses_total",
            "sglang:http_requests_active",
            "sglang:rust_server_requests_total",
            "sglang:rust_server_inflight_requests",
            "sglang:rust_server_ingress_ring_push_total",
            "sglang:rust_server_egress_frames_total",
            "sglang:rust_server_ring_capacity",
            "sglang:rust_server_threads",
        ]:
            self.assertIn(name, metrics_text, f"Missing metric: {name}")

        self.assertGreaterEqual(
            _sum_samples(
                metrics,
                "sglang:rust_server_requests_total",
                {
                    "kind": "generate",
                    "input_source": "text",
                    "stream": "false",
                },
            ),
            1,
        )
        self.assertGreaterEqual(
            _sum_samples(
                metrics,
                "sglang:rust_server_requests_total",
                {
                    "kind": "generate",
                    "input_source": "text",
                    "stream": "true",
                },
            ),
            1,
        )
        self.assertGreaterEqual(
            _sum_samples(
                metrics,
                "sglang:rust_server_requests_total",
                {
                    "kind": "health_generate",
                    "input_source": "input_ids",
                    "stream": "false",
                },
            ),
            1,
        )
        self.assertGreaterEqual(
            _sum_samples(
                metrics,
                "sglang:http_requests_total",
                {"endpoint": "/generate", "method": "POST"},
            ),
            2,
        )

        for sample in metrics.get("sglang:http_requests_active", []):
            self.assertEqual(sample.value, 0, sample)
        for sample in metrics.get("sglang:rust_server_inflight_requests", []):
            self.assertEqual(sample.value, 0, sample)

        self.assertNotIn('rid="', metrics_text)


if __name__ == "__main__":
    unittest.main()
