"""Integration test: the EPD encoder server exports sglang:encoder_* metrics."""

import unittest
from typing import Dict, List

import requests
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from sglang.srt.utils import kill_process_tree
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
        try:
            health = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(health.status_code, 200)

            metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
            self.assertEqual(metrics_response.status_code, 200)
            metrics_text = metrics_response.text

            self.assertIn("sglang:encoder_model_forward_seconds", metrics_text)
            self.assertIn(f'model_name="{_MODEL_NAME}"', metrics_text)

            metrics = _parse_prometheus_metrics(metrics_text)
            forward_count = metrics.get(
                "sglang:encoder_model_forward_seconds_count", []
            )
            self.assertGreater(sum(s.value for s in forward_count), 0)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
