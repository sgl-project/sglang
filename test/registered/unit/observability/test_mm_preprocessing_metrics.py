# SPDX-License-Identifier: Apache-2.0
import threading
import unittest
from unittest import mock

from prometheus_client import CollectorRegistry

from sglang.srt.observability import mm_preprocessing_metrics as mpm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMMPreprocessingMetrics(unittest.TestCase):
    """Unit tests for the multimodal preprocessing Prometheus metrics."""

    def setUp(self):
        self.registry = CollectorRegistry()
        mpm._reset_for_test(enabled=True, registry=self.registry)

    def tearDown(self):
        mpm._reset_for_test(enabled=False)

    def test_observe_media_download_records_latency_and_bytes(self):
        mpm.observe_mm_media_download("image", 0.5, 2048)
        mpm.observe_mm_media_download("image", 1.5, 4096)

        self.assertEqual(
            self.registry.get_sample_value(
                "sglang:mm_media_download_seconds_count", {"modality": "image"}
            ),
            2.0,
        )
        self.assertAlmostEqual(
            self.registry.get_sample_value(
                "sglang:mm_media_download_seconds_sum", {"modality": "image"}
            ),
            2.0,
        )
        self.assertEqual(
            self.registry.get_sample_value(
                "sglang:mm_media_download_bytes_count", {"modality": "image"}
            ),
            2.0,
        )
        self.assertEqual(
            self.registry.get_sample_value(
                "sglang:mm_media_download_bytes_sum", {"modality": "image"}
            ),
            6144.0,
        )

    def test_observe_media_load_per_modality_labels(self):
        mpm.observe_mm_media_load("image", 0.1)
        mpm.observe_mm_media_load("video", 2.0)
        mpm.observe_mm_media_load("audio", 0.05)

        for modality, expected_sum in (
            ("image", 0.1),
            ("video", 2.0),
            ("audio", 0.05),
        ):
            self.assertEqual(
                self.registry.get_sample_value(
                    "sglang:mm_media_load_seconds_count", {"modality": modality}
                ),
                1.0,
            )
            self.assertAlmostEqual(
                self.registry.get_sample_value(
                    "sglang:mm_media_load_seconds_sum", {"modality": modality}
                ),
                expected_sum,
            )
        # No cross-label contamination.
        self.assertIsNone(
            self.registry.get_sample_value(
                "sglang:mm_media_load_seconds_count", {"modality": "text"}
            )
        )

    def test_observe_load_data_and_processor(self):
        mpm.observe_mm_load_data(0.25)
        mpm.observe_mm_load_data(0.75)
        mpm.observe_mm_processor(1.25)

        self.assertEqual(
            self.registry.get_sample_value("sglang:mm_load_data_seconds_count"), 2.0
        )
        self.assertAlmostEqual(
            self.registry.get_sample_value("sglang:mm_load_data_seconds_sum"), 1.0
        )
        self.assertEqual(
            self.registry.get_sample_value("sglang:mm_processor_seconds_count"), 1.0
        )
        self.assertAlmostEqual(
            self.registry.get_sample_value("sglang:mm_processor_seconds_sum"), 1.25
        )

    def test_disabled_metrics_are_noop(self):
        mpm._reset_for_test(enabled=False)

        # Must not raise, and must not touch the registry.
        mpm.observe_mm_media_download("image", 1.0, 10)
        mpm.observe_mm_media_load("image", 1.0)
        mpm.observe_mm_load_data(1.0)
        mpm.observe_mm_processor(1.0)

        self.assertIsNone(
            self.registry.get_sample_value(
                "sglang:mm_media_load_seconds_count", {"modality": "image"}
            )
        )
        # Label-less histograms create their timeseries at construction time
        # (during setUp), so the count exists but must stay at 0.
        self.assertEqual(
            self.registry.get_sample_value("sglang:mm_processor_seconds_count"), 0.0
        )

    def test_concurrent_observations_are_thread_safe(self):
        n_threads, n_obs = 8, 50

        def worker():
            for _ in range(n_obs):
                mpm.observe_mm_media_load("image", 0.01)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(
            self.registry.get_sample_value(
                "sglang:mm_media_load_seconds_count", {"modality": "image"}
            ),
            float(n_threads * n_obs),
        )

    def test_metrics_turned_on_reflects_server_args(self):
        fake_args = mock.Mock(enable_metrics=True)
        with mock.patch(
            "sglang.srt.runtime_context.get_server_args", return_value=fake_args
        ):
            self.assertTrue(mpm._metrics_turned_on())

        fake_args.enable_metrics = False
        with mock.patch(
            "sglang.srt.runtime_context.get_server_args", return_value=fake_args
        ):
            self.assertFalse(mpm._metrics_turned_on())

        # Server args not published (unit tests, offline tools) -> disabled,
        # no exception.
        with mock.patch(
            "sglang.srt.runtime_context.get_server_args",
            side_effect=ValueError("Global server args is not set yet!"),
        ):
            self.assertFalse(mpm._metrics_turned_on())


if __name__ == "__main__":
    unittest.main()
