"""
Test for Grafana Dashboard metric prefix consistency.

This test ensures that the Grafana dashboard JSON uses the correct
metric prefix (sglang_) that matches the /metrics API output.

Related Issue: https://github.com/sgl-project/sglang/issues/12618
"""

import json
import os
import re
import unittest


class TestGrafanaDashboard(unittest.TestCase):
    """Test suite for Grafana dashboard configuration."""

    DASHBOARD_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "monitoring",
        "grafana",
        "dashboards",
        "json",
        "sglang-dashboard.json",
    )

    def setUp(self):
        """Load the dashboard JSON file."""
        with open(self.DASHBOARD_PATH, "r") as f:
            self.dashboard = json.load(f)

    def test_dashboard_file_exists(self):
        """Test that the dashboard file exists."""
        self.assertTrue(
            os.path.exists(self.DASHBOARD_PATH),
            f"Dashboard file not found at {self.DASHBOARD_PATH}",
        )

    def test_no_old_metric_prefix(self):
        """Test that no old 'sglang:' prefix exists in the dashboard.

        The /metrics API changed from 'sglang:' to 'sglang_' prefix.
        All Prometheus queries should use the new prefix.
        """
        dashboard_str = json.dumps(self.dashboard)
        old_prefix_matches = re.findall(r"sglang:[a-z_]+", dashboard_str)

        self.assertEqual(
            len(old_prefix_matches),
            0,
            f"Found old 'sglang:' prefix in dashboard. "
            f"Matches: {old_prefix_matches}. "
            f"Please replace 'sglang:' with 'sglang_'.",
        )

    def test_new_metric_prefix_exists(self):
        """Test that the new 'sglang_' prefix is used in the dashboard."""
        dashboard_str = json.dumps(self.dashboard)
        new_prefix_matches = re.findall(r"sglang_[a-z_]+", dashboard_str)

        self.assertGreater(
            len(new_prefix_matches),
            0,
            "No 'sglang_' metrics found in dashboard. "
            "Dashboard should contain Prometheus queries with sglang_ prefix.",
        )

    def test_expected_metrics_present(self):
        """Test that expected metrics are present in the dashboard."""
        dashboard_str = json.dumps(self.dashboard)

        expected_metrics = [
            "sglang_e2e_request_latency_seconds",
            "sglang_time_to_first_token_seconds",
            "sglang_num_running_reqs",
            "sglang_gen_throughput",
            "sglang_cache_hit_rate",
            "sglang_num_queue_reqs",
        ]

        for metric in expected_metrics:
            self.assertIn(
                metric,
                dashboard_str,
                f"Expected metric '{metric}' not found in dashboard.",
            )

    def test_dashboard_is_valid_json(self):
        """Test that the dashboard is valid JSON with required fields."""
        self.assertIn("panels", self.dashboard)
        self.assertIn("title", self.dashboard)
        self.assertEqual(self.dashboard["title"], "SGLang Dashboard")

    def test_all_panels_have_targets(self):
        """Test that all panels have Prometheus targets defined."""
        panels = self.dashboard.get("panels", [])
        self.assertGreater(len(panels), 0, "Dashboard has no panels")

        for panel in panels:
            if panel.get("type") in ["timeseries", "heatmap"]:
                self.assertIn(
                    "targets",
                    panel,
                    f"Panel '{panel.get('title')}' has no targets",
                )
                self.assertGreater(
                    len(panel["targets"]),
                    0,
                    f"Panel '{panel.get('title')}' has empty targets",
                )


if __name__ == "__main__":
    unittest.main()
