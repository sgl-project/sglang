import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_monitor.py"
)


def load_monitor_module():
    spec = importlib.util.spec_from_file_location("pd_flip_monitor", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(self):
        self.metrics = {
            "http://node0": self._metrics(ttft_good=8, ttft_total=10, tpot_good=0, tpot_total=0),
            "http://node1": self._metrics(ttft_good=9, ttft_total=10, tpot_good=0, tpot_total=0),
            "http://node2": self._metrics(ttft_good=0, ttft_total=0, tpot_good=19, tpot_total=20),
        }
        self.loads = {
            "http://node0": {"loads": [{"num_running_reqs": 1, "num_waiting_reqs": 2, "num_total_tokens": 128, "token_usage": 0.1}]},
            "http://node1": {"loads": [{"num_running_reqs": 2, "num_waiting_reqs": 0, "num_total_tokens": 256, "token_usage": 0.2}]},
            "http://node2": {"loads": [{"num_running_reqs": 3, "num_waiting_reqs": 1, "num_total_tokens": 512, "token_usage": 0.4}]},
        }

    def get_text(self, base_url, path):
        if path != "/metrics":
            raise AssertionError(path)
        return self.metrics[base_url]

    def get_json(self, base_url, path):
        if path != "/v1/loads?include=all":
            raise AssertionError(path)
        return self.loads[base_url]

    @staticmethod
    def _metrics(ttft_good, ttft_total, tpot_good, tpot_total):
        return f"""
sglang:time_to_first_token_seconds_bucket{{le="0.2"}} {ttft_good}
sglang:time_to_first_token_seconds_bucket{{le="+Inf"}} {ttft_total}
sglang:inter_token_latency_seconds_bucket{{le="0.02"}} {tpot_good}
sglang:inter_token_latency_seconds_bucket{{le="+Inf"}} {tpot_total}
"""


class TestPDFlipMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = load_monitor_module()

    def test_parse_histogram_counts_computes_attainment(self):
        metrics = """
sglang:time_to_first_token_seconds_bucket{le="0.1"} 2
sglang:time_to_first_token_seconds_bucket{le="0.2"} 7
sglang:time_to_first_token_seconds_bucket{le="+Inf"} 10
"""

        counts = self.monitor.parse_histogram_counts(
            metrics, "sglang:time_to_first_token_seconds", 0.2
        )

        self.assertEqual(counts.good, 7)
        self.assertEqual(counts.total, 10)
        self.assertEqual(counts.attainment, 0.7)

    def test_window_aggregates_role_specific_attainment(self):
        window = self.monitor.SLOWindow(window_seconds=10.0)
        window.add(
            self.monitor.NodeSLOSample(
                timestamp=1.0,
                name="node0",
                role="prefill",
                ttft=self.monitor.SampleCounts(good=8, total=10),
            )
        )
        window.add(
            self.monitor.NodeSLOSample(
                timestamp=1.5,
                name="node1",
                role="decode",
                tpot=self.monitor.SampleCounts(good=18, total=20),
            )
        )

        snapshot = window.snapshot(timestamp=2.0)

        self.assertEqual(snapshot.prefill_nodes, 1)
        self.assertEqual(snapshot.decode_nodes, 1)
        self.assertEqual(snapshot.prefill_slo_attainment, 0.8)
        self.assertEqual(snapshot.decode_slo_attainment, 0.9)

    def test_monitor_collects_cluster_snapshot_from_workers(self):
        monitor = self.monitor.PDFlipSLOMonitor(
            ttft_slo_seconds=0.2,
            tpot_slo_seconds=0.02,
            window_seconds=30.0,
            client=FakeClient(),
            time_fn=lambda: 10.0,
        )

        snapshot = monitor.collect_cluster(
            [
                ("node0", "http://node0", "prefill"),
                ("node1", "http://node1", "prefill"),
                ("node2", "http://node2", "decode"),
            ]
        )

        self.assertEqual(snapshot.prefill_nodes, 2)
        self.assertEqual(snapshot.decode_nodes, 1)
        self.assertEqual(snapshot.prefill_slo_attainment, 17 / 20)
        self.assertEqual(snapshot.decode_slo_attainment, 19 / 20)
        by_name = {sample.name: sample for sample in snapshot.nodes}
        self.assertEqual(by_name["node2"].running_reqs, 3)
        self.assertEqual(by_name["node2"].waiting_reqs, 1)
        self.assertEqual(by_name["node2"].token_usage, 0.4)


if __name__ == "__main__":
    unittest.main()
