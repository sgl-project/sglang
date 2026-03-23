import ast
import unittest
from pathlib import Path


class TestKvTransferMetrics(unittest.TestCase):
    def test_scheduler_metrics_collector_preinitializes_kv_transfer_metrics(self):
        source_path = (
            Path(__file__).resolve().parents[3]
            / "python"
            / "sglang"
            / "srt"
            / "observability"
            / "metrics_collector.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        collector_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SchedulerMetricsCollector"
        )
        init_fn = next(
            node
            for node in collector_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        helper_fn = next(
            node
            for node in collector_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_initialize_kv_transfer_metric_labels"
        )

        self.assertTrue(
            any(
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and isinstance(stmt.value.func.value, ast.Name)
                and stmt.value.func.value.id == "self"
                and stmt.value.func.attr == "_initialize_kv_transfer_metric_labels"
                for stmt in init_fn.body
            )
        )

        helper_source = ast.unparse(helper_fn)
        expected_metrics = [
            "num_decode_transfer_queue_reqs",
            "num_decode_prealloc_queue_reqs",
            "num_prefill_prealloc_queue_reqs",
            "num_prefill_inflight_queue_reqs",
            "num_bootstrap_failed_reqs",
            "num_transfer_failed_reqs",
            "num_prefill_retries_total",
            "kv_transfer_speed_gb_s",
            "kv_transfer_latency_ms",
            "kv_transfer_bootstrap_ms",
            "kv_transfer_alloc_ms",
            "kv_transfer_total_mb",
        ]
        for metric_name in expected_metrics:
            self.assertIn(metric_name, helper_source)


if __name__ == "__main__":
    unittest.main()
