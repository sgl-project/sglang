"""Explorer checks use small recordings, without GPUs or external services."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import explore
from metrics import counter_rate, hit_percentages, metric_values, occupancy


def counters(device=0, host=0, storage=0, uncached=0, rank="0"):
    return metric_values(
        "\n".join(
            f'sglang:prefill_effective_tokens_total{{mode="{mode}",dp_rank="{rank}"}} {value}'
            for mode, value in zip(
                ("device_hit", "host_hit", "storage_hit", "input"),
                (device, host, storage, uncached),
            )
        )
    )


class MetricTests(unittest.TestCase):
    def test_occupancy_includes_evictable_and_matches_labels(self):
        samples = metric_values("""
sglang:kv_used_tokens{rank="0"} 20
sglang:kv_evictable_tokens{rank="0"} 30
sglang:max_total_num_tokens{rank="0"} 100
sglang:hicache_host_used_tokens{rank="0"} 80
sglang:hicache_host_total_tokens{rank="0"} 200
""")
        names = ("kv_used_tokens", "kv_evictable_tokens", "max_total_num_tokens")
        self.assertEqual(occupancy(samples, names), [20, 30, 100, 50])
        self.assertEqual(
            occupancy(
                samples, ("hicache_host_used_tokens", "hicache_host_total_tokens")
            ),
            [80, 200, 40],
        )
        samples.update(metric_values('sglang:kv_used_tokens{rank="1"} 1'))
        self.assertEqual(occupancy(samples, names), [None] * 4)

    def test_hits_include_storage_and_uncached_denominator(self):
        self.assertEqual(
            hit_percentages(counters(), counters(40, 20, 10, 30)), [40, 20, 70]
        )
        self.assertEqual(
            hit_percentages(counters(), counters(80, 0, 0, 20)), [80, 0, 80]
        )
        self.assertEqual(hit_percentages(counters(), counters()), [None] * 3)
        missing = counters(80, 0, 0, 20)
        missing.pop(next(k for k in missing if dict(k[1])["mode"] == "host_hit"))
        self.assertEqual(hit_percentages(counters(), missing), [None] * 3)

    def test_per_label_reset_cannot_be_hidden_by_another_worker(self):
        before = counters(20) | counters(20, rank="1")
        after = counters(19) | counters(100, rank="1")
        self.assertEqual(hit_percentages(before, after), [None] * 3)
        self.assertIsNone(
            counter_rate(before, after, "sglang:prefill_effective_tokens_total", 1)
        )
        self.assertIsNone(
            counter_rate({}, after, "sglang:prefill_effective_tokens_total", 1)
        )

    def test_scrape_error_breaks_rates(self):
        records = [
            {"timestamp": 0, "text": "sglang:generation_tokens_total 1"},
            {"timestamp": 1, "error": "timeout"},
            {"timestamp": 2, "text": "sglang:generation_tokens_total 9"},
        ]
        panels = explore.panels_for([], records, 0, 3)
        rate = next(p for p in panels if p["title"] == "Output Throughput")
        self.assertEqual(rate["series"][0]["points"], [[1, None], [2, None]])

    def test_nonfinite_missing_and_zero_capacity(self):
        self.assertEqual(metric_values("sglang:kv_used_tokens NaN"), {})
        self.assertEqual(occupancy({}, ("a", "b")), [None] * 3)
        self.assertEqual(
            occupancy(metric_values("sglang:a 1\nsglang:b 0"), ("a", "b")), [1, 0, None]
        )


class RecordingTests(unittest.TestCase):
    def test_old_failed_recording_and_offline_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "manifest.json").write_text(
                json.dumps(
                    {
                        "started_at": 10,
                        "status": "failed",
                        "finished_at": 15,
                        "server_info": {"dp_size": 1},
                        "arguments": {
                            "tokenizer": "</script><script>alert(1)</script>"
                        },
                    }
                )
            )
            (path / "requests.jsonl").write_text(
                json.dumps(
                    {
                        "conversation": 7,
                        "turn": 2,
                        "submitted_at": 11,
                        "failed_at": 14,
                        "ttft_s": 0.5,
                        "first_token_at": 11.5,
                        "events": [[0.5, 1], [1, 2]],
                        "error": "disconnected",
                        "meta_info": {"completion_tokens": 2},
                    }
                )
                + "\n"
            )
            (path / "metrics.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "timestamp": 12,
                            "url": url,
                            "text": f"sglang:num_running_reqs {n}",
                        }
                    )
                    for url, n in (("one", 2), ("two", 3))
                )
            )
            output = path / "view.html"
            data = explore.build(path, output)
            row = data["rows"][0]
            self.assertEqual(row["worker"], 0)
            self.assertEqual(row["unavailable"], ["Tool call", "Wait"])
            self.assertEqual(
                row["phases"], [{"type": "Sampling", "start": 1, "end": 4}]
            )
            self.assertEqual(data["summary"]["errors"], 1)
            self.assertEqual(set(data["exporters"]), {"one", "two"})
            for url, expected in (("one", 2), ("two", 3)):
                panels = data["exporters"][url]
                self.assertEqual(len(panels), 15)
                running = next(
                    p for p in panels if p["title"] == "Running And Queued Requests"
                )
                self.assertEqual(running["series"][0]["points"], [[2, expected]])
                ttft = next(p for p in panels if p["title"] == "TTFT")
                self.assertTrue(any(v == 500 for _, v in ttft["series"][0]["points"]))
            html = output.read_text()
            self.assertNotIn("</script><script>alert", html)
            self.assertNotIn("<script src", html)
            self.assertNotIn("<link", html)
            self.assertNotIn("fetch(", html)
            self.assertNotIn("__REAL_DATA__", html)
            self.assertIn("Timeline", html)
            self.assertIn("Engine Metrics", html)

    def test_interrupted_empty_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "manifest.json").write_text('{"started_at":1,"status":"running"}')
            (path / "requests.jsonl").write_text('{"conversation":')
            data = explore.load_run(path)
            self.assertEqual(data["summary"]["status"], "incomplete")
            self.assertEqual(data["summary"]["errors"], 1)
            self.assertEqual(data["rows"][0]["phases"], [])
            self.assertIsNone(data["rows"][0]["worker"])


if __name__ == "__main__":
    unittest.main()
