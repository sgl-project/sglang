import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_trace_slo.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("pd_flip_trace_slo_reset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class FakeClient:
    def get_json(self, base_url, path):
        assert path == "/v1/loads?include=all"
        return {"num_running_reqs": 1, "num_waiting_reqs": 0}


def append_record(path, **record):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record) + "\n")


class TraceSLOResetTest(unittest.TestCase):
    def test_reset_window_excludes_old_records_without_truncating_ledger(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "trace_slo_ledger.jsonl"
            clock = Clock(100.0)
            monitor = module.TraceSLOMonitor(
                ledger_path=str(ledger),
                window_seconds=30,
                client=FakeClient(),
                time_fn=clock,
            )
            nodes = [
                ("node0", "http://node0", "prefill"),
                ("node1", "http://node1", "decode"),
            ]

            append_record(
                ledger,
                request_id="old",
                event_time=100.0,
                ttft_seconds=9.0,
                ttft_slo_seconds=8.0,
                good_tpot_intervals=0,
                total_tpot_intervals=1,
            )
            clock.value = 101.0
            monitor.reset_window()
            append_record(
                ledger,
                request_id="new",
                event_time=102.0,
                ttft_seconds=1.0,
                ttft_slo_seconds=8.0,
                good_tpot_intervals=2,
                total_tpot_intervals=2,
            )
            clock.value = 103.0

            snapshot = monitor.collect_cluster(nodes)

            self.assertEqual(snapshot.prefill_counts.total, 1)
            self.assertEqual(snapshot.prefill_counts.good, 1)
            self.assertEqual(snapshot.decode_counts.total, 2)
            self.assertEqual(snapshot.decode_counts.good, 2)
            self.assertEqual(
                len(ledger.read_text(encoding="utf-8").splitlines()), 2
            )


if __name__ == "__main__":
    unittest.main()
